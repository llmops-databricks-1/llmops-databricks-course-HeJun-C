"""Local integration for a Databricks-managed Genie MCP server.

This module lets local Python code call a Genie Space via the Databricks MCP
server URL, so other modules (and future agents) can reuse it as a "tool".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
import time
import os
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient


def _space_id_from_server_url(server_url: str) -> str:
    path = urlparse(server_url).path.rstrip("/")
    space_id = path.split("/")[-1]
    if not space_id:
        raise ValueError("Could not parse space_id from server_url")
    return space_id


def _joined_text(mcp_response) -> str:
    # databricks_mcp responses contain a list of content blocks with .text
    return "".join(getattr(c, "text", "") for c in getattr(mcp_response, "content", []))


def _best_effort_parse(text: str) -> dict:
    text = text.strip()
    if not text:
        return {"raw_text": ""}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        return {"data": obj, "raw_text": text}
    except Exception:
        return {"raw_text": text}


def _get_first(d: dict, *keys: str) -> str | None:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v:
            return v
    return None


def _extract_conversation_and_message_ids(response_dict: dict) -> tuple[str | None, str | None]:
    conversation_id = _get_first(response_dict, "conversation_id", "conversationId")
    message_id = _get_first(response_dict, "message_id", "messageId")
    return conversation_id, message_id


def _workspace_client(
    *,
    profile: str,
    host: str | None = None,
    token: str | None = None,
) -> WorkspaceClient:
    # Prefer explicit host/token if provided (avoids local CLI OAuth cache issues).
    if host and token:
        return WorkspaceClient(host=host, token=token)
    return WorkspaceClient(profile=profile)


@dataclass(frozen=True)
class GenieMcpConfig:
    server_url: str
    profile: str = "llmops-course"
    host: str | None = None
    token: str | None = None

    @staticmethod
    def from_env(
        *,
        server_url_env: str = "GENIE_MCP_SERVER_URL",
        profile_env: str = "DATABRICKS_CONFIG_PROFILE",
        host_env: str = "DATABRICKS_HOST",
        token_env: str = "DATABRICKS_TOKEN",
    ) -> "GenieMcpConfig":
        server_url = os.environ.get(server_url_env, "").strip()
        if not server_url:
            raise ValueError(f"Missing {server_url_env} in environment")
        profile = os.environ.get(profile_env, "llmops-course").strip() or "llmops-course"
        host = (os.environ.get(host_env) or "").strip() or None
        token = (os.environ.get(token_env) or "").strip() or None
        return GenieMcpConfig(server_url=server_url, profile=profile, host=host, token=token)


class GenieMcpClient:
    def __init__(self, config: GenieMcpConfig):
        self.config = config
        self.space_id = _space_id_from_server_url(config.server_url)

        w = _workspace_client(profile=config.profile, host=config.host, token=config.token)
        self._mcp = DatabricksMCPClient(server_url=config.server_url, workspace_client=w)

        self.query_tool = f"query_space_{self.space_id}"
        self.poll_tool = f"poll_response_{self.space_id}"

    def list_tools(self) -> list[dict]:
        tools = self._mcp.list_tools()
        out: list[dict] = []
        for t in tools:
            out.append(
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.inputSchema,
                }
            )
        return out

    def query(self, query: str, conversation_id: str | None = None) -> dict:
        payload: dict[str, str] = {"query": query}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        resp = self._mcp.call_tool(self.query_tool, payload)
        text = _joined_text(resp)
        data = _best_effort_parse(text)
        return {
            "tool": self.query_tool,
            "space_id": self.space_id,
            "request": payload,
            "response": data,
        }

    def poll(self, *, conversation_id: str, message_id: str) -> dict:
        payload = {"conversation_id": conversation_id, "message_id": message_id}
        resp = self._mcp.call_tool(self.poll_tool, payload)
        text = _joined_text(resp)
        data = _best_effort_parse(text)
        return {
            "tool": self.poll_tool,
            "space_id": self.space_id,
            "request": payload,
            "response": data,
        }

    def ask(
        self,
        query: str,
        *,
        conversation_id: str | None = None,
        timeout_s: float = 300.0,
        poll_interval_s: float = 2.0,
    ) -> dict:
        """Query Genie and poll until completion (best-effort).

        Returns a dict including the final polled response (or last response if timeout).
        """
        initial = self.query(query, conversation_id=conversation_id)
        resp = initial.get("response", {})
        if not isinstance(resp, dict):
            return initial

        conv_id, msg_id = _extract_conversation_and_message_ids(resp)
        if not conv_id or not msg_id:
            return initial

        deadline = time.time() + timeout_s
        last = initial
        while time.time() < deadline:
            polled = self.poll(conversation_id=conv_id, message_id=msg_id)
            last = polled
            r = polled.get("response", {})
            if isinstance(r, dict):
                status = r.get("status")
                if status in {"COMPLETED", "FAILED"}:
                    break
            time.sleep(poll_interval_s)

        return {
            "initial": initial,
            "final": last,
            "conversation_id": conv_id,
            "message_id": msg_id,
        }

