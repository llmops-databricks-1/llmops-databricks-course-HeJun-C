"""Thin wrapper around the Databricks Genie Conversation API.

Provides a single ``ask`` method that starts a conversation, polls until
completion, and returns the generated SQL together with the query result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mlflow
from databricks.sdk import WorkspaceClient
from loguru import logger

_POLL_MAX_SECONDS = 300
_POLL_INITIAL_WAIT = 2.0
_POLL_BACKOFF_FACTOR = 2.0
_POLL_MAX_WAIT = 30.0


@dataclass
class GenieResult:
    """Structured result returned by :func:`GenieClient.ask`."""

    sql: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, str]] = field(default_factory=list)
    status: str = "UNKNOWN"
    description: str = ""


class GenieClient:
    """Stateless client that sends one-shot questions to a Genie space."""

    def __init__(
        self,
        space_id: str,
        *,
        profile: str | None = None,
    ) -> None:
        self.space_id = space_id
        kwargs: dict[str, str] = {}
        if profile:
            kwargs["profile"] = profile
        self._ws = WorkspaceClient(**kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @mlflow.trace(span_type="TOOL", name="genie_ask")
    def ask(self, question: str) -> GenieResult:
        """Send *question* to Genie and return the result."""
        logger.info("Genie ← {!r}", question)

        conv = self._ws.genie.start_conversation(
            space_id=self.space_id,
            content=question,
        )
        conversation_id: str = conv.conversation_id
        message_id: str = conv.message_id
        logger.debug(
            "conversation={} message={}", conversation_id, message_id
        )

        msg = self._poll_until_done(conversation_id, message_id)

        status = (msg.status or "UNKNOWN").value if hasattr(msg.status, "value") else str(msg.status or "UNKNOWN")

        if status != "COMPLETED":
            logger.warning("Genie finished with status={}", status)
            return GenieResult(status=status)

        return self._extract_result(
            conversation_id, message_id, msg
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _poll_until_done(
        self,
        conversation_id: str,
        message_id: str,
    ) -> object:
        wait = _POLL_INITIAL_WAIT
        elapsed = 0.0

        while elapsed < _POLL_MAX_SECONDS:
            msg = self._ws.genie.get_message(
                space_id=self.space_id,
                conversation_id=conversation_id,
                message_id=message_id,
            )
            raw_status = (
                msg.status.value
                if hasattr(msg.status, "value")
                else str(msg.status)
            )
            if raw_status in ("COMPLETED", "FAILED", "CANCELLED"):
                return msg

            logger.debug(
                "Genie polling … status={} elapsed={:.0f}s",
                raw_status,
                elapsed,
            )
            time.sleep(wait)
            elapsed += wait
            wait = min(wait * _POLL_BACKOFF_FACTOR, _POLL_MAX_WAIT)

        raise TimeoutError(
            f"Genie did not complete within {_POLL_MAX_SECONDS}s"
        )

    def _extract_result(
        self,
        conversation_id: str,
        message_id: str,
        msg: object,
    ) -> GenieResult:
        sql = ""
        description = ""
        columns: list[str] = []
        rows: list[dict[str, str]] = []

        attachments = getattr(msg, "attachments", None) or []
        for att in attachments:
            query = getattr(att, "query", None)
            if not query:
                continue
            sql = getattr(query, "query", "") or ""
            description = getattr(query, "description", "") or ""

            att_id = getattr(att, "attachment_id", None)
            if att_id is None:
                continue
            try:
                qr = self._ws.genie.get_message_attachment_query_result(
                    space_id=self.space_id,
                    conversation_id=conversation_id,
                    message_id=message_id,
                    attachment_id=att_id,
                )
                # Path: qr.statement_response.manifest.schema.columns
                stmt = getattr(qr, "statement_response", None)
                if stmt is None:
                    continue
                manifest = getattr(stmt, "manifest", None)
                if manifest:
                    schema = getattr(manifest, "schema", None)
                    if schema:
                        columns = [
                            c.name
                            for c in (
                                getattr(schema, "columns", None) or []
                            )
                        ]
                # Path: qr.statement_response.result.data_array
                result = getattr(stmt, "result", None)
                if result:
                    raw_rows = (
                        getattr(result, "data_array", None) or []
                    )
                    rows = [
                        dict(zip(columns, r)) for r in raw_rows
                    ]
            except Exception:
                logger.opt(exception=True).warning(
                    "Failed to fetch query result for attachment {}",
                    att_id,
                )

        logger.info(
            "Genie → {} cols, {} rows, sql_len={}",
            len(columns),
            len(rows),
            len(sql),
        )
        return GenieResult(
            sql=sql,
            columns=columns,
            rows=rows,
            status="COMPLETED",
            description=description,
        )
