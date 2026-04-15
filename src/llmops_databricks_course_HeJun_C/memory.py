"""Lakebase-backed session memory for the EDA Agent.

Stores conversation history (user questions + assistant answers) in a
Databricks Lakebase (managed PostgreSQL) instance, keyed by session ID.

Follows the pattern from Lecture 3 — Agent memory with Lakebase.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from typing import Any
from uuid import uuid4

import psycopg
from databricks.sdk import WorkspaceClient
from loguru import logger
from psycopg_pool import ConnectionPool


class LakebaseMemory:
    """Handles session message persistence using Lakebase (PostgreSQL)."""

    def __init__(
        self,
        host: str,
        instance_name: str,
    ) -> None:
        self.host = host
        self.instance_name = instance_name
        self._pool: ConnectionPool | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_connection_string(self) -> str:
        """Build connection string for Lakebase.

        Supports two authentication modes:
        - SPN (production): Set DATABRICKS_CLIENT_ID,
          DATABRICKS_CLIENT_SECRET, DATABRICKS_HOST
        - User (local testing): Uses default WorkspaceClient auth
          (e.g., ~/.databrickscfg)
        """
        w = WorkspaceClient()

        client_id = os.environ.get("DATABRICKS_CLIENT_ID")
        if client_id:
            username = client_id
        else:
            user = w.current_user.me()
            username = urllib.parse.quote_plus(user.user_name)

        pg_credential = w.database.generate_database_credential(
            request_id=str(uuid4()),
            instance_names=[self.instance_name],
        )

        return (
            f"postgresql://{username}:{pg_credential.token}"
            f"@{self.host}:5432/"
            "databricks_postgres?sslmode=require"
        )

    def _get_pool(self) -> ConnectionPool:
        """Get or create connection pool (lazy init)."""
        if self._pool is None:
            conn_string = self._get_connection_string()
            self._pool = ConnectionPool(
                conninfo=conn_string, min_size=1, max_size=5
            )
        return self._pool

    def _reset_pool(self) -> None:
        """Reset pool to force new credentials on next use."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def _ensure_table(self, conn: psycopg.Connection[Any]) -> None:
        """Create the session_messages table if it doesn't exist."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_messages (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                message_data JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_messages_session_id
            ON session_messages(session_id)
        """)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_messages(
        self, session_id: str
    ) -> list[dict[str, Any]]:
        """Load previous messages for a session, ordered by time."""
        try:
            with self._get_pool().connection() as conn:
                self._ensure_table(conn)
                result = conn.execute(
                    """
                    SELECT message_data FROM session_messages
                    WHERE session_id = %s
                    ORDER BY created_at ASC
                    """,
                    (session_id,),
                ).fetchall()
                return [row[0] for row in result]
        except psycopg.OperationalError:
            self._reset_pool()
            raise
        except Exception as exc:
            logger.warning("Failed to load session messages: {}", exc)
            return []

    def save_messages(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """Append messages to a session."""
        try:
            with self._get_pool().connection() as conn:
                self._ensure_table(conn)
                for msg in messages:
                    conn.execute(
                        "INSERT INTO session_messages "
                        "(session_id, message_data) VALUES (%s, %s)",
                        (session_id, json.dumps(msg)),
                    )
        except psycopg.OperationalError:
            self._reset_pool()
            raise
        except Exception as exc:
            logger.warning("Failed to save session messages: {}", exc)
