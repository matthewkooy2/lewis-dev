from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import DB_PATH
from app.db import connect_database, initialize_database


@dataclass(frozen=True)
class Memory:
    id: int
    approval_id: int
    memory_type: str
    content: str
    source: str
    tags: list[str]
    status: str
    created_at: str
    updated_at: str
    archived_at: str | None
    archive_reason: str | None


@dataclass(frozen=True)
class MemoryEvent:
    id: int
    memory_id: int | None
    approval_id: int | None
    event_type: str
    actor: str
    details: dict[str, Any]
    created_at: str


class MemoryStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        initialize_database(self.db_path)

    def get_by_approval_id(self, approval_id: int) -> Memory | None:
        with connect_database(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE approval_id = ?", (approval_id,)
            ).fetchone()
        return self._row_to_memory(row) if row else None

    def get(self, memory_id: int) -> Memory | None:
        with connect_database(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return self._row_to_memory(row) if row else None

    def list_active(self) -> list[Memory]:
        with connect_database(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE status = 'active' ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def list_recent_events(self, limit: int = 20) -> list[MemoryEvent]:
        with connect_database(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM memory_events
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (max(limit, 1),),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def list_events(self, approval_id: int | None = None) -> list[MemoryEvent]:
        with connect_database(self.db_path) as conn:
            if approval_id is None:
                rows = conn.execute(
                    "SELECT * FROM memory_events ORDER BY created_at ASC, id ASC"
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM memory_events
                    WHERE approval_id = ?
                    ORDER BY created_at ASC, id ASC
                    """,
                    (approval_id,),
                ).fetchall()
        return [self._row_to_event(row) for row in rows]

    @staticmethod
    def create_active_memory_once(
        conn: sqlite3.Connection,
        *,
        approval_id: int,
        memory_type: str,
        content: str,
        source: str,
        tags: list[str],
    ) -> tuple[int, bool]:
        existing = conn.execute(
            "SELECT id FROM memories WHERE approval_id = ?", (approval_id,)
        ).fetchone()
        if existing is not None:
            return int(existing["id"]), False

        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            """
            INSERT INTO memories (
                approval_id, memory_type, content, source, tags_json, status,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
            """,
            (
                approval_id,
                memory_type,
                content,
                source,
                json.dumps(tags, sort_keys=True),
                now,
                now,
            ),
        )
        return int(cur.lastrowid), True

    @staticmethod
    def get_for_update(conn: sqlite3.Connection, memory_id: int) -> Memory | None:
        row = conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return MemoryStore._row_to_memory(row) if row else None

    @staticmethod
    def get_by_approval_id_for_update(
        conn: sqlite3.Connection, approval_id: int
    ) -> Memory | None:
        row = conn.execute(
            "SELECT * FROM memories WHERE approval_id = ?", (approval_id,)
        ).fetchone()
        return MemoryStore._row_to_memory(row) if row else None

    @staticmethod
    def archive_active_memory(
        conn: sqlite3.Connection,
        *,
        memory_id: int,
        reason: str | None = None,
    ) -> Memory:
        memory = MemoryStore.get_for_update(conn, memory_id)
        if memory is None:
            raise KeyError(f"Memory {memory_id} not found.")
        if memory.status != "active":
            raise ValueError(f"Memory {memory_id} is not active.")

        now = datetime.now(timezone.utc).isoformat()
        normalized_reason = reason.strip() if reason else None
        conn.execute(
            """
            UPDATE memories
            SET status = 'archived',
                updated_at = ?,
                archived_at = ?,
                archive_reason = ?
            WHERE id = ? AND status = 'active'
            """,
            (now, now, normalized_reason, memory_id),
        )
        updated = MemoryStore.get_for_update(conn, memory_id)
        if updated is None:
            raise RuntimeError("Failed to archive memory.")
        return updated

    @staticmethod
    def record_event(
        conn: sqlite3.Connection,
        *,
        event_type: str,
        actor: str,
        details: dict[str, Any] | None = None,
        memory_id: int | None = None,
        approval_id: int | None = None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO memory_events (
                memory_id, approval_id, event_type, actor, details_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                approval_id,
                event_type,
                actor,
                json.dumps(details or {}, sort_keys=True),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        tags = json.loads(str(row["tags_json"]))
        if not isinstance(tags, list):
            tags = []
        return Memory(
            id=int(row["id"]),
            approval_id=int(row["approval_id"]),
            memory_type=str(row["memory_type"]),
            content=str(row["content"]),
            source=str(row["source"]),
            tags=[str(tag) for tag in tags],
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            archived_at=row["archived_at"],
            archive_reason=row["archive_reason"],
        )

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> MemoryEvent:
        details = json.loads(str(row["details_json"]))
        if not isinstance(details, dict):
            details = {}
        memory_id = row["memory_id"]
        approval_id = row["approval_id"]
        return MemoryEvent(
            id=int(row["id"]),
            memory_id=int(memory_id) if memory_id is not None else None,
            approval_id=int(approval_id) if approval_id is not None else None,
            event_type=str(row["event_type"]),
            actor=str(row["actor"]),
            details=details,
            created_at=str(row["created_at"]),
        )
