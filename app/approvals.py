from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import DB_PATH
from app.db import connect_database, initialize_database

APPROVAL_STATUSES = {
    "pending",
    "approved",
    "rejected",
    "completed",
    "execution_previewed",
}


@dataclass(frozen=True)
class ApprovalItem:
    id: int
    action_type: str
    title: str
    payload: dict[str, Any]
    status: str
    created_at: str
    updated_at: str


class ApprovalStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        initialize_database(self.db_path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = connect_database(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create(
        self, action_type: str, title: str, payload: dict[str, Any]
    ) -> ApprovalItem:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO approvals (action_type, title, payload, status, created_at, updated_at)
                VALUES (?, ?, ?, 'pending', ?, ?)
                """,
                (action_type, title, json.dumps(payload, sort_keys=True), now, now),
            )
            approval_id = int(cur.lastrowid)
        item = self.get(approval_id)
        if item is None:
            raise RuntimeError("Failed to create approval item.")
        return item

    def list(self, status: str | None = None) -> list[ApprovalItem]:
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM approvals WHERE status = ? ORDER BY created_at DESC, id DESC",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM approvals ORDER BY created_at DESC, id DESC"
                ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def get(self, approval_id: int) -> ApprovalItem | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM approvals WHERE id = ?", (approval_id,)
            ).fetchone()
        return self._row_to_item(row) if row else None

    def update_status(self, approval_id: int, status: str) -> ApprovalItem:
        if status not in APPROVAL_STATUSES:
            raise ValueError(
                "Status must be pending, approved, rejected, completed, "
                "or execution_previewed."
            )
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE approvals SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, approval_id),
            )
        item = self.get(approval_id)
        if item is None:
            raise KeyError(f"Approval item {approval_id} not found.")
        return item

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> ApprovalItem:
        return ApprovalItem(
            id=int(row["id"]),
            action_type=str(row["action_type"]),
            title=str(row["title"]),
            payload=json.loads(str(row["payload"])),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
