from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    apply: Callable[[sqlite3.Connection], None]


RUNS_ADDITIVE_COLUMNS = {
    "prompt_cache_hit_tokens": "INTEGER NOT NULL DEFAULT 0",
    "prompt_cache_miss_tokens": "INTEGER NOT NULL DEFAULT 0",
    "reasoning_tokens": "INTEGER NOT NULL DEFAULT 0",
    "api_calls": "INTEGER NOT NULL DEFAULT 0",
}


def connect_database(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(db_path: Path) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect_database(path) as conn:
        _ensure_schema_migrations(conn)
        applied_versions = _applied_versions(conn)
        for migration in MIGRATIONS:
            if migration.version in applied_versions:
                continue
            migration.apply(conn)
            _record_migration(conn, migration)


def _ensure_schema_migrations(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )


def _applied_versions(conn: sqlite3.Connection) -> set[int]:
    rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    return {int(row["version"]) for row in rows}


def _record_migration(conn: sqlite3.Connection, migration: Migration) -> None:
    conn.execute(
        """
        INSERT INTO schema_migrations (version, name, applied_at)
        VALUES (?, ?, ?)
        """,
        (
            migration.version,
            migration.name,
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def _baseline_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS approvals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action_type TEXT NOT NULL,
            title TEXT NOT NULL,
            payload TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_input TEXT NOT NULL,
            model TEXT NOT NULL,
            tool_calls TEXT NOT NULL,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            estimated_cost REAL NOT NULL DEFAULT 0.0
        )
        """
    )
    _ensure_runs_columns(conn)


def _ensure_runs_columns(conn: sqlite3.Connection) -> None:
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(runs)").fetchall()
    }
    for name, definition in RUNS_ADDITIVE_COLUMNS.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {name} {definition}")


def _memory_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            approval_id INTEGER NOT NULL UNIQUE,
            memory_type TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'chat',
            tags_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            archived_at TEXT,
            archive_reason TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER,
            approval_id INTEGER,
            event_type TEXT NOT NULL,
            actor TEXT NOT NULL,
            details_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        )
        """
    )


def _coding_agents_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS coding_agent_repositories (
            root TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS coding_agent_jobs (
            id TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            repository_root TEXT NOT NULL,
            repository_label TEXT NOT NULL,
            goal TEXT NOT NULL,
            model TEXT,
            effort TEXT,
            max_turns INTEGER NOT NULL,
            timeout_seconds INTEGER NOT NULL,
            external_session_id TEXT,
            state TEXT NOT NULL,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            updated_at TEXT NOT NULL,
            input_tokens INTEGER,
            cached_input_tokens INTEGER,
            output_tokens INTEGER,
            reasoning_tokens INTEGER,
            token_usage_source TEXT,
            estimated_cost_usd REAL,
            cost_source TEXT,
            duration_seconds REAL,
            failure_category TEXT,
            cancellation_requested INTEGER NOT NULL DEFAULT 0,
            cancellation_requested_at TEXT,
            last_error TEXT,
            FOREIGN KEY(repository_root) REFERENCES coding_agent_repositories(root)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS coding_agent_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            summary TEXT NOT NULL DEFAULT '',
            content_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(job_id) REFERENCES coding_agent_jobs(id)
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_coding_agent_events_job_sequence
        ON coding_agent_events(job_id, sequence)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_coding_agent_jobs_updated
        ON coding_agent_jobs(updated_at DESC)
        """
    )


MIGRATIONS = (
    Migration(1, "baseline_schema", _baseline_schema),
    Migration(2, "memory_schema", _memory_schema),
    Migration(3, "coding_agents_schema", _coding_agents_schema),
)
