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



def _projects_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            repository_root TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            color TEXT NOT NULL DEFAULT '',
            icon TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            archived_at TEXT,
            last_map_refreshed_at TEXT,
            current_branch TEXT,
            head_commit TEXT,
            language_summary_json TEXT NOT NULL DEFAULT '{}',
            mapping_state TEXT NOT NULL DEFAULT 'not_indexed',
            mapping_detail TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS project_tags (
            project_id TEXT NOT NULL,
            tag TEXT NOT NULL,
            PRIMARY KEY(project_id, tag),
            FOREIGN KEY(project_id) REFERENCES projects(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS project_note_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(project_id, name),
            FOREIGN KEY(project_id) REFERENCES projects(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS project_notes (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            title TEXT NOT NULL,
            content_markdown TEXT NOT NULL DEFAULT '',
            category_id INTEGER,
            is_pinned INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            archived_at TEXT,
            FOREIGN KEY(project_id) REFERENCES projects(id),
            FOREIGN KEY(category_id) REFERENCES project_note_categories(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS project_note_tags (
            note_id TEXT NOT NULL,
            tag TEXT NOT NULL,
            PRIMARY KEY(note_id, tag),
            FOREIGN KEY(note_id) REFERENCES project_notes(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS project_map_runs (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            run_type TEXT NOT NULL,
            status TEXT NOT NULL,
            current_stage TEXT NOT NULL DEFAULT '',
            stage_index INTEGER NOT NULL DEFAULT 0,
            stage_count INTEGER NOT NULL DEFAULT 0,
            progress_percent INTEGER NOT NULL DEFAULT 0,
            files_scanned INTEGER NOT NULL DEFAULT 0,
            files_indexed INTEGER NOT NULL DEFAULT 0,
            files_reused INTEGER NOT NULL DEFAULT 0,
            files_changed INTEGER NOT NULL DEFAULT 0,
            files_deleted INTEGER NOT NULL DEFAULT 0,
            files_skipped INTEGER NOT NULL DEFAULT 0,
            unsupported_files INTEGER NOT NULL DEFAULT 0,
            warnings_json TEXT NOT NULL DEFAULT '[]',
            diagnostics_json TEXT NOT NULL DEFAULT '{}',
            revision_branch TEXT,
            revision_commit TEXT,
            previous_run_id TEXT,
            full_rebuild INTEGER NOT NULL DEFAULT 0,
            deterministic_complete INTEGER NOT NULL DEFAULT 0,
            model_summaries_enabled INTEGER NOT NULL DEFAULT 0,
            cancellation_requested INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            updated_at TEXT NOT NULL,
            last_error TEXT,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS project_map_stages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            stage_key TEXT NOT NULL,
            name TEXT NOT NULL,
            stage_index INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            progress_percent INTEGER NOT NULL DEFAULT 0,
            diagnostics_json TEXT NOT NULL DEFAULT '{}',
            started_at TEXT,
            completed_at TEXT,
            updated_at TEXT NOT NULL,
            UNIQUE(run_id, stage_key),
            FOREIGN KEY(run_id) REFERENCES project_map_runs(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS project_indexed_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            path TEXT NOT NULL,
            name TEXT NOT NULL,
            parent_path TEXT NOT NULL DEFAULT '',
            language TEXT NOT NULL DEFAULT '',
            role TEXT NOT NULL DEFAULT '',
            file_type TEXT NOT NULL DEFAULT '',
            size_bytes INTEGER NOT NULL DEFAULT 0,
            line_count INTEGER NOT NULL DEFAULT 0,
            content_hash TEXT NOT NULL,
            git_status TEXT NOT NULL DEFAULT '',
            safety_classification TEXT NOT NULL DEFAULT 'safe',
            imports_json TEXT NOT NULL DEFAULT '[]',
            summary TEXT NOT NULL DEFAULT '',
            summary_source TEXT NOT NULL DEFAULT 'deterministic',
            indexed_at TEXT NOT NULL,
            UNIQUE(project_id, path),
            FOREIGN KEY(project_id) REFERENCES projects(id),
            FOREIGN KEY(run_id) REFERENCES project_map_runs(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS project_indexed_symbols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            line INTEGER NOT NULL DEFAULT 0,
            end_line INTEGER NOT NULL DEFAULT 0,
            parent TEXT,
            signature TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            deterministic INTEGER NOT NULL DEFAULT 1,
            indexed_at TEXT NOT NULL,
            FOREIGN KEY(project_id) REFERENCES projects(id),
            FOREIGN KEY(run_id) REFERENCES project_map_runs(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS project_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            relationship_type TEXT NOT NULL,
            source_type TEXT NOT NULL DEFAULT 'file',
            source_path TEXT NOT NULL DEFAULT '',
            source_symbol TEXT,
            target_type TEXT NOT NULL DEFAULT 'file',
            target_path TEXT,
            target_symbol TEXT,
            resolution TEXT NOT NULL DEFAULT 'inferred',
            summary TEXT NOT NULL DEFAULT '',
            deterministic INTEGER NOT NULL DEFAULT 1,
            indexed_at TEXT NOT NULL,
            FOREIGN KEY(project_id) REFERENCES projects(id),
            FOREIGN KEY(run_id) REFERENCES project_map_runs(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS project_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            run_id TEXT,
            scope_type TEXT NOT NULL,
            scope_path TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'deterministic',
            evidence_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS project_diagrams (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            map_run_id TEXT,
            title TEXT NOT NULL,
            diagram_type TEXT NOT NULL,
            definition_json TEXT NOT NULL DEFAULT '{}',
            generation_source TEXT NOT NULL DEFAULT 'deterministic',
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            archived_at TEXT,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS project_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            summary TEXT NOT NULL,
            details_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        )
        """
    )
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(coding_agent_jobs)")
    }
    if "project_id" not in columns:
        conn.execute("ALTER TABLE coding_agent_jobs ADD COLUMN project_id TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status, updated_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_project_notes_project_status ON project_notes(project_id, status, updated_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_project_map_runs_project_status ON project_map_runs(project_id, status, updated_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_project_indexed_files_project_path ON project_indexed_files(project_id, path)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_project_indexed_symbols_project_name ON project_indexed_symbols(project_id, name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_project_relationships_project_source ON project_relationships(project_id, source_path)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_project_activity_project_created ON project_activity(project_id, created_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_coding_agent_jobs_project ON coding_agent_jobs(project_id, updated_at DESC)")


MIGRATIONS = (
    Migration(1, "baseline_schema", _baseline_schema),
    Migration(2, "memory_schema", _memory_schema),
    Migration(3, "coding_agents_schema", _coding_agents_schema),
    Migration(4, "projects_schema", _projects_schema),
)
