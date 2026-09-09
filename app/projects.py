"""Projects workspace persistence, repository indexing, and integrations."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import subprocess
import threading
import time
from typing import Any
import uuid

from app.code_proposal_path_safety import validate_relative_project_path
from app.config import DB_PATH, Settings
from app.db import connect_database, initialize_database
from .dev import coding_agents
from .dev.coding_agents import ApprovedRepository, RepositoryRefusedError
from .dev.freestyle_policy import GENERATED_OR_CACHE_PARTS, contains_sensitive_content
from .dev.repo_map import _is_approval_sensitive
from .dev.repo_map_extract import ExtractedFile, extract_file
from app.trace.service import CodeTracer

PROJECT_STATUSES = {"active", "paused", "archived"}
MAX_TEXT = 2_000
MAX_NOTE = 200_000
MAX_GIT_OUTPUT_BYTES = 120_000
GIT_TIMEOUT_SECONDS = 10
MAP_STAGES: tuple[tuple[str, str], ...] = (
    ("inventory", "Repository inventory"),
    ("classification", "Language and file classification"),
    ("symbols", "Deterministic symbol extraction"),
    ("dependencies", "Import and dependency mapping"),
    ("relationships", "Cross-file relationship analysis"),
    ("tests", "Test/source association"),
    ("architecture", "Architectural grouping"),
    ("summaries", "Bounded deterministic summaries"),
    ("consistency", "Final consistency check"),
)
INDEXABLE_SUFFIXES = {
    ".cfg", ".css", ".csv", ".html", ".htm", ".ini", ".js", ".json",
    ".jsx", ".md", ".mjs", ".py", ".rst", ".sql", ".toml", ".ts",
    ".tsx", ".txt", ".yaml", ".yml",
}
DATABASE_SUFFIXES = {".sqlite", ".sqlite3", ".db"}
RESUMABLE_RUN_STATES = {"failed", "cancelled", "paused"}


@dataclass(frozen=True)
class Project:
    id: str
    display_name: str
    repository_root: str
    description: str
    status: str
    color: str
    icon: str
    created_at: str
    updated_at: str
    archived_at: str | None
    last_map_refreshed_at: str | None
    current_branch: str | None
    head_commit: str | None
    language_summary: dict[str, int]
    mapping_state: str
    mapping_detail: str
    tags: tuple[str, ...]
    notes_count: int = 0
    active_notes_count: int = 0
    latest_run_id: str | None = None
    latest_run_status: str | None = None
    latest_run_error: str | None = None


@dataclass(frozen=True)
class ProjectNote:
    id: str
    project_id: str
    title: str
    content_markdown: str
    category_id: int | None
    category_name: str | None
    is_pinned: bool
    status: str
    created_at: str
    updated_at: str
    archived_at: str | None
    tags: tuple[str, ...]


@dataclass(frozen=True)
class MapRun:
    id: str
    project_id: str
    run_type: str
    status: str
    current_stage: str
    stage_index: int
    stage_count: int
    progress_percent: int
    files_scanned: int
    files_indexed: int
    files_reused: int
    files_changed: int
    files_deleted: int
    files_skipped: int
    unsupported_files: int
    warnings: tuple[str, ...]
    diagnostics: dict[str, Any]
    revision_branch: str | None
    revision_commit: str | None
    previous_run_id: str | None
    full_rebuild: bool
    deterministic_complete: bool
    model_summaries_enabled: bool
    cancellation_requested: bool
    created_at: str
    started_at: str | None
    completed_at: str | None
    updated_at: str
    last_error: str | None


@dataclass(frozen=True)
class IndexedFile:
    path: str
    name: str
    parent_path: str
    language: str
    role: str
    file_type: str
    size_bytes: int
    line_count: int
    content_hash: str
    git_status: str
    safety_classification: str
    imports: tuple[str, ...]
    summary: str
    summary_source: str
    indexed_at: str


@dataclass(frozen=True)
class IndexedSymbol:
    id: int
    file_path: str
    name: str
    kind: str
    line: int
    end_line: int
    parent: str | None
    summary: str
    deterministic: bool


@dataclass(frozen=True)
class Relationship:
    id: int
    relationship_type: str
    source_path: str
    source_symbol: str | None
    target_path: str | None
    target_symbol: str | None
    resolution: str
    summary: str
    deterministic: bool


@dataclass(frozen=True)
class Diagram:
    id: str
    project_id: str
    map_run_id: str | None
    title: str
    diagram_type: str
    definition: dict[str, Any]
    generation_source: str
    status: str
    created_at: str
    updated_at: str
    archived_at: str | None


@dataclass(frozen=True)
class CandidateFile:
    relative: str
    path: Path
    text: str
    reason: str | None = None


class ProjectError(RuntimeError):
    """Raised when a Projects request is refused or cannot complete safely."""


class ProjectCancelled(RuntimeError):
    pass


class ProjectStore:
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

    def create_project(
        self,
        *,
        display_name: str,
        repository: ApprovedRepository,
        description: str = "",
        tags: Sequence[str] = (),
        status: str = "active",
        color: str = "",
        icon: str = "",
    ) -> Project:
        name = _required(display_name, 120, "Project name")
        clean_status = _project_status(status)
        now = _utc_now()
        metadata = read_git_metadata(repository.root)
        project_id = uuid.uuid4().hex
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM projects WHERE repository_root = ?",
                (str(repository.root),),
            ).fetchone()
            if row:
                raise ProjectError("A project already exists for that repository root.")
            conn.execute(
                """
                INSERT INTO projects (
                    id, display_name, repository_root, description, status,
                    color, icon, created_at, updated_at, archived_at,
                    last_map_refreshed_at, current_branch, head_commit,
                    language_summary_json, mapping_state, mapping_detail
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, '{}',
                          'not_indexed', ?)
                """,
                (
                    project_id, name, str(repository.root), _text(description, MAX_TEXT),
                    clean_status, _text(color, 40), _text(icon, 40), now, now,
                    metadata.get("branch"), metadata.get("head_commit"),
                    "Repository map has not been generated yet.",
                ),
            )
            self._replace_tags(conn, "project_tags", "project_id", project_id, tags)
            self._activity(conn, project_id, "project_created", f"Created project {name}.", {"repository_root": str(repository.root)})
        return self.require_project(project_id)

    def update_project(
        self,
        project_id: str,
        *,
        display_name: str | None = None,
        description: str | None = None,
        tags: Sequence[str] | None = None,
        status: str | None = None,
        color: str | None = None,
        icon: str | None = None,
    ) -> Project:
        project = self.require_project(project_id)
        updates: dict[str, object] = {"updated_at": _utc_now()}
        if display_name is not None:
            updates["display_name"] = _required(display_name, 120, "Project name")
        if description is not None:
            updates["description"] = _text(description, MAX_TEXT)
        if status is not None:
            updates["status"] = _project_status(status)
            updates["archived_at"] = _utc_now() if status == "archived" else None
        if color is not None:
            updates["color"] = _text(color, 40)
        if icon is not None:
            updates["icon"] = _text(icon, 40)
        with self._connect() as conn:
            assignments = ", ".join(f"{column} = ?" for column in updates)
            conn.execute(
                f"UPDATE projects SET {assignments} WHERE id = ?",
                (*updates.values(), project.id),
            )
            if tags is not None:
                self._replace_tags(conn, "project_tags", "project_id", project.id, tags)
            self._activity(conn, project.id, "project_updated", "Project metadata updated.", {"status": updates.get("status", project.status)})
        return self.require_project(project.id)

    def archive_project(self, project_id: str, *, confirm: str) -> Project:
        if confirm != "archive-project":
            raise ProjectError("Explicit archive confirmation is required.")
        return self.update_project(project_id, status="archived")

    def restore_project(self, project_id: str) -> Project:
        return self.update_project(project_id, status="active")

    def require_project(self, project_id: str) -> Project:
        if not _is_hex_id(project_id):
            raise ProjectError("Project id is not valid.")
        project = self.get_project(project_id)
        if project is None:
            raise ProjectError("Project was not found.")
        return project

    def get_project(self, project_id: str) -> Project | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
            if row is None:
                return None
            return self._project_from_row(conn, row)

    def list_projects(
        self,
        *,
        query: str = "",
        status: str = "active",
        tag: str = "",
        include_archived: bool = False,
        limit: int = 100,
    ) -> list[Project]:
        filters: list[str] = []
        values: list[object] = []
        if query.strip():
            like = f"%{query.strip()}%"
            filters.append("(display_name LIKE ? OR description LIKE ? OR repository_root LIKE ?)")
            values.extend([like, like, like])
        if status and status != "all":
            filters.append("status = ?")
            values.append(_project_status(status))
        elif not include_archived:
            filters.append("status != 'archived'")
        if tag.strip():
            filters.append("id IN (SELECT project_id FROM project_tags WHERE tag = ?)")
            values.append(_tag(tag))
        where = " WHERE " + " AND ".join(filters) if filters else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM projects{where} ORDER BY updated_at DESC LIMIT ?",
                (*values, max(1, min(limit, 200))),
            ).fetchall()
            return [self._project_from_row(conn, row) for row in rows]

    def list_project_tags(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT tag FROM project_tags ORDER BY tag COLLATE NOCASE").fetchall()
        return [str(row["tag"]) for row in rows]

    def list_categories(self, project_id: str) -> list[str]:
        self.require_project(project_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT name FROM project_note_categories
                WHERE project_id = ? AND status = 'active'
                ORDER BY name COLLATE NOCASE
                """,
                (project_id,),
            ).fetchall()
        return [str(row["name"]) for row in rows]

    def create_note(
        self,
        project_id: str,
        *,
        title: str,
        content_markdown: str,
        category: str = "",
        tags: Sequence[str] = (),
        is_pinned: bool = False,
    ) -> ProjectNote:
        project = self.require_project(project_id)
        note_id = uuid.uuid4().hex
        now = _utc_now()
        with self._connect() as conn:
            category_id = self._ensure_category(conn, project.id, category)
            conn.execute(
                """
                INSERT INTO project_notes (
                    id, project_id, title, content_markdown, category_id,
                    is_pinned, status, created_at, updated_at, archived_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, NULL)
                """,
                (
                    note_id, project.id, _required(title, 180, "Note title"),
                    _text(content_markdown, MAX_NOTE), category_id, int(is_pinned),
                    now, now,
                ),
            )
            self._replace_tags(conn, "project_note_tags", "note_id", note_id, tags)
            self._activity(conn, project.id, "note_created", f"Created note {title}.", {"note_id": note_id})
        return self.require_note(project.id, note_id)

    def update_note(
        self,
        project_id: str,
        note_id: str,
        *,
        title: str,
        content_markdown: str,
        category: str = "",
        tags: Sequence[str] = (),
        is_pinned: bool = False,
    ) -> ProjectNote:
        note = self.require_note(project_id, note_id, include_archived=True)
        now = _utc_now()
        with self._connect() as conn:
            category_id = self._ensure_category(conn, project_id, category)
            conn.execute(
                """
                UPDATE project_notes
                SET title = ?, content_markdown = ?, category_id = ?,
                    is_pinned = ?, updated_at = ?
                WHERE project_id = ? AND id = ?
                """,
                (
                    _required(title, 180, "Note title"), _text(content_markdown, MAX_NOTE),
                    category_id, int(is_pinned), now, project_id, note.id,
                ),
            )
            self._replace_tags(conn, "project_note_tags", "note_id", note.id, tags)
            self._activity(conn, project_id, "note_updated", f"Updated note {title}.", {"note_id": note.id})
        return self.require_note(project_id, note.id, include_archived=True)

    def duplicate_note(self, project_id: str, note_id: str) -> ProjectNote:
        note = self.require_note(project_id, note_id, include_archived=True)
        return self.create_note(
            project_id,
            title=f"{note.title} copy",
            content_markdown=note.content_markdown,
            category=note.category_name or "",
            tags=note.tags,
        )

    def archive_note(self, project_id: str, note_id: str) -> None:
        note = self.require_note(project_id, note_id)
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE project_notes SET status = 'archived', archived_at = ?, updated_at = ? WHERE project_id = ? AND id = ?",
                (now, now, project_id, note.id),
            )
            self._activity(conn, project_id, "note_archived", f"Archived note {note.title}.", {"note_id": note.id})

    def restore_note(self, project_id: str, note_id: str) -> None:
        note = self.require_note(project_id, note_id, include_archived=True)
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE project_notes SET status = 'active', archived_at = NULL, updated_at = ? WHERE project_id = ? AND id = ?",
                (now, project_id, note.id),
            )
            self._activity(conn, project_id, "note_restored", f"Restored note {note.title}.", {"note_id": note.id})

    def delete_note_permanently(self, project_id: str, note_id: str, *, confirm: str) -> None:
        if confirm != "delete-note":
            raise ProjectError("Explicit note deletion confirmation is required.")
        note = self.require_note(project_id, note_id, include_archived=True)
        with self._connect() as conn:
            conn.execute("DELETE FROM project_note_tags WHERE note_id = ?", (note.id,))
            conn.execute("DELETE FROM project_notes WHERE project_id = ? AND id = ?", (project_id, note.id))
            self._activity(conn, project_id, "note_deleted", f"Permanently deleted note {note.title}.", {"note_id": note.id})

    def list_notes(
        self,
        project_id: str,
        *,
        query: str = "",
        category: str = "",
        include_archived: bool = False,
        sort: str = "updated",
        limit: int = 100,
    ) -> list[ProjectNote]:
        self.require_project(project_id)
        filters = ["n.project_id = ?"]
        values: list[object] = [project_id]
        if not include_archived:
            filters.append("n.status = 'active'")
        if query.strip():
            like = f"%{query.strip()}%"
            filters.append("(n.title LIKE ? OR n.content_markdown LIKE ?)")
            values.extend([like, like])
        if category.strip():
            filters.append("c.name = ?")
            values.append(category.strip())
        order = {
            "title": "n.is_pinned DESC, n.title COLLATE NOCASE ASC",
            "created": "n.is_pinned DESC, n.created_at DESC",
            "updated": "n.is_pinned DESC, n.updated_at DESC",
        }.get(sort, "n.is_pinned DESC, n.updated_at DESC")
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT n.*, c.name AS category_name
                FROM project_notes n
                LEFT JOIN project_note_categories c ON c.id = n.category_id
                WHERE {' AND '.join(filters)}
                ORDER BY {order}
                LIMIT ?
                """,
                (*values, max(1, min(limit, 200))),
            ).fetchall()
            return [self._note_from_row(conn, row) for row in rows]

    def require_note(self, project_id: str, note_id: str, *, include_archived: bool = False) -> ProjectNote:
        if not _is_hex_id(note_id):
            raise ProjectError("Note id is not valid.")
        status_sql = "" if include_archived else "AND n.status = 'active'"
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT n.*, c.name AS category_name
                FROM project_notes n
                LEFT JOIN project_note_categories c ON c.id = n.category_id
                WHERE n.project_id = ? AND n.id = ? {status_sql}
                """,
                (project_id, note_id),
            ).fetchone()
            if row is None:
                raise ProjectError("Note was not found.")
            return self._note_from_row(conn, row)

    def create_map_run(
        self,
        project_id: str,
        *,
        run_type: str,
        full_rebuild: bool,
        previous_run_id: str | None = None,
        model_summaries_enabled: bool = False,
    ) -> MapRun:
        project = self.require_project(project_id)
        run_id = uuid.uuid4().hex
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO project_map_runs (
                    id, project_id, run_type, status, current_stage,
                    stage_index, stage_count, progress_percent, files_scanned,
                    files_indexed, files_reused, files_changed, files_deleted,
                    files_skipped, unsupported_files, warnings_json,
                    diagnostics_json, revision_branch, revision_commit,
                    previous_run_id, full_rebuild, deterministic_complete,
                    model_summaries_enabled, cancellation_requested,
                    created_at, started_at, completed_at, updated_at, last_error
                ) VALUES (?, ?, ?, 'queued', '', 0, ?, 0, 0, 0, 0, 0, 0, 0, 0,
                          '[]', '{}', ?, ?, ?, ?, 0, ?, 0, ?, NULL, NULL, ?, NULL)
                """,
                (
                    run_id, project.id, _text(run_type, 40) or "incremental",
                    len(MAP_STAGES), project.current_branch, project.head_commit,
                    previous_run_id, int(full_rebuild), int(model_summaries_enabled),
                    now, now,
                ),
            )
            for index, (key, name) in enumerate(MAP_STAGES, start=1):
                conn.execute(
                    """
                    INSERT INTO project_map_stages (
                        run_id, stage_key, name, stage_index, status,
                        progress_percent, diagnostics_json, started_at,
                        completed_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'pending', 0, '{}', NULL, NULL, ?)
                    """,
                    (run_id, key, name, index, now),
                )
            conn.execute(
                "UPDATE projects SET mapping_state = 'queued', mapping_detail = ?, updated_at = ? WHERE id = ?",
                ("Repository map refresh queued.", now, project.id),
            )
            self._activity(conn, project.id, "map_queued", "Repository map refresh queued.", {"run_id": run_id})
        return self.require_map_run(run_id)

    def require_map_run(self, run_id: str) -> MapRun:
        if not _is_hex_id(run_id):
            raise ProjectError("Map run id is not valid.")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM project_map_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise ProjectError("Map run was not found.")
        return self._map_run_from_row(row)

    def latest_map_run(self, project_id: str) -> MapRun | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM project_map_runs WHERE project_id = ? ORDER BY created_at DESC LIMIT 1",
                (project_id,),
            ).fetchone()
        return self._map_run_from_row(row) if row else None

    def list_map_runs(self, project_id: str, limit: int = 10) -> list[MapRun]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM project_map_runs WHERE project_id = ? ORDER BY created_at DESC LIMIT ?",
                (project_id, max(1, min(limit, 50))),
            ).fetchall()
        return [self._map_run_from_row(row) for row in rows]

    def list_map_stages(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM project_map_stages WHERE run_id = ? ORDER BY stage_index ASC",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def request_map_cancellation(self, run_id: str) -> None:
        run = self.require_map_run(run_id)
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE project_map_runs SET cancellation_requested = 1, updated_at = ? WHERE id = ?",
                (now, run.id),
            )
            self._activity(conn, run.project_id, "map_cancel_requested", "Repository map cancellation requested.", {"run_id": run.id})

    def recover_interrupted_map_runs(self) -> int:
        now = _utc_now()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, project_id FROM project_map_runs WHERE status IN ('queued', 'running')"
            ).fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE project_map_runs SET status = 'paused', updated_at = ?, last_error = ? WHERE id = ?",
                    (now, "Lewis restarted before this map run finished.", row["id"]),
                )
                conn.execute(
                    "UPDATE projects SET mapping_state = 'paused', mapping_detail = ?, updated_at = ? WHERE id = ?",
                    ("Previous map run was interrupted and can be resumed.", now, row["project_id"]),
                )
        return len(rows)

    def mark_run_started(self, run_id: str) -> None:
        run = self.require_map_run(run_id)
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE project_map_runs SET status = 'running', started_at = COALESCE(started_at, ?), updated_at = ? WHERE id = ?",
                (now, now, run.id),
            )
            conn.execute(
                "UPDATE projects SET mapping_state = 'running', mapping_detail = ?, updated_at = ? WHERE id = ?",
                ("Repository map refresh is running.", now, run.project_id),
            )

    def update_stage(
        self,
        run_id: str,
        stage_key: str,
        *,
        status: str,
        progress_percent: int = 0,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        now = _utc_now()
        stage_number = next((i for i, (key, _name) in enumerate(MAP_STAGES, 1) if key == stage_key), 0)
        completed = now if status in {"completed", "failed", "cancelled"} else None
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE project_map_stages
                SET status = ?, progress_percent = ?, diagnostics_json = ?,
                    started_at = COALESCE(started_at, ?),
                    completed_at = COALESCE(?, completed_at), updated_at = ?
                WHERE run_id = ? AND stage_key = ?
                """,
                (
                    status, _percent(progress_percent), _json(diagnostics or {}),
                    now, completed, now, run_id, stage_key,
                ),
            )
            conn.execute(
                "UPDATE project_map_runs SET current_stage = ?, stage_index = ?, progress_percent = ?, updated_at = ? WHERE id = ?",
                (stage_key, stage_number, _overall_progress(stage_number, progress_percent), now, run_id),
            )

    def update_run_counts(self, run_id: str, **counts: int) -> None:
        allowed = {
            "files_scanned", "files_indexed", "files_reused", "files_changed",
            "files_deleted", "files_skipped", "unsupported_files",
        }
        assignments = []
        values: list[object] = []
        for key, value in counts.items():
            if key in allowed:
                assignments.append(f"{key} = ?")
                values.append(max(0, int(value)))
        if not assignments:
            return
        assignments.append("updated_at = ?")
        values.append(_utc_now())
        values.append(run_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE project_map_runs SET {', '.join(assignments)} WHERE id = ?", tuple(values))

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        warnings: Sequence[str] = (),
        diagnostics: dict[str, Any] | None = None,
        branch: str | None = None,
        commit: str | None = None,
        language_summary: dict[str, int] | None = None,
        last_error: str | None = None,
    ) -> None:
        run = self.require_map_run(run_id)
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE project_map_runs
                SET status = ?, progress_percent = ?, warnings_json = ?,
                    diagnostics_json = ?, revision_branch = ?, revision_commit = ?,
                    deterministic_complete = ?, completed_at = ?, updated_at = ?,
                    last_error = ?
                WHERE id = ?
                """,
                (
                    status, 100 if status in {"completed", "partial"} else run.progress_percent,
                    _json(list(warnings)[:100]), _json(diagnostics or {}), branch, commit,
                    int(status == "completed"), now, now, _text(last_error or "", 500) or None,
                    run.id,
                ),
            )
            detail = _mapping_detail(status, last_error)
            conn.execute(
                """
                UPDATE projects
                SET mapping_state = ?, mapping_detail = ?, last_map_refreshed_at = ?,
                    current_branch = COALESCE(?, current_branch),
                    head_commit = COALESCE(?, head_commit),
                    language_summary_json = COALESCE(?, language_summary_json),
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    status, detail, now if status in {"completed", "partial"} else run.completed_at,
                    branch, commit, _json(language_summary) if language_summary is not None else None,
                    now, run.project_id,
                ),
            )
            self._activity(conn, run.project_id, f"map_{status}", detail, {"run_id": run.id, "diagnostics": diagnostics or {}})

    def indexed_hashes(self, project_id: str) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT path, content_hash FROM project_indexed_files WHERE project_id = ?", (project_id,)).fetchall()
        return {str(row["path"]): str(row["content_hash"]) for row in rows}

    def replace_indexed_file(
        self,
        project_id: str,
        run_id: str,
        extracted: ExtractedFile,
        *,
        git_status: str,
        indexed_at: str,
    ) -> None:
        descriptor = extracted.descriptor.to_record()
        path = str(descriptor["path"])
        parent = PurePosixPath(path).parent.as_posix()
        parent = "" if parent == "." else parent
        imports = [str(item) for item in descriptor.get("imports", [])]
        with self._connect() as conn:
            self._delete_path(conn, project_id, path)
            conn.execute(
                """
                INSERT INTO project_indexed_files (
                    project_id, run_id, path, name, parent_path, language,
                    role, file_type, size_bytes, line_count, content_hash,
                    git_status, safety_classification, imports_json,
                    summary, summary_source, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'deterministic', ?)
                """,
                (
                    project_id, run_id, path, PurePosixPath(path).name, parent,
                    descriptor.get("language", "other"), descriptor.get("role", "normal_source"),
                    PurePosixPath(path).suffix.lstrip(".") or "file",
                    int(descriptor.get("size_bytes", 0) or 0), int(descriptor.get("line_count", 0) or 0),
                    descriptor.get("content_hash", ""), git_status,
                    descriptor.get("safety_classification", "safe"), _json(imports),
                    descriptor.get("responsibility_summary", ""), indexed_at,
                ),
            )
            for symbol in extracted.symbols:
                conn.execute(
                    """
                    INSERT INTO project_indexed_symbols (
                        project_id, run_id, file_path, name, kind, line,
                        end_line, parent, signature, summary, deterministic,
                        indexed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                    """,
                    (
                        project_id, run_id, path, _text(symbol.get("name", ""), 200),
                        _text(symbol.get("kind", "symbol"), 80), int(symbol.get("line", 0) or 0),
                        int(symbol.get("end_line", symbol.get("line", 0)) or 0),
                        _optional(symbol.get("parent"), 200), _optional(symbol.get("signature"), 500) or "",
                        _symbol_summary(symbol), indexed_at,
                    ),
                )
            for route in extracted.routes:
                self._insert_relationship(
                    conn, project_id, run_id, "route", path, str(route.get("route", "")),
                    path, _optional(route.get("handler"), 200), "framework-registered",
                    f"{route.get('method', 'GET')} {route.get('route', '')}", True, indexed_at,
                )
            for test in extracted.tests:
                for module in test.get("referenced_modules", []):
                    self._insert_relationship(
                        conn, project_id, run_id, "test_references_module", path, None,
                        None, str(module), "inferred", f"Test references module {module}.", True, indexed_at,
                    )
            for rel in extracted.relationships:
                source_path, source_symbol = _split_endpoint(str(rel.get("source", "")))
                target_path, target_symbol = _split_endpoint(str(rel.get("target", "")))
                self._insert_relationship(
                    conn, project_id, run_id, _text(rel.get("kind", "related"), 80),
                    source_path or path, source_symbol, target_path, target_symbol,
                    "inferred", _text(rel.get("kind", "related"), 500), True, indexed_at,
                )

    def replace_import_relationships(self, project_id: str, run_id: str, relationships: Sequence[Relationship]) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("DELETE FROM project_relationships WHERE project_id = ? AND relationship_type = 'import'", (project_id,))
            for rel in relationships:
                self._insert_relationship(
                    conn, project_id, run_id, rel.relationship_type, rel.source_path,
                    rel.source_symbol, rel.target_path, rel.target_symbol,
                    rel.resolution, rel.summary, rel.deterministic, now,
                )

    def delete_indexed_paths(self, project_id: str, paths: Sequence[str]) -> None:
        with self._connect() as conn:
            for path in paths[:10_000]:
                self._delete_path(conn, project_id, path)

    def files_for_project(self, project_id: str, *, language: str = "", role: str = "", limit: int = 300) -> list[IndexedFile]:
        filters = ["project_id = ?"]
        values: list[object] = [project_id]
        if language:
            filters.append("language = ?")
            values.append(language)
        if role:
            filters.append("role = ?")
            values.append(role)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM project_indexed_files WHERE {' AND '.join(filters)} ORDER BY path COLLATE NOCASE LIMIT ?",
                (*values, max(1, min(limit, 1_000))),
            ).fetchall()
        return [self._file_from_row(row) for row in rows]

    def get_indexed_file(self, project_id: str, path: str) -> IndexedFile | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM project_indexed_files WHERE project_id = ? AND path = ?", (project_id, path)).fetchone()
        return self._file_from_row(row) if row else None

    def symbols_for_file(self, project_id: str, path: str, limit: int = 200) -> list[IndexedSymbol]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM project_indexed_symbols WHERE project_id = ? AND file_path = ? ORDER BY line ASC, name COLLATE NOCASE LIMIT ?",
                (project_id, path, max(1, min(limit, 500))),
            ).fetchall()
        return [self._symbol_from_row(row) for row in rows]

    def relationships_for_file(self, project_id: str, path: str, limit: int = 200) -> list[Relationship]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM project_relationships
                WHERE project_id = ? AND (source_path = ? OR target_path = ?)
                ORDER BY relationship_type, source_path, target_path LIMIT ?
                """,
                (project_id, path, path, max(1, min(limit, 500))),
            ).fetchall()
        return [self._relationship_from_row(row) for row in rows]

    def search_map(
        self,
        project_id: str,
        *,
        query: str = "",
        language: str = "",
        role: str = "",
        symbol_type: str = "",
        limit: int = 80,
    ) -> dict[str, Any]:
        q = query.strip()
        like = f"%{q}%"
        file_filters = ["project_id = ?"]
        file_values: list[object] = [project_id]
        if q:
            file_filters.append("(path LIKE ? OR summary LIKE ? OR imports_json LIKE ?)")
            file_values.extend([like, like, like])
        if language:
            file_filters.append("language = ?")
            file_values.append(language)
        if role:
            file_filters.append("role = ?")
            file_values.append(role)
        symbol_filters = ["project_id = ?"]
        symbol_values: list[object] = [project_id]
        if q:
            symbol_filters.append("(name LIKE ? OR file_path LIKE ? OR summary LIKE ?)")
            symbol_values.extend([like, like, like])
        if symbol_type:
            symbol_filters.append("kind = ?")
            symbol_values.append(symbol_type)
        rel_filters = ["project_id = ?"]
        rel_values: list[object] = [project_id]
        if q:
            rel_filters.append("(source_path LIKE ? OR target_path LIKE ? OR source_symbol LIKE ? OR target_symbol LIKE ? OR summary LIKE ?)")
            rel_values.extend([like, like, like, like, like])
        with self._connect() as conn:
            files = conn.execute(
                f"SELECT * FROM project_indexed_files WHERE {' AND '.join(file_filters)} ORDER BY path COLLATE NOCASE LIMIT ?",
                (*file_values, max(1, min(limit, 200))),
            ).fetchall()
            symbols = conn.execute(
                f"SELECT * FROM project_indexed_symbols WHERE {' AND '.join(symbol_filters)} ORDER BY file_path COLLATE NOCASE, line ASC LIMIT ?",
                (*symbol_values, max(1, min(limit, 200))),
            ).fetchall()
            relationships = conn.execute(
                f"SELECT * FROM project_relationships WHERE {' AND '.join(rel_filters)} ORDER BY relationship_type, source_path LIMIT ?",
                (*rel_values, max(1, min(limit, 200))),
            ).fetchall()
        return {
            "files": [self._file_from_row(row) for row in files],
            "symbols": [self._symbol_from_row(row) for row in symbols],
            "relationships": [self._relationship_from_row(row) for row in relationships],
        }

    def language_options(self, project_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT language FROM project_indexed_files WHERE project_id = ? AND language != '' ORDER BY language COLLATE NOCASE",
                (project_id,),
            ).fetchall()
        return [str(row["language"]) for row in rows]

    def role_options(self, project_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT role FROM project_indexed_files WHERE project_id = ? AND role != '' ORDER BY role COLLATE NOCASE",
                (project_id,),
            ).fetchall()
        return [str(row["role"]) for row in rows]

    def symbol_kind_options(self, project_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT kind FROM project_indexed_symbols WHERE project_id = ? AND kind != '' ORDER BY kind COLLATE NOCASE",
                (project_id,),
            ).fetchall()
        return [str(row["kind"]) for row in rows]

    def important_files(self, project_id: str, limit: int = 20) -> list[IndexedFile]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM project_indexed_files
                WHERE project_id = ?
                ORDER BY
                    CASE role
                        WHEN 'web_route' THEN 1
                        WHEN 'web_page' THEN 2
                        WHEN 'normal_source' THEN 3
                        WHEN 'test' THEN 4
                        ELSE 5
                    END,
                    size_bytes DESC
                LIMIT ?
                """,
                (project_id, max(1, min(limit, 50))),
            ).fetchall()
        return [self._file_from_row(row) for row in rows]

    def save_architecture_summaries(self, project_id: str, run_id: str, repo_summary: str, directory_summaries: dict[str, str]) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("DELETE FROM project_summaries WHERE project_id = ? AND source = 'deterministic'", (project_id,))
            conn.execute(
                """
                INSERT INTO project_summaries
                    (project_id, run_id, scope_type, scope_path, summary, source,
                     evidence_json, created_at, updated_at)
                VALUES (?, ?, 'repository', '', ?, 'deterministic', '{}', ?, ?)
                """,
                (project_id, run_id, repo_summary[:2_000], now, now),
            )
            for path, summary in sorted(directory_summaries.items())[:500]:
                conn.execute(
                    """
                    INSERT INTO project_summaries
                        (project_id, run_id, scope_type, scope_path, summary, source,
                         evidence_json, created_at, updated_at)
                    VALUES (?, ?, 'directory', ?, ?, 'deterministic', '{}', ?, ?)
                    """,
                    (project_id, run_id, path[:500], summary[:1_000], now, now),
                )

    def repository_summary(self, project_id: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT summary FROM project_summaries
                WHERE project_id = ? AND scope_type = 'repository'
                ORDER BY updated_at DESC LIMIT 1
                """,
                (project_id,),
            ).fetchone()
        return str(row["summary"]) if row else ""

    def generate_diagram(self, project_id: str, *, title: str, diagram_type: str = "module_dependency") -> Diagram:
        project = self.require_project(project_id)
        latest = self.latest_map_run(project.id)
        diagram_id = uuid.uuid4().hex
        now = _utc_now()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT relationship_type, source_path, target_path, source_symbol,
                       target_symbol, resolution
                FROM project_relationships
                WHERE project_id = ? AND relationship_type IN
                    ('import', 'route', 'test_references_module', 'page_to_asset')
                ORDER BY relationship_type, source_path, target_path
                LIMIT 240
                """,
                (project.id,),
            ).fetchall()
            definition = _diagram_definition(rows)
            conn.execute(
                """
                INSERT INTO project_diagrams (
                    id, project_id, map_run_id, title, diagram_type,
                    definition_json, generation_source, status, created_at,
                    updated_at, archived_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'deterministic', 'active', ?, ?, NULL)
                """,
                (
                    diagram_id, project.id, latest.id if latest else None,
                    _required(title, 160, "Diagram title"), _text(diagram_type, 80),
                    _json(definition), now, now,
                ),
            )
            self._activity(conn, project.id, "diagram_generated", f"Generated diagram {title}.", {"diagram_id": diagram_id})
        return self.require_diagram(project.id, diagram_id)

    def list_diagrams(self, project_id: str, include_archived: bool = False) -> list[Diagram]:
        filters = ["project_id = ?"]
        if not include_archived:
            filters.append("status = 'active'")
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM project_diagrams WHERE {' AND '.join(filters)} ORDER BY updated_at DESC LIMIT 100",
                (project_id,),
            ).fetchall()
        return [self._diagram_from_row(row) for row in rows]

    def require_diagram(self, project_id: str, diagram_id: str) -> Diagram:
        if not _is_hex_id(diagram_id):
            raise ProjectError("Diagram id is not valid.")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM project_diagrams WHERE project_id = ? AND id = ?", (project_id, diagram_id)).fetchone()
        if row is None:
            raise ProjectError("Diagram was not found.")
        return self._diagram_from_row(row)

    def archive_diagram(self, project_id: str, diagram_id: str, *, confirm: str) -> None:
        if confirm != "archive-diagram":
            raise ProjectError("Explicit diagram archive confirmation is required.")
        diagram = self.require_diagram(project_id, diagram_id)
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE project_diagrams SET status = 'archived', archived_at = ?, updated_at = ? WHERE project_id = ? AND id = ?",
                (now, now, project_id, diagram.id),
            )
            self._activity(conn, project_id, "diagram_archived", f"Archived diagram {diagram.title}.", {"diagram_id": diagram.id})

    def recent_activity(self, project_id: str, limit: int = 40) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM project_activity WHERE project_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
                (project_id, max(1, min(limit, 200))),
            ).fetchall()
        return [dict(row) | {"details": _dict(row["details_json"])} for row in rows]

    def related_coding_agent_jobs(self, project_id: str, limit: int = 10) -> list[dict[str, Any]]:
        project = self.require_project(project_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM coding_agent_jobs
                WHERE project_id = ? OR repository_root = ?
                ORDER BY updated_at DESC, created_at DESC LIMIT ?
                """,
                (project.id, project.repository_root, max(1, min(limit, 50))),
            ).fetchall()
        return [dict(row) for row in rows]

    def trace_symbol(self, project_id: str, *, symbol: str, mode: str = "quick", file: str | None = None) -> dict[str, Any]:
        project = self.require_project(project_id)
        outcome = CodeTracer(Path(project.repository_root)).trace(symbol, mode=mode, file=file)
        with self._connect() as conn:
            self._activity(conn, project.id, "trace_run", f"Ran {mode} trace for {symbol}.", {"symbol": symbol, "mode": mode, "ok": outcome.result.ok})
        return {"result": outcome.result, "report": outcome.report}

    def _project_from_row(self, conn: sqlite3.Connection, row: sqlite3.Row) -> Project:
        project_id = str(row["id"])
        tag_rows = conn.execute("SELECT tag FROM project_tags WHERE project_id = ? ORDER BY tag COLLATE NOCASE", (project_id,)).fetchall()
        counts = conn.execute(
            "SELECT COUNT(*) AS total, SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active FROM project_notes WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        latest = conn.execute(
            "SELECT id, status, last_error FROM project_map_runs WHERE project_id = ? ORDER BY created_at DESC LIMIT 1",
            (project_id,),
        ).fetchone()
        return Project(
            id=project_id,
            display_name=str(row["display_name"]),
            repository_root=str(row["repository_root"]),
            description=str(row["description"] or ""),
            status=str(row["status"]),
            color=str(row["color"] or ""),
            icon=str(row["icon"] or ""),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            archived_at=row["archived_at"],
            last_map_refreshed_at=row["last_map_refreshed_at"],
            current_branch=row["current_branch"],
            head_commit=row["head_commit"],
            language_summary=_dict(row["language_summary_json"]),
            mapping_state=str(row["mapping_state"]),
            mapping_detail=str(row["mapping_detail"] or ""),
            tags=tuple(str(tag["tag"]) for tag in tag_rows),
            notes_count=int(counts["total"] or 0),
            active_notes_count=int(counts["active"] or 0),
            latest_run_id=str(latest["id"]) if latest else None,
            latest_run_status=str(latest["status"]) if latest else None,
            latest_run_error=str(latest["last_error"]) if latest and latest["last_error"] else None,
        )

    def _note_from_row(self, conn: sqlite3.Connection, row: sqlite3.Row) -> ProjectNote:
        tags = conn.execute("SELECT tag FROM project_note_tags WHERE note_id = ? ORDER BY tag COLLATE NOCASE", (row["id"],)).fetchall()
        return ProjectNote(
            id=str(row["id"]), project_id=str(row["project_id"]), title=str(row["title"]),
            content_markdown=str(row["content_markdown"] or ""),
            category_id=int(row["category_id"]) if row["category_id"] is not None else None,
            category_name=row["category_name"], is_pinned=bool(row["is_pinned"]),
            status=str(row["status"]), created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]), archived_at=row["archived_at"],
            tags=tuple(str(tag["tag"]) for tag in tags),
        )

    def _map_run_from_row(self, row: sqlite3.Row) -> MapRun:
        return MapRun(
            id=str(row["id"]), project_id=str(row["project_id"]), run_type=str(row["run_type"]),
            status=str(row["status"]), current_stage=str(row["current_stage"] or ""),
            stage_index=int(row["stage_index"] or 0), stage_count=int(row["stage_count"] or len(MAP_STAGES)),
            progress_percent=int(row["progress_percent"] or 0), files_scanned=int(row["files_scanned"] or 0),
            files_indexed=int(row["files_indexed"] or 0), files_reused=int(row["files_reused"] or 0),
            files_changed=int(row["files_changed"] or 0), files_deleted=int(row["files_deleted"] or 0),
            files_skipped=int(row["files_skipped"] or 0), unsupported_files=int(row["unsupported_files"] or 0),
            warnings=tuple(str(item) for item in _list(row["warnings_json"])), diagnostics=_dict(row["diagnostics_json"]),
            revision_branch=row["revision_branch"], revision_commit=row["revision_commit"],
            previous_run_id=row["previous_run_id"], full_rebuild=bool(row["full_rebuild"]),
            deterministic_complete=bool(row["deterministic_complete"]), model_summaries_enabled=bool(row["model_summaries_enabled"]),
            cancellation_requested=bool(row["cancellation_requested"]), created_at=str(row["created_at"]),
            started_at=row["started_at"], completed_at=row["completed_at"], updated_at=str(row["updated_at"]),
            last_error=row["last_error"],
        )

    def _file_from_row(self, row: sqlite3.Row) -> IndexedFile:
        return IndexedFile(
            path=str(row["path"]), name=str(row["name"]), parent_path=str(row["parent_path"] or ""),
            language=str(row["language"] or ""), role=str(row["role"] or ""), file_type=str(row["file_type"] or ""),
            size_bytes=int(row["size_bytes"] or 0), line_count=int(row["line_count"] or 0),
            content_hash=str(row["content_hash"] or ""), git_status=str(row["git_status"] or ""),
            safety_classification=str(row["safety_classification"] or "safe"),
            imports=tuple(str(item) for item in _list(row["imports_json"])), summary=str(row["summary"] or ""),
            summary_source=str(row["summary_source"] or "deterministic"), indexed_at=str(row["indexed_at"] or ""),
        )

    def _symbol_from_row(self, row: sqlite3.Row) -> IndexedSymbol:
        return IndexedSymbol(
            id=int(row["id"]), file_path=str(row["file_path"]), name=str(row["name"]),
            kind=str(row["kind"]), line=int(row["line"] or 0), end_line=int(row["end_line"] or 0),
            parent=row["parent"], summary=str(row["summary"] or ""), deterministic=bool(row["deterministic"]),
        )

    def _relationship_from_row(self, row: sqlite3.Row) -> Relationship:
        return Relationship(
            id=int(row["id"]), relationship_type=str(row["relationship_type"]),
            source_path=str(row["source_path"] or ""), source_symbol=row["source_symbol"],
            target_path=row["target_path"], target_symbol=row["target_symbol"],
            resolution=str(row["resolution"] or "inferred"), summary=str(row["summary"] or ""),
            deterministic=bool(row["deterministic"]),
        )

    def _diagram_from_row(self, row: sqlite3.Row) -> Diagram:
        return Diagram(
            id=str(row["id"]), project_id=str(row["project_id"]), map_run_id=row["map_run_id"],
            title=str(row["title"]), diagram_type=str(row["diagram_type"]), definition=_dict(row["definition_json"]),
            generation_source=str(row["generation_source"]), status=str(row["status"]),
            created_at=str(row["created_at"]), updated_at=str(row["updated_at"]), archived_at=row["archived_at"],
        )

    def _ensure_category(self, conn: sqlite3.Connection, project_id: str, name: str) -> int | None:
        clean = _text(name, 80).strip()
        if not clean:
            return None
        row = conn.execute(
            "SELECT id FROM project_note_categories WHERE project_id = ? AND lower(name) = lower(?)",
            (project_id, clean),
        ).fetchone()
        now = _utc_now()
        if row:
            conn.execute("UPDATE project_note_categories SET status = 'active', updated_at = ? WHERE id = ?", (now, int(row["id"])))
            return int(row["id"])
        cursor = conn.execute(
            "INSERT INTO project_note_categories (project_id, name, status, created_at, updated_at) VALUES (?, ?, 'active', ?, ?)",
            (project_id, clean, now, now),
        )
        return int(cursor.lastrowid)

    def _replace_tags(self, conn: sqlite3.Connection, table: str, owner_column: str, owner_id: str, tags: Sequence[str]) -> None:
        conn.execute(f"DELETE FROM {table} WHERE {owner_column} = ?", (owner_id,))
        for tag in _tags(tags):
            conn.execute(f"INSERT OR IGNORE INTO {table} ({owner_column}, tag) VALUES (?, ?)", (owner_id, tag))

    def _delete_path(self, conn: sqlite3.Connection, project_id: str, path: str) -> None:
        conn.execute("DELETE FROM project_indexed_files WHERE project_id = ? AND path = ?", (project_id, path))
        conn.execute("DELETE FROM project_indexed_symbols WHERE project_id = ? AND file_path = ?", (project_id, path))
        conn.execute(
            "DELETE FROM project_relationships WHERE project_id = ? AND (source_path = ? OR target_path = ?)",
            (project_id, path, path),
        )

    def _insert_relationship(
        self,
        conn: sqlite3.Connection,
        project_id: str,
        run_id: str,
        relationship_type: str,
        source_path: str,
        source_symbol: str | None,
        target_path: str | None,
        target_symbol: str | None,
        resolution: str,
        summary: str,
        deterministic: bool,
        indexed_at: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO project_relationships (
                project_id, run_id, relationship_type, source_type, source_path,
                source_symbol, target_type, target_path, target_symbol,
                resolution, summary, deterministic, indexed_at
            ) VALUES (?, ?, ?, 'file', ?, ?, 'file', ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id, run_id, relationship_type[:80], source_path[:500],
                _optional(source_symbol, 300), _optional(target_path, 500),
                _optional(target_symbol, 300), resolution[:80], summary[:1_000],
                int(deterministic), indexed_at,
            ),
        )

    def _activity(self, conn: sqlite3.Connection, project_id: str, event_type: str, summary: str, details: dict[str, Any] | None = None) -> None:
        conn.execute(
            "INSERT INTO project_activity (project_id, event_type, summary, details_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (project_id, event_type[:80], _text(summary, 500), _bounded_json(details or {}, 4_000), _utc_now()),
        )


class ProjectMapService:
    def __init__(self, settings: Settings, *, store: ProjectStore | None = None):
        self.settings = settings
        self.store = store or ProjectStore(settings.database_path)
        self._active: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self.store.recover_interrupted_map_runs()

    def start_refresh(self, project_id: str, *, full_rebuild: bool = False, resume_run_id: str | None = None) -> MapRun:
        project = self.store.require_project(project_id)
        latest = self.store.latest_map_run(project.id)
        if latest and latest.status in {"queued", "running"}:
            raise ProjectError("A repository map refresh is already queued or running.")
        previous = None
        if resume_run_id:
            previous = self.store.require_map_run(resume_run_id)
            if previous.project_id != project.id or previous.status not in RESUMABLE_RUN_STATES:
                raise ProjectError("Only a paused, failed, or cancelled map run can be resumed.")
        run = self.store.create_map_run(
            project.id,
            run_type="full" if full_rebuild else ("resume" if previous else "incremental"),
            full_rebuild=full_rebuild,
            previous_run_id=previous.id if previous else (latest.id if latest else None),
            model_summaries_enabled=self.settings.project_map_model_summaries,
        )
        cancel_event = threading.Event()
        with self._lock:
            self._active[run.id] = cancel_event
        thread = threading.Thread(target=self._run_refresh, args=(run.id, cancel_event), daemon=True)
        thread.start()
        return run

    def cancel(self, run_id: str) -> None:
        self.store.request_map_cancellation(run_id)
        with self._lock:
            event = self._active.get(run_id)
        if event:
            event.set()

    def _run_refresh(self, run_id: str, cancel_event: threading.Event) -> None:
        try:
            ProjectMapWorker(self.settings, self.store, cancel_event).run(run_id)
        finally:
            with self._lock:
                self._active.pop(run_id, None)


class ProjectMapWorker:
    def __init__(self, settings: Settings, store: ProjectStore, cancel_event: threading.Event):
        self.settings = settings
        self.store = store
        self.cancel_event = cancel_event
        self.started = time.monotonic()

    def run(self, run_id: str) -> None:
        run = self.store.require_map_run(run_id)
        project = self.store.require_project(run.project_id)
        self.store.mark_run_started(run.id)
        warnings: list[str] = []
        diagnostics: dict[str, Any] = {"model_generated": False, "deterministic_facts": True}
        try:
            repository = validate_project_repository(self.settings, repository_path=project.repository_root)
            root = repository.root
            metadata = read_git_metadata(root)
            tracked = git_tracked_paths(root)
            status_by_path = git_status_by_path(root)
            existing_hashes = self.store.indexed_hashes(project.id)
            prior_hashes = {} if run.full_rebuild else existing_hashes
            seen_paths: set[str] = set()
            extracted_records: list[tuple[ExtractedFile, str]] = []
            counts = {
                "files_scanned": 0,
                "files_indexed": 0,
                "files_reused": 0,
                "files_changed": 0,
                "files_deleted": 0,
                "files_skipped": 0,
                "unsupported_files": 0,
            }
            skipped: dict[str, int] = {}

            self._stage(run.id, "inventory", "running", 5)
            candidates = list(iter_project_files(root, self.settings))
            self._check_cancel(run.id)
            self._stage(run.id, "inventory", "completed", 100, {"candidates": len(candidates)})

            self._stage(run.id, "classification", "running", 10)
            for item in candidates:
                self._check_cancel(run.id)
                if item.reason:
                    skipped[item.reason] = skipped.get(item.reason, 0) + 1
                    counts["files_skipped"] += 1
                    if item.reason == "unsupported_file_type":
                        counts["unsupported_files"] += 1
                    continue
                counts["files_scanned"] += 1
                seen_paths.add(item.relative)
                digest = hashlib.sha256(item.text.encode("utf-8")).hexdigest()
                if prior_hashes.get(item.relative) == digest:
                    counts["files_reused"] += 1
                    continue
                approval_sensitive = _is_approval_sensitive(item.relative, PurePosixPath(item.relative).name, item.text)
                extracted = extract_file(item.relative, item.text, approval_sensitive=approval_sensitive)
                git_status = status_by_path.get(item.relative, "tracked" if item.relative in tracked else "untracked")
                extracted_records.append((extracted, git_status))
                counts["files_changed"] += 1
            deleted = sorted(set(existing_hashes) - seen_paths)
            counts["files_deleted"] = len(deleted)
            self.store.update_run_counts(run.id, **counts)
            self._stage(run.id, "classification", "completed", 100, {"skipped": skipped})

            self._stage(run.id, "symbols", "running", 10)
            now = _utc_now()
            self.store.delete_indexed_paths(project.id, deleted)
            for index, (extracted, git_status) in enumerate(extracted_records, start=1):
                self._check_cancel(run.id)
                self.store.replace_indexed_file(project.id, run.id, extracted, git_status=git_status, indexed_at=now)
                counts["files_indexed"] += 1
                if index % 20 == 0:
                    self.store.update_run_counts(run.id, **counts)
                    self._stage(run.id, "symbols", "running", int(index / max(1, len(extracted_records)) * 100))
            self.store.update_run_counts(run.id, **counts)
            self._stage(run.id, "symbols", "completed", 100)

            self._stage(run.id, "dependencies", "running", 20)
            files = self.store.files_for_project(project.id, limit=self.settings.project_map_max_files)
            imports = derive_import_relationships(files)
            self.store.replace_import_relationships(project.id, run.id, imports)
            self._stage(run.id, "dependencies", "completed", 100, {"relationships": len(imports)})

            self._stage(run.id, "relationships", "completed", 100)
            self._stage(run.id, "tests", "completed", 100)
            self._stage(run.id, "architecture", "running", 50)
            language_summary = language_counts(files)
            repo_summary, directory_summaries = build_deterministic_summaries(files)
            self.store.save_architecture_summaries(project.id, run.id, repo_summary, directory_summaries)
            self._stage(run.id, "architecture", "completed", 100)

            if skipped.get("scan_limit"):
                warnings.append("File discovery reached the configured maximum file limit.")
            if skipped.get("oversized"):
                warnings.append("Some files exceeded the configured file-size limit.")
            if self.settings.project_map_model_summaries and self.settings.project_map_max_model_calls <= 0:
                warnings.append("Model summaries were enabled but model call budget is zero; deterministic summaries were stored only.")
            diagnostics.update({"skipped": skipped, "changed_paths": [r.descriptor.path for r, _ in extracted_records[:200]], "deleted_paths": deleted[:200]})
            self._stage(run.id, "summaries", "completed", 100, {"source": "deterministic"})
            self._stage(run.id, "consistency", "completed", 100)
            self.store.finish_run(
                run.id,
                status="partial" if warnings else "completed",
                warnings=warnings,
                diagnostics=diagnostics,
                branch=metadata.get("branch"),
                commit=metadata.get("head_commit"),
                language_summary=language_summary,
            )
        except ProjectCancelled:
            self.store.finish_run(run.id, status="cancelled", warnings=warnings, diagnostics=diagnostics, last_error="Repository map refresh was cancelled.")
        except Exception as exc:
            self.store.finish_run(run.id, status="failed", warnings=warnings, diagnostics=diagnostics, last_error=str(exc)[:500])

    def _stage(self, run_id: str, key: str, status: str, progress: int, diagnostics: dict[str, Any] | None = None) -> None:
        self._check_timeout()
        self.store.update_stage(run_id, key, status=status, progress_percent=progress, diagnostics=diagnostics)

    def _check_cancel(self, run_id: str) -> None:
        if self.cancel_event.is_set() or self.store.require_map_run(run_id).cancellation_requested:
            raise ProjectCancelled()
        self._check_timeout()

    def _check_timeout(self) -> None:
        if time.monotonic() - self.started > max(30, self.settings.project_map_timeout_seconds):
            raise ProjectError("Repository map refresh exceeded the configured timeout.")


def configured_project_repository_options(settings: Settings) -> list[ApprovedRepository]:
    repo_settings = replace(settings, coding_agent_repositories=settings.approved_project_repository_roots)
    return coding_agents.configured_repository_options(repo_settings)


def approved_project_repositories(settings: Settings) -> list[ApprovedRepository]:
    repo_settings = replace(settings, coding_agent_repositories=settings.approved_project_repository_roots)
    return coding_agents.approved_repositories(repo_settings)


def validate_project_repository(settings: Settings, *, repository_id: str = "", repository_path: str = "") -> ApprovedRepository:
    repo_settings = replace(settings, coding_agent_repositories=settings.approved_project_repository_roots)
    if repository_id.strip():
        return coding_agents.repository_by_id(repo_settings, repository_id.strip())
    if not repository_path.strip():
        raise RepositoryRefusedError("Choose or enter one approved repository root.")
    try:
        requested = Path(repository_path).expanduser().resolve(strict=False)
    except OSError as exc:
        raise RepositoryRefusedError("Repository path could not be resolved.") from exc
    for repository in coding_agents.approved_repositories(repo_settings):
        if repository.valid and str(repository.root).casefold() == str(requested).casefold():
            return repository
    raise RepositoryRefusedError("Repository path must exactly match one configured approved root.")


def read_git_metadata(root: Path) -> dict[str, str | None]:
    branch = _git_stdout(["git", "rev-parse", "--abbrev-ref", "HEAD"], root)
    commit = _git_stdout(["git", "rev-parse", "HEAD"], root)
    return {"branch": branch or None, "head_commit": commit or None}


def git_tracked_paths(root: Path) -> set[str]:
    output = _git_bytes(["git", "ls-files", "-z"], root)
    return {item.decode("utf-8", errors="replace") for item in output.split(b"\x00") if item}


def git_status_by_path(root: Path) -> dict[str, str]:
    output = _git_bytes(["git", "--no-optional-locks", "status", "--porcelain=v1", "-z", "--untracked-files=all"], root)
    values: dict[str, str] = {}
    chunks = [item.decode("utf-8", errors="replace") for item in output.split(b"\x00") if item]
    index = 0
    while index < len(chunks):
        item = chunks[index]
        code = item[:2].strip() or "modified"
        path = item[3:] if len(item) > 3 else item
        if code.startswith("R") and index + 1 < len(chunks):
            values[chunks[index + 1]] = "renamed-from"
            values[path] = "renamed"
            index += 2
            continue
        values[path] = _git_status_label(code)
        index += 1
    return values


def iter_project_files(root: Path, settings: Settings) -> Iterator[CandidateFile]:
    max_files = max(1, settings.project_map_max_files)
    ignored = {item.casefold() for item in settings.project_map_ignored_directories}
    ignored.update(part.casefold() for part in GENERATED_OR_CACHE_PARTS)
    scanned = 0
    for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        kept_dirs = []
        for name in sorted(directory_names):
            child = directory_path / name
            if name.casefold() in ignored or name.startswith(".") or child.is_symlink():
                continue
            kept_dirs.append(name)
        directory_names[:] = kept_dirs
        for name in sorted(file_names):
            if scanned >= max_files:
                yield CandidateFile("<scan-limit>", directory_path / name, "", "scan_limit")
                return
            scanned += 1
            path = directory_path / name
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                yield CandidateFile(name, path, "", "outside_root")
                continue
            text, reason = read_indexable_text(relative, path, root, settings)
            yield CandidateFile(relative, path, text, reason)


def read_indexable_text(relative: str, path: Path, root: Path, settings: Settings) -> tuple[str, str | None]:
    posix = PurePosixPath(relative)
    parts = tuple(part.casefold() for part in posix.parts)
    ignored = {item.casefold() for item in settings.project_map_ignored_directories}
    if any(part in ignored or part.startswith(".") for part in parts):
        return "", "ignored_path"
    if path.is_symlink():
        return "", "symlink"
    if posix.suffix.casefold() in DATABASE_SUFFIXES:
        return "", "database_file"
    if posix.name.casefold().startswith(".env"):
        return "", "secret_file"
    if posix.suffix.casefold() not in INDEXABLE_SUFFIXES:
        return "", "unsupported_file_type"
    try:
        validate_relative_project_path(relative, root)
    except (ValueError, OSError):
        return "", "unsafe_path"
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
        if not resolved.is_file() or path.is_symlink():
            return "", "unsupported_file_type"
        if resolved.stat().st_size > max(1, settings.project_map_max_file_bytes):
            return "", "oversized"
        data = resolved.read_bytes()
    except (OSError, ValueError):
        return "", "unreadable"
    if b"\x00" in data[:4096]:
        return "", "binary"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return "", "binary"
    if contains_sensitive_content(text):
        return "", "sensitive_content"
    return text, None


def derive_import_relationships(files: Sequence[IndexedFile]) -> list[Relationship]:
    by_python_module: dict[str, str] = {}
    by_path = {record.path: record for record in files}
    for record in files:
        if record.language == "python" and record.path.endswith(".py"):
            module = record.path[:-3].replace("/", ".")
            if module.endswith(".__init__"):
                module = module[: -len(".__init__")]
            by_python_module[module] = record.path
    relationships = []
    next_id = 1
    for record in files:
        for imported in record.imports[:200]:
            target = _resolve_import(record, imported, by_python_module, by_path)
            relationships.append(
                Relationship(
                    id=next_id,
                    relationship_type="import",
                    source_path=record.path,
                    source_symbol=None,
                    target_path=target,
                    target_symbol=imported,
                    resolution="confirmed" if target else "external",
                    summary=f"{record.path} imports {imported}" + (f" from {target}." if target else " outside the indexed repository."),
                    deterministic=True,
                )
            )
            next_id += 1
    return relationships[:5_000]


def language_counts(files: Sequence[IndexedFile]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in files:
        counts[record.language] = counts.get(record.language, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def build_deterministic_summaries(files: Sequence[IndexedFile]) -> tuple[str, dict[str, str]]:
    languages = language_counts(files)
    roles: dict[str, int] = {}
    directories: dict[str, int] = {}
    for record in files:
        roles[record.role] = roles.get(record.role, 0) + 1
        top = record.path.split("/", 1)[0] if "/" in record.path else "(root)"
        directories[top] = directories.get(top, 0) + 1
    language_text = ", ".join(f"{name}: {count}" for name, count in list(languages.items())[:8]) or "none"
    role_text = ", ".join(f"{name}: {count}" for name, count in sorted(roles.items())[:8]) or "none"
    repo = (
        f"Deterministic repository map with {len(files)} indexed files. "
        f"Languages: {language_text}. Roles: {role_text}. "
        "No model-generated architecture claims are included in this summary."
    )
    directory_summaries = {
        path: f"{path} contains {count} indexed files in the current deterministic map."
        for path, count in sorted(directories.items(), key=lambda item: (-item[1], item[0]))
    }
    return repo, directory_summaries


def _resolve_import(record: IndexedFile, imported: str, by_python_module: dict[str, str], by_path: dict[str, IndexedFile]) -> str | None:
    if record.language == "python":
        parts = imported.split(".")
        while parts:
            module = ".".join(parts)
            if module in by_python_module:
                return by_python_module[module]
            parts.pop()
        return None
    if record.language in {"javascript", "typescript"} and imported.startswith("."):
        base = PurePosixPath(record.path).parent
        normalized = _normalize_relative(base, imported)
        if not normalized:
            return None
        candidates = [normalized]
        if not PurePosixPath(normalized).suffix:
            candidates.extend(f"{normalized}{suffix}" for suffix in (".ts", ".tsx", ".js", ".jsx", ".mjs"))
            candidates.extend(f"{normalized}/index{suffix}" for suffix in (".ts", ".tsx", ".js", ".jsx"))
        for candidate in candidates:
            if candidate in by_path:
                return candidate
    return None


def _normalize_relative(base: PurePosixPath, specifier: str) -> str | None:
    parts = list(base.parts)
    for segment in specifier.split("/"):
        if segment in {"", "."}:
            continue
        if segment == "..":
            if not parts:
                return None
            parts.pop()
            continue
        parts.append(segment)
    return "/".join(parts)


def _diagram_definition(rows: Sequence[sqlite3.Row]) -> dict[str, Any]:
    nodes: dict[str, dict[str, str]] = {}
    edges: list[dict[str, str]] = []
    for row in rows:
        source = str(row["source_path"] or "")
        target = str(row["target_path"] or row["target_symbol"] or "")
        if not source or not target:
            continue
        if len(nodes) >= 80 and (source not in nodes or target not in nodes):
            continue
        nodes.setdefault(source, {"id": source, "label": _short_label(source)})
        nodes.setdefault(target, {"id": target, "label": _short_label(target)})
        if len(edges) < 120:
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "kind": str(row["relationship_type"]),
                    "resolution": str(row["resolution"]),
                }
            )
    return {
        "schema": "lewis-project-diagram/1",
        "renderer": "bounded-list-graph",
        "nodes": list(nodes.values()),
        "edges": edges,
        "truncated": len(edges) >= 120 or len(nodes) >= 80,
    }


def _git_stdout(args: Sequence[str], root: Path) -> str:
    return _git_bytes(args, root).decode("utf-8", errors="replace").strip()


def _git_bytes(args: Sequence[str], root: Path) -> bytes:
    if not _allowed_git_command(args):
        raise ProjectError("Internal error: forbidden Git metadata command.")
    try:
        completed = subprocess.run(
            list(args),
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProjectError("Git metadata command could not complete safely.") from exc
    if completed.returncode != 0:
        return b""
    return bytes(completed.stdout or b"")[:MAX_GIT_OUTPUT_BYTES]


def _allowed_git_command(args: Sequence[str]) -> bool:
    values = tuple(args)
    return values in {
        ("git", "rev-parse", "--abbrev-ref", "HEAD"),
        ("git", "rev-parse", "HEAD"),
        ("git", "ls-files", "-z"),
        ("git", "--no-optional-locks", "status", "--porcelain=v1", "-z", "--untracked-files=all"),
    }


def _git_status_label(code: str) -> str:
    if code == "??":
        return "untracked"
    if "D" in code:
        return "deleted"
    if "A" in code:
        return "added"
    if "M" in code:
        return "modified"
    if "R" in code:
        return "renamed"
    return "tracked"


def _mapping_detail(status: str, last_error: str | None) -> str:
    if status == "completed":
        return "Repository map is complete for the recorded revision."
    if status == "partial":
        return "Repository map completed with bounded skips or warnings."
    if status == "cancelled":
        return "Repository map refresh was cancelled; partial results remain available."
    if status == "failed":
        return f"Repository map refresh failed: {last_error or 'unknown error'}"
    return status


def _overall_progress(stage_number: int, stage_progress: int) -> int:
    if stage_number <= 0:
        return _percent(stage_progress)
    return _percent(int((((stage_number - 1) + stage_progress / 100) / len(MAP_STAGES)) * 100))


def _percent(value: int) -> int:
    return max(0, min(100, int(value)))


def _split_endpoint(endpoint: str) -> tuple[str | None, str | None]:
    if not endpoint:
        return None, None
    if endpoint.startswith(".") or endpoint.startswith("#"):
        return None, endpoint
    if ":" in endpoint:
        path, symbol = endpoint.split(":", 1)
        return path or None, symbol or None
    if "/" in endpoint or "." in PurePosixPath(endpoint).name:
        return endpoint, None
    return None, endpoint


def _symbol_summary(symbol: dict[str, object]) -> str:
    return f"Deterministic {symbol.get('kind', 'symbol')} definition {symbol.get('name', '')}."[:500]


def _short_label(path: str) -> str:
    pure = PurePosixPath(path)
    if len(pure.parts) <= 2:
        return path
    return "/".join(pure.parts[-2:])


def _project_status(status: str) -> str:
    clean = str(status or "active").strip().lower()
    if clean not in PROJECT_STATUSES:
        raise ProjectError("Project status must be active, paused, or archived.")
    return clean


def _tags(tags: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        clean = _tag(str(tag))
        if not clean or clean.casefold() in seen:
            continue
        seen.add(clean.casefold())
        normalized.append(clean)
        if len(normalized) >= 20:
            break
    return tuple(normalized)


def _tag(tag: str) -> str:
    return re.sub(r"\s+", " ", tag.strip())[:48]


def _required(value: object, limit: int, label: str) -> str:
    clean = _text(value, limit).strip()
    if not clean:
        raise ProjectError(f"{label} is required.")
    return clean


def _text(value: object, limit: int) -> str:
    return ("" if value is None else str(value)).replace("\x00", "")[:limit]


def _optional(value: object, limit: int) -> str | None:
    clean = _text(value, limit).strip()
    return clean or None


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def _bounded_json(value: object, limit: int) -> str:
    encoded = _json(value)
    if len(encoded) <= limit:
        return encoded
    return _json({"truncated": True, "preview": encoded[: max(0, limit - 80)]})


def _list(value: object) -> list[Any]:
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _dict(value: object) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _is_hex_id(value: str) -> bool:
    return isinstance(value, str) and len(value) == 32 and all(ch in "0123456789abcdef" for ch in value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
