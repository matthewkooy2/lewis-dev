"""Bounded read-only coding-agent jobs for Codex and Claude CLIs."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import subprocess
import threading
import time
from typing import Any
import uuid

from app.config import DB_PATH, ROOT_DIR, Settings
from app.db import connect_database, initialize_database

PROVIDER_CODEX = "codex"
PROVIDER_CLAUDE = "claude"
KNOWN_PROVIDERS = frozenset({PROVIDER_CODEX, PROVIDER_CLAUDE})
EVENT_TYPES = frozenset(
    {
        "job_started",
        "agent_message",
        "command_started",
        "command_finished",
        "file_observed",
        "usage_updated",
        "job_completed",
        "job_failed",
        "job_cancelled",
    }
)
CLAUDE_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
MAX_GOAL_LENGTH = 10_000
MAX_MODEL_LENGTH = 120
MAX_TURNS_LIMIT = 20
MIN_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 3_600
DEFAULT_TIMEOUT_SECONDS = 600
MAX_PROCESS_OUTPUT_BYTES = 1_000_000
MAX_EVENT_SUMMARY_CHARS = 1_000
MAX_EVENT_CONTENT_CHARS = 4_000
MAX_REPOSITORY_SCAN_ENTRIES = 50_000
STATUS_OUTPUT_LIMIT_BYTES = 20_000
STATUS_TIMEOUT_SECONDS = 10
_OUTPUT_TRUNCATION_MARKER = "\n[output truncated]\n"
_MODEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,119}")
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:api[_-]?key|auth|bearer|credential|password|secret|token)", re.I
)
_REASONING_KEY_NAMES = {
    "reasoning",
    "thinking",
    "thought",
    "chain_of_thought",
    "hidden_reasoning",
    "encrypted_content",
}
_ALLOWED_ENV_KEYS = {
    "ALLUSERSPROFILE",
    "APPDATA",
    "CODEX_HOME",
    "HOME",
    "LOCALAPPDATA",
    "PATH",
    "PATHEXT",
    "PROGRAMDATA",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "WINDIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
}
_GENERATED_OR_CACHE_PARTS = {
    ".cache",
    ".lewis-dev",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "htmlcov",
    "node_modules",
    "runtime",
    "target",
    "vendor",
}


class CodingAgentError(RuntimeError):
    """Raised when a coding-agent request is refused or cannot run safely."""


class RepositoryRefusedError(CodingAgentError):
    """Raised when a configured repository root fails the allowlist policy."""


class ProviderOutputError(CodingAgentError):
    """Raised when a provider emits malformed JSONL/stream output."""


@dataclass(frozen=True)
class ApprovedRepository:
    repo_id: str
    root: Path
    label: str
    valid: bool = True
    error: str = ""


@dataclass(frozen=True)
class UsageValues:
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    estimated_cost_usd: float | None = None
    token_usage_source: str | None = None
    cost_source: str | None = None

    def any_tokens(self) -> bool:
        return any(
            value is not None
            for value in (
                self.input_tokens,
                self.cached_input_tokens,
                self.output_tokens,
                self.reasoning_tokens,
            )
        )


@dataclass(frozen=True)
class NormalizedEvent:
    event_type: str
    summary: str = ""
    content: dict[str, Any] | None = None
    external_session_id: str | None = None
    usage: UsageValues | None = None
    duration_seconds: float | None = None
    failure_category: str | None = None

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"Unknown coding-agent event type: {self.event_type}")


@dataclass(frozen=True)
class AgentInvocation:
    provider: str
    args: tuple[str, ...]
    cwd: Path
    stdin_text: str
    timeout_seconds: int
    max_output_bytes: int = MAX_PROCESS_OUTPUT_BYTES


@dataclass(frozen=True)
class ProcessResult:
    returncode: int | None
    duration_seconds: float
    timed_out: bool = False
    cancelled: bool = False
    output_limited: bool = False
    callback_error: str | None = None
    stderr: str = ""


@dataclass(frozen=True)
class CodingAgentJob:
    id: str
    provider: str
    repository_root: str
    repository_label: str
    goal: str
    model: str | None
    effort: str | None
    max_turns: int
    timeout_seconds: int
    external_session_id: str | None
    state: str
    created_at: str
    started_at: str | None
    completed_at: str | None
    updated_at: str
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    token_usage_source: str | None
    estimated_cost_usd: float | None
    cost_source: str | None
    duration_seconds: float | None
    failure_category: str | None
    cancellation_requested: bool
    cancellation_requested_at: str | None
    last_error: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> CodingAgentJob:
        return cls(
            id=str(row["id"]),
            provider=str(row["provider"]),
            repository_root=str(row["repository_root"]),
            repository_label=str(row["repository_label"]),
            goal=str(row["goal"]),
            model=row["model"],
            effort=row["effort"],
            max_turns=int(row["max_turns"]),
            timeout_seconds=int(row["timeout_seconds"]),
            external_session_id=row["external_session_id"],
            state=str(row["state"]),
            created_at=str(row["created_at"]),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            updated_at=str(row["updated_at"]),
            input_tokens=_optional_int(row["input_tokens"]),
            cached_input_tokens=_optional_int(row["cached_input_tokens"]),
            output_tokens=_optional_int(row["output_tokens"]),
            reasoning_tokens=_optional_int(row["reasoning_tokens"]),
            token_usage_source=row["token_usage_source"],
            estimated_cost_usd=_optional_float(row["estimated_cost_usd"]),
            cost_source=row["cost_source"],
            duration_seconds=_optional_float(row["duration_seconds"]),
            failure_category=row["failure_category"],
            cancellation_requested=bool(row["cancellation_requested"]),
            cancellation_requested_at=row["cancellation_requested_at"],
            last_error=row["last_error"],
        )


@dataclass(frozen=True)
class CodingAgentEventRecord:
    id: int
    job_id: str
    sequence: int
    event_type: str
    summary: str
    content: dict[str, Any]
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> CodingAgentEventRecord:
        try:
            content = json.loads(str(row["content_json"] or "{}"))
        except json.JSONDecodeError:
            content = {"malformed": True}
        if not isinstance(content, dict):
            content = {"value": content}
        return cls(
            id=int(row["id"]),
            job_id=str(row["job_id"]),
            sequence=int(row["sequence"]),
            event_type=str(row["event_type"]),
            summary=str(row["summary"]),
            content=content,
            created_at=str(row["created_at"]),
        )


@dataclass(frozen=True)
class ProviderStatus:
    provider: str
    installed: bool
    executable_path: str | None = None
    version: str | None = None
    authenticated: bool | None = None
    auth_status: str = "not checked"
    usage_status: str = "unavailable"
    checked_at: str | None = None


@dataclass(frozen=True)
class LaunchRequest:
    repository_id: str
    provider: str
    goal: str
    model: str | None = None
    effort: str | None = None
    max_turns: int = 6
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS


class CodingAgentStore:
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

    def upsert_repository(self, repository: ApprovedRepository) -> None:
        now = _utc_now()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT root FROM coding_agent_repositories WHERE root = ?",
                (str(repository.root),),
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE coding_agent_repositories
                    SET label = ?, updated_at = ?, last_seen_at = ?
                    WHERE root = ?
                    """,
                    (repository.label, now, now, str(repository.root)),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO coding_agent_repositories
                        (root, label, created_at, updated_at, last_seen_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (str(repository.root), repository.label, now, now, now),
                )

    def create_job(self, job: CodingAgentJob) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO coding_agent_jobs (
                    id, provider, repository_root, repository_label, goal, model,
                    effort, max_turns, timeout_seconds, external_session_id, state,
                    created_at, started_at, completed_at, updated_at,
                    input_tokens, cached_input_tokens, output_tokens,
                    reasoning_tokens, token_usage_source, estimated_cost_usd,
                    cost_source, duration_seconds, failure_category,
                    cancellation_requested, cancellation_requested_at, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.provider,
                    job.repository_root,
                    job.repository_label,
                    job.goal,
                    job.model,
                    job.effort,
                    job.max_turns,
                    job.timeout_seconds,
                    job.external_session_id,
                    job.state,
                    job.created_at,
                    job.started_at,
                    job.completed_at,
                    job.updated_at,
                    job.input_tokens,
                    job.cached_input_tokens,
                    job.output_tokens,
                    job.reasoning_tokens,
                    job.token_usage_source,
                    job.estimated_cost_usd,
                    job.cost_source,
                    job.duration_seconds,
                    job.failure_category,
                    int(job.cancellation_requested),
                    job.cancellation_requested_at,
                    job.last_error,
                ),
            )

    def get_job(self, job_id: str) -> CodingAgentJob | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM coding_agent_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return CodingAgentJob.from_row(row) if row else None

    def list_jobs(self, limit: int = 25) -> list[CodingAgentJob]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM coding_agent_jobs
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """,
                (max(1, min(limit, 100)),),
            ).fetchall()
        return [CodingAgentJob.from_row(row) for row in rows]

    def list_events(self, job_id: str, limit: int = 100) -> list[CodingAgentEventRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM coding_agent_events
                WHERE job_id = ?
                ORDER BY sequence ASC
                LIMIT ?
                """,
                (job_id, max(1, min(limit, 200))),
            ).fetchall()
        return [CodingAgentEventRecord.from_row(row) for row in rows]

    def append_event(self, job_id: str, event: NormalizedEvent) -> None:
        summary = _bounded_text(event.summary, MAX_EVENT_SUMMARY_CHARS)
        content_json = _bounded_event_json(event.content or {})
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence FROM coding_agent_events WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            sequence = int(row["next_sequence"] if row else 1)
            conn.execute(
                """
                INSERT INTO coding_agent_events
                    (job_id, sequence, event_type, summary, content_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, sequence, event.event_type, summary, content_json, _utc_now()),
            )

    def mark_started(self, job_id: str) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE coding_agent_jobs
                SET state = 'running', started_at = ?, updated_at = ?
                WHERE id = ? AND state IN ('queued', 'running')
                """,
                (now, now, job_id),
            )

    def set_external_session_id(self, job_id: str, external_id: str | None) -> None:
        if not external_id:
            return
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE coding_agent_jobs
                SET external_session_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (_bounded_text(external_id, 200), _utc_now(), job_id),
            )

    def update_usage(self, job_id: str, usage: UsageValues) -> None:
        assignments: list[str] = []
        values: list[object] = []
        for column, value in (
            ("input_tokens", usage.input_tokens),
            ("cached_input_tokens", usage.cached_input_tokens),
            ("output_tokens", usage.output_tokens),
            ("reasoning_tokens", usage.reasoning_tokens),
            ("estimated_cost_usd", usage.estimated_cost_usd),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                values.append(value)
        if usage.token_usage_source:
            assignments.append("token_usage_source = ?")
            values.append(usage.token_usage_source)
        if usage.cost_source:
            assignments.append("cost_source = ?")
            values.append(usage.cost_source)
        if not assignments:
            return
        assignments.append("updated_at = ?")
        values.append(_utc_now())
        values.append(job_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE coding_agent_jobs SET {', '.join(assignments)} WHERE id = ?",
                tuple(values),
            )

    def request_cancellation(self, job_id: str) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE coding_agent_jobs
                SET cancellation_requested = 1,
                    cancellation_requested_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, job_id),
            )

    def mark_terminal(
        self,
        job_id: str,
        *,
        state: str,
        duration_seconds: float,
        failure_category: str | None = None,
        last_error: str | None = None,
    ) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE coding_agent_jobs
                SET state = ?, completed_at = ?, updated_at = ?,
                    duration_seconds = ?, failure_category = ?, last_error = ?
                WHERE id = ?
                """,
                (
                    state,
                    now,
                    now,
                    round(max(0.0, duration_seconds), 3),
                    failure_category,
                    _bounded_text(last_error or "", 500) or None,
                    job_id,
                ),
            )


class SafeProcessRunner:
    """Launch fixed argv without a shell and support timeout/cancellation."""

    def run(
        self,
        invocation: AgentInvocation,
        *,
        on_stdout_line: Callable[[str], None],
        on_stderr_line: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ProcessResult:
        _assert_safe_invocation(invocation.provider, invocation.args)
        cancel_event = cancel_event or threading.Event()
        env = sanitized_process_environment()
        start = time.monotonic()
        stderr_chunks: list[str] = []
        total_output_bytes = 0
        output_limited = threading.Event()
        callback_error: list[str] = []
        creationflags = 0
        preexec_fn = None
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            preexec_fn = os.setsid
        try:
            proc = subprocess.Popen(
                list(invocation.args),
                cwd=str(invocation.cwd),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                text=False,
                creationflags=creationflags,
                preexec_fn=preexec_fn,
            )
        except OSError as exc:
            return ProcessResult(
                returncode=None,
                duration_seconds=round(time.monotonic() - start, 3),
                callback_error=f"process_start_failed: {exc}",
            )

        def count_bytes(raw: bytes) -> bool:
            nonlocal total_output_bytes
            total_output_bytes += len(raw)
            if total_output_bytes > invocation.max_output_bytes:
                output_limited.set()
                return False
            return True

        def stdout_reader() -> None:
            assert proc.stdout is not None
            try:
                for raw in iter(proc.stdout.readline, b""):
                    if not count_bytes(raw):
                        break
                    try:
                        on_stdout_line(raw.decode("utf-8", errors="replace"))
                    except Exception as exc:
                        callback_error.append(str(exc))
                        break
            finally:
                try:
                    proc.stdout.close()
                except OSError:
                    pass

        def stderr_reader() -> None:
            assert proc.stderr is not None
            try:
                for raw in iter(proc.stderr.readline, b""):
                    if not count_bytes(raw):
                        break
                    text = raw.decode("utf-8", errors="replace")
                    stderr_chunks.append(text)
                    if on_stderr_line is not None:
                        try:
                            on_stderr_line(text)
                        except Exception as exc:
                            callback_error.append(str(exc))
                            break
            finally:
                try:
                    proc.stderr.close()
                except OSError:
                    pass

        out_thread = threading.Thread(target=stdout_reader, daemon=True)
        err_thread = threading.Thread(target=stderr_reader, daemon=True)
        out_thread.start()
        err_thread.start()
        if proc.stdin is not None:
            try:
                proc.stdin.write(invocation.stdin_text.encode("utf-8"))
                proc.stdin.close()
            except OSError:
                pass

        timed_out = False
        cancelled = False
        while proc.poll() is None:
            if cancel_event.is_set():
                cancelled = True
                _terminate_process_tree(proc)
                break
            if output_limited.is_set() or callback_error:
                _terminate_process_tree(proc)
                break
            if time.monotonic() - start > invocation.timeout_seconds:
                timed_out = True
                _terminate_process_tree(proc)
                break
            time.sleep(0.05)
        try:
            returncode = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(proc)
            returncode = proc.poll()
        out_thread.join(timeout=2)
        err_thread.join(timeout=2)
        stderr = _bounded_text("".join(stderr_chunks), 2_000)
        return ProcessResult(
            returncode=returncode,
            duration_seconds=round(time.monotonic() - start, 3),
            timed_out=timed_out,
            cancelled=cancelled,
            output_limited=output_limited.is_set(),
            callback_error=callback_error[0] if callback_error else None,
            stderr=stderr,
        )


class CodexAdapter:
    provider = PROVIDER_CODEX

    def build_invocation(
        self,
        executable: Path,
        request: LaunchRequest,
        repository: ApprovedRepository,
    ) -> AgentInvocation:
        args: list[str] = [
            str(executable),
            "exec",
            "--json",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--cd",
            str(repository.root),
        ]
        if request.model:
            args.extend(["--model", request.model])
        args.append("-")
        invocation = AgentInvocation(
            provider=self.provider,
            args=tuple(args),
            cwd=repository.root,
            stdin_text=_read_only_prompt(request.goal, provider="Codex"),
            timeout_seconds=request.timeout_seconds,
        )
        _assert_safe_invocation(self.provider, invocation.args)
        return invocation

    def parse_line(self, line: str) -> list[NormalizedEvent]:
        payload = _parse_json_line(line)
        event_type = _payload_type(payload)
        events: list[NormalizedEvent] = []
        external_id = _extract_external_id(payload)
        if external_id and any(marker in event_type for marker in ("thread", "session")):
            events.append(
                NormalizedEvent(
                    "job_started",
                    "Codex session initialized.",
                    {"provider_event_type": event_type},
                    external_session_id=external_id,
                )
            )
        if _payload_error_text(payload):
            events.append(
                NormalizedEvent(
                    "job_failed",
                    _payload_error_text(payload),
                    {"provider_event_type": event_type},
                    external_session_id=external_id,
                    failure_category="provider_error",
                )
            )
        text = _extract_text(payload)
        if text and not _is_reasoning_event(event_type, payload):
            events.append(
                NormalizedEvent(
                    "agent_message",
                    text,
                    {"provider_event_type": event_type},
                    external_session_id=external_id,
                )
            )
        command = _extract_command(payload)
        if command:
            normalized_type = (
                "command_finished"
                if any(marker in event_type for marker in ("finished", "completed", "exit", "end"))
                else "command_started"
            )
            events.append(
                NormalizedEvent(
                    normalized_type,
                    command,
                    _safe_subset(payload, {"exit_code", "status", "duration_ms"}),
                    external_session_id=external_id,
                )
            )
        file_path = _extract_file_path(payload)
        if file_path:
            events.append(
                NormalizedEvent(
                    "file_observed",
                    file_path,
                    {"provider_event_type": event_type},
                    external_session_id=external_id,
                )
            )
        usage = _extract_usage(payload)
        if usage.any_tokens() or usage.estimated_cost_usd is not None:
            events.append(
                NormalizedEvent(
                    "usage_updated",
                    "Codex reported usage values.",
                    {"provider_event_type": event_type},
                    external_session_id=external_id,
                    usage=usage,
                )
            )
        if any(marker in event_type for marker in ("completed", "done", "success")):
            events.append(
                NormalizedEvent(
                    "job_completed",
                    "Codex reported completion.",
                    {"provider_event_type": event_type},
                    external_session_id=external_id,
                )
            )
        return events


class ClaudeAdapter:
    provider = PROVIDER_CLAUDE

    def build_invocation(
        self,
        executable: Path,
        request: LaunchRequest,
        repository: ApprovedRepository,
    ) -> AgentInvocation:
        args: list[str] = [
            str(executable),
            "-p",
            "--output-format",
            "stream-json",
            "--permission-mode",
            "plan",
            "--tools",
            "Read,Grep,Glob,LS",
            "--disallowedTools",
            "Bash,Edit,MultiEdit,Write,NotebookEdit,WebFetch,WebSearch,TodoWrite",
            "--safe-mode",
            "--disable-slash-commands",
            "--no-chrome",
        ]
        if request.model:
            args.extend(["--model", request.model])
        if request.effort:
            args.extend(["--effort", request.effort])
        invocation = AgentInvocation(
            provider=self.provider,
            args=tuple(args),
            cwd=repository.root,
            stdin_text=_read_only_prompt(request.goal, provider="Claude Code"),
            timeout_seconds=request.timeout_seconds,
        )
        _assert_safe_invocation(self.provider, invocation.args)
        return invocation

    def parse_line(self, line: str) -> list[NormalizedEvent]:
        payload = _parse_json_line(line)
        event_type = _payload_type(payload)
        events: list[NormalizedEvent] = []
        external_id = _extract_external_id(payload)
        if event_type == "system" or payload.get("subtype") == "init":
            events.append(
                NormalizedEvent(
                    "job_started",
                    "Claude session initialized.",
                    _safe_subset(payload, {"type", "subtype", "model"}),
                    external_session_id=external_id,
                )
            )
        message = payload.get("message")
        if isinstance(message, dict):
            for content in _as_list(message.get("content")):
                if not isinstance(content, dict):
                    continue
                content_type = str(content.get("type") or "")
                if content_type == "text":
                    text = str(content.get("text") or "").strip()
                    if text:
                        events.append(
                            NormalizedEvent(
                                "agent_message",
                                text,
                                {"provider_event_type": event_type},
                                external_session_id=external_id,
                            )
                        )
                elif content_type == "tool_use":
                    tool_name = str(content.get("name") or "tool")[:100]
                    events.append(
                        NormalizedEvent(
                            "command_started",
                            tool_name,
                            {
                                "provider_event_type": event_type,
                                "tool_id": _bounded_text(str(content.get("id") or ""), 200),
                                "tool_name": tool_name,
                            },
                            external_session_id=external_id,
                        )
                    )
                elif content_type == "tool_result":
                    events.append(
                        NormalizedEvent(
                            "command_finished",
                            "tool_result",
                            {
                                "provider_event_type": event_type,
                                "tool_use_id": _bounded_text(str(content.get("tool_use_id") or ""), 200),
                                "is_error": bool(content.get("is_error", False)),
                            },
                            external_session_id=external_id,
                        )
                    )
            usage = _extract_usage(message)
            if usage.any_tokens() or usage.estimated_cost_usd is not None:
                events.append(
                    NormalizedEvent(
                        "usage_updated",
                        "Claude reported message usage values.",
                        {"provider_event_type": event_type},
                        external_session_id=external_id,
                        usage=usage,
                    )
                )
        if event_type == "result":
            usage = _extract_usage(payload)
            if usage.any_tokens() or usage.estimated_cost_usd is not None:
                events.append(
                    NormalizedEvent(
                        "usage_updated",
                        "Claude reported result usage values.",
                        {"provider_event_type": event_type},
                        external_session_id=external_id,
                        usage=usage,
                    )
                )
            duration = _duration_seconds_from_payload(payload)
            is_error = bool(payload.get("is_error")) or str(payload.get("subtype") or "").startswith("error")
            if is_error:
                events.append(
                    NormalizedEvent(
                        "job_failed",
                        _bounded_text(str(payload.get("result") or "Claude reported an error."), 500),
                        {"provider_event_type": event_type},
                        external_session_id=external_id,
                        duration_seconds=duration,
                        failure_category="provider_error",
                    )
                )
            else:
                result_text = str(payload.get("result") or "").strip()
                if result_text:
                    events.append(
                        NormalizedEvent(
                            "agent_message",
                            result_text,
                            {"provider_event_type": event_type, "final": True},
                            external_session_id=external_id,
                        )
                    )
                events.append(
                    NormalizedEvent(
                        "job_completed",
                        "Claude reported completion.",
                        {"provider_event_type": event_type},
                        external_session_id=external_id,
                        duration_seconds=duration,
                    )
                )
        if event_type == "error" or _payload_error_text(payload):
            events.append(
                NormalizedEvent(
                    "job_failed",
                    _payload_error_text(payload) or "Claude stream reported an error.",
                    {"provider_event_type": event_type},
                    external_session_id=external_id,
                    failure_category="provider_error",
                )
            )
        return events


class CodingAgentService:
    def __init__(
        self,
        settings: Settings,
        *,
        store: CodingAgentStore | None = None,
        runner: SafeProcessRunner | None = None,
        executable_resolver: Callable[[str], Path] | None = None,
        status_runner: Callable[[Sequence[str], Path], subprocess.CompletedProcess] | None = None,
    ):
        self.settings = settings
        self.store = store or CodingAgentStore(settings.database_path)
        self.runner = runner or SafeProcessRunner()
        self._executable_resolver = executable_resolver
        self._status_runner = status_runner
        self._active: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._status_cache: dict[str, ProviderStatus] = {}

    def repositories(self) -> list[ApprovedRepository]:
        return configured_repository_options(self.settings)

    def provider_statuses(self) -> dict[str, ProviderStatus]:
        statuses: dict[str, ProviderStatus] = {}
        for provider in sorted(KNOWN_PROVIDERS):
            cached = self._status_cache.get(provider)
            installed_path = _safe_resolve_executable(self.settings, provider, strict=False)
            if cached is not None:
                statuses[provider] = cached
            else:
                statuses[provider] = ProviderStatus(
                    provider=provider,
                    installed=installed_path is not None,
                    executable_path=str(installed_path) if installed_path else None,
                    usage_status="unavailable",
                )
        return statuses

    def refresh_provider_statuses(self) -> dict[str, ProviderStatus]:
        statuses = {
            PROVIDER_CODEX: self._refresh_codex_status(),
            PROVIDER_CLAUDE: self._refresh_claude_status(),
        }
        self._status_cache.update(statuses)
        return statuses

    def launch(self, request: LaunchRequest) -> CodingAgentJob:
        request = _validated_launch_request(request, self.settings)
        repository = repository_by_id(self.settings, request.repository_id)
        executable = self._resolve_executable(request.provider)
        self.store.upsert_repository(repository)
        now = _utc_now()
        job = CodingAgentJob(
            id=uuid.uuid4().hex,
            provider=request.provider,
            repository_root=str(repository.root),
            repository_label=repository.label,
            goal=request.goal,
            model=request.model,
            effort=request.effort,
            max_turns=request.max_turns,
            timeout_seconds=request.timeout_seconds,
            external_session_id=None,
            state="queued",
            created_at=now,
            started_at=None,
            completed_at=None,
            updated_at=now,
            input_tokens=None,
            cached_input_tokens=None,
            output_tokens=None,
            reasoning_tokens=None,
            token_usage_source=None,
            estimated_cost_usd=None,
            cost_source=None,
            duration_seconds=None,
            failure_category=None,
            cancellation_requested=False,
            cancellation_requested_at=None,
            last_error=None,
        )
        self.store.create_job(job)
        cancel_event = threading.Event()
        with self._lock:
            self._active[job.id] = cancel_event
        thread = threading.Thread(
            target=self._run_job,
            args=(job.id, request, repository, executable, cancel_event),
            daemon=True,
        )
        thread.start()
        return job

    def cancel(self, job_id: str) -> None:
        if not _is_job_id(job_id):
            raise CodingAgentError("Job id must be a 32-character lowercase hex id.")
        self.store.request_cancellation(job_id)
        with self._lock:
            cancel_event = self._active.get(job_id)
        if cancel_event is None:
            job = self.store.get_job(job_id)
            if job and job.state == "running":
                self.store.append_event(
                    job_id,
                    NormalizedEvent(
                        "job_failed",
                        "Cancellation requested, but no active local process handle was found.",
                        failure_category="missing_process",
                    ),
                )
                self.store.mark_terminal(
                    job_id,
                    state="failed",
                    duration_seconds=job.duration_seconds or 0.0,
                    failure_category="missing_process",
                    last_error="No active local process handle was found.",
                )
            return
        cancel_event.set()

    def _run_job(
        self,
        job_id: str,
        request: LaunchRequest,
        repository: ApprovedRepository,
        executable: Path,
        cancel_event: threading.Event,
    ) -> None:
        adapter = CodexAdapter() if request.provider == PROVIDER_CODEX else ClaudeAdapter()
        invocation = adapter.build_invocation(executable, request, repository)
        self.store.mark_started(job_id)
        self.store.append_event(
            job_id,
            NormalizedEvent(
                "job_started",
                f"Started {request.provider} read-only job.",
                {
                    "provider": request.provider,
                    "cwd": str(invocation.cwd),
                    "argv": _safe_argv_display(invocation.args),
                    "timeout_seconds": invocation.timeout_seconds,
                    "max_turns": request.max_turns,
                },
            ),
        )
        turn_count = 0

        def on_stdout_line(line: str) -> None:
            nonlocal turn_count
            events = adapter.parse_line(line)
            for event in events:
                self.store.append_event(job_id, event)
                if event.external_session_id:
                    self.store.set_external_session_id(job_id, event.external_session_id)
                if event.usage:
                    self.store.update_usage(job_id, event.usage)
                if event.event_type in {"agent_message", "command_started"}:
                    turn_count += 1
                    if turn_count > request.max_turns:
                        raise ProviderOutputError(
                            f"Job exceeded the configured maximum of {request.max_turns} streamed turns."
                        )

        def on_stderr_line(line: str) -> None:
            text = line.strip()
            if text:
                self.store.append_event(
                    job_id,
                    NormalizedEvent(
                        "agent_message",
                        _bounded_text(text, 500),
                        {"stream": "stderr"},
                    ),
                )

        result = self.runner.run(
            invocation,
            on_stdout_line=on_stdout_line,
            on_stderr_line=on_stderr_line,
            cancel_event=cancel_event,
        )
        try:
            self._finalize_result(job_id, result)
        finally:
            with self._lock:
                self._active.pop(job_id, None)

    def _finalize_result(self, job_id: str, result: ProcessResult) -> None:
        if result.cancelled:
            self.store.append_event(
                job_id,
                NormalizedEvent("job_cancelled", "Job was cancelled by request."),
            )
            self.store.mark_terminal(
                job_id,
                state="cancelled",
                duration_seconds=result.duration_seconds,
                failure_category="cancelled",
            )
            return
        if result.timed_out:
            message = "Job exceeded its configured timeout."
            self.store.append_event(
                job_id,
                NormalizedEvent("job_failed", message, failure_category="timeout"),
            )
            self.store.mark_terminal(
                job_id,
                state="failed",
                duration_seconds=result.duration_seconds,
                failure_category="timeout",
                last_error=message,
            )
            return
        if result.output_limited:
            message = "Provider output exceeded the configured byte limit."
            self.store.append_event(
                job_id,
                NormalizedEvent("job_failed", message, failure_category="output_limit"),
            )
            self.store.mark_terminal(
                job_id,
                state="failed",
                duration_seconds=result.duration_seconds,
                failure_category="output_limit",
                last_error=message,
            )
            return
        if result.callback_error:
            message = _bounded_text(result.callback_error, 500)
            category = "malformed_output"
            if "maximum" in message and "turn" in message:
                category = "max_turns"
            self.store.append_event(
                job_id,
                NormalizedEvent("job_failed", message, failure_category=category),
            )
            self.store.mark_terminal(
                job_id,
                state="failed",
                duration_seconds=result.duration_seconds,
                failure_category=category,
                last_error=message,
            )
            return
        if result.returncode not in {0, None}:
            message = result.stderr or f"Provider process exited with code {result.returncode}."
            self.store.append_event(
                job_id,
                NormalizedEvent("job_failed", message, failure_category="process_error"),
            )
            self.store.mark_terminal(
                job_id,
                state="failed",
                duration_seconds=result.duration_seconds,
                failure_category="process_error",
                last_error=message,
            )
            return
        self.store.append_event(
            job_id,
            NormalizedEvent("job_completed", "Provider process completed successfully."),
        )
        self.store.mark_terminal(
            job_id,
            state="completed",
            duration_seconds=result.duration_seconds,
        )

    def _resolve_executable(self, provider: str) -> Path:
        if self._executable_resolver is not None:
            return self._executable_resolver(provider)
        path = _safe_resolve_executable(self.settings, provider, strict=True)
        assert path is not None
        return path

    def _run_status_command(self, args: Sequence[str], cwd: Path) -> subprocess.CompletedProcess:
        if self._status_runner is not None:
            return self._status_runner(args, cwd)
        return _run_status_command(args, cwd)

    def _refresh_codex_status(self) -> ProviderStatus:
        path = _safe_resolve_executable(self.settings, PROVIDER_CODEX, strict=False)
        if path is None:
            return ProviderStatus(
                PROVIDER_CODEX,
                installed=False,
                auth_status="executable unavailable",
                checked_at=_utc_now(),
            )
        version = _status_stdout(self._run_status_command([str(path), "--version"], ROOT_DIR))
        status = self._run_status_command([str(path), "login", "status"], ROOT_DIR)
        text = _status_stdout(status) or _status_stderr(status)
        authenticated = status.returncode == 0 and "logged in" in text.casefold()
        return ProviderStatus(
            provider=PROVIDER_CODEX,
            installed=True,
            executable_path=str(path),
            version=version or None,
            authenticated=authenticated,
            auth_status="authenticated" if authenticated else _bounded_text(text or "not authenticated", 200),
            usage_status=(
                "unavailable; Codex app-server usage/rate-limit integration "
                "is not required for Phase 1"
            ),
            checked_at=_utc_now(),
        )

    def _refresh_claude_status(self) -> ProviderStatus:
        path = _safe_resolve_executable(self.settings, PROVIDER_CLAUDE, strict=False)
        if path is None:
            return ProviderStatus(
                PROVIDER_CLAUDE,
                installed=False,
                auth_status="executable unavailable",
                checked_at=_utc_now(),
            )
        version = _status_stdout(self._run_status_command([str(path), "--version"], ROOT_DIR))
        status = self._run_status_command([str(path), "auth", "status", "--json"], ROOT_DIR)
        authenticated = False
        auth_status = "not authenticated"
        try:
            payload = json.loads(_status_stdout(status) or "{}")
        except json.JSONDecodeError:
            payload = {}
        if status.returncode == 0 and isinstance(payload, dict):
            authenticated = bool(payload.get("loggedIn"))
            method = str(payload.get("authMethod") or payload.get("apiProvider") or "authenticated")
            subscription = str(payload.get("subscriptionType") or "")
            auth_status = method if not subscription else f"{method}; subscription={subscription}"
        elif status.returncode != 0:
            auth_status = _bounded_text(_status_stderr(status) or _status_stdout(status), 200) or "not authenticated"
        return ProviderStatus(
            provider=PROVIDER_CLAUDE,
            installed=True,
            executable_path=str(path),
            version=version or None,
            authenticated=authenticated,
            auth_status="authenticated" if authenticated and not auth_status else auth_status,
            usage_status="subscription-window usage unavailable via documented local JSON status",
            checked_at=_utc_now(),
        )


def configured_repository_options(settings: Settings) -> list[ApprovedRepository]:
    repositories: list[ApprovedRepository] = []
    seen: set[str] = set()
    for configured in settings.coding_agent_repositories:
        root = configured.expanduser()
        try:
            resolved = root.resolve(strict=False)
        except OSError:
            resolved = root.absolute()
        key = str(resolved).casefold()
        if key in seen:
            continue
        seen.add(key)
        repositories.append(
            ApprovedRepository(
                repo_id=_repository_id(resolved),
                root=resolved,
                label=root.name or str(root),
                valid=True,
                error="not validated until launch",
            )
        )
    return repositories


def approved_repositories(settings: Settings) -> list[ApprovedRepository]:
    repositories: list[ApprovedRepository] = []
    seen: set[str] = set()
    for configured in settings.coding_agent_repositories:
        try:
            repository = _validate_configured_repository(configured)
        except RepositoryRefusedError as exc:
            root = configured.expanduser()
            try:
                resolved = root.resolve(strict=False)
            except OSError:
                resolved = root.absolute()
            repo_id = _repository_id(resolved)
            repository = ApprovedRepository(
                repo_id=repo_id,
                root=resolved,
                label=root.name or str(root),
                valid=False,
                error=str(exc),
            )
        key = str(repository.root).casefold()
        if key in seen:
            continue
        seen.add(key)
        repositories.append(repository)
    return repositories


def repository_by_id(settings: Settings, repository_id: str) -> ApprovedRepository:
    for repository in approved_repositories(settings):
        if repository.repo_id == repository_id:
            if not repository.valid:
                raise RepositoryRefusedError(repository.error or "Repository is not approved.")
            return repository
    raise RepositoryRefusedError("Repository selection must be one configured approved repository.")


def sanitized_process_environment(parent: dict[str, str] | None = None) -> dict[str, str]:
    source = parent or os.environ
    env: dict[str, str] = {}
    for key, value in source.items():
        normalized = key.upper()
        if normalized not in _ALLOWED_ENV_KEYS:
            continue
        if _SENSITIVE_KEY_PATTERN.search(key):
            continue
        if not isinstance(value, str) or "\x00" in value or len(value) > 4_000:
            continue
        env[key] = value
    env.setdefault("NO_COLOR", "1")
    env.setdefault("TERM", "dumb")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONNOUSERSITE", "1")
    return env


def _validated_launch_request(request: LaunchRequest, settings: Settings) -> LaunchRequest:
    provider = str(request.provider or "").strip().casefold()
    if provider not in KNOWN_PROVIDERS:
        raise CodingAgentError("Provider must be codex or claude.")
    goal = _validate_goal(request.goal)
    model = _validate_model(request.model)
    effort = str(request.effort or "").strip().casefold() or None
    if provider == PROVIDER_CLAUDE:
        if effort is not None and effort not in CLAUDE_EFFORTS:
            raise CodingAgentError("Claude effort must be low, medium, high, xhigh, or max.")
    elif effort is not None:
        raise CodingAgentError("Effort is only accepted for Claude jobs in this phase.")
    max_turns = _bounded_int(request.max_turns, default=6, minimum=1, maximum=MAX_TURNS_LIMIT)
    configured_max = _bounded_int(
        settings.coding_agent_max_timeout_seconds,
        default=MAX_TIMEOUT_SECONDS,
        minimum=MIN_TIMEOUT_SECONDS,
        maximum=MAX_TIMEOUT_SECONDS,
    )
    default_timeout = _bounded_int(
        settings.coding_agent_default_timeout_seconds,
        default=DEFAULT_TIMEOUT_SECONDS,
        minimum=MIN_TIMEOUT_SECONDS,
        maximum=configured_max,
    )
    timeout = _bounded_int(
        request.timeout_seconds,
        default=default_timeout,
        minimum=MIN_TIMEOUT_SECONDS,
        maximum=configured_max,
    )
    return LaunchRequest(
        repository_id=str(request.repository_id),
        provider=provider,
        goal=goal,
        model=model,
        effort=effort,
        max_turns=max_turns,
        timeout_seconds=timeout,
    )


def _validate_goal(goal: str) -> str:
    if not isinstance(goal, str):
        raise CodingAgentError("Goal must be text.")
    clean = goal.strip()
    if not clean:
        raise CodingAgentError("Goal must not be empty.")
    if len(clean) > MAX_GOAL_LENGTH:
        raise CodingAgentError(f"Goal exceeds the maximum length of {MAX_GOAL_LENGTH} characters.")
    if "\x00" in clean:
        raise CodingAgentError("Goal must not contain NUL bytes.")
    return clean


def _validate_model(model: str | None) -> str | None:
    if model is None:
        return None
    clean = str(model).strip()
    if not clean:
        return None
    if len(clean) > MAX_MODEL_LENGTH or _MODEL_PATTERN.fullmatch(clean) is None:
        raise CodingAgentError("Model must be a bounded model name without spaces or shell characters.")
    return clean


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        parsed = default
    return min(maximum, max(minimum, parsed))


def _validate_configured_repository(path: Path) -> ApprovedRepository:
    if not isinstance(path, Path):
        raise RepositoryRefusedError("Approved repository roots must come from local configuration.")
    if path.is_symlink():
        raise RepositoryRefusedError("Approved repository root must not be a symlink.")
    try:
        root = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise RepositoryRefusedError("Approved repository root does not exist.") from exc
    if not root.is_dir():
        raise RepositoryRefusedError("Approved repository root must be a directory.")
    _require_git_toplevel(root)
    _validate_repository_tree(root)
    return ApprovedRepository(repo_id=_repository_id(root), root=root, label=root.name or str(root))


def _require_git_toplevel(root: Path) -> None:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        check=False,
        timeout=STATUS_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise RepositoryRefusedError("Approved repository root must be a Git work tree.")
    try:
        top = Path(result.stdout.decode("utf-8", errors="replace").strip()).resolve(strict=True)
    except OSError as exc:
        raise RepositoryRefusedError("Git top-level path could not be resolved.") from exc
    if top != root:
        raise RepositoryRefusedError("Jobs may run only at the approved Git repository top level.")


def _validate_repository_tree(root: Path) -> None:
    scanned = 0
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError as exc:
            raise RepositoryRefusedError("Approved repository tree could not be inspected safely.") from exc
        for entry in entries:
            scanned += 1
            if scanned > MAX_REPOSITORY_SCAN_ENTRIES:
                raise RepositoryRefusedError("Approved repository tree exceeds the Phase 1 safety scan limit.")
            if entry.name == ".git":
                if entry.parent != root:
                    raise RepositoryRefusedError("Nested Git repositories are not accepted for Phase 1 jobs.")
                continue
            if entry.is_symlink():
                try:
                    target = entry.resolve(strict=True)
                    target.relative_to(root)
                except (OSError, ValueError) as exc:
                    raise RepositoryRefusedError("Repository symlinks must not escape the approved root.") from exc
                continue
            if entry.is_dir():
                if entry.name in _GENERATED_OR_CACHE_PARTS:
                    continue
                stack.append(entry)


def _repository_id(root: Path) -> str:
    return hashlib.sha256(str(root).casefold().encode("utf-8")).hexdigest()[:16]


def _safe_resolve_executable(settings: Settings, provider: str, *, strict: bool) -> Path | None:
    if provider == PROVIDER_CODEX:
        configured = settings.codex_cli_path
        executable_name = "codex"
        expected = {"codex", "codex.exe"}
    elif provider == PROVIDER_CLAUDE:
        configured = settings.claude_cli_path
        executable_name = "claude"
        expected = {"claude", "claude.exe"}
    else:
        raise CodingAgentError("Unknown provider.")
    if configured is not None:
        path = _validate_executable_path(configured, expected)
        if path is not None:
            return path
        if strict:
            raise CodingAgentError(f"Configured {provider} executable is not usable.")
        return None
    found = shutil.which(executable_name)
    if not found:
        if strict:
            raise CodingAgentError(f"{provider} executable was not found on PATH.")
        return None
    path = _validate_executable_path(Path(found), expected)
    if path is None and strict:
        raise CodingAgentError(f"PATH {provider} executable is not usable.")
    return path


def _validate_executable_path(path: Path, expected_names: set[str]) -> Path | None:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError:
        return None
    if not resolved.is_file() or resolved.is_symlink():
        return None
    if resolved.name.casefold() not in {name.casefold() for name in expected_names}:
        return None
    if os.name == "nt" and resolved.suffix.casefold() != ".exe":
        return None
    return resolved


def _assert_safe_invocation(provider: str, args: Sequence[str]) -> None:
    values = tuple(str(arg) for arg in args)
    joined = "\n".join(values).casefold()
    forbidden_global = (
        "danger-full-access",
        "workspace-write",
        "full-auto",
        "bypasspermissions",
        "dangerously-bypass",
        "dangerously-skip-permissions",
        "allow-dangerously-skip-permissions",
        "--add-dir",
        "--plugin-url",
        "--worktree",
        "--bg",
        "--background",
    )
    if any(token in joined for token in forbidden_global):
        raise CodingAgentError("Unsafe provider invocation was refused.")
    if provider == PROVIDER_CODEX:
        if "exec" not in values or "--json" not in values:
            raise CodingAgentError("Codex invocation must use exec --json.")
        if "--sandbox" not in values:
            raise CodingAgentError("Codex invocation must set --sandbox read-only.")
        sandbox_index = values.index("--sandbox")
        if sandbox_index + 1 >= len(values) or values[sandbox_index + 1] != "read-only":
            raise CodingAgentError("Codex invocation must set --sandbox read-only.")
        if "--skip-git-repo-check" in values:
            raise CodingAgentError("Codex invocation must not skip the Git repository check.")
    elif provider == PROVIDER_CLAUDE:
        if "-p" not in values and "--print" not in values:
            raise CodingAgentError("Claude invocation must use print mode.")
        if "--output-format" not in values:
            raise CodingAgentError("Claude invocation must use stream-json output.")
        output_index = values.index("--output-format")
        if output_index + 1 >= len(values) or values[output_index + 1] != "stream-json":
            raise CodingAgentError("Claude invocation must use stream-json output.")
        if "--permission-mode" not in values:
            raise CodingAgentError("Claude invocation must set permission mode plan.")
        mode_index = values.index("--permission-mode")
        if mode_index + 1 >= len(values) or values[mode_index + 1] != "plan":
            raise CodingAgentError("Claude invocation must set permission mode plan.")
        if "--tools" not in values:
            raise CodingAgentError("Claude invocation must restrict tools.")
        tools_index = values.index("--tools")
        if tools_index + 1 >= len(values) or values[tools_index + 1] != "Read,Grep,Glob,LS":
            raise CodingAgentError("Claude invocation must restrict tools to read-only built-ins.")
    else:
        raise CodingAgentError("Unknown provider invocation.")


def _safe_argv_display(args: Sequence[str]) -> list[str]:
    display = []
    for arg in args:
        text = str(arg)
        if _SENSITIVE_KEY_PATTERN.search(text):
            display.append("[redacted]")
        else:
            display.append(_bounded_text(text, 240))
    return display


def _read_only_prompt(goal: str, *, provider: str) -> str:
    return (
        f"You are running under Lewis Coding Agents Phase 1 using {provider}.\n"
        "This is a bounded read-only analysis and proposal job. Do not edit files, "
        "write files, create branches, switch branches, create worktrees, install "
        "packages, run package managers, commit, pull, fetch, push, merge, rebase, "
        "open PRs, change external services, or ask for secrets. Inspect only the "
        "approved repository and produce analysis, risks, and proposed next steps.\n\n"
        "Goal:\n"
        f"{goal}\n"
    )


def _parse_json_line(line: str) -> dict[str, Any]:
    stripped = line.strip()
    if not stripped:
        return {}
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ProviderOutputError("Provider emitted malformed JSON output.") from exc
    if not isinstance(payload, dict):
        raise ProviderOutputError("Provider emitted a non-object JSON event.")
    return payload


def _payload_type(payload: dict[str, Any]) -> str:
    return str(payload.get("type") or payload.get("event") or payload.get("event_type") or "").strip().casefold()


def _extract_external_id(payload: dict[str, Any]) -> str | None:
    keys = (
        "thread_id",
        "threadId",
        "session_id",
        "sessionId",
        "conversation_id",
        "conversationId",
    )
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return _bounded_text(value.strip(), 200)
    nested = payload.get("message")
    if isinstance(nested, dict):
        return _extract_external_id(nested)
    return None


def _payload_error_text(payload: dict[str, Any]) -> str:
    for key in ("error", "error_message", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip() and (key != "message" or "error" in _payload_type(payload)):
            return _bounded_text(value, 500)
        if isinstance(value, dict):
            nested = value.get("message") or value.get("error")
            if isinstance(nested, str) and nested.strip():
                return _bounded_text(nested, 500)
    return ""


def _extract_text(payload: dict[str, Any]) -> str:
    for key in ("text", "content", "message", "delta", "output"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return _bounded_text(value, MAX_EVENT_SUMMARY_CHARS)
    message = payload.get("message")
    if isinstance(message, dict):
        text = _extract_text(message)
        if text:
            return text
    return ""


def _extract_command(payload: dict[str, Any]) -> str:
    event_type = _payload_type(payload)
    if not any(marker in event_type for marker in ("command", "exec", "tool")):
        return ""
    for key in ("command", "cmd", "name", "tool", "tool_name"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return _bounded_text(value, 500)
        if isinstance(value, list):
            return _bounded_text(" ".join(str(item) for item in value), 500)
    return _bounded_text(event_type or "command", 500)


def _extract_file_path(payload: dict[str, Any]) -> str:
    event_type = _payload_type(payload)
    if not any(marker in event_type for marker in ("file", "read", "grep", "search")):
        return ""
    for key in ("path", "file", "file_path", "filepath"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return _bounded_text(value, 500)
    return ""


def _extract_usage(payload: dict[str, Any]) -> UsageValues:
    candidates: list[dict[str, Any]] = []
    for key in ("usage", "token_usage"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    response = payload.get("response")
    if isinstance(response, dict) and isinstance(response.get("usage"), dict):
        candidates.append(response["usage"])
    if not candidates:
        candidates = [payload]
    combined: dict[str, Any] = {}
    for candidate in candidates:
        combined.update(candidate)
    input_tokens = _first_int(
        combined,
        "input_tokens",
        "prompt_tokens",
        "prompt_input_tokens",
    )
    cached_input_tokens = _sum_ints(
        combined,
        "cached_input_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    )
    output_tokens = _first_int(combined, "output_tokens", "completion_tokens")
    reasoning_tokens = _first_int(combined, "reasoning_tokens")
    details = combined.get("output_tokens_details")
    if reasoning_tokens is None and isinstance(details, dict):
        reasoning_tokens = _first_int(details, "reasoning_tokens")
    cost = _first_float(combined, "total_cost_usd", "cost_usd", "estimated_cost_usd")
    return UsageValues(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        estimated_cost_usd=cost,
        token_usage_source=(
            "reported"
            if any(
                value is not None
                for value in (input_tokens, cached_input_tokens, output_tokens, reasoning_tokens)
            )
            else None
        ),
        cost_source="reported" if cost is not None else None,
    )


def _first_int(mapping: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def _sum_ints(mapping: dict[str, Any], *keys: str) -> int | None:
    total = 0
    found = False
    for key in keys:
        value = _first_int(mapping, key)
        if value is not None:
            total += value
            found = True
    return total if found else None


def _first_float(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _duration_seconds_from_payload(payload: dict[str, Any]) -> float | None:
    for key in ("duration_seconds", "duration_s"):
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    value = payload.get("duration_ms")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) / 1000.0
    return None


def _is_reasoning_event(event_type: str, payload: dict[str, Any]) -> bool:
    if any(marker in event_type for marker in ("reasoning", "thinking", "thought")):
        return True
    return any(str(key).casefold() in _REASONING_KEY_NAMES for key in payload)


def _safe_subset(payload: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: payload[key] for key in keys if key in payload}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _bounded_event_json(content: dict[str, Any]) -> str:
    redacted = _redact_event_value(content)
    encoded = json.dumps(redacted, sort_keys=True, ensure_ascii=True)
    if len(encoded) <= MAX_EVENT_CONTENT_CHARS:
        return encoded
    preview = encoded[: MAX_EVENT_CONTENT_CHARS - len(_OUTPUT_TRUNCATION_MARKER)]
    return json.dumps(
        {"truncated": True, "preview": preview + _OUTPUT_TRUNCATION_MARKER},
        sort_keys=True,
        ensure_ascii=True,
    )


def _redact_event_value(value: Any) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            lowered = key_text.casefold()
            if _SENSITIVE_KEY_PATTERN.search(key_text):
                clean[key_text] = "[redacted]"
            elif lowered in _REASONING_KEY_NAMES:
                clean[key_text] = "[omitted]"
            else:
                clean[key_text] = _redact_event_value(child)
        return clean
    if isinstance(value, list):
        return [_redact_event_value(item) for item in value[:50]]
    if isinstance(value, str):
        return _bounded_text(value, 1_000)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _bounded_text(str(value), 500)


def _bounded_text(text: object, limit: int) -> str:
    clean = " ".join(str(text).strip().split())
    return clean[:limit]


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_job_id(value: str) -> bool:
    return isinstance(value, str) and len(value) == 32 and all(character in "0123456789abcdef" for character in value)


def _run_status_command(args: Sequence[str], cwd: Path) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        list(args),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        check=False,
        timeout=STATUS_TIMEOUT_SECONDS,
    )
    return _cap_completed_output(completed)


def _cap_completed_output(completed: subprocess.CompletedProcess) -> subprocess.CompletedProcess:
    stdout = bytes(completed.stdout or b"")[:STATUS_OUTPUT_LIMIT_BYTES]
    stderr = bytes(completed.stderr or b"")[:STATUS_OUTPUT_LIMIT_BYTES]
    return subprocess.CompletedProcess(completed.args, completed.returncode, stdout=stdout, stderr=stderr)


def _status_stdout(completed: subprocess.CompletedProcess) -> str:
    return bytes(completed.stdout or b"").decode("utf-8", errors="replace").strip()


def _status_stderr(completed: subprocess.CompletedProcess) -> str:
    return bytes(completed.stderr or b"").decode("utf-8", errors="replace").strip()


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
            except OSError:
                pass
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except OSError:
        try:
            proc.terminate()
        except OSError:
            return
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass
