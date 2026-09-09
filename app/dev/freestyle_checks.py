"""Allowlisted, isolated check execution for freestyle coding jobs.

This developer-only slice runs checks inside one approved, successfully
materialized freestyle worktree and persists a bounded check-run artifact in
ignored local dev state. Only checks from the static in-code allowlist below
can execute: callers select stable check ids and can never supply command
strings, arguments, executables, environment values, shell fragments, or a
working directory. Every check is spawned through one narrow runner as the
current Python interpreter with ``-m`` and a fixed argv list, ``shell=False``,
no stdin, a constructed minimal environment that never forwards parent
process environment values, and the validated isolated worktree as its only
working directory. The slice does not call a model, read ``.env`` or secret
files, touch approvals or SQLite, run package managers or repository scripts,
or perform any Git write, remote, or network action.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import hashlib
import importlib.util
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import time
from typing import Any
import uuid

from app.config import ROOT_DIR
from app.dev.freestyle_artifacts import encode_json_artifact, write_artifact_bytes
from app.dev.freestyle_coding import (
    FreestyleJob,
    FreestyleJobRefusedError,
    _bounded_error,
    _require_explicit_developer_approval,
    _utc_now,
    _validated_job_worktree,
    _write_job,
    get_freestyle_job,
)
from app.dev.freestyle_policy import recovery_message
from app.dev.freestyle_review import (
    _ChangedPath,
    _path_omission_reason,
    _read_changed_paths,
)
from app.dev.freestyle_state import (
    require_materialized,
    require_no_active_edit_application,
    resolved_roots,
)

CHECKS_SCHEMA_VERSION = 1
MAX_CHECKS_ARTIFACT_BYTES = 1_000_000
MAX_CHECK_TIMEOUT_SECONDS = 3_600
MAX_CHECK_OUTPUT_BYTES_LIMIT = 200_000
_HASH_CHUNK_BYTES = 64 * 1024
_OUTPUT_TRUNCATION_MARKER = "\n[output truncated]\n"
# Content of secret-like or runtime-data paths is never read, even for
# hashing; their snapshot fingerprint comes from lstat metadata only.
_UNHASHED_OMISSION_REASONS = {"secret_or_env", "database_or_runtime_data"}
_ALLOWED_CHECK_MODULES = frozenset({"pytest", "ruff"})
_FORBIDDEN_CHECK_TOKENS = frozenset(
    {
        "apt",
        "bash",
        "brew",
        "cmake",
        "conda",
        "curl",
        "dash",
        "docker",
        "git",
        "install",
        "make",
        "npm",
        "npx",
        "pip",
        "pip3",
        "pnpm",
        "poetry",
        "powershell",
        "python",
        "python3",
        "sh",
        "ssh",
        "uv",
        "wget",
        "yarn",
        "zsh",
    }
)
_SHELL_METACHARACTERS = ";|&$`<>*?(){}[]\"'\\\n\r\t "
_SAFE_PATH = "/usr/bin:/bin" if os.name == "posix" else os.defpath
# Constructed from scratch on purpose: filtering the parent environment is
# fail-open, so no parent process key or value is ever forwarded to a check.
_CHECK_ENVIRONMENT = {
    "LC_ALL": "C",
    "PATH": _SAFE_PATH,
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONNOUSERSITE": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
}
_RECOVERY_MESSAGE = recovery_message("retrying checks")
_CHECKS_EXECUTION_STATEMENT = (
    "This check run executed only static allowlisted checks inside the "
    "isolated job worktree: no commit, merge, push, pull, fetch, rebase, "
    "PR creation, auto-merge, package installation, repository script, "
    "shell command, or model call was performed, and the main checkout "
    "was not touched."
)


@dataclass(frozen=True)
class CheckDefinition:
    """One statically allowlisted check; only these definitions can execute."""

    check_id: str
    description: str
    module: str
    argv_suffix: tuple[str, ...]
    timeout_seconds: int
    max_output_bytes: int
    expected_failure: str
    requires_module: str


CHECK_REGISTRY: tuple[CheckDefinition, ...] = (
    CheckDefinition(
        check_id="pytest-freestyle",
        description=(
            "Run the focused freestyle pytest tests from the isolated job "
            "worktree's tests directory."
        ),
        module="pytest",
        argv_suffix=("-q", "-p", "no:cacheprovider", "-k", "freestyle", "tests"),
        timeout_seconds=600,
        max_output_bytes=32_000,
        expected_failure=(
            "A nonzero exit code means failing, missing, or uncollectable "
            "focused tests; pytest uses 1 for failures, 4 for usage errors, "
            "and 5 when no tests were collected."
        ),
        requires_module="pytest",
    ),
    CheckDefinition(
        check_id="ruff-check",
        description=(
            "Run ruff lint over the isolated job worktree without fixing "
            "files or writing a cache."
        ),
        module="ruff",
        argv_suffix=("check", ".", "--no-cache", "--no-fix"),
        timeout_seconds=120,
        max_output_bytes=32_000,
        expected_failure=(
            "A nonzero exit code means lint violations (1) or a ruff "
            "invocation error (2)."
        ),
        requires_module="ruff",
    ),
)
CHECKS_BY_ID: dict[str, CheckDefinition] = {
    definition.check_id: definition for definition in CHECK_REGISTRY
}


def _require_safe_registry() -> None:
    """Fail at import time if the static allowlist ever drifts unsafe."""
    if len(CHECKS_BY_ID) != len(CHECK_REGISTRY):
        raise RuntimeError("Freestyle check registry contains duplicate ids.")
    for definition in CHECK_REGISTRY:
        identifier = definition.check_id
        if (
            not identifier
            or len(identifier) > 64
            or identifier != identifier.strip()
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
                for character in identifier
            )
        ):
            raise RuntimeError("Freestyle check registry contains an unsafe id.")
        if definition.module not in _ALLOWED_CHECK_MODULES:
            raise RuntimeError("Freestyle check registry names a forbidden module.")
        if definition.requires_module not in _ALLOWED_CHECK_MODULES:
            raise RuntimeError(
                "Freestyle check registry names a forbidden requirement."
            )
        tokens = (definition.module, *definition.argv_suffix)
        for token in tokens:
            if (
                not token
                or not token.isprintable()
                or token.casefold() in _FORBIDDEN_CHECK_TOKENS
                or any(character in _SHELL_METACHARACTERS for character in token)
            ):
                raise RuntimeError(
                    "Freestyle check registry contains a forbidden argv token."
                )
        if not 0 < definition.timeout_seconds <= MAX_CHECK_TIMEOUT_SECONDS:
            raise RuntimeError("Freestyle check registry timeout is out of bounds.")
        if not 0 < definition.max_output_bytes <= MAX_CHECK_OUTPUT_BYTES_LIMIT:
            raise RuntimeError("Freestyle check registry output cap is out of bounds.")


_require_safe_registry()


def run_freestyle_checks(
    job_id: str,
    *,
    check_ids: Sequence[str] | None = None,
    project_root: Path | None = None,
    state_root: Path | None = None,
) -> FreestyleJob:
    """Run only static allowlisted checks inside one isolated job worktree."""
    root, state = resolved_roots(project_root or ROOT_DIR, state_root)
    job = get_freestyle_job(job_id, project_root=root, state_root=state)
    if job.checks_status == "running":
        # Refuse without persisting so the "running" lock survives until the
        # developer verifies the worktree and resets the job's check state.
        raise FreestyleJobRefusedError(
            f"Freestyle job {job_id!r} already has a check run in progress or "
            f"interrupted. {_RECOVERY_MESSAGE}"
        )

    started_at = _utc_now()
    try:
        _require_checks_prerequisites(job)
        selected, selection_mode = _validated_check_selection(check_ids)
    except Exception as exc:
        _persist_checks_failure(job, state, exc, started_at=started_at)
        raise FreestyleJobRefusedError(
            f"Could not run checks for freestyle job {job_id}; the error was "
            "recorded in local dev state."
        ) from exc

    started = replace(
        job,
        checks_status="running",
        checks_started_at=started_at,
        checks_completed_at=None,
        checks_artifact_path=None,
        checks_selected_ids=tuple(definition.check_id for definition in selected),
        checks_passed_count=0,
        checks_failed_count=0,
        checks_timeout_count=0,
        checks_infra_error_count=0,
        checks_modified_worktree=False,
        checks_errors=(),
        updated_at=started_at,
    )
    _write_job(started, state)
    try:
        worktree = _validated_job_worktree(started, root, state)
        before = _snapshot_worktree_status(worktree)
        results = [_execute_check(definition, worktree) for definition in selected]
        after = _snapshot_worktree_status(worktree)
        _validated_job_worktree(started, root, state)
        modified_paths = _modified_snapshot_paths(before, after)
        artifact = _build_checks_artifact(
            started,
            worktree,
            requested_check_ids=check_ids,
            selection_mode=selection_mode,
            selected=selected,
            results=results,
            before=before,
            after=after,
            modified_paths=modified_paths,
            started_at=started_at,
        )
        artifact_path = _write_checks_artifact(artifact, state, started.job_id)
    except Exception as exc:
        _persist_checks_failure(
            started,
            state,
            exc,
            started_at=started_at,
            selected_ids=tuple(definition.check_id for definition in selected),
        )
        raise FreestyleJobRefusedError(
            f"Could not run checks for freestyle job {job_id}; the error was "
            "recorded in local dev state."
        ) from exc

    completed_at = artifact["generated_at"]
    counts = artifact["summary"]
    completed = replace(
        started,
        checks_status=counts["overall_status"],
        checks_completed_at=completed_at,
        checks_artifact_path=artifact_path,
        checks_passed_count=counts["passed"],
        checks_failed_count=counts["failed"],
        checks_timeout_count=counts["timeout"],
        checks_infra_error_count=counts["infra_error"],
        checks_modified_worktree=bool(modified_paths),
        checks_errors=(),
        updated_at=completed_at,
    )
    _write_job(completed, state)
    return completed


def _require_checks_prerequisites(job: FreestyleJob) -> None:
    _require_explicit_developer_approval(job, "run checks")
    require_materialized(
        job, "must be successfully materialized before checks can run."
    )
    require_no_active_edit_application(job, f". {_RECOVERY_MESSAGE}")


def _validated_check_selection(
    check_ids: Sequence[str] | None,
) -> tuple[tuple[CheckDefinition, ...], str]:
    if check_ids is None:
        available = tuple(
            definition
            for definition in CHECK_REGISTRY
            if _module_available(definition.requires_module)
        )
        if not available:
            raise FreestyleJobRefusedError(
                "No allowlisted checks are available in this environment."
            )
        return available, "all_available"
    if isinstance(check_ids, (str, bytes)) or not isinstance(check_ids, (list, tuple)):
        raise FreestyleJobRefusedError(
            "Check selection must be a list of allowlisted check ids."
        )
    if not check_ids:
        raise FreestyleJobRefusedError(
            "Check selection must include at least one allowlisted check id."
        )
    selected: list[CheckDefinition] = []
    seen: set[str] = set()
    for candidate in check_ids:
        if not isinstance(candidate, str):
            raise FreestyleJobRefusedError(
                "Check selection accepts allowlisted check ids only."
            )
        definition = CHECKS_BY_ID.get(candidate)
        if definition is None:
            raise FreestyleJobRefusedError(
                f"Unknown check id {_sanitized_check_id(candidate)!r}; allowed "
                f"ids: {', '.join(sorted(CHECKS_BY_ID))}."
            )
        if candidate in seen:
            raise FreestyleJobRefusedError(
                f"Check id {candidate!r} was selected more than once."
            )
        seen.add(candidate)
        selected.append(definition)
    return tuple(selected), "explicit"


def _sanitized_check_id(candidate: str) -> str:
    printable = "".join(character for character in candidate if character.isprintable())
    return printable[:80]


def _module_available(module_name: str) -> bool:
    if module_name not in _ALLOWED_CHECK_MODULES:
        raise FreestyleJobRefusedError("Internal error: forbidden check requirement.")
    return importlib.util.find_spec(module_name) is not None


def _execute_check(definition: CheckDefinition, worktree: Path) -> dict[str, Any]:
    """Spawn one allowlisted check; the only subprocess call in this slice."""
    if (
        CHECKS_BY_ID.get(definition.check_id) is not definition
        or definition.module not in _ALLOWED_CHECK_MODULES
    ):
        raise FreestyleJobRefusedError("Internal error: forbidden check execution.")
    argv = [sys.executable, "-m", definition.module, *definition.argv_suffix]
    result: dict[str, Any] = {
        "check_id": definition.check_id,
        "description": definition.description,
        "argv": argv,
        "expected_failure": definition.expected_failure,
        "timeout_seconds": definition.timeout_seconds,
        "started_at": _utc_now(),
        "completed_at": None,
        "status": "infra_error",
        "exit_code": None,
        "duration_seconds": 0.0,
        "timed_out": False,
        "stdout": "",
        "stdout_truncated": False,
        "stderr": "",
        "stderr_truncated": False,
        "note": None,
    }
    if not _module_available(definition.requires_module):
        result["completed_at"] = _utc_now()
        result["note"] = (
            f"Required module {definition.requires_module!r} is not importable "
            "in the current interpreter; the check was not executed and "
            "nothing was installed."
        )
        return result
    start = time.monotonic()
    stdout_bytes = b""
    stderr_bytes = b""
    try:
        completed = subprocess.run(
            argv,
            cwd=str(worktree),
            env=dict(_CHECK_ENVIRONMENT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
            timeout=definition.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_bytes = _coerced_output_bytes(exc.stdout)
        stderr_bytes = _coerced_output_bytes(exc.stderr)
        result["status"] = "timeout"
        result["timed_out"] = True
        result["note"] = (
            f"Check exceeded its {definition.timeout_seconds}s timeout and "
            "was terminated."
        )
    except OSError:
        result["status"] = "infra_error"
        result["note"] = "Check process could not start."
    else:
        stdout_bytes = _coerced_output_bytes(completed.stdout)
        stderr_bytes = _coerced_output_bytes(completed.stderr)
        result["exit_code"] = completed.returncode
        result["status"] = "passed" if completed.returncode == 0 else "failed"
    result["duration_seconds"] = round(time.monotonic() - start, 3)
    result["completed_at"] = _utc_now()
    result["stdout"], result["stdout_truncated"] = _bounded_output(
        stdout_bytes, definition.max_output_bytes
    )
    result["stderr"], result["stderr_truncated"] = _bounded_output(
        stderr_bytes, definition.max_output_bytes
    )
    return result


def _coerced_output_bytes(raw: Any) -> bytes:
    if raw is None:
        return b""
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, str):
        return raw.encode("utf-8", errors="replace")
    return b""


def _bounded_output(raw: bytes, limit: int) -> tuple[str, bool]:
    if len(raw) <= limit:
        return raw.decode("utf-8", errors="replace"), False
    truncated = raw[:limit].decode("utf-8", errors="replace")
    return truncated + _OUTPUT_TRUNCATION_MARKER, True


def _snapshot_worktree_status(worktree: Path) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "path": item.path,
            "status": item.status,
            "git_status": item.git_status,
            "content_digest": _content_digest(worktree, item),
        }
        for item in _read_changed_paths(worktree)
    )


def _content_digest(worktree: Path, item: _ChangedPath) -> str:
    omission_reason = _path_omission_reason(item.path)
    target = worktree.joinpath(*PurePosixPath(item.path).parts)
    try:
        info = target.lstat()
    except OSError:
        return "missing"
    if stat.S_ISLNK(info.st_mode):
        return "symlink"
    if not stat.S_ISREG(info.st_mode):
        return "non_regular_file"
    if omission_reason in _UNHASHED_OMISSION_REASONS:
        return f"stat:{info.st_size}:{info.st_mtime_ns}"
    open_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0)
    )
    hasher = hashlib.sha256()
    try:
        descriptor = os.open(str(target), open_flags)
    except OSError:
        return "unreadable"
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return "non_regular_file"
        while chunk := os.read(descriptor, _HASH_CHUNK_BYTES):
            hasher.update(chunk)
    except OSError:
        return "unreadable"
    finally:
        os.close(descriptor)
    return f"sha256:{hasher.hexdigest()}"


def _modified_snapshot_paths(
    before: tuple[dict[str, str], ...], after: tuple[dict[str, str], ...]
) -> tuple[dict[str, str], ...]:
    before_map = {entry["path"]: entry for entry in before}
    after_map = {entry["path"]: entry for entry in after}
    modified: list[dict[str, str]] = []
    for path in sorted(
        set(before_map) | set(after_map), key=lambda value: value.casefold()
    ):
        previous = before_map.get(path)
        current = after_map.get(path)
        if previous is None:
            modified.append({"path": path, "change": "appeared"})
        elif current is None:
            modified.append({"path": path, "change": "disappeared"})
        elif previous["content_digest"] != current["content_digest"]:
            modified.append({"path": path, "change": "content_changed"})
        elif previous["git_status"] != current["git_status"]:
            modified.append({"path": path, "change": "status_changed"})
    return tuple(modified)


def _build_checks_artifact(
    job: FreestyleJob,
    worktree: Path,
    *,
    requested_check_ids: Sequence[str] | None,
    selection_mode: str,
    selected: tuple[CheckDefinition, ...],
    results: list[dict[str, Any]],
    before: tuple[dict[str, str], ...],
    after: tuple[dict[str, str], ...],
    modified_paths: tuple[dict[str, str], ...],
    started_at: str,
) -> dict[str, Any]:
    statuses = [result["status"] for result in results]
    summary = {
        "total": len(results),
        "passed": statuses.count("passed"),
        "failed": statuses.count("failed"),
        "timeout": statuses.count("timeout"),
        "infra_error": statuses.count("infra_error"),
        "overall_status": (
            "passed" if all(status == "passed" for status in statuses) else "failed"
        ),
    }
    artifact = {
        "schema_version": CHECKS_SCHEMA_VERSION,
        "artifact_type": "freestyle_check_run",
        "run_id": uuid.uuid4().hex[:16],
        "generated_at": _utc_now(),
        "started_at": started_at,
        "job": {
            "job_id": job.job_id,
            "goal": job.goal,
            "branch": job.branch,
            "worktree_path": str(worktree),
        },
        "base_commit": job.base_commit,
        "current_head": {"commit": job.base_commit, "branch": job.branch},
        "selection": {
            "requested_check_ids": (
                None if requested_check_ids is None else list(requested_check_ids)
            ),
            "selected_check_ids": [definition.check_id for definition in selected],
            "selection_mode": selection_mode,
        },
        "checks": results,
        "summary": summary,
        "environment": {
            "policy": "constructed_allowlist_only",
            "variables": sorted(_CHECK_ENVIRONMENT),
            "parent_environment_forwarded": False,
        },
        "worktree_status": {
            "before": list(before),
            "after": list(after),
            "modified_by_checks": bool(modified_paths),
            "modified_paths": list(modified_paths),
        },
        "execution_statement": _CHECKS_EXECUTION_STATEMENT,
    }
    encoded = encode_json_artifact(artifact)
    if len(encoded) > MAX_CHECKS_ARTIFACT_BYTES:
        raise FreestyleJobRefusedError(
            "Check-run artifact exceeds the total size limit."
        )
    return artifact


def _write_checks_artifact(artifact: dict[str, Any], state: Path, job_id: str) -> str:
    encoded = encode_json_artifact(artifact)
    if len(encoded) > MAX_CHECKS_ARTIFACT_BYTES:
        raise FreestyleJobRefusedError(
            "Check-run artifact exceeds the total size limit."
        )
    return write_artifact_bytes(
        state,
        "checks",
        job_id,
        f"checks-{artifact['run_id']}.json",
        encoded,
        label="check",
    )


def _persist_checks_failure(
    job: FreestyleJob,
    state: Path,
    exc: Exception,
    *,
    started_at: str,
    selected_ids: tuple[str, ...] = (),
) -> None:
    completed_at = _utc_now()
    failed = replace(
        job,
        checks_status="error",
        checks_started_at=started_at,
        checks_completed_at=completed_at,
        checks_artifact_path=None,
        checks_selected_ids=tuple(selected_ids),
        checks_passed_count=0,
        checks_failed_count=0,
        checks_timeout_count=0,
        checks_infra_error_count=0,
        checks_modified_worktree=False,
        checks_errors=(_bounded_error(str(exc), "Check run failed."),),
        updated_at=completed_at,
    )
    _write_job(failed, state)
