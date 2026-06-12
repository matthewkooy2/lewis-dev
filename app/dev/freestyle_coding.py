"""Developer-only freestyle coding job scaffold.

Creates an isolated local branch and no-checkout worktree from the active
checkout's committed HEAD. The no-checkout form prevents repository-defined
checkout hooks and filters from executing. The proposal slice may call DeepSeek
after materialization, and the generate-edits slice may ask DeepSeek for a
validated structured edit bundle, but neither edits source files. The
apply-edits slice deterministically applies one validated structured edit
bundle inside the job's isolated worktree only. The review slice
deterministically persists a bounded read-only diff artifact. The run-checks
slice executes only checks from a static in-code allowlist inside the isolated
worktree with a sanitized environment. No slice reads secrets, touches SQLite,
or exposes itself through normal chat/runtime/web paths.

Invocation:

    python -m app.dev.freestyle_coding create --approve --goal "<goal>"
    python -m app.dev.freestyle_coding list
    python -m app.dev.freestyle_coding show <job-id>
    python -m app.dev.freestyle_coding status <job-id> [--json]
    python -m app.dev.freestyle_coding reapprove <job-id> --approve
    python -m app.dev.freestyle_coding materialize <job-id>
    python -m app.dev.freestyle_coding propose <job-id>
    python -m app.dev.freestyle_coding generate-edits <job-id>
    python -m app.dev.freestyle_coding apply-edits <job-id> <bundle.json>
    python -m app.dev.freestyle_coding review <job-id>
    python -m app.dev.freestyle_coding run-checks <job-id> [--check <check-id> ...]
    python -m app.dev.freestyle_coding run <job-id>
    python -m app.dev.freestyle_coding run <job-id> \
        --apply-latest-generated --confirm-apply [--run-checks]
    python -m app.dev.freestyle_coding cleanup-status
    python -m app.dev.freestyle_coding recover <job-id> --confirm
    python -m app.dev.freestyle_coding cleanup <job-id> --confirm
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import subprocess
import sys
import tarfile
from typing import Any
import uuid

from app.code_proposal_path_safety import (
    BLOCKED_FILE_NAMES,
    BLOCKED_PATH_PARTS,
    BLOCKED_SUFFIXES,
    SECRET_NAME_PATTERN,
)
from app.config import ROOT_DIR
from app.dev.freestyle_policy import is_job_id

MAX_GOAL_LENGTH = 10_000
MAX_TITLE_LENGTH = 120
MAX_MATERIALIZED_FILES = 20_000
MAX_MATERIALIZED_FILE_BYTES = 5_000_000
MAX_MATERIALIZED_TOTAL_BYTES = 100_000_000
MAX_ARCHIVE_BYTES = 125_000_000
MAX_ARCHIVE_PATHS_PER_COMMAND = 400
DEV_STATE_DIR_NAME = ".lewis-dev"
BRANCH_PREFIX = "lewis/freestyle-"
_GIT_TIMEOUT_SECONDS = 60
_MAX_CLI_ERROR_LENGTH = 500
_SENSITIVE_ENV_NAME = re.compile(
    r"(?:api|auth|credential|key|password|secret|token)", re.IGNORECASE
)
_CLI_COMMANDS = {
    "apply-edits",
    "cleanup",
    "cleanup-status",
    "create",
    "generate-edits",
    "list",
    "materialize",
    "propose",
    "recover",
    "reapprove",
    "review",
    "run",
    "run-checks",
    "show",
    "status",
}
_WINDOWS_RESERVED_NAMES = {
    "aux",
    "clock$",
    "con",
    "conin$",
    "conout$",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}
_ALLOWED_GIT_PREFIXES = (
    ("git", "rev-parse", "--show-toplevel"),
    ("git", "rev-parse", "HEAD"),
    ("git", "rev-parse", "--abbrev-ref", "HEAD"),
    (
        "git",
        "--no-optional-locks",
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
    ),
)
_MATERIALIZATION_GIT_PREFIX = (
    "git",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.fsmonitor=false",
)


class FreestyleJobRefusedError(RuntimeError):
    """Raised when a freestyle job request or setup operation is refused."""


class FreestyleCliInputError(ValueError):
    """Raised for malformed freestyle CLI arguments."""


class FreestyleArgumentParser(argparse.ArgumentParser):
    """Argument parser that lets the CLI format expected input errors."""

    def error(self, message: str) -> None:
        raise FreestyleCliInputError(message)


@dataclass(frozen=True)
class GitCommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class GitBytesCommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: str


@dataclass(frozen=True)
class SnapshotBlob:
    path: str
    object_id: str
    mode: int


@dataclass(frozen=True)
class SnapshotPaths:
    materialized: tuple[SnapshotBlob, ...]
    omitted: tuple[str, ...]

    @property
    def materialized_paths(self) -> tuple[str, ...]:
        return tuple(blob.path for blob in self.materialized)


@dataclass(frozen=True)
class FreestyleJob:
    job_id: str
    title: str
    goal: str
    branch: str
    worktree_path: str
    base_commit: str
    status: str
    created_at: str
    updated_at: str
    developer_approved: Any = None
    errors: tuple[str, ...] = ()
    materialization_status: str = "not_materialized"
    materialization_started_at: str | None = None
    materialized_at: str | None = None
    materialized_file_count: int = 0
    proposal_status: str = "not_proposed"
    proposal_started_at: str | None = None
    proposal_completed_at: str | None = None
    proposal_artifact_path: str | None = None
    proposal_failure_artifact_path: str | None = None
    proposal_inspected_file_count: int = 0
    proposal_model: str | None = None
    proposal_errors: tuple[str, ...] = ()
    edit_generation_status: str = "not_generated"
    edit_generation_started_at: str | None = None
    edit_generation_completed_at: str | None = None
    edit_generation_artifact_path: str | None = None
    edit_bundle_path: str | None = None
    edit_bundle_validation_status: str = "not_validated"
    edit_bundle_candidate_file_count: int = 0
    edit_bundle_goal_fit_warnings: tuple[str, ...] = ()
    edit_generation_model: str | None = None
    edit_generation_errors: tuple[str, ...] = ()
    edit_apply_status: str = "not_applied"
    edit_apply_started_at: str | None = None
    edit_applied_at: str | None = None
    edit_applied_file_count: int = 0
    edit_artifact_path: str | None = None
    edit_errors: tuple[str, ...] = ()
    review_status: str = "not_reviewed"
    review_started_at: str | None = None
    review_completed_at: str | None = None
    review_artifact_path: str | None = None
    review_changed_file_count: int = 0
    review_errors: tuple[str, ...] = ()
    checks_status: str = "not_run"
    checks_started_at: str | None = None
    checks_completed_at: str | None = None
    checks_artifact_path: str | None = None
    checks_selected_ids: tuple[str, ...] = ()
    checks_passed_count: int = 0
    checks_failed_count: int = 0
    checks_timeout_count: int = 0
    checks_infra_error_count: int = 0
    checks_modified_worktree: bool = False
    checks_errors: tuple[str, ...] = ()
    latest_run_id: str | None = None
    latest_run_status: str = "not_run"
    latest_run_started_at: str | None = None
    latest_run_completed_at: str | None = None
    latest_run_artifact_path: str | None = None
    latest_run_completed_stages: tuple[str, ...] = ()
    latest_run_errors: tuple[str, ...] = ()

    @property
    def approval_state(self) -> str:
        if self.developer_approved is True:
            return "approved"
        if self.developer_approved is False:
            return "not_approved"
        if self.developer_approved is None:
            return "missing"
        return "invalid"


def create_freestyle_job(
    goal: str,
    *,
    title: str | None = None,
    developer_approved: bool = False,
    project_root: Path | None = None,
    state_root: Path | None = None,
) -> FreestyleJob:
    """Create one approved, local-only freestyle job scaffold."""
    if developer_approved is not True:
        raise FreestyleJobRefusedError(
            "Explicit developer approval is required to create a freestyle job."
        )
    clean_goal = validate_goal(goal)
    clean_title = validate_title(title, clean_goal)
    root = (project_root or ROOT_DIR).resolve()
    state = (state_root or root / DEV_STATE_DIR_NAME).resolve()
    _require_state_inside_project(state, root)
    _require_git_toplevel(root)
    base_commit = _current_clean_head_commit(root)

    job_id = uuid.uuid4().hex
    branch = f"{BRANCH_PREFIX}{job_id[:12]}"
    worktree = state / "worktrees" / job_id
    now = _utc_now()
    job = FreestyleJob(
        job_id=job_id,
        title=clean_title,
        goal=clean_goal,
        branch=branch,
        worktree_path=str(worktree),
        base_commit=base_commit,
        status="preparing",
        created_at=now,
        updated_at=now,
        developer_approved=True,
    )
    _write_job(job, state)

    result = _run_git(
        [
            "git",
            "worktree",
            "add",
            "--no-checkout",
            "-b",
            branch,
            str(worktree),
            base_commit,
        ],
        root,
    )
    if result.returncode != 0:
        failed = replace(
            job,
            status="error",
            updated_at=_utc_now(),
            errors=(_bounded_error(result.stderr or result.stdout),),
        )
        _write_job(failed, state)
        raise FreestyleJobRefusedError(
            f"Could not create isolated worktree for job {job_id}; "
            "the error was recorded in local dev state."
        )

    ready = replace(job, status="ready", updated_at=_utc_now())
    _write_job(ready, state)
    return ready


def reapprove_freestyle_job(
    job_id: str,
    *,
    developer_approved: bool = False,
    project_root: Path | None = None,
    state_root: Path | None = None,
) -> FreestyleJob:
    """Explicitly reapprove one existing job by updating only its dev metadata."""
    if developer_approved is not True:
        raise FreestyleJobRefusedError(
            "Explicit developer approval is required to reapprove a freestyle job."
        )
    root = (project_root or ROOT_DIR).resolve()
    state = (state_root or root / DEV_STATE_DIR_NAME).resolve()
    _require_state_inside_project(state, root)
    job = get_freestyle_job(job_id, project_root=root, state_root=state)
    approved = replace(job, developer_approved=True, updated_at=_utc_now())
    _write_job(approved, state)
    return approved


def materialize_freestyle_job(
    job_id: str,
    *,
    project_root: Path | None = None,
    state_root: Path | None = None,
) -> FreestyleJob:
    """Safely populate one existing no-checkout freestyle worktree."""
    root = (project_root or ROOT_DIR).resolve()
    state = (state_root or root / DEV_STATE_DIR_NAME).resolve()
    _require_state_inside_project(state, root)
    job = get_freestyle_job(job_id, project_root=root, state_root=state)
    _require_explicit_developer_approval(job, "materialize")
    _require_git_toplevel(root)
    if job.status == "materialized" and job.materialization_status == "materialized":
        raise FreestyleJobRefusedError(
            f"Freestyle job {job_id!r} is already materialized."
        )
    if job.status != "ready" or job.materialization_status != "not_materialized":
        raise FreestyleJobRefusedError(
            f"Freestyle job {job_id!r} is not ready for materialization."
        )

    started = replace(
        job,
        status="materializing",
        materialization_status="materializing",
        materialization_started_at=_utc_now(),
        updated_at=_utc_now(),
        errors=(),
    )
    _write_job(started, state)
    try:
        worktree = _validated_job_worktree(started, root, state)
        _require_empty_no_checkout_worktree(worktree)
        snapshot_paths = _list_snapshot_paths(started.base_commit, root)
        archives = _archive_snapshot(snapshot_paths.materialized, root)
        archive_members = tuple(_preflight_archive(archive) for archive in archives)
        materialized_paths = tuple(
            path for members in archive_members for path in members
        )
        if set(materialized_paths) != set(snapshot_paths.materialized_paths):
            raise FreestyleJobRefusedError(
                "Git archive paths did not match the validated repository snapshot."
            )
        file_count = 0
        for archive, members in zip(archives, archive_members):
            file_count += _extract_preflighted_archive(archive, members, worktree)
        _synchronize_materialized_worktree_index(
            started, snapshot_paths, worktree, root, state
        )
    except (FreestyleJobRefusedError, OSError, tarfile.TarError) as exc:
        failed = replace(
            started,
            status="materialization_error",
            materialization_status="error",
            updated_at=_utc_now(),
            errors=(_bounded_error(str(exc), "Materialization failed."),),
        )
        _write_job(failed, state)
        raise FreestyleJobRefusedError(
            f"Could not materialize freestyle job {job_id}; "
            "the error was recorded in local dev state."
        ) from exc

    completed_at = _utc_now()
    materialized = replace(
        started,
        status="materialized",
        materialization_status="materialized",
        materialized_at=completed_at,
        materialized_file_count=file_count,
        updated_at=completed_at,
        errors=(),
    )
    _write_job(materialized, state)
    return materialized


def propose_freestyle_job(
    job_id: str,
    *,
    project_root: Path | None = None,
    state_root: Path | None = None,
    deepseek_config: Any | None = None,
    deepseek_client: Any | None = None,
) -> FreestyleJob:
    """Generate a structured DeepSeek proposal for one materialized job."""
    from app.dev.freestyle_proposal import generate_freestyle_proposal

    return generate_freestyle_proposal(
        job_id,
        project_root=project_root,
        state_root=state_root,
        deepseek_config=deepseek_config,
        deepseek_client=deepseek_client,
    )


def apply_freestyle_edit_bundle(
    job_id: str,
    bundle_path: str,
    *,
    project_root: Path | None = None,
    state_root: Path | None = None,
) -> FreestyleJob:
    """Apply one structured edit bundle inside one materialized job worktree."""
    from app.dev.freestyle_edit_bundle import apply_edit_bundle

    return apply_edit_bundle(
        job_id,
        bundle_path,
        project_root=project_root,
        state_root=state_root,
    )


def generate_freestyle_edit_bundle(
    job_id: str,
    *,
    project_root: Path | None = None,
    state_root: Path | None = None,
    deepseek_config: Any | None = None,
    deepseek_client: Any | None = None,
) -> FreestyleJob:
    """Generate and validate one DeepSeek edit bundle without applying it."""
    from app.dev.freestyle_edit_generation import generate_edit_bundle

    return generate_edit_bundle(
        job_id,
        project_root=project_root,
        state_root=state_root,
        deepseek_config=deepseek_config,
        deepseek_client=deepseek_client,
    )


def review_freestyle_job(
    job_id: str,
    *,
    project_root: Path | None = None,
    state_root: Path | None = None,
) -> FreestyleJob:
    """Persist one deterministic, read-only review of an isolated job worktree."""
    from app.dev.freestyle_review import generate_freestyle_review

    return generate_freestyle_review(
        job_id,
        project_root=project_root,
        state_root=state_root,
    )


def run_freestyle_checks(
    job_id: str,
    *,
    check_ids: Sequence[str] | None = None,
    project_root: Path | None = None,
    state_root: Path | None = None,
) -> FreestyleJob:
    """Run only static allowlisted checks inside one isolated job worktree."""
    from app.dev.freestyle_checks import run_freestyle_checks as run_checks

    return run_checks(
        job_id,
        check_ids=check_ids,
        project_root=project_root,
        state_root=state_root,
    )


def run_freestyle_job(
    job_id: str,
    *,
    apply_latest_generated: bool = False,
    confirm_apply: bool = False,
    run_checks: bool = False,
    check_ids: Sequence[str] | None = None,
    project_root: Path | None = None,
    state_root: Path | None = None,
    deepseek_config: Any | None = None,
    deepseek_client: Any | None = None,
) -> FreestyleJob:
    """Advance one approved job through existing safe stages."""
    from app.dev.freestyle_run import run_freestyle_job as orchestrate

    return orchestrate(
        job_id,
        apply_latest_generated=apply_latest_generated,
        confirm_apply=confirm_apply,
        run_checks=run_checks,
        check_ids=check_ids,
        project_root=project_root,
        state_root=state_root,
        deepseek_config=deepseek_config,
        deepseek_client=deepseek_client,
    )


def get_freestyle_handoff(
    job_id: str,
    *,
    project_root: Path | None = None,
    state_root: Path | None = None,
) -> dict[str, Any]:
    """Build a read-only human status/handoff summary for one job."""
    from app.dev.freestyle_handoff import get_freestyle_handoff as build_handoff

    return build_handoff(
        job_id,
        project_root=project_root or ROOT_DIR,
        state_root=state_root,
    )


def get_freestyle_cleanup_status(
    *,
    project_root: Path | None = None,
    state_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Return a read-only inventory of local freestyle maintenance state."""
    from app.dev.freestyle_maintenance import get_freestyle_cleanup_status as inspect

    return inspect(project_root=project_root or ROOT_DIR, state_root=state_root)


def cleanup_freestyle_job(
    job_id: str,
    *,
    confirm: bool = False,
    discard_worktree_changes: bool = False,
    project_root: Path | None = None,
    state_root: Path | None = None,
) -> dict[str, Any]:
    """Clean one explicitly confirmed local freestyle job."""
    from app.dev.freestyle_maintenance import cleanup_freestyle_job as cleanup

    return cleanup(
        job_id,
        confirm=confirm,
        discard_worktree_changes=discard_worktree_changes,
        project_root=project_root or ROOT_DIR,
        state_root=state_root,
    )


def recover_freestyle_job(
    job_id: str,
    *,
    confirm: bool = False,
    confirm_inconsistent: bool = False,
    project_root: Path | None = None,
    state_root: Path | None = None,
) -> FreestyleJob:
    """Recover explicitly confirmed stale lock metadata for one job."""
    from app.dev.freestyle_maintenance import recover_freestyle_job as recover

    return recover(
        job_id,
        confirm=confirm,
        confirm_inconsistent=confirm_inconsistent,
        project_root=project_root or ROOT_DIR,
        state_root=state_root,
    )


def validate_goal(goal: str) -> str:
    if not isinstance(goal, str):
        raise FreestyleJobRefusedError("Goal must be text.")
    clean = goal.strip()
    if not clean:
        raise FreestyleJobRefusedError("Goal must not be empty.")
    if len(clean) > MAX_GOAL_LENGTH:
        raise FreestyleJobRefusedError(
            f"Goal exceeds the maximum length of {MAX_GOAL_LENGTH} characters."
        )
    if "\x00" in clean:
        raise FreestyleJobRefusedError("Goal must not contain NUL bytes.")
    return clean


def validate_title(title: str | None, goal: str) -> str:
    if title is None:
        return goal.splitlines()[0][:MAX_TITLE_LENGTH].strip()
    if not isinstance(title, str):
        raise FreestyleJobRefusedError("Title must be text.")
    clean = title.strip()
    if not clean:
        return goal.splitlines()[0][:MAX_TITLE_LENGTH].strip()
    if len(clean) > MAX_TITLE_LENGTH:
        raise FreestyleJobRefusedError(
            f"Title exceeds the maximum length of {MAX_TITLE_LENGTH} characters."
        )
    if any(character in clean for character in ("\x00", "\r", "\n")):
        raise FreestyleJobRefusedError("Title must be a single line without NUL bytes.")
    return clean


def list_freestyle_jobs(
    *,
    project_root: Path | None = None,
    state_root: Path | None = None,
) -> list[FreestyleJob]:
    root = (project_root or ROOT_DIR).resolve()
    state = (state_root or root / DEV_STATE_DIR_NAME).resolve()
    _require_state_inside_project(state, root)
    jobs_dir = _validated_jobs_dir(state, create=False)
    if not jobs_dir.exists():
        return []
    jobs = [_read_job(path) for path in jobs_dir.glob("*.json")]
    return sorted(jobs, key=lambda item: item.created_at, reverse=True)


def get_freestyle_job(
    job_id: str,
    *,
    project_root: Path | None = None,
    state_root: Path | None = None,
) -> FreestyleJob:
    _validate_job_id(job_id)
    root = (project_root or ROOT_DIR).resolve()
    state = (state_root or root / DEV_STATE_DIR_NAME).resolve()
    _require_state_inside_project(state, root)
    path = _validated_jobs_dir(state, create=False) / f"{job_id}.json"
    if not path.is_file() or path.is_symlink():
        raise FreestyleJobRefusedError(f"Freestyle job {job_id!r} was not found.")
    return _read_job(path)


def _run_git(args: list[str], root: Path) -> GitCommandResult:
    """Run only the fixed git operations needed for worktree setup."""
    completed = _run_git_bytes(args, root)
    return GitCommandResult(
        args=completed.args,
        returncode=completed.returncode,
        stdout=completed.stdout.decode("utf-8", errors="replace"),
        stderr=completed.stderr,
    )


def _run_git_bytes(args: list[str], root: Path) -> GitBytesCommandResult:
    """Run one constrained local Git operation, optionally returning binary output."""
    command = tuple(args)
    if not _is_allowed_git_command(command):
        raise FreestyleJobRefusedError("Internal error: forbidden git operation.")
    try:
        completed = subprocess.run(
            args,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return GitBytesCommandResult(command, 124, b"", "Git operation timed out.")
    except OSError:
        return GitBytesCommandResult(command, 1, b"", "Git operation could not start.")
    return GitBytesCommandResult(
        args=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr.decode("utf-8", errors="replace"),
    )


def _run_materialization_git_bytes(
    args: list[str],
    worktree: Path,
    *,
    omitted_paths: tuple[str, ...] = (),
) -> GitBytesCommandResult:
    """Run one exact index-only Git operation inside a job worktree."""
    command = tuple(args)
    if not _is_allowed_materialization_git_command(command):
        raise FreestyleJobRefusedError(
            "Internal error: forbidden materialization git operation."
        )
    input_bytes: bytes | None = None
    update_index = command == _MATERIALIZATION_GIT_PREFIX + (
        "update-index",
        "-z",
        "--skip-worktree",
        "--stdin",
    )
    if update_index:
        if not omitted_paths:
            raise FreestyleJobRefusedError(
                "Internal error: omitted index paths are required."
            )
        for path in omitted_paths:
            try:
                validated = _validated_archive_member_path(path)
            except FreestyleJobRefusedError as exc:
                raise FreestyleJobRefusedError(
                    "Internal error: invalid omitted index path."
                ) from exc
            if validated is not None:
                raise FreestyleJobRefusedError(
                    "Internal error: materialized paths cannot be skipped in the index."
                )
        input_bytes = b"".join(path.encode("utf-8") + b"\x00" for path in omitted_paths)
    elif omitted_paths:
        raise FreestyleJobRefusedError(
            "Internal error: index paths are not allowed for this git operation."
        )
    try:
        completed = subprocess.run(
            args,
            cwd=str(worktree),
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return GitBytesCommandResult(command, 124, b"", "Git operation timed out.")
    except OSError:
        return GitBytesCommandResult(command, 1, b"", "Git operation could not start.")
    return GitBytesCommandResult(
        args=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr.decode("utf-8", errors="replace"),
    )


def _run_git_blob_batch(
    blobs: tuple[SnapshotBlob, ...], root: Path, *, check_only: bool = False
) -> GitBytesCommandResult:
    """Read exact committed blob bytes without checkout or attribute filters."""
    command = ("git", "cat-file", "--batch-check" if check_only else "--batch")
    if (
        not blobs
        or len(blobs) > MAX_ARCHIVE_PATHS_PER_COMMAND
        or any(not _is_commit_id(blob.object_id) for blob in blobs)
    ):
        raise FreestyleJobRefusedError("Internal error: invalid git blob batch.")
    input_bytes = b"".join(
        blob.object_id.encode("ascii") + b"\n" for blob in blobs
    )
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(root),
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return GitBytesCommandResult(command, 124, b"", "Git operation timed out.")
    except OSError:
        return GitBytesCommandResult(command, 1, b"", "Git operation could not start.")
    return GitBytesCommandResult(
        args=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr.decode("utf-8", errors="replace"),
    )


def _is_allowed_git_command(args: tuple[str, ...]) -> bool:
    if args in _ALLOWED_GIT_PREFIXES:
        return True
    if (
        len(args) == 6
        and args[:5] == ("git", "ls-tree", "-r", "-z", "--full-tree")
        and _is_commit_id(args[5])
    ):
        return True
    return (
        len(args) == 8
        and args[:3] == ("git", "worktree", "add")
        and args[3:5] == ("--no-checkout", "-b")
        and args[5].startswith(BRANCH_PREFIX)
        and len(args[5]) == len(BRANCH_PREFIX) + 12
        and all(character in "0123456789abcdef" for character in args[5][-12:])
        and _is_commit_id(args[7])
    )


def _is_allowed_materialization_git_command(args: tuple[str, ...]) -> bool:
    suffix = args[len(_MATERIALIZATION_GIT_PREFIX) :]
    if args[: len(_MATERIALIZATION_GIT_PREFIX)] != _MATERIALIZATION_GIT_PREFIX:
        return False
    if suffix == (
        "update-index",
        "-z",
        "--skip-worktree",
        "--stdin",
    ):
        return True
    if suffix == ("ls-files", "-v", "-z"):
        return True
    return (
        len(suffix) == 2
        and suffix[0] == "read-tree"
        and _is_commit_id(suffix[1])
    ) or (
        len(suffix) == 5
        and suffix[:3] == ("diff-index", "--cached", "--quiet")
        and _is_commit_id(suffix[3])
        and suffix[4] == "--"
    )


def _require_git_toplevel(root: Path) -> None:
    result = _run_git(["git", "rev-parse", "--show-toplevel"], root)
    if result.returncode != 0:
        raise FreestyleJobRefusedError(f"Project root {root} is not a git work tree.")
    if Path(result.stdout.strip()).resolve() != root:
        raise FreestyleJobRefusedError(f"Project root {root} is not the git top-level.")


def _head_commit(root: Path) -> str:
    result = _run_git(["git", "rev-parse", "HEAD"], root)
    commit = result.stdout.strip()
    if (
        result.returncode != 0
        or not _is_commit_id(commit)
    ):
        raise FreestyleJobRefusedError("Could not read a valid committed HEAD.")
    return commit


def _current_clean_head_commit(root: Path) -> str:
    """Resolve current checkout HEAD only while the main checkout stays clean."""
    _require_clean_main_checkout(root)
    commit = _head_commit(root)
    _require_clean_main_checkout(root)
    if _head_commit(root) != commit:
        raise FreestyleJobRefusedError(
            "Cannot create a freestyle job because the main checkout HEAD changed "
            "during base selection. Retry from a stable current clean repo HEAD."
        )
    return commit


def _require_clean_main_checkout(root: Path) -> None:
    result = _run_git_bytes(
        [
            "git",
            "--no-optional-locks",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        root,
    )
    if result.returncode != 0:
        raise FreestyleJobRefusedError(
            _bounded_error(result.stderr, "Could not verify that the main checkout is clean.")
        )
    if result.stdout:
        raise FreestyleJobRefusedError(
            "Cannot create a freestyle job because the main checkout has "
            "uncommitted tracked or untracked changes. Commit or remove them, "
            "then create a fresh job from the current clean repo HEAD."
        )


def _require_state_inside_project(state: Path, root: Path) -> None:
    try:
        state.relative_to(root)
    except ValueError as exc:
        raise FreestyleJobRefusedError(
            "Freestyle dev state must stay inside the project root."
        ) from exc
    if state == root:
        raise FreestyleJobRefusedError("Freestyle dev state cannot be the project root.")


def _validated_job_worktree(job: FreestyleJob, root: Path, state: Path) -> Path:
    expected = state / "worktrees" / job.job_id
    supplied = Path(job.worktree_path)
    if not supplied.is_absolute() or supplied != expected:
        raise FreestyleJobRefusedError("Job worktree path does not match local dev state.")
    if not expected.is_dir() or expected.is_symlink():
        raise FreestyleJobRefusedError("Job worktree is missing or is a symlink.")
    try:
        expected.resolve(strict=True).relative_to(state)
    except (OSError, ValueError) as exc:
        raise FreestyleJobRefusedError(
            "Job worktree must stay inside local dev state."
        ) from exc
    _require_git_toplevel(expected)
    if _head_commit(expected) != job.base_commit:
        raise FreestyleJobRefusedError("Job worktree HEAD does not match its base commit.")
    branch = _run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"], expected)
    if branch.returncode != 0 or branch.stdout.strip() != job.branch:
        raise FreestyleJobRefusedError("Job worktree branch does not match its metadata.")
    if job.branch != f"{BRANCH_PREFIX}{job.job_id[:12]}":
        raise FreestyleJobRefusedError("Job branch does not match its generated job id.")
    if not _is_commit_id(job.base_commit):
        raise FreestyleJobRefusedError("Job base commit is invalid.")
    return expected


def _require_empty_no_checkout_worktree(worktree: Path) -> None:
    entries = list(worktree.iterdir())
    if (
        len(entries) != 1
        or entries[0].name != ".git"
        or entries[0].is_symlink()
        or not entries[0].is_file()
    ):
        raise FreestyleJobRefusedError(
            "Job worktree is already dirty or contains materialized files."
        )


def _list_snapshot_paths(commit: str, root: Path) -> SnapshotPaths:
    result = _run_git_bytes(
        ["git", "ls-tree", "-r", "-z", "--full-tree", commit],
        root,
    )
    if result.returncode != 0:
        raise FreestyleJobRefusedError(
            _bounded_error(result.stderr, "Git tree listing failed.")
        )
    materialized_blobs: list[SnapshotBlob] = []
    omitted_paths: list[str] = []
    for raw_entry in result.stdout.split(b"\x00"):
        if not raw_entry:
            continue
        try:
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ")
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise FreestyleJobRefusedError(
                "Git tree listing contained a malformed path entry."
            ) from exc
        if mode == "120000":
            raise FreestyleJobRefusedError("Repository snapshot contains a symlink.")
        if mode == "160000" or object_type == "commit":
            raise FreestyleJobRefusedError("Repository snapshot contains a submodule.")
        if mode not in {"100644", "100755"} or object_type != "blob":
            raise FreestyleJobRefusedError(
                "Repository snapshot contains an unsupported file type."
            )
        if not _is_commit_id(object_id):
            raise FreestyleJobRefusedError(
                "Git tree listing contained an invalid object id."
            )
        safe_path = _validated_archive_member_path(path)
        if safe_path is None:
            omitted_paths.append(path)
        else:
            materialized_blobs.append(
                SnapshotBlob(
                    path=safe_path,
                    object_id=object_id,
                    mode=0o755 if mode == "100755" else 0o644,
                )
            )
    all_paths = [blob.path for blob in materialized_blobs] + omitted_paths
    _require_non_conflicting_paths(all_paths)
    if len(all_paths) > MAX_MATERIALIZED_FILES:
        raise FreestyleJobRefusedError("Repository snapshot contains too many files.")
    return SnapshotPaths(tuple(materialized_blobs), tuple(omitted_paths))


def _synchronize_materialized_worktree_index(
    job: FreestyleJob,
    snapshot_paths: SnapshotPaths,
    worktree: Path,
    root: Path,
    state: Path,
) -> None:
    read_tree = _run_materialization_git_bytes(
        [*_MATERIALIZATION_GIT_PREFIX, "read-tree", job.base_commit],
        worktree,
    )
    if read_tree.returncode != 0:
        raise FreestyleJobRefusedError(
            _bounded_error(read_tree.stderr, "Git index synchronization failed.")
        )
    if snapshot_paths.omitted:
        skip_omitted = _run_materialization_git_bytes(
            [
                *_MATERIALIZATION_GIT_PREFIX,
                "update-index",
                "-z",
                "--skip-worktree",
                "--stdin",
            ],
            worktree,
            omitted_paths=snapshot_paths.omitted,
        )
        if skip_omitted.returncode != 0:
            raise FreestyleJobRefusedError(
                _bounded_error(
                    skip_omitted.stderr, "Git omitted-path index synchronization failed."
                )
            )

    _validated_job_worktree(job, root, state)
    index_diff = _run_materialization_git_bytes(
        [
            *_MATERIALIZATION_GIT_PREFIX,
            "diff-index",
            "--cached",
            "--quiet",
            job.base_commit,
            "--",
        ],
        worktree,
    )
    if index_diff.returncode != 0:
        raise FreestyleJobRefusedError(
            _bounded_error(
                index_diff.stderr, "Git index consistency verification failed."
            )
        )
    _require_expected_materialized_index(snapshot_paths, worktree)


def _require_expected_materialized_index(
    snapshot_paths: SnapshotPaths, worktree: Path
) -> None:
    listed = _run_materialization_git_bytes(
        [*_MATERIALIZATION_GIT_PREFIX, "ls-files", "-v", "-z"],
        worktree,
    )
    if listed.returncode != 0:
        raise FreestyleJobRefusedError(
            _bounded_error(listed.stderr, "Git index path verification failed.")
        )
    expected = {
        **{path: "H" for path in snapshot_paths.materialized_paths},
        **{path: "S" for path in snapshot_paths.omitted},
    }
    actual: dict[str, str] = {}
    for raw_entry in listed.stdout.split(b"\x00"):
        if not raw_entry:
            continue
        try:
            tag, raw_path = raw_entry.split(b" ", 1)
            path = raw_path.decode("utf-8")
            marker = tag.decode("ascii")
        except (UnicodeDecodeError, ValueError) as exc:
            raise FreestyleJobRefusedError(
                "Git index path verification returned malformed output."
            ) from exc
        if marker not in {"H", "S"} or path in actual:
            raise FreestyleJobRefusedError(
                "Git index path verification returned unexpected entries."
            )
        actual[path] = marker
    if actual != expected:
        raise FreestyleJobRefusedError(
            "Git index paths do not match the materialized repository snapshot."
        )

    for path in snapshot_paths.materialized_paths:
        target = worktree.joinpath(*PurePosixPath(path).parts)
        if not target.is_file() or target.is_symlink():
            raise FreestyleJobRefusedError(
                "A materialized repository file is missing after index synchronization."
            )
    for path in snapshot_paths.omitted:
        target = worktree.joinpath(*PurePosixPath(path).parts)
        if target.exists() or target.is_symlink():
            raise FreestyleJobRefusedError(
                "An omitted repository path appeared during index synchronization."
            )


def _archive_snapshot(
    blobs: tuple[SnapshotBlob, ...], root: Path
) -> tuple[bytes, ...]:
    archives: list[bytes] = []
    total_archive_bytes = 0
    total_blob_bytes = 0
    for offset in range(0, len(blobs), MAX_ARCHIVE_PATHS_PER_COMMAND):
        chunk = blobs[offset : offset + MAX_ARCHIVE_PATHS_PER_COMMAND]
        checked = _run_git_blob_batch(chunk, root, check_only=True)
        if checked.returncode != 0:
            raise FreestyleJobRefusedError(
                _bounded_error(checked.stderr, "Git blob size check failed.")
            )
        sizes = _parse_git_blob_batch_check(checked.stdout, chunk)
        for size in sizes:
            total_blob_bytes += size
            if size > MAX_MATERIALIZED_FILE_BYTES:
                raise FreestyleJobRefusedError(
                    "Repository snapshot contains an oversized file."
                )
            if total_blob_bytes > MAX_MATERIALIZED_TOTAL_BYTES:
                raise FreestyleJobRefusedError(
                    "Repository snapshot exceeds the materialization size limit."
                )
        result = _run_git_blob_batch(chunk, root)
        if result.returncode != 0:
            raise FreestyleJobRefusedError(
                _bounded_error(result.stderr, "Git blob extraction failed.")
            )
        contents = _parse_git_blob_batch(result.stdout, chunk, sizes)
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w") as archive:
            for blob, content in zip(chunk, contents):
                member = tarfile.TarInfo(blob.path)
                member.mode = blob.mode
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        payload = output.getvalue()
        total_archive_bytes += len(payload)
        if total_archive_bytes > MAX_ARCHIVE_BYTES:
            raise FreestyleJobRefusedError(
                "Git archive exceeds the materialization limit."
            )
        archives.append(payload)
    return tuple(archives)


def _parse_git_blob_batch_check(
    output: bytes, blobs: tuple[SnapshotBlob, ...]
) -> tuple[int, ...]:
    raw_lines = output.splitlines()
    if len(raw_lines) != len(blobs):
        raise FreestyleJobRefusedError("Git blob size check returned malformed output.")
    sizes: list[int] = []
    for raw_line, blob in zip(raw_lines, blobs):
        header = raw_line.split(b" ")
        try:
            object_id = header[0].decode("ascii")
            object_type = header[1].decode("ascii")
            size = int(header[2])
        except (IndexError, UnicodeDecodeError, ValueError) as exc:
            raise FreestyleJobRefusedError(
                "Git blob size check returned malformed output."
            ) from exc
        if (
            len(header) != 3
            or object_id != blob.object_id
            or object_type != "blob"
            or size < 0
        ):
            raise FreestyleJobRefusedError(
                "Git blob size check returned malformed output."
            )
        sizes.append(size)
    return tuple(sizes)


def _parse_git_blob_batch(
    output: bytes, blobs: tuple[SnapshotBlob, ...], expected_sizes: tuple[int, ...]
) -> tuple[bytes, ...]:
    contents: list[bytes] = []
    offset = 0
    if len(blobs) != len(expected_sizes):
        raise FreestyleJobRefusedError("Git blob extraction returned malformed output.")
    for blob, expected_size in zip(blobs, expected_sizes):
        header_end = output.find(b"\n", offset)
        if header_end < 0:
            raise FreestyleJobRefusedError("Git blob extraction returned malformed output.")
        header = output[offset:header_end].split(b" ")
        try:
            object_id = header[0].decode("ascii")
            object_type = header[1].decode("ascii")
            size = int(header[2])
        except (IndexError, UnicodeDecodeError, ValueError) as exc:
            raise FreestyleJobRefusedError(
                "Git blob extraction returned malformed output."
            ) from exc
        content_start = header_end + 1
        content_end = content_start + size
        if (
            len(header) != 3
            or object_id != blob.object_id
            or object_type != "blob"
            or size < 0
            or size != expected_size
            or content_end >= len(output)
            or output[content_end : content_end + 1] != b"\n"
        ):
            raise FreestyleJobRefusedError("Git blob extraction returned malformed output.")
        contents.append(output[content_start:content_end])
        offset = content_end + 1
    if offset != len(output):
        raise FreestyleJobRefusedError("Git blob extraction returned malformed output.")
    return tuple(contents)


def _preflight_archive(archive: bytes) -> tuple[str, ...]:
    safe_files: list[str] = []
    seen: set[str] = set()
    total_bytes = 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as snapshot:
        for member in snapshot:
            path = _validated_archive_member_path(member.name)
            if member.issym() or member.islnk():
                raise FreestyleJobRefusedError("Repository archive contains a symlink.")
            if not member.isfile() and not member.isdir():
                raise FreestyleJobRefusedError(
                    "Repository archive contains an unsupported file type."
                )
            if path is None or member.isdir():
                continue
            if member.mode & ~0o777:
                raise FreestyleJobRefusedError(
                    "Repository archive contains an unsupported file mode."
                )
            folded = path.casefold()
            if folded in seen:
                raise FreestyleJobRefusedError(
                    "Repository archive contains duplicate file paths."
                )
            seen.add(folded)
            if member.size > MAX_MATERIALIZED_FILE_BYTES:
                raise FreestyleJobRefusedError(
                    "Repository archive contains an oversized file."
                )
            total_bytes += member.size
            if total_bytes > MAX_MATERIALIZED_TOTAL_BYTES:
                raise FreestyleJobRefusedError(
                    "Repository archive exceeds the materialization size limit."
                )
            safe_files.append(path)
            if len(safe_files) > MAX_MATERIALIZED_FILES:
                raise FreestyleJobRefusedError(
                    "Repository archive contains too many files."
                )
    _require_non_conflicting_paths(safe_files)
    return tuple(safe_files)


def _require_non_conflicting_paths(paths: list[str]) -> None:
    folded_paths = sorted(path.casefold() for path in paths)
    if len(folded_paths) != len(set(folded_paths)):
        raise FreestyleJobRefusedError(
            "Repository snapshot contains duplicate file paths."
        )
    for previous, current in zip(folded_paths, folded_paths[1:]):
        if current.startswith(previous + "/"):
            raise FreestyleJobRefusedError(
                "Repository snapshot contains a file and directory path conflict."
            )


def _validated_archive_member_path(raw_path: str) -> str | None:
    raw_parts = raw_path.rstrip("/").split("/")
    if (
        not raw_path
        or "\x00" in raw_path
        or "\\" in raw_path
        or any(ord(character) < 32 for character in raw_path)
        or any(part in {"", ".", ".."} for part in raw_parts)
        or any(
            any(character in '<>:"|?*' for character in part)
            or part.endswith((" ", "."))
            or part.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_NAMES
            for part in raw_parts
        )
    ):
        raise FreestyleJobRefusedError(
            "Repository archive contains an invalid path or path traversal."
        )
    path = PurePosixPath(raw_path)
    windows_path = PureWindowsPath(raw_path)
    if path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise FreestyleJobRefusedError("Repository archive contains an absolute path.")
    lowered = tuple(part.lower() for part in path.parts)
    if ".git" in lowered:
        raise FreestyleJobRefusedError("Repository archive attempted .git tampering.")
    name = lowered[-1]
    if (
        any(part.startswith(".") or part in BLOCKED_PATH_PARTS for part in lowered)
        or name in BLOCKED_FILE_NAMES
        or path.suffix.lower() in BLOCKED_SUFFIXES
        or any(SECRET_NAME_PATTERN.search(part) for part in path.parts)
    ):
        return None
    return path.as_posix()


def _extract_preflighted_archive(
    archive: bytes, members: tuple[str, ...], worktree: Path
) -> int:
    allowed = set(members)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as snapshot:
        for member in snapshot:
            if member.name not in allowed:
                continue
            target = worktree.joinpath(*PurePosixPath(member.name).parts)
            _require_safe_extraction_target(target, worktree)
            target.parent.mkdir(parents=True, exist_ok=True)
            _require_safe_extraction_target(target, worktree)
            source = snapshot.extractfile(member)
            if source is None:
                raise FreestyleJobRefusedError("Could not read a repository archive file.")
            with source, target.open("xb") as destination:
                while chunk := source.read(64 * 1024):
                    destination.write(chunk)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)
    return len(members)


def _require_safe_extraction_target(target: Path, worktree: Path) -> None:
    try:
        target.relative_to(worktree)
    except ValueError as exc:
        raise FreestyleJobRefusedError(
            "Repository archive attempted path traversal."
        ) from exc
    current = target.parent
    while current != worktree:
        if current.exists() and (current.is_symlink() or not current.is_dir()):
            raise FreestyleJobRefusedError(
                "Repository archive attempted a symlink or file escape."
            )
        if current.exists():
            try:
                current.resolve(strict=True).relative_to(worktree.resolve(strict=True))
            except (OSError, ValueError) as exc:
                raise FreestyleJobRefusedError(
                    "Repository archive attempted a symlink or file escape."
                ) from exc
        current = current.parent
    if target.exists() or target.is_symlink():
        raise FreestyleJobRefusedError(
            "Repository archive attempted to overwrite an existing path."
        )


def _write_job(job: FreestyleJob, state: Path) -> None:
    jobs_dir = _validated_jobs_dir(state, create=True)
    path = jobs_dir / f"{job.job_id}.json"
    temporary = path.with_suffix(".json.tmp")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise FreestyleJobRefusedError("Freestyle job metadata path is unsafe.")
    if temporary.exists() or temporary.is_symlink():
        raise FreestyleJobRefusedError("Freestyle job temporary metadata path is unsafe.")
    temporary.write_text(
        json.dumps(asdict(job), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_job(path: Path) -> FreestyleJob:
    if path.is_symlink() or not path.is_file():
        raise FreestyleJobRefusedError(f"Could not read freestyle job {path.stem!r}.")
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Freestyle job metadata must be an object.")
        payload.setdefault("developer_approved", None)
        payload["errors"] = tuple(payload.get("errors", ()))
        payload.setdefault("materialization_status", "not_materialized")
        payload.setdefault("materialization_started_at", None)
        payload.setdefault("materialized_at", None)
        payload.setdefault("materialized_file_count", 0)
        payload.setdefault("proposal_status", "not_proposed")
        payload.setdefault("proposal_started_at", None)
        payload.setdefault("proposal_completed_at", None)
        payload.setdefault("proposal_artifact_path", None)
        payload.setdefault("proposal_failure_artifact_path", None)
        payload.setdefault("proposal_inspected_file_count", 0)
        payload.setdefault("proposal_model", None)
        payload["proposal_errors"] = tuple(payload.get("proposal_errors", ()))
        payload.setdefault("edit_generation_status", "not_generated")
        payload.setdefault("edit_generation_started_at", None)
        payload.setdefault("edit_generation_completed_at", None)
        payload.setdefault("edit_generation_artifact_path", None)
        payload.setdefault("edit_bundle_path", None)
        payload.setdefault("edit_bundle_validation_status", "not_validated")
        payload.setdefault("edit_bundle_candidate_file_count", 0)
        payload["edit_bundle_goal_fit_warnings"] = tuple(
            payload.get("edit_bundle_goal_fit_warnings", ())
        )
        payload.setdefault("edit_generation_model", None)
        payload["edit_generation_errors"] = tuple(
            payload.get("edit_generation_errors", ())
        )
        payload.setdefault("edit_apply_status", "not_applied")
        payload.setdefault("edit_apply_started_at", None)
        payload.setdefault("edit_applied_at", None)
        payload.setdefault("edit_applied_file_count", 0)
        payload.setdefault("edit_artifact_path", None)
        payload["edit_errors"] = tuple(payload.get("edit_errors", ()))
        payload.setdefault("review_status", "not_reviewed")
        payload.setdefault("review_started_at", None)
        payload.setdefault("review_completed_at", None)
        payload.setdefault("review_artifact_path", None)
        payload.setdefault("review_changed_file_count", 0)
        payload["review_errors"] = tuple(payload.get("review_errors", ()))
        payload.setdefault("checks_status", "not_run")
        payload.setdefault("checks_started_at", None)
        payload.setdefault("checks_completed_at", None)
        payload.setdefault("checks_artifact_path", None)
        payload["checks_selected_ids"] = tuple(payload.get("checks_selected_ids", ()))
        payload.setdefault("checks_passed_count", 0)
        payload.setdefault("checks_failed_count", 0)
        payload.setdefault("checks_timeout_count", 0)
        payload.setdefault("checks_infra_error_count", 0)
        payload.setdefault("checks_modified_worktree", False)
        payload["checks_errors"] = tuple(payload.get("checks_errors", ()))
        payload.setdefault("latest_run_id", None)
        payload.setdefault("latest_run_status", "not_run")
        payload.setdefault("latest_run_started_at", None)
        payload.setdefault("latest_run_completed_at", None)
        payload.setdefault("latest_run_artifact_path", None)
        payload["latest_run_completed_stages"] = tuple(
            payload.get("latest_run_completed_stages", ())
        )
        payload["latest_run_errors"] = tuple(payload.get("latest_run_errors", ()))
        return FreestyleJob(**payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FreestyleJobRefusedError(f"Could not read freestyle job {path.stem!r}.") from exc


def _validated_jobs_dir(state: Path, *, create: bool) -> Path:
    jobs_dir = state / "jobs"
    if create:
        jobs_dir.mkdir(parents=True, exist_ok=True)
    if not jobs_dir.exists():
        return jobs_dir
    if jobs_dir.is_symlink() or not jobs_dir.is_dir():
        raise FreestyleJobRefusedError("Freestyle job metadata directory is unsafe.")
    try:
        jobs_dir.resolve(strict=True).relative_to(state)
    except (OSError, ValueError) as exc:
        raise FreestyleJobRefusedError(
            "Freestyle job metadata directory must stay inside local dev state."
        ) from exc
    return jobs_dir


def _validate_job_id(job_id: str) -> None:
    if not is_job_id(job_id):
        raise FreestyleJobRefusedError("Job id must be a 32-character lowercase hex id.")


def _is_commit_id(value: str) -> bool:
    return (
        len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _bounded_error(text: str, fallback: str = "Git worktree creation failed.") -> str:
    clean = " ".join(text.strip().split())
    return (clean or fallback)[:500]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_explicit_developer_approval(
    job: FreestyleJob, operation: str
) -> None:
    if job.developer_approved is not True:
        raise FreestyleJobRefusedError(
            f"Freestyle job {job.job_id!r} cannot {operation}: explicit persisted "
            f"developer approval is required; approval state is {job.approval_state}."
        )


def _print_job(job: FreestyleJob) -> None:
    payload = asdict(job)
    payload["approval_state"] = job.approval_state
    print(json.dumps(payload, indent=2, sort_keys=True))


def _redact_sensitive_env_values(text: str) -> str:
    sensitive_values = {
        value
        for name, value in os.environ.items()
        if value and value in text and _SENSITIVE_ENV_NAME.search(name)
    }
    if not sensitive_values:
        return text
    pattern = "|".join(
        re.escape(value) for value in sorted(sensitive_values, key=len, reverse=True)
    )
    return re.sub(pattern, "[REDACTED]", text)


def _bounded_cli_text(text: str, *, limit: int) -> str:
    clean = " ".join(text.strip().split())
    return _redact_sensitive_env_values(clean)[:limit]


def _bounded_cli_error(exc: Exception) -> str:
    text = str(exc)
    cause = exc.__cause__
    if isinstance(cause, FreestyleJobRefusedError) and str(cause) != text:
        text = f"{text} Reason: {cause}"
    message = _bounded_cli_text(text, limit=_MAX_CLI_ERROR_LENGTH)
    return message or "Freestyle command was refused."


def _display_job_id(job_id: str | None) -> str | None:
    if job_id is None:
        return None
    display = _bounded_cli_text(job_id, limit=128)
    return display or None


def _cli_context(arguments: Sequence[str]) -> tuple[str | None, str | None, bool]:
    command = arguments[0] if arguments and arguments[0] in _CLI_COMMANDS else None
    job_id = None
    if command not in {None, "cleanup-status", "create", "list"} and len(arguments) > 1:
        candidate = arguments[1]
        if not candidate.startswith("-"):
            job_id = candidate
    human_output = command == "status" and "--json" not in arguments
    return command, job_id, human_output


def _failed_artifact_path(command: str | None, job_id: str | None) -> str | None:
    if job_id is None:
        return None
    artifact_fields = {
        "propose": ("proposal_status", {"error"}, "proposal_failure_artifact_path"),
        "generate-edits": (
            "edit_generation_status",
            {"error"},
            "edit_generation_artifact_path",
        ),
        "apply-edits": (
            "edit_apply_status",
            {"error", "inconsistent"},
            "edit_artifact_path",
        ),
        "review": ("review_status", {"error", "inconsistent"}, "review_artifact_path"),
        "run-checks": ("checks_status", {"error"}, "checks_artifact_path"),
        "run": (
            "latest_run_status",
            {"error", "failed", "refused"},
            "latest_run_artifact_path",
        ),
    }
    artifact_spec = artifact_fields.get(command)
    if artifact_spec is None:
        return None
    try:
        job = get_freestyle_job(job_id)
    except FreestyleJobRefusedError:
        return None
    status_field, failed_statuses, artifact_field = artifact_spec
    if getattr(job, status_field) not in failed_statuses:
        return None
    artifact_path = getattr(job, artifact_field)
    if not isinstance(artifact_path, str):
        return None
    display = _bounded_cli_text(artifact_path, limit=_MAX_CLI_ERROR_LENGTH)
    path = PurePosixPath(display)
    if (
        not display
        or "\\" in display
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        return None
    return display


def _print_cli_error(
    exc: Exception,
    *,
    error_type: str,
    command: str | None,
    job_id: str | None,
    human_output: bool,
) -> None:
    message = _bounded_cli_error(exc)
    display_job_id = _display_job_id(job_id)
    artifact_path = _failed_artifact_path(command, job_id)
    if human_output:
        details = [f"ERROR [{error_type}]: {message}"]
        if command:
            details.append(f"command: {command}")
        if display_job_id:
            details.append(f"job_id: {display_job_id}")
        if artifact_path:
            details.append(f"artifact_path: {artifact_path}")
        print("\n".join(details), file=sys.stderr)
        return
    payload: dict[str, Any] = {
        "ok": False,
        "error_type": error_type,
        "message": message,
    }
    if command:
        payload["command"] = command
        payload["stage"] = command
    if display_job_id:
        payload["job_id"] = display_job_id
    if artifact_path:
        payload["artifact_path"] = artifact_path
    print(json.dumps(payload, sort_keys=True), file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    command, job_id, human_output = _cli_context(arguments)
    parser = FreestyleArgumentParser(
        description="Manage developer-only freestyle jobs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create an isolated job.")
    create_parser.add_argument("--goal", required=True)
    create_parser.add_argument("--title")
    create_parser.add_argument(
        "--approve",
        action="store_true",
        help="Confirm this developer explicitly approves creating the local job.",
    )

    subparsers.add_parser("list", help="List local freestyle jobs.")
    subparsers.add_parser(
        "cleanup-status",
        help="Read-only inventory of jobs, worktrees, branches, and artifacts.",
    )
    show_parser = subparsers.add_parser("show", help="Show one local freestyle job.")
    show_parser.add_argument("job_id")
    status_parser = subparsers.add_parser(
        "status", help="Show a concise read-only human handoff for one local job."
    )
    status_parser.add_argument("job_id")
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the handoff as JSON instead of Markdown.",
    )
    reapprove_parser = subparsers.add_parser(
        "reapprove", help="Explicitly reapprove one existing local freestyle job."
    )
    reapprove_parser.add_argument("job_id")
    reapprove_parser.add_argument(
        "--approve",
        action="store_true",
        help="Confirm this developer explicitly reapproves the local job.",
    )
    materialize_parser = subparsers.add_parser(
        "materialize", help="Safely populate one isolated job worktree."
    )
    materialize_parser.add_argument("job_id")
    propose_parser = subparsers.add_parser(
        "propose", help="Generate a DeepSeek-only proposal for one materialized job."
    )
    propose_parser.add_argument("job_id")
    generate_parser = subparsers.add_parser(
        "generate-edits",
        help="Generate and validate a DeepSeek-only edit bundle without applying it.",
    )
    generate_parser.add_argument("job_id")
    apply_parser = subparsers.add_parser(
        "apply-edits",
        help="Deterministically apply one structured edit bundle to one "
        "materialized job worktree.",
    )
    apply_parser.add_argument("job_id")
    apply_parser.add_argument("bundle_path")
    review_parser = subparsers.add_parser(
        "review",
        help="Persist a deterministic read-only diff review for one isolated job.",
    )
    review_parser.add_argument("job_id")
    checks_parser = subparsers.add_parser(
        "run-checks",
        help="Run only static allowlisted checks inside one isolated job worktree.",
    )
    checks_parser.add_argument("job_id")
    checks_parser.add_argument(
        "--check",
        action="append",
        dest="check_ids",
        metavar="CHECK_ID",
        help="Allowlisted check id to run (repeatable); defaults to every "
        "available allowlisted check.",
    )
    run_parser = subparsers.add_parser(
        "run",
        help="Advance existing safe stages and stop at explicit human boundaries.",
    )
    run_parser.add_argument("job_id")
    run_parser.add_argument(
        "--apply-latest-generated",
        action="store_true",
        help="Request deterministic application of the latest valid generated bundle.",
    )
    run_parser.add_argument(
        "--confirm-apply",
        action="store_true",
        help="Explicitly confirm applying the latest generated bundle.",
    )
    run_parser.add_argument(
        "--run-checks",
        action="store_true",
        help="Run allowlisted checks after edits have been applied and reviewed.",
    )
    run_parser.add_argument(
        "--check",
        action="append",
        dest="check_ids",
        metavar="CHECK_ID",
        help="Allowlisted check id to run (repeatable); requires --run-checks.",
    )
    cleanup_parser = subparsers.add_parser(
        "cleanup", help="Explicitly clean one safe local freestyle job."
    )
    cleanup_parser.add_argument("job_id")
    cleanup_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm removal of this one local freestyle job.",
    )
    cleanup_parser.add_argument(
        "--discard-worktree-changes",
        action="store_true",
        help="Also confirm deletion of unreviewed isolated worktree changes.",
    )
    recover_parser = subparsers.add_parser(
        "recover", help="Explicitly clear stale lock metadata to failed states."
    )
    recover_parser.add_argument("job_id")
    recover_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm recovery of stale lock metadata.",
    )
    recover_parser.add_argument(
        "--confirm-inconsistent",
        action="store_true",
        help="Claim manual verification before clearing inconsistent edit metadata.",
    )

    try:
        args = parser.parse_args(arguments)
        if args.command == "create":
            _print_job(
                create_freestyle_job(
                    args.goal,
                    title=args.title,
                    developer_approved=args.approve,
                )
            )
        elif args.command == "list":
            for job in list_freestyle_jobs():
                _print_job(job)
        elif args.command == "cleanup-status":
            print(json.dumps(get_freestyle_cleanup_status(), indent=2, sort_keys=True))
        elif args.command == "show":
            _print_job(get_freestyle_job(args.job_id))
        elif args.command == "status":
            handoff = get_freestyle_handoff(args.job_id)
            if args.json:
                print(json.dumps(handoff, indent=2, sort_keys=True))
            else:
                from app.dev.freestyle_handoff import render_freestyle_handoff_markdown

                print(render_freestyle_handoff_markdown(handoff), end="")
        elif args.command == "reapprove":
            _print_job(
                reapprove_freestyle_job(
                    args.job_id,
                    developer_approved=args.approve,
                )
            )
        elif args.command == "materialize":
            _print_job(materialize_freestyle_job(args.job_id))
        elif args.command == "propose":
            _print_job(propose_freestyle_job(args.job_id))
        elif args.command == "generate-edits":
            _print_job(generate_freestyle_edit_bundle(args.job_id))
        elif args.command == "apply-edits":
            _print_job(apply_freestyle_edit_bundle(args.job_id, args.bundle_path))
        elif args.command == "review":
            _print_job(review_freestyle_job(args.job_id))
        elif args.command == "run-checks":
            _print_job(run_freestyle_checks(args.job_id, check_ids=args.check_ids))
        elif args.command == "cleanup":
            print(
                json.dumps(
                    cleanup_freestyle_job(
                        args.job_id,
                        confirm=args.confirm,
                        discard_worktree_changes=args.discard_worktree_changes,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "recover":
            _print_job(
                recover_freestyle_job(
                    args.job_id,
                    confirm=args.confirm,
                    confirm_inconsistent=args.confirm_inconsistent,
                )
            )
        else:
            _print_job(
                run_freestyle_job(
                    args.job_id,
                    apply_latest_generated=args.apply_latest_generated,
                    confirm_apply=args.confirm_apply,
                    run_checks=args.run_checks,
                    check_ids=args.check_ids,
                )
            )
    except FreestyleCliInputError as exc:
        _print_cli_error(
            exc,
            error_type="invalid_cli_input",
            command=command,
            job_id=job_id,
            human_output=human_output,
        )
        return 2
    except FreestyleJobRefusedError as exc:
        _print_cli_error(
            exc,
            error_type="freestyle_job_refused",
            command=command,
            job_id=job_id,
            human_output=human_output,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
