"""Developer-only inspection, cleanup, and stale-lock recovery for freestyle jobs."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any
import uuid

from app.config import ROOT_DIR
from app.dev.freestyle_coding import (
    BRANCH_PREFIX,
    DEV_STATE_DIR_NAME,
    FreestyleJob,
    FreestyleJobRefusedError,
    GitCommandResult,
    _bounded_error,
    _is_commit_id,
    _require_git_toplevel,
    _utc_now,
    _validate_job_id,
    _write_job,
    get_freestyle_job,
    list_freestyle_jobs,
)
from app.dev.freestyle_policy import (
    NO_INTEGRATION_STATEMENT_MAINTENANCE,
    is_job_id,
)
from app.dev.freestyle_state import (
    ACTIVE_STATES,
    INCONSISTENT_STATES,
    resolved_roots,
)

MAINTENANCE_SCHEMA_VERSION = 1
STALE_AFTER = timedelta(hours=1)
MAX_MAINTENANCE_ARTIFACT_BYTES = 250_000
_NO_INTEGRATION_STATEMENT = NO_INTEGRATION_STATEMENT_MAINTENANCE
_JOB_ARTIFACT_DIRS = (
    "proposals",
    "edit-bundles",
    "edits",
    "reviews",
    "checks",
    "runs",
)
_MAINTENANCE_GIT_PREFIX = (
    "git",
    "--no-optional-locks",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.fsmonitor=false",
)


def get_freestyle_cleanup_status(
    *,
    project_root: Path | None = None,
    state_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Return a read-only inventory of jobs and their local maintenance state."""
    root, state = _maintenance_roots(project_root, state_root)
    _require_git_toplevel(root)
    jobs = {job.job_id: job for job in list_freestyle_jobs(
        project_root=root, state_root=state
    )}
    worktrees = _registered_worktrees(root)
    branches = _freestyle_branches(root)
    job_ids = set(jobs)
    job_ids.update(_artifact_job_ids(state))
    job_ids.update(_job_id_from_branch(branch) for branch in branches)
    job_ids.update(_job_id_from_worktree_path(path, state) for path in worktrees)
    job_ids.discard(None)
    statuses = [
        _job_cleanup_status(
            job_id,
            jobs.get(job_id),
            root=root,
            state=state,
            registered_worktrees=worktrees,
            branches=branches,
        )
        for job_id in sorted(job_ids)
    ]
    represented_branches = {item["branch"] for item in statuses}
    statuses.extend(
        _orphan_branch_status(branch, commit)
        for branch, commit in sorted(branches.items())
        if branch not in represented_branches
    )
    return statuses


def cleanup_freestyle_job(
    job_id: str,
    *,
    confirm: bool = False,
    discard_worktree_changes: bool = False,
    project_root: Path | None = None,
    state_root: Path | None = None,
) -> dict[str, Any]:
    """Remove one validated job's local artifacts, worktree, branch, and metadata."""
    if confirm is not True:
        raise FreestyleJobRefusedError(
            "Explicit --confirm is required to clean a freestyle job."
        )
    if not isinstance(discard_worktree_changes, bool):
        raise FreestyleJobRefusedError("Cleanup flags must be booleans.")
    _validate_job_id(job_id)
    root, state = _maintenance_roots(project_root, state_root)
    _require_git_toplevel(root)
    job = get_freestyle_job(job_id, project_root=root, state_root=state)
    status = _single_cleanup_status(job_id, root, state)
    if status["active"] or status["inconsistent"]:
        reason = (
            "Freestyle cleanup refuses active or inconsistent jobs. "
            "Use recover with explicit confirmation first."
        )
        _write_refusal_artifact("cleanup", job, state, status, reason)
        raise FreestyleJobRefusedError(reason)
    if status["mismatched"]:
        reason = "Freestyle cleanup refuses mismatched worktree or branch metadata."
        _write_refusal_artifact("cleanup", job, state, status, reason)
        raise FreestyleJobRefusedError(reason)
    if status["dirty"] and not discard_worktree_changes:
        reason = (
            "Isolated worktree has unreviewed source changes or local commits; "
            "--discard-worktree-changes is required to delete it."
        )
        _write_refusal_artifact("cleanup", job, state, status, reason)
        raise FreestyleJobRefusedError(reason)
    for path in _job_artifact_paths(state, job_id):
        if path.exists() or path.is_symlink():
            _validate_allowed_state_path(path, state)

    actions: list[str] = []
    removed: list[str] = []
    kept: list[str] = []
    git_operations: list[dict[str, Any]] = []
    worktree = state / "worktrees" / job_id
    if status["worktree_registered"]:
        args = [
            *_MAINTENANCE_GIT_PREFIX,
            "worktree",
            "remove",
            *(["--force"] if status["dirty"] else []),
            str(worktree),
        ]
        result = _run_maintenance_git(args, root)
        git_operations.append(_git_operation_record(result))
        if result.returncode != 0:
            reason = _bounded_error(result.stderr or result.stdout,
                                    "Could not remove isolated worktree.")
            _write_refusal_artifact(
                "cleanup", job, state, status, reason,
                git_operations=git_operations,
            )
            raise FreestyleJobRefusedError(reason)
        actions.append("removed_validated_isolated_worktree")
        removed.append(_relative_state_path(worktree, state))
    elif worktree.exists() or worktree.is_symlink():
        reason = "Expected worktree path exists but is not a registered matching worktree."
        _write_refusal_artifact("cleanup", job, state, status, reason)
        raise FreestyleJobRefusedError(reason)

    if status["branch_exists"]:
        args = [
            *_MAINTENANCE_GIT_PREFIX,
            "branch",
            "-D",
            job.branch,
        ]
        result = _run_maintenance_git(args, root)
        git_operations.append(_git_operation_record(result))
        if result.returncode != 0:
            reason = _bounded_error(result.stderr or result.stdout,
                                    "Could not remove local freestyle branch.")
            _write_refusal_artifact(
                "cleanup", job, state, status, reason,
                git_operations=git_operations,
            )
            raise FreestyleJobRefusedError(reason)
        actions.append("removed_matching_local_freestyle_branch")
        removed.append(f"refs/heads/{job.branch}")

    for path in _job_artifact_paths(state, job_id):
        if path.exists() or path.is_symlink():
            _remove_allowed_state_path(path, state)
            removed.append(_relative_state_path(path, state))
    metadata = state / "jobs" / f"{job_id}.json"
    if metadata.is_file() and not metadata.is_symlink():
        metadata.unlink()
        removed.append(_relative_state_path(metadata, state))
        actions.append("removed_job_metadata")
    maintenance = state / "maintenance" / job_id
    kept.append(_relative_state_path(maintenance, state))
    artifact = _write_maintenance_artifact(
        "cleanup",
        job,
        state,
        pre_state=status,
        actions=actions,
        paths_removed=removed,
        paths_kept=kept,
        git_operations=git_operations,
        refusal_reasons=[],
        extra={"discard_worktree_changes_confirmed": discard_worktree_changes},
    )
    return {
        "job_id": job_id,
        "operation": "cleanup",
        "artifact_path": artifact,
        "actions_taken": actions,
        "paths_removed": removed,
        "paths_kept": kept,
        "integration_statement": _NO_INTEGRATION_STATEMENT,
    }


def recover_freestyle_job(
    job_id: str,
    *,
    confirm: bool = False,
    confirm_inconsistent: bool = False,
    project_root: Path | None = None,
    state_root: Path | None = None,
) -> FreestyleJob:
    """Clear only explicitly acknowledged stale metadata locks to failed states."""
    if confirm is not True:
        raise FreestyleJobRefusedError(
            "Explicit --confirm is required to recover a freestyle job."
        )
    if not isinstance(confirm_inconsistent, bool):
        raise FreestyleJobRefusedError("Recovery flags must be booleans.")
    _validate_job_id(job_id)
    root, state = _maintenance_roots(project_root, state_root)
    _require_git_toplevel(root)
    job = get_freestyle_job(job_id, project_root=root, state_root=state)
    status = _single_cleanup_status(job_id, root, state)
    if status["mismatched"] or not status["worktree_registered"]:
        reason = "Recovery requires a registered worktree matching the job metadata."
        _write_refusal_artifact("recover", job, state, status, reason)
        raise FreestyleJobRefusedError(reason)
    if status["inconsistent"] and not confirm_inconsistent:
        reason = (
            "Recovery will not clear inconsistent state without "
            "--confirm-inconsistent."
        )
        _write_refusal_artifact("recover", job, state, status, reason)
        raise FreestyleJobRefusedError(reason)

    note = _bounded_error(
        "Developer-confirmed maintenance recovery cleared stale lock metadata; "
        "the interrupted operation was not marked successful and the isolated "
        "worktree still requires manual verification."
    )
    changes: dict[str, Any] = {"updated_at": _utc_now()}
    actions: list[str] = []
    if job.latest_run_status == "running":
        changes.update(
            latest_run_status="failed",
            latest_run_completed_at=_utc_now(),
            latest_run_errors=_append_bounded(job.latest_run_errors, note),
        )
        actions.append("marked_stale_run_failed")
    if job.checks_status == "running":
        changes.update(
            checks_status="error",
            checks_completed_at=_utc_now(),
            checks_errors=_append_bounded(job.checks_errors, note),
        )
        actions.append("marked_stale_checks_failed")
    if job.edit_apply_status == "applying":
        changes.update(
            edit_apply_status="error",
            edit_errors=_append_bounded(job.edit_errors, note),
        )
        actions.append("marked_stale_apply_failed")
    if job.edit_apply_status == "inconsistent" and confirm_inconsistent:
        inconsistent_note = _bounded_error(
            f"{note} Developer explicitly claimed manual verification of the "
            "previous inconsistent state."
        )
        changes.update(
            edit_apply_status="error",
            edit_errors=_append_bounded(job.edit_errors, inconsistent_note),
        )
        actions.append("marked_manually_verified_inconsistent_apply_failed")
    if not actions:
        reason = "Freestyle job has no recoverable stale lock state."
        _write_refusal_artifact("recover", job, state, status, reason)
        raise FreestyleJobRefusedError(reason)

    artifact = _write_maintenance_artifact(
        "recover",
        job,
        state,
        pre_state=status,
        actions=actions,
        paths_removed=[],
        paths_kept=[_relative_state_path(state / "worktrees" / job_id, state)],
        git_operations=[],
        refusal_reasons=[],
        extra={
            "confirm_inconsistent": confirm_inconsistent,
            "manual_verification_claimed": bool(
                job.edit_apply_status == "inconsistent" and confirm_inconsistent
            ),
            "recovery_note": note,
        },
    )
    changes["errors"] = _append_bounded(
        job.errors, f"{note} Recovery artifact: {artifact}"
    )
    recovered = replace(job, **changes)
    _write_job(recovered, state)
    return recovered


def _single_cleanup_status(job_id: str, root: Path, state: Path) -> dict[str, Any]:
    statuses = get_freestyle_cleanup_status(project_root=root, state_root=state)
    for status in statuses:
        if status["job_id"] == job_id:
            return status
    raise FreestyleJobRefusedError(f"Freestyle job {job_id!r} was not found.")


def _job_cleanup_status(
    job_id: str,
    job: FreestyleJob | None,
    *,
    root: Path,
    state: Path,
    registered_worktrees: dict[Path, str | None],
    branches: dict[str, str],
) -> dict[str, Any]:
    expected_branch = f"{BRANCH_PREFIX}{job_id[:12]}"
    expected_worktree = state / "worktrees" / job_id
    recorded_branch = job.branch if job else expected_branch
    recorded_worktree = Path(job.worktree_path) if job else expected_worktree
    worktree_exists = expected_worktree.exists() or expected_worktree.is_symlink()
    registered_branch = registered_worktrees.get(expected_worktree)
    worktree_registered = expected_worktree in registered_worktrees
    branch_exists = expected_branch in branches
    mismatched = bool(
        job
        and (
            recorded_branch != expected_branch
            or recorded_worktree != expected_worktree
            or (worktree_registered and registered_branch != expected_branch)
        )
    )
    active_fields = _matching_states(job, ACTIVE_STATES)
    inconsistent_fields = _matching_states(job, INCONSISTENT_STATES)
    dirty_paths: list[str] = []
    status_error: str | None = None
    branch_diverged = bool(
        job and branch_exists and branches[expected_branch] != job.base_commit
    )
    if worktree_registered and not mismatched and job:
        if job.materialization_status == "not_materialized":
            entries = [
                path.name for path in expected_worktree.iterdir()
                if path.name != ".git"
            ] if expected_worktree.is_dir() and not expected_worktree.is_symlink() else []
            dirty_paths = sorted(entries)
        else:
            result = _run_maintenance_git(
                [*_MAINTENANCE_GIT_PREFIX, "status", "--porcelain=v1", "-z",
                 "--untracked-files=all"],
                expected_worktree,
            )
            if result.returncode == 0:
                dirty_paths = _parse_status_paths(result.stdout)
            else:
                status_error = _bounded_error(
                    result.stderr or result.stdout, "Could not inspect worktree status."
                )
    dirty = bool(dirty_paths or branch_diverged or status_error)
    orphaned = job is None or bool(
        job and (
            not worktree_exists
            or not worktree_registered
            or not branch_exists
        )
    )
    stale = bool(active_fields and job and _is_stale(job.updated_at))
    active = bool(active_fields)
    inconsistent = bool(inconsistent_fields)
    cleanable = bool(job and not active and not inconsistent and not mismatched)
    artifact_dirs = [
        _relative_state_path(path, state)
        for path in _job_artifact_paths(state, job_id)
        if path.exists() or path.is_symlink()
    ]
    return {
        "job_id": job_id,
        "title": job.title if job else None,
        "status": job.status if job else "orphaned",
        "branch": recorded_branch,
        "branch_exists": branch_exists,
        "branch_diverged": branch_diverged,
        "worktree_path": str(recorded_worktree),
        "worktree_path_exists": worktree_exists,
        "worktree_registered": worktree_registered,
        "latest_states": {
            "run": job.latest_run_status if job else None,
            "checks": job.checks_status if job else None,
            "apply": job.edit_apply_status if job else None,
        },
        "artifact_directories_present": artifact_dirs,
        "active": active,
        "active_fields": active_fields,
        "stale": stale,
        "orphaned": orphaned,
        "dirty": dirty,
        "dirty_paths": dirty_paths[:100],
        "inconsistent": inconsistent,
        "mismatched": mismatched,
        "cleanable": cleanable,
        "inspection_error": status_error,
        "recommended_manual_action": _recommended_action(
            job=job,
            active=active,
            stale=stale,
            inconsistent=inconsistent,
            mismatched=mismatched,
            dirty=dirty,
            orphaned=orphaned,
        ),
    }


def _recommended_action(
    *,
    job: FreestyleJob | None,
    active: bool,
    stale: bool,
    inconsistent: bool,
    mismatched: bool,
    dirty: bool,
    orphaned: bool,
) -> str:
    if job is None:
        return "Inspect orphaned local artifacts manually; cleanup requires job metadata."
    if mismatched:
        return "Inspect mismatched worktree and branch metadata manually."
    if inconsistent:
        return "Manually verify the isolated worktree, then use recover with stronger confirmation."
    if active:
        return (
            "Use recover --confirm after verifying the operation is abandoned."
            if stale else
            "Verify the operation is abandoned before using recover --confirm."
        )
    if dirty:
        return "Review isolated changes, or clean with explicit discard confirmation."
    if orphaned:
        return "Inspect missing local worktree or branch state before cleanup."
    return "Job is cleanable with cleanup --confirm."


def _orphan_branch_status(branch: str, commit: str) -> dict[str, Any]:
    return {
        "job_id": None,
        "title": None,
        "status": "orphaned_branch",
        "branch": branch,
        "branch_exists": True,
        "branch_diverged": None,
        "branch_commit": commit,
        "worktree_path": None,
        "worktree_path_exists": False,
        "worktree_registered": False,
        "latest_states": {"run": None, "checks": None, "apply": None},
        "artifact_directories_present": [],
        "active": False,
        "active_fields": {},
        "stale": False,
        "orphaned": True,
        "dirty": None,
        "dirty_paths": [],
        "inconsistent": False,
        "mismatched": False,
        "cleanable": False,
        "inspection_error": None,
        "recommended_manual_action": (
            "Inspect orphaned local freestyle branch manually; no full job id is "
            "available for constrained cleanup."
        ),
    }


def _maintenance_roots(
    project_root: Path | None, state_root: Path | None
) -> tuple[Path, Path]:
    return resolved_roots(project_root or ROOT_DIR, state_root)


def _matching_states(
    job: FreestyleJob | None, definitions: dict[str, set[str]]
) -> dict[str, str]:
    if job is None:
        return {}
    return {
        field: value
        for field, blocked in definitions.items()
        if (value := getattr(job, field)) in blocked
    }


def _is_stale(timestamp: str) -> bool:
    try:
        updated = datetime.fromisoformat(timestamp)
        if updated.tzinfo is None:
            return True
        return datetime.now(timezone.utc) - updated.astimezone(timezone.utc) >= STALE_AFTER
    except (TypeError, ValueError):
        return True


def _artifact_job_ids(state: Path) -> set[str]:
    found: set[str] = set()
    for name in (*_JOB_ARTIFACT_DIRS, "worktrees"):
        parent = state / name
        if parent.is_dir() and not parent.is_symlink():
            for child in parent.iterdir():
                if is_job_id(child.name):
                    found.add(child.name)
    return found


def _job_artifact_paths(state: Path, job_id: str) -> tuple[Path, ...]:
    return tuple(state / name / job_id for name in _JOB_ARTIFACT_DIRS)


def _registered_worktrees(root: Path) -> dict[Path, str | None]:
    result = _run_maintenance_git(
        [*_MAINTENANCE_GIT_PREFIX, "worktree", "list", "--porcelain"], root
    )
    if result.returncode != 0:
        raise FreestyleJobRefusedError(
            _bounded_error(result.stderr or result.stdout, "Could not list worktrees.")
        )
    found: dict[Path, str | None] = {}
    path: Path | None = None
    branch: str | None = None
    for line in result.stdout.splitlines() + [""]:
        if line.startswith("worktree "):
            path = Path(line.removeprefix("worktree ")).resolve()
            branch = None
        elif line.startswith("branch refs/heads/"):
            branch = line.removeprefix("branch refs/heads/")
        elif not line and path is not None:
            found[path] = branch
            path = None
            branch = None
    return found


def _freestyle_branches(root: Path) -> dict[str, str]:
    result = _run_maintenance_git(
        [
            *_MAINTENANCE_GIT_PREFIX,
            "for-each-ref",
            "--format=%(refname:short)%00%(objectname)",
            f"refs/heads/{BRANCH_PREFIX}*",
        ],
        root,
    )
    if result.returncode != 0:
        raise FreestyleJobRefusedError(
            _bounded_error(result.stderr or result.stdout, "Could not list branches.")
        )
    found: dict[str, str] = {}
    for line in result.stdout.splitlines():
        try:
            branch, commit = line.split("\x00", 1)
        except ValueError as exc:
            raise FreestyleJobRefusedError("Git returned malformed branch metadata.") from exc
        if branch.startswith(BRANCH_PREFIX) and _is_commit_id(commit):
            found[branch] = commit
    return found


def _job_id_from_branch(branch: str) -> str | None:
    suffix = branch.removeprefix(BRANCH_PREFIX)
    return suffix if is_job_id(suffix) else None


def _job_id_from_worktree_path(path: Path, state: Path) -> str | None:
    try:
        relative = path.relative_to(state / "worktrees")
    except ValueError:
        return None
    return relative.name if len(relative.parts) == 1 and is_job_id(relative.name) else None


def _parse_status_paths(output: str) -> list[str]:
    paths: list[str] = []
    for entry in output.split("\x00"):
        if not entry:
            continue
        path = entry[3:] if len(entry) >= 4 else entry
        paths.append(path[:500])
    return sorted(set(paths))


def _remove_allowed_state_path(path: Path, state: Path) -> None:
    _validate_allowed_state_path(path, state)
    if path.is_symlink():
        path.unlink()
    else:
        shutil.rmtree(path)


def _validate_allowed_state_path(path: Path, state: Path) -> None:
    allowed_parents = {state / name for name in _JOB_ARTIFACT_DIRS}
    if path.parent not in allowed_parents or not is_job_id(path.name):
        raise FreestyleJobRefusedError("Cleanup path is outside allowed local artifacts.")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise FreestyleJobRefusedError("Cleanup artifact parent is unsafe.")
    try:
        path.parent.resolve(strict=True).relative_to(state)
    except (OSError, ValueError) as exc:
        raise FreestyleJobRefusedError(
            "Cleanup artifact parent escaped local dev state."
        ) from exc
    if not path.is_symlink() and not path.is_dir():
        raise FreestyleJobRefusedError("Cleanup artifact path is unsafe.")


def _relative_state_path(path: Path, state: Path) -> str:
    try:
        return path.relative_to(state).as_posix()
    except ValueError as exc:
        raise FreestyleJobRefusedError("Maintenance path escaped local dev state.") from exc


def _append_bounded(values: tuple[str, ...], note: str) -> tuple[str, ...]:
    return (*values[-19:], _bounded_error(note, "Maintenance recovery recorded."))


def _write_refusal_artifact(
    operation: str,
    job: FreestyleJob,
    state: Path,
    pre_state: dict[str, Any],
    reason: str,
    *,
    git_operations: list[dict[str, Any]] | None = None,
) -> str:
    return _write_maintenance_artifact(
        operation,
        job,
        state,
        pre_state=pre_state,
        actions=[],
        paths_removed=[],
        paths_kept=[],
        git_operations=git_operations or [],
        refusal_reasons=[_bounded_error(reason, "Maintenance operation refused.")],
        extra={},
    )


def _write_maintenance_artifact(
    operation: str,
    job: FreestyleJob,
    state: Path,
    *,
    pre_state: dict[str, Any],
    actions: list[str],
    paths_removed: list[str],
    paths_kept: list[str],
    git_operations: list[dict[str, Any]],
    refusal_reasons: list[str],
    extra: dict[str, Any],
) -> str:
    directory = state / "maintenance" / job.job_id
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise FreestyleJobRefusedError("Maintenance artifact directory is unsafe.")
    directory.resolve(strict=True).relative_to(state)
    timestamp = _utc_now()
    artifact = {
        "schema_version": MAINTENANCE_SCHEMA_VERSION,
        "operation": operation,
        "timestamp": timestamp,
        "job_id": job.job_id,
        "pre_state": pre_state,
        "job_metadata_pre_state": asdict(job),
        "actions_taken": actions,
        "paths_removed": paths_removed,
        "paths_kept": paths_kept,
        "git_operations_attempted": git_operations,
        "refusal_reasons": refusal_reasons,
        "integration_statement": _NO_INTEGRATION_STATEMENT,
        **extra,
    }
    payload = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if len(payload.encode("utf-8")) > MAX_MAINTENANCE_ARTIFACT_BYTES:
        raise FreestyleJobRefusedError("Maintenance artifact exceeds size limit.")
    path = directory / f"{operation}-{uuid.uuid4().hex[:16]}.json"
    if path.exists() or path.is_symlink():
        raise FreestyleJobRefusedError("Maintenance artifact path is unsafe.")
    path.write_text(payload, encoding="utf-8")
    return _relative_state_path(path, state)


def _git_operation_record(result: GitCommandResult) -> dict[str, Any]:
    return {
        "args": list(result.args),
        "returncode": result.returncode,
        "stdout": _bounded_error(result.stdout, ""),
        "stderr": _bounded_error(result.stderr, ""),
    }


def _run_maintenance_git(args: list[str], cwd: Path) -> GitCommandResult:
    command = tuple(args)
    if not _is_allowed_maintenance_git(command):
        raise FreestyleJobRefusedError("Internal error: forbidden maintenance git operation.")
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return GitCommandResult(command, 124, "", "Git operation timed out.")
    except OSError:
        return GitCommandResult(command, 1, "", "Git operation could not start.")
    return GitCommandResult(
        command,
        completed.returncode,
        completed.stdout.decode("utf-8", errors="replace"),
        completed.stderr.decode("utf-8", errors="replace"),
    )


def _is_allowed_maintenance_git(args: tuple[str, ...]) -> bool:
    if args[: len(_MAINTENANCE_GIT_PREFIX)] != _MAINTENANCE_GIT_PREFIX:
        return False
    suffix = args[len(_MAINTENANCE_GIT_PREFIX):]
    if suffix in {
        ("worktree", "list", "--porcelain"),
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
    }:
        return True
    if suffix == (
        "for-each-ref",
        "--format=%(refname:short)%00%(objectname)",
        f"refs/heads/{BRANCH_PREFIX}*",
    ):
        return True
    if (
        len(suffix) in {3, 4}
        and suffix[:2] == ("worktree", "remove")
        and (len(suffix) == 3 or suffix[2] == "--force")
    ):
        path = Path(suffix[-1])
        return (
            path.is_absolute()
            and is_job_id(path.name)
            and path.parent.name == "worktrees"
            and path.parent.parent.name == DEV_STATE_DIR_NAME
        )
    return (
        len(suffix) == 3
        and suffix[:2] in {("branch", "-d"), ("branch", "-D")}
        and suffix[2].startswith(BRANCH_PREFIX)
        and len(suffix[2]) == len(BRANCH_PREFIX) + 12
        and all(
            character in "0123456789abcdef"
            for character in suffix[2][len(BRANCH_PREFIX):]
        )
    )
