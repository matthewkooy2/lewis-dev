"""Shared root resolution and fail-closed job-state guards for freestyle modules.

These helpers centralize the dev-state containment check and the common
approval-adjacent state guards that every downstream freestyle slice
performed inline. They never write job metadata, never run subprocesses,
and refuse rather than repair unexpected state.
"""

from __future__ import annotations

from pathlib import Path

from app.dev.freestyle_coding import (
    DEV_STATE_DIR_NAME,
    FreestyleJob,
    FreestyleJobRefusedError,
    _require_state_inside_project,
)

# Job-state fields whose listed values mark an operation as still in
# progress. The orchestration slice additionally treats an "inconsistent"
# edit application as unsafe to run over.
ACTIVE_STATES: dict[str, set[str]] = {
    "status": {"preparing", "materializing"},
    "materialization_status": {"materializing"},
    "proposal_status": {"proposing"},
    "edit_generation_status": {"generating"},
    "edit_apply_status": {"applying"},
    "review_status": {"reviewing"},
    "checks_status": {"running"},
    "latest_run_status": {"running"},
    "latest_refinement_status": {"running"},
}
INCONSISTENT_STATES: dict[str, set[str]] = {"edit_apply_status": {"inconsistent"}}


def resolved_roots(project_root: Path, state_root: Path | None) -> tuple[Path, Path]:
    """Resolve one project root and its contained dev-state root, fail-closed."""
    root = project_root.resolve()
    state = (state_root or root / DEV_STATE_DIR_NAME).resolve()
    _require_state_inside_project(state, root)
    return root, state


def is_materialized(job: FreestyleJob) -> bool:
    return job.status == "materialized" and job.materialization_status == "materialized"


def require_materialized(job: FreestyleJob, requirement: str) -> None:
    """Refuse unless the job was successfully materialized."""
    if not is_materialized(job):
        raise FreestyleJobRefusedError(f"Freestyle job {job.job_id!r} {requirement}")


def require_no_active_edit_application(job: FreestyleJob, detail: str) -> None:
    """Refuse while an edit application is in progress or inconsistent."""
    if job.edit_apply_status in {"applying", "inconsistent"}:
        raise FreestyleJobRefusedError(
            f"Freestyle job {job.job_id!r} has an in-progress or inconsistent "
            f"edit application{detail}"
        )
