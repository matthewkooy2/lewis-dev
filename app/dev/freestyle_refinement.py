"""Refine an approved freestyle job from its current isolated worktree state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
import uuid

from app.config import ROOT_DIR
from app.dev.freestyle_artifacts import (
    encode_json_artifact,
    safe_artifact_reference,
    write_artifact_bytes,
)
from app.dev.freestyle_coding import (
    FreestyleJob,
    FreestyleJobRefusedError,
    _require_explicit_developer_approval,
    _utc_now,
    _validated_job_worktree,
    _write_job,
    apply_freestyle_edit_bundle,
    generate_freestyle_edit_bundle,
    get_freestyle_job,
    propose_freestyle_job,
    review_freestyle_job,
    run_freestyle_checks,
    validate_goal,
)
from app.dev.freestyle_run import _required_bundle_path, _validate_requested_flags
from app.dev.freestyle_state import (
    ACTIVE_STATES,
    INCONSISTENT_STATES,
    require_materialized,
    resolved_roots,
)

MAX_REFINEMENT_ARTIFACT_BYTES = 300_000


def refine_freestyle_job(
    job_id: str,
    goal: str,
    *,
    developer_approved: bool = False,
    apply_latest_generated: bool = False,
    confirm_apply: bool = False,
    run_checks: bool = False,
    check_ids: Sequence[str] | None = None,
    project_root: Path | None = None,
    state_root: Path | None = None,
    deepseek_config: Any | None = None,
    deepseek_client: Any | None = None,
) -> FreestyleJob:
    """Generate a new proposal and edit bundle from an existing worktree."""
    if developer_approved is not True:
        raise FreestyleJobRefusedError(
            "Explicit developer approval is required to refine a freestyle job."
        )
    refinement_goal = validate_goal(goal)
    _validate_requested_flags(
        apply_latest_generated=apply_latest_generated,
        confirm_apply=confirm_apply,
        run_checks=run_checks,
        check_ids=check_ids,
    )
    if run_checks and not apply_latest_generated:
        raise FreestyleJobRefusedError(
            "Refinement checks require explicit application of the new bundle."
        )

    root, state = resolved_roots(project_root or ROOT_DIR, state_root)
    job = get_freestyle_job(job_id, project_root=root, state_root=state)
    _preflight_refinement(job, root, state)
    refinement_id = uuid.uuid4().hex[:16]
    effective_goal = _effective_refinement_goal(job, refinement_goal)
    metadata = {
        "refinement_id": refinement_id,
        "iteration": job.refinement_count + 1,
        "original_title": job.title,
        "original_goal": job.goal,
        "refinement_goal": refinement_goal,
        "current_isolated_worktree": True,
    }
    started_at = _utc_now()
    original_artifacts = job.original_artifacts or _artifact_snapshot(job)
    original_stage_statuses = job.original_stage_statuses or _stage_snapshot(job)
    record = {
        **metadata,
        "status": "running",
        "started_at": started_at,
        "completed_at": None,
        "revision": 0,
        "artifact_path": None,
        "artifacts": {},
    }
    started = replace(
        job,
        original_artifacts=original_artifacts,
        original_stage_statuses=original_stage_statuses,
        refinement_count=job.refinement_count + 1,
        latest_refinement_id=refinement_id,
        latest_refinement_goal=refinement_goal,
        latest_refinement_status="running",
        latest_refinement_artifact_path=None,
        latest_refinement_proposal_artifact_path=None,
        latest_refinement_edit_generation_artifact_path=None,
        latest_refinement_edit_bundle_path=None,
        latest_refinement_edit_artifact_path=None,
        latest_refinement_review_artifact_path=None,
        latest_refinement_checks_artifact_path=None,
        refinement_history=(*job.refinement_history, record),
        proposal_status="not_proposed",
        proposal_started_at=None,
        proposal_completed_at=None,
        proposal_artifact_path=None,
        proposal_failure_artifact_path=None,
        proposal_inspected_file_count=0,
        proposal_model=None,
        proposal_errors=(),
        edit_generation_status="not_generated",
        edit_generation_started_at=None,
        edit_generation_completed_at=None,
        edit_generation_artifact_path=None,
        edit_bundle_path=None,
        edit_bundle_validation_status="not_validated",
        edit_bundle_candidate_file_count=0,
        edit_bundle_goal_fit_warnings=(),
        edit_generation_model=None,
        edit_generation_errors=(),
        edit_apply_status="not_applied",
        edit_apply_started_at=None,
        edit_applied_at=None,
        edit_applied_file_count=0,
        edit_artifact_path=None,
        edit_errors=(),
        review_status="not_reviewed",
        review_started_at=None,
        review_completed_at=None,
        review_artifact_path=None,
        review_changed_file_count=0,
        review_errors=(),
        checks_status="not_run",
        checks_started_at=None,
        checks_completed_at=None,
        checks_artifact_path=None,
        checks_selected_ids=(),
        checks_passed_count=0,
        checks_failed_count=0,
        checks_timeout_count=0,
        checks_infra_error_count=0,
        checks_modified_worktree=False,
        checks_errors=(),
        updated_at=started_at,
    )
    _write_job(started, state)
    current = sync_latest_refinement(started, project_root=root, state_root=state)
    try:
        current = propose_freestyle_job(
            job_id,
            project_root=root,
            state_root=state,
            deepseek_config=deepseek_config,
            deepseek_client=deepseek_client,
            goal_override=effective_goal,
            title_override=f"{job.title} refinement",
            artifact_name=f"refinement-{refinement_id}-proposal.json",
            failure_artifact_name=f"refinement-{refinement_id}-proposal-failure.json",
            artifact_metadata=metadata,
        )
        current = generate_freestyle_edit_bundle(
            job_id,
            project_root=root,
            state_root=state,
            deepseek_config=deepseek_config,
            deepseek_client=deepseek_client,
            goal_override=effective_goal,
            title_override=f"{job.title} refinement",
            artifact_metadata=metadata,
        )
        if apply_latest_generated:
            current = apply_freestyle_edit_bundle(
                job_id,
                str(_required_bundle_path(current, state)),
                project_root=root,
                state_root=state,
            )
            current = review_freestyle_job(
                job_id, project_root=root, state_root=state
            )
        if run_checks:
            current = run_freestyle_checks(
                job_id,
                check_ids=check_ids,
                project_root=root,
                state_root=state,
            )
    except Exception as exc:
        current = get_freestyle_job(job_id, project_root=root, state_root=state)
        failed = replace(
            current,
            latest_refinement_status="failed",
            updated_at=_utc_now(),
        )
        _write_job(failed, state)
        sync_latest_refinement(failed, project_root=root, state_root=state)
        raise FreestyleJobRefusedError(
            f"Could not complete refinement {refinement_id} for freestyle job "
            f"{job_id}; the bounded failure was recorded in local dev state."
        ) from exc
    return current


def sync_latest_refinement(
    job: FreestyleJob,
    *,
    project_root: Path | None = None,
    state_root: Path | None = None,
) -> FreestyleJob:
    """Persist an immutable manifest revision for the latest refinement."""
    if job.latest_refinement_id is None:
        return job
    root, state = resolved_roots(project_root or ROOT_DIR, state_root)
    _require_consistent_refinement_metadata(job)
    if not job.refinement_history:
        raise FreestyleJobRefusedError("Latest refinement metadata has no history.")
    record = dict(job.refinement_history[-1])
    if record.get("refinement_id") != job.latest_refinement_id:
        raise FreestyleJobRefusedError("Latest refinement metadata is inconsistent.")
    prior_revision = record.get("revision", 0)
    if (
        not isinstance(prior_revision, int)
        or isinstance(prior_revision, bool)
        or not 0 <= prior_revision < 10_000
    ):
        raise FreestyleJobRefusedError("Latest refinement revision is invalid.")
    revision = prior_revision + 1
    status = (
        "failed"
        if job.latest_refinement_status == "failed"
        else _refinement_status(job)
    )
    artifacts = _artifact_snapshot(job)
    completed_at = None if status == "running" else _utc_now()
    manifest = {
        "schema_version": 1,
        "artifact_type": "freestyle_refinement",
        "refinement_id": job.latest_refinement_id,
        "iteration": job.refinement_count,
        "revision": revision,
        "status": status,
        "original_job": {
            "job_id": job.job_id,
            "title": job.title,
            "goal": job.goal,
            "branch": job.branch,
            "worktree_path": job.worktree_path,
        },
        "refinement_goal": job.latest_refinement_goal,
        "uses_current_isolated_worktree": True,
        "artifacts": artifacts,
        "original_artifacts": job.original_artifacts or {},
        "original_stage_statuses": job.original_stage_statuses or {},
        "source_mutation_performed": job.edit_apply_status == "applied",
        "integration_statement": (
            "No commit, merge, push, pull, fetch, PR, remote contact, or main "
            "checkout mutation is performed by refinement."
        ),
    }
    encoded = encode_json_artifact(manifest)
    if len(encoded) > MAX_REFINEMENT_ARTIFACT_BYTES:
        raise FreestyleJobRefusedError("Refinement artifact exceeds the size limit.")
    artifact_path = write_artifact_bytes(
        state,
        "refinements",
        job.job_id,
        f"refinement-{job.latest_refinement_id}-{revision:04d}.json",
        encoded,
        label="refinement",
    )
    record.update(
        status=status,
        completed_at=completed_at,
        revision=revision,
        artifact_path=artifact_path,
        artifacts=artifacts,
    )
    updated = replace(
        job,
        latest_refinement_status=status,
        latest_refinement_artifact_path=artifact_path,
        latest_refinement_proposal_artifact_path=artifacts["proposal"],
        latest_refinement_edit_generation_artifact_path=artifacts["edit_generation"],
        latest_refinement_edit_bundle_path=artifacts["generated_bundle"],
        latest_refinement_edit_artifact_path=artifacts["applied_edit"],
        latest_refinement_review_artifact_path=artifacts["review"],
        latest_refinement_checks_artifact_path=artifacts["checks"],
        refinement_history=(*job.refinement_history[:-1], record),
        updated_at=_utc_now(),
    )
    _write_job(updated, state)
    return updated


def _preflight_refinement(job: FreestyleJob, root: Path, state: Path) -> None:
    _require_explicit_developer_approval(job, "refine")
    require_materialized(job, "must be successfully materialized before refinement.")
    _require_consistent_refinement_metadata(job)
    for field in set(ACTIVE_STATES) | set(INCONSISTENT_STATES):
        value = getattr(job, field)
        blocked = ACTIVE_STATES.get(field, set()) | INCONSISTENT_STATES.get(
            field, set()
        )
        if value in blocked:
            raise FreestyleJobRefusedError(
                f"Freestyle job {job.job_id!r} cannot be refined while "
                f"{field} is {value!r}."
            )
    _validated_job_worktree(job, root, state)
    from app.dev.freestyle_maintenance import _registered_worktrees

    registered = _registered_worktrees(root)
    expected = state / "worktrees" / job.job_id
    if registered.get(expected) != job.branch:
        raise FreestyleJobRefusedError(
            "Refinement requires a registered worktree branch matching job metadata."
        )


def _require_consistent_refinement_metadata(job: FreestyleJob) -> None:
    if (
        not isinstance(job.refinement_count, int)
        or isinstance(job.refinement_count, bool)
        or job.refinement_count < 0
        or job.refinement_count != len(job.refinement_history)
    ):
        raise FreestyleJobRefusedError("Refinement history metadata is inconsistent.")
    if job.original_artifacts is not None and not isinstance(
        job.original_artifacts, dict
    ):
        raise FreestyleJobRefusedError("Original artifact metadata is inconsistent.")
    if job.original_stage_statuses is not None and not isinstance(
        job.original_stage_statuses, dict
    ):
        raise FreestyleJobRefusedError("Original stage metadata is inconsistent.")
    for iteration, record in enumerate(job.refinement_history, start=1):
        if (
            record.get("iteration") != iteration
            or not _is_refinement_id(record.get("refinement_id"))
        ):
            raise FreestyleJobRefusedError("Refinement history metadata is inconsistent.")
    if job.refinement_count == 0:
        if job.latest_refinement_id is not None:
            raise FreestyleJobRefusedError("Latest refinement metadata is inconsistent.")
        return
    if (
        not _is_refinement_id(job.latest_refinement_id)
        or job.refinement_history[-1].get("refinement_id")
        != job.latest_refinement_id
    ):
        raise FreestyleJobRefusedError("Latest refinement metadata is inconsistent.")


def _is_refinement_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 16
        and all(character in "0123456789abcdef" for character in value)
    )


def _effective_refinement_goal(job: FreestyleJob, refinement_goal: str) -> str:
    return (
        "This is a refinement of the current isolated worktree, not a fresh job. "
        "Preserve existing working changes unless this refinement requires modifying "
        "them, and prefer minimal targeted edits.\n\n"
        f"Original job title: {job.title}\n"
        f"Original job goal: {job.goal}\n\n"
        f"New refinement goal: {refinement_goal}"
    )


def _artifact_snapshot(job: FreestyleJob) -> dict[str, str | None]:
    return {
        "proposal": safe_artifact_reference(job.proposal_artifact_path),
        "proposal_failure": safe_artifact_reference(job.proposal_failure_artifact_path),
        "edit_generation": safe_artifact_reference(job.edit_generation_artifact_path),
        "generated_bundle": safe_artifact_reference(job.edit_bundle_path),
        "applied_edit": safe_artifact_reference(job.edit_artifact_path),
        "review": safe_artifact_reference(job.review_artifact_path),
        "checks": safe_artifact_reference(job.checks_artifact_path),
        "orchestration_run": safe_artifact_reference(job.latest_run_artifact_path),
    }


def _stage_snapshot(job: FreestyleJob) -> dict[str, str]:
    return {
        "proposal": job.proposal_status,
        "edit_generation": job.edit_generation_status,
        "edit_apply": job.edit_apply_status,
        "review": job.review_status,
        "checks": job.checks_status,
        "run": job.latest_run_status,
    }


def _refinement_status(job: FreestyleJob) -> str:
    if any(
        status in {"error", "inconsistent"}
        for status in (
            job.proposal_status,
            job.edit_generation_status,
            job.edit_apply_status,
            job.review_status,
            job.checks_status,
        )
    ):
        return "failed"
    if job.checks_status in {"passed", "failed"}:
        return "checked"
    if job.review_status == "reviewed":
        return "reviewed"
    if job.edit_apply_status == "applied":
        return "applied"
    if (
        job.edit_generation_status == "generated"
        and job.edit_bundle_validation_status == "valid"
    ):
        return "generated"
    if job.proposal_status == "proposed":
        return "proposed"
    return "running"
