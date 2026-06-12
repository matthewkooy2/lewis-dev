"""Developer-only orchestration for one bounded freestyle coding run.

This module adds no capabilities of its own. It preflights one explicit run,
then calls the existing materialize, propose, generate, apply, review, and
allowlisted-check operations in order. The default path stops before source
mutation, and application requires two explicit flags.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
import json
from pathlib import Path, PurePosixPath
from typing import Any
import uuid

from app.config import ROOT_DIR
from app.dev.freestyle_coding import (
    DEV_STATE_DIR_NAME,
    FreestyleJob,
    FreestyleJobRefusedError,
    _bounded_error,
    _require_explicit_developer_approval,
    _require_state_inside_project,
    _utc_now,
    _validated_job_worktree,
    _write_job,
    apply_freestyle_edit_bundle,
    generate_freestyle_edit_bundle,
    get_freestyle_job,
    materialize_freestyle_job,
    propose_freestyle_job,
    review_freestyle_job,
    run_freestyle_checks,
)

RUN_SCHEMA_VERSION = 1
MAX_RUN_ARTIFACT_BYTES = 250_000
MAX_RUN_ERRORS = 3
_NO_INTEGRATION_STATEMENT = (
    "No commit, merge, push, pull, fetch, remote operation, branch switch, PR "
    "creation, or auto-merge occurred."
)
_RECOVERY_GUIDANCE = (
    "Recover or verify the isolated worktree and job metadata before starting "
    "another orchestration run."
)
_ACTIVE_STATES = {
    "status": {"preparing", "materializing"},
    "materialization_status": {"materializing"},
    "proposal_status": {"proposing"},
    "edit_generation_status": {"generating"},
    "edit_apply_status": {"applying", "inconsistent"},
    "review_status": {"reviewing"},
    "checks_status": {"running"},
    "latest_run_status": {"running"},
}


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
    """Run existing safe stages in order and stop at human review boundaries."""
    _validate_requested_flags(
        apply_latest_generated=apply_latest_generated,
        confirm_apply=confirm_apply,
        run_checks=run_checks,
        check_ids=check_ids,
    )
    root = (project_root or ROOT_DIR).resolve()
    state = (state_root or root / DEV_STATE_DIR_NAME).resolve()
    _require_state_inside_project(state, root)
    job = get_freestyle_job(job_id, project_root=root, state_root=state)
    _preflight_job(
        job,
        root,
        state,
        apply_latest_generated=apply_latest_generated,
        run_checks=run_checks,
    )

    run_id = uuid.uuid4().hex[:16]
    started_at = _utc_now()
    started = replace(
        job,
        latest_run_id=run_id,
        latest_run_status="running",
        latest_run_started_at=started_at,
        latest_run_completed_at=None,
        latest_run_artifact_path=None,
        latest_run_completed_stages=(),
        latest_run_errors=(),
        updated_at=started_at,
    )
    _write_job(started, state)
    stages: list[dict[str, Any]] = []
    completed_stages: list[str] = []
    errors: list[str] = []
    current = started
    apply_completed = False
    checks_completed = False

    try:
        current = _run_or_skip(
            "materialize",
            current,
            stages,
            completed_stages,
            skip=_is_materialized(current),
            skip_reason="already_materialized",
            operation=lambda: materialize_freestyle_job(
                job_id, project_root=root, state_root=state
            ),
        )
        current = _run_or_skip(
            "propose",
            current,
            stages,
            completed_stages,
            skip=_has_proposal(current),
            skip_reason="already_proposed",
            operation=lambda: propose_freestyle_job(
                job_id,
                project_root=root,
                state_root=state,
                deepseek_config=deepseek_config,
                deepseek_client=deepseek_client,
            ),
        )
        current = _run_or_skip(
            "generate_edits",
            current,
            stages,
            completed_stages,
            skip=_has_valid_generated_bundle(current, state),
            skip_reason="valid_generated_bundle_exists",
            operation=lambda: generate_freestyle_edit_bundle(
                job_id,
                project_root=root,
                state_root=state,
                deepseek_config=deepseek_config,
                deepseek_client=deepseek_client,
            ),
        )

        if apply_latest_generated:
            was_applied = current.edit_apply_status == "applied"
            current = _run_or_skip(
                "apply_edits",
                current,
                stages,
                completed_stages,
                skip=was_applied,
                skip_reason="edits_already_applied",
                operation=lambda: apply_freestyle_edit_bundle(
                    job_id,
                    str(_required_bundle_path(current, state)),
                    project_root=root,
                    state_root=state,
                ),
            )
            apply_completed = not was_applied
        else:
            stages.append(_skipped_stage("apply_edits", "explicit_confirmation_required"))

        if current.edit_apply_status == "applied":
            review_was_current = current.review_status == "reviewed" and not apply_completed
            current = _run_or_skip(
                "review",
                current,
                stages,
                completed_stages,
                skip=review_was_current,
                skip_reason="review_artifact_exists",
                operation=lambda: review_freestyle_job(
                    job_id, project_root=root, state_root=state
                ),
            )
        else:
            stages.append(_skipped_stage("review", "edits_not_applied"))

        if run_checks:
            checks_were_current = _checks_are_current(current, check_ids, apply_completed)
            current = _run_or_skip(
                "run_checks",
                current,
                stages,
                completed_stages,
                skip=checks_were_current,
                skip_reason="matching_check_artifact_exists",
                operation=lambda: run_freestyle_checks(
                    job_id,
                    check_ids=check_ids,
                    project_root=root,
                    state_root=state,
                ),
            )
            checks_completed = not checks_were_current
        else:
            stages.append(_skipped_stage("run_checks", "not_explicitly_requested"))
    except Exception as exc:
        current = _refresh_job(job_id, root, state, current)
        error = _bounded_error(str(exc), "Freestyle orchestration failed.")
        errors.append(error)
        if stages and stages[-1].get("status") == "failed":
            stages[-1]["artifact_references"] = _artifact_references(current)
        else:
            stages.append(
                {
                    "stage": "orchestration",
                    "status": "failed",
                    "error": error,
                    "artifact_references": _artifact_references(current),
                }
            )
        return _persist_run_result(
            current,
            state,
            run_id=run_id,
            started_at=started_at,
            status="failed",
            stages=stages,
            completed_stages=completed_stages,
            errors=errors,
            apply_latest_generated=apply_latest_generated,
            confirm_apply=confirm_apply,
            run_checks=run_checks,
            check_ids=check_ids,
            edits_applied_by_run=apply_completed,
            checks_run_by_run=checks_completed,
            raise_failure=True,
        )

    return _persist_run_result(
        current,
        state,
        run_id=run_id,
        started_at=started_at,
        status="completed",
        stages=stages,
        completed_stages=completed_stages,
        errors=errors,
        apply_latest_generated=apply_latest_generated,
        confirm_apply=confirm_apply,
        run_checks=run_checks,
        check_ids=check_ids,
        edits_applied_by_run=apply_completed,
        checks_run_by_run=checks_completed,
        raise_failure=False,
    )


def _validate_requested_flags(
    *,
    apply_latest_generated: bool,
    confirm_apply: bool,
    run_checks: bool,
    check_ids: Sequence[str] | None,
) -> None:
    if any(
        not isinstance(value, bool)
        for value in (apply_latest_generated, confirm_apply, run_checks)
    ):
        raise FreestyleJobRefusedError("Orchestration flags must be booleans.")
    if apply_latest_generated != confirm_apply:
        raise FreestyleJobRefusedError(
            "Applying edits requires both --apply-latest-generated and "
            "--confirm-apply; neither flag may be used alone."
        )
    if check_ids is not None and not run_checks:
        raise FreestyleJobRefusedError("--check requires --run-checks.")
    if run_checks:
        from app.dev.freestyle_checks import _validated_check_selection

        _validated_check_selection(check_ids)


def _preflight_job(
    job: FreestyleJob,
    root: Path,
    state: Path,
    *,
    apply_latest_generated: bool,
    run_checks: bool,
) -> None:
    _require_explicit_developer_approval(job, "run orchestration")
    for field, blocked_values in _ACTIVE_STATES.items():
        value = getattr(job, field)
        if value in blocked_values:
            raise FreestyleJobRefusedError(
                f"Freestyle job {job.job_id!r} has unsafe {field} {value!r}. "
                f"{_RECOVERY_GUIDANCE}"
            )
    _validated_job_worktree(job, root, state)
    if _is_materialized(job):
        if job.status != "materialized":
            raise FreestyleJobRefusedError(
                f"Freestyle job {job.job_id!r} has inconsistent materialization "
                f"metadata. {_RECOVERY_GUIDANCE}"
            )
    elif job.status != "ready" or job.materialization_status != "not_materialized":
        raise FreestyleJobRefusedError(
            f"Freestyle job {job.job_id!r} is not safely ready or materialized. "
            f"{_RECOVERY_GUIDANCE}"
        )
    if run_checks and job.edit_apply_status != "applied" and not apply_latest_generated:
        raise FreestyleJobRefusedError(
            "Orchestration checks require previously applied edits or the explicit "
            "apply flags in the same run."
        )


def _run_or_skip(
    stage: str,
    job: FreestyleJob,
    stages: list[dict[str, Any]],
    completed_stages: list[str],
    *,
    skip: bool,
    skip_reason: str,
    operation: Callable[[], FreestyleJob],
) -> FreestyleJob:
    if skip:
        stages.append(
            {
                **_skipped_stage(stage, skip_reason),
                "artifact_references": _artifact_references(job),
            }
        )
        return job
    attempted = {
        "stage": stage,
        "status": "attempted",
        "artifact_references": _artifact_references(job),
    }
    stages.append(attempted)
    try:
        completed = operation()
    except Exception as exc:
        attempted["status"] = "failed"
        attempted["error"] = _bounded_error(str(exc), f"{stage} failed.")
        raise
    attempted["status"] = "completed"
    attempted["artifact_references"] = _artifact_references(completed)
    completed_stages.append(stage)
    return completed


def _persist_run_result(
    job: FreestyleJob,
    state: Path,
    *,
    run_id: str,
    started_at: str,
    status: str,
    stages: list[dict[str, Any]],
    completed_stages: list[str],
    errors: list[str],
    apply_latest_generated: bool,
    confirm_apply: bool,
    run_checks: bool,
    check_ids: Sequence[str] | None,
    edits_applied_by_run: bool,
    checks_run_by_run: bool,
    raise_failure: bool,
) -> FreestyleJob:
    completed_at = _utc_now()
    bounded_errors = tuple(
        _bounded_error(error, "Freestyle orchestration failed.")
        for error in errors[:MAX_RUN_ERRORS]
    )
    artifact = {
        "schema_version": RUN_SCHEMA_VERSION,
        "artifact_type": "freestyle_orchestration_run",
        "run_id": run_id,
        "status": status,
        "started_at": started_at,
        "completed_at": completed_at,
        "job": {
            "job_id": job.job_id,
            "goal": job.goal,
            "branch": job.branch,
            "worktree_path": job.worktree_path,
        },
        "requested": {
            "stages": [
                "materialize",
                "propose",
                "generate_edits",
                *(["apply_edits", "review"] if apply_latest_generated else []),
                *(["run_checks"] if run_checks else []),
            ],
            "apply_latest_generated": apply_latest_generated,
            "confirm_apply": confirm_apply,
            "run_checks": run_checks,
            "check_ids": None if check_ids is None else list(check_ids),
        },
        "stages": stages,
        "completed_stages": list(completed_stages),
        "artifact_references": _artifact_references(job),
        "generated_bundle_validation_status": job.edit_bundle_validation_status,
        "edits_applied": job.edit_apply_status == "applied",
        "edits_applied_by_run": edits_applied_by_run,
        "final_edit_apply_status": job.edit_apply_status,
        "checks_run": job.checks_status not in {"not_run", "error"},
        "checks_run_by_run": checks_run_by_run,
        "final_checks_status": job.checks_status,
        "final_changed_file_summary": {
            "review_status": job.review_status,
            "changed_file_count": (
                job.review_changed_file_count
                if job.review_status == "reviewed"
                else None
            ),
        },
        "errors": list(bounded_errors),
        "human_next_steps": _human_next_steps(job, status),
        "integration_statement": _NO_INTEGRATION_STATEMENT,
    }
    artifact_path = _write_run_artifact(artifact, state, job.job_id, run_id)
    updated = replace(
        job,
        latest_run_id=run_id,
        latest_run_status=status,
        latest_run_started_at=started_at,
        latest_run_completed_at=completed_at,
        latest_run_artifact_path=artifact_path,
        latest_run_completed_stages=tuple(completed_stages),
        latest_run_errors=bounded_errors,
        updated_at=completed_at,
    )
    _write_job(updated, state)
    if raise_failure:
        raise FreestyleJobRefusedError(
            f"Could not complete orchestration run {run_id} for freestyle job "
            f"{job.job_id}; the bounded failure was recorded in local dev state."
        )
    return updated


def _write_run_artifact(
    artifact: dict[str, Any], state: Path, job_id: str, run_id: str
) -> str:
    runs_dir = state / "runs"
    job_dir = runs_dir / job_id
    for directory in (runs_dir, job_dir):
        if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
            raise FreestyleJobRefusedError(
                "Freestyle orchestration artifact path is unsafe."
            )
        directory.mkdir(parents=True, exist_ok=True)
        try:
            directory.resolve(strict=True).relative_to(state)
        except (OSError, ValueError) as exc:
            raise FreestyleJobRefusedError(
                "Freestyle orchestration artifact path must stay inside local dev state."
            ) from exc
    name = f"run-{run_id}.json"
    path = job_dir / name
    temporary = job_dir / f"{name}.tmp"
    for candidate in (path, temporary):
        if candidate.exists() or candidate.is_symlink():
            raise FreestyleJobRefusedError(
                "Freestyle orchestration artifact path is unsafe."
            )
    encoded = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) > MAX_RUN_ARTIFACT_BYTES:
        raise FreestyleJobRefusedError(
            "Freestyle orchestration artifact exceeds the size limit."
        )
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return path.relative_to(state).as_posix()


def _artifact_references(job: FreestyleJob) -> dict[str, str | None]:
    return {
        "proposal": _safe_artifact_reference(job.proposal_artifact_path),
        "edit_generation": _safe_artifact_reference(job.edit_generation_artifact_path),
        "generated_bundle": _safe_artifact_reference(job.edit_bundle_path),
        "applied_edits": _safe_artifact_reference(job.edit_artifact_path),
        "review": _safe_artifact_reference(job.review_artifact_path),
        "checks": _safe_artifact_reference(job.checks_artifact_path),
    }


def _human_next_steps(job: FreestyleJob, status: str) -> list[str]:
    if status == "failed":
        return [
            _RECOVERY_GUIDANCE,
            "Inspect the bounded run error and referenced stage artifacts.",
            _NO_INTEGRATION_STATEMENT,
        ]
    steps: list[str] = []
    if job.edit_apply_status != "applied":
        steps.extend(
            [
                f"Review the generated bundle at {job.edit_bundle_path!r}.",
                "To apply it, rerun with --apply-latest-generated --confirm-apply.",
            ]
        )
    else:
        steps.append(f"Review the diff artifact at {job.review_artifact_path!r}.")
        if job.checks_status == "not_run":
            steps.append("Run allowlisted checks separately with --run-checks.")
        else:
            steps.append(f"Inspect the check artifact at {job.checks_artifact_path!r}.")
    steps.append(_NO_INTEGRATION_STATEMENT)
    return steps


def _is_materialized(job: FreestyleJob) -> bool:
    return job.status == "materialized" and job.materialization_status == "materialized"


def _has_proposal(job: FreestyleJob) -> bool:
    return job.proposal_status == "proposed" and bool(job.proposal_artifact_path)


def _has_valid_generated_bundle(job: FreestyleJob, state: Path) -> bool:
    bundle_id = _generated_reference_id(
        job.edit_bundle_path, "bundle-", job.job_id
    )
    artifact_id = _generated_reference_id(
        job.edit_generation_artifact_path, "generation-", job.job_id
    )
    return (
        job.edit_generation_status == "generated"
        and job.edit_bundle_validation_status == "valid"
        and bundle_id is not None
        and bundle_id == artifact_id
        and _is_safe_generated_file(job.edit_bundle_path, state)
        and _is_safe_generated_file(job.edit_generation_artifact_path, state)
    )


def _required_bundle_path(job: FreestyleJob, state: Path) -> Path:
    if not _has_valid_generated_bundle(job, state) or not job.edit_bundle_path:
        raise FreestyleJobRefusedError(
            "The latest generated edit bundle is not valid and available."
        )
    return state.joinpath(*PurePosixPath(job.edit_bundle_path).parts)


def _generated_reference_id(
    value: Any, prefix: str, job_id: str
) -> str | None:
    safe = _safe_artifact_reference(value)
    if safe is None:
        return None
    parts = PurePosixPath(safe).parts
    if len(parts) != 3 or parts[:2] != ("edit-bundles", job_id):
        return None
    name = parts[2]
    if not name.startswith(prefix) or not name.endswith(".json"):
        return None
    identifier = name[len(prefix) : -len(".json")]
    if len(identifier) != 16 or any(
        character not in "0123456789abcdef" for character in identifier
    ):
        return None
    return identifier


def _is_safe_generated_file(value: Any, state: Path) -> bool:
    safe = _safe_artifact_reference(value)
    if safe is None:
        return False
    relative = PurePosixPath(safe)
    path = state.joinpath(*relative.parts)
    parents = (state / "edit-bundles", state / "edit-bundles" / relative.parts[1])
    if any(
        parent.is_symlink() or not parent.is_dir()
        for parent in parents
    ):
        return False
    if path.is_symlink() or not path.is_file():
        return False
    try:
        path.resolve(strict=True).relative_to(state.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def _safe_artifact_reference(value: Any) -> str | None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 500
        or "\x00" in value
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()


def _checks_are_current(
    job: FreestyleJob, check_ids: Sequence[str] | None, apply_completed: bool
) -> bool:
    if apply_completed or job.checks_status not in {"passed", "failed"}:
        return False
    if not job.checks_artifact_path:
        return False
    return check_ids is None or tuple(check_ids) == job.checks_selected_ids


def _skipped_stage(stage: str, reason: str) -> dict[str, Any]:
    return {"stage": stage, "status": "skipped", "reason": reason}


def _refresh_job(
    job_id: str, root: Path, state: Path, fallback: FreestyleJob
) -> FreestyleJob:
    try:
        return get_freestyle_job(job_id, project_root=root, state_root=state)
    except FreestyleJobRefusedError:
        return fallback
