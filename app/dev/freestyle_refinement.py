"""Refine an approved freestyle job from its current isolated worktree state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import json
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
    observed_ui_feedback: str | None = None,
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
    ui_feedback = _validate_observed_ui_feedback(observed_ui_feedback)
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
    memory = _previous_attempt_memory(job, state, root)
    effective_goal = _effective_refinement_goal(
        job, refinement_goal, memory, ui_feedback
    )
    metadata = {
        "refinement_id": refinement_id,
        "iteration": job.refinement_count + 1,
        "original_title": job.title,
        "original_goal": job.goal,
        "refinement_goal": refinement_goal,
        "observed_ui_feedback": ui_feedback,
        "current_isolated_worktree": True,
        "previous_attempt_memory": memory,
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
        implementation_plan_artifact_path=None,
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
        review_ui_warnings=(),
        review_plan_warnings=(),
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
            current = review_freestyle_job(job_id, project_root=root, state_root=state)
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
        if record.get("iteration") != iteration or not _is_refinement_id(
            record.get("refinement_id")
        ):
            raise FreestyleJobRefusedError(
                "Refinement history metadata is inconsistent."
            )
    if job.refinement_count == 0:
        if job.latest_refinement_id is not None:
            raise FreestyleJobRefusedError(
                "Latest refinement metadata is inconsistent."
            )
        return
    if (
        not _is_refinement_id(job.latest_refinement_id)
        or job.refinement_history[-1].get("refinement_id") != job.latest_refinement_id
    ):
        raise FreestyleJobRefusedError("Latest refinement metadata is inconsistent.")


def _is_refinement_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 16
        and all(character in "0123456789abcdef" for character in value)
    )


MAX_MEMORY_ARTIFACT_BYTES = 300_000
MAX_MEMORY_FILES = 20


MAX_OBSERVED_UI_FEEDBACK_CHARS = 4_000


def _validate_observed_ui_feedback(value: str | None) -> str | None:
    """Bound optional human-reported runtime/UI feedback; ``None`` when absent."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise FreestyleJobRefusedError("Observed UI feedback must be text.")
    clean = value.strip()
    if not clean:
        return None
    if "\x00" in clean:
        raise FreestyleJobRefusedError(
            "Observed UI feedback must not contain NUL bytes."
        )
    if len(clean) > MAX_OBSERVED_UI_FEEDBACK_CHARS:
        raise FreestyleJobRefusedError(
            "Observed UI feedback exceeds the maximum length of "
            f"{MAX_OBSERVED_UI_FEEDBACK_CHARS} characters."
        )
    return clean


def _effective_refinement_goal(
    job: FreestyleJob,
    refinement_goal: str,
    memory: dict[str, Any] | None = None,
    observed_ui_feedback: str | None = None,
) -> str:
    base = (
        "This is a refinement of the current isolated worktree, not a fresh job. "
        "Preserve existing working changes unless this refinement requires modifying "
        "them, and prefer minimal targeted edits.\n\n"
        f"Original job title: {job.title}\n"
        f"Original job goal: {job.goal}\n\n"
    )
    base += _observed_ui_feedback_section(observed_ui_feedback)
    base += _memory_prompt_section(memory)
    return base + f"New refinement goal: {refinement_goal}"


def _observed_ui_feedback_section(observed_ui_feedback: str | None) -> str:
    """Render human-reported runtime/UI problems as first-class refinement input.

    Kept separate from the refinement goal text so the next attempt treats an
    observed rendered-UI failure as concrete evidence to fix, not as optional
    prose buried in the goal.
    """
    if not observed_ui_feedback:
        return ""
    return (
        "Observed runtime/UI feedback (reported by a human reviewer after seeing "
        "the rendered result; treat this as a concrete defect to repair, not as "
        "optional context):\n"
        f"{observed_ui_feedback}\n\n"
    )


def _memory_prompt_section(memory: dict[str, Any] | None) -> str:
    """Render the previous attempt's behaviour so a refinement does not repeat it."""
    if not memory:
        return ""
    lines: list[str] = []
    touched = memory.get("previous_bundle_touched_files") or []
    likely = memory.get("previous_proposal_likely_files") or []
    warnings = memory.get("previous_warnings") or []
    if touched:
        lines.append(
            "The previous bundle changed these files: " + ", ".join(touched) + "."
        )
    if likely:
        lines.append(
            "The previous proposal expected to edit: " + ", ".join(likely) + "."
        )
    if warnings:
        lines.append(
            "The previous attempt was flagged with: "
            + " ".join(warnings)
            + " Do not repeat those rejected patterns; satisfy the refinement "
            "goal with a real, wired implementation change."
        )
    ui_warnings = memory.get("previous_ui_review_warnings") or []
    if ui_warnings:
        lines.append(
            "The previous attempt's rendered UI review flagged: "
            + " ".join(ui_warnings)
            + " Fix these rendered-layout/selector problems in this refinement."
        )
    repo_map_warnings = memory.get("previous_repo_map_review_warnings") or []
    if repo_map_warnings:
        lines.append(
            "The repo map flagged: "
            + " ".join(repo_map_warnings)
            + " Consider the connected implementation surfaces, not just the "
            "files named in the prompt."
        )
    connected = memory.get("repo_map_connected_surfaces") or {}
    if connected:
        rendered = "; ".join(
            f"{path} connects to {', '.join(related)}"
            for path, related in connected.items()
        )
        lines.append(
            "Repo map context used: the previously changed files connect to "
            f"these related implementation surfaces: {rendered}."
        )
    prior_plan = memory.get("previous_implementation_plan") or {}
    if isinstance(prior_plan, dict) and prior_plan.get("recommended_strategy"):
        failed = prior_plan.get("failed_strategy_patterns") or []
        suffix = (
            f" If the same warnings remain, that approach did not work — do not "
            f"repeat the failed strategy ({', '.join(failed)}); inspect the "
            "evidence and change approach."
            if failed
            else " If the same warnings remain, that approach did not work; "
            "change approach rather than repeating it."
        )
        lines.append(
            "The previous attempt's implementation plan recommended: "
            + str(prior_plan["recommended_strategy"])
            + suffix
        )
    if not lines:
        return ""
    return (
        "Previous attempt memory (do not treat this refinement as unrelated):\n"
        + "\n".join(f"- {line}" for line in lines)
        + "\n\n"
    )


def _previous_attempt_memory(
    job: FreestyleJob, state: Path, root: Path | None = None
) -> dict[str, Any]:
    """Collect a bounded record of what the previous attempt did and why it was
    flagged, so refinement generation can avoid repeating a rejected pattern.

    Best-effort and fail-open: any unreadable or malformed artifact yields an
    empty field rather than blocking the refinement. Only bounded, safe path
    strings, the previous attempt's own advisory warnings, and the repo map's
    connected implementation surfaces for the changed files are carried; no file
    contents, secrets, or prompts are read.
    """
    touched = _previous_bundle_touched_files(job, state)
    likely = _previous_proposal_likely_files(job, state)
    warnings = [
        str(warning)[:500]
        for warning in (
            *job.edit_bundle_goal_fit_warnings,
            *job.edit_bundle_semantic_warnings,
        )
        if isinstance(warning, str) and warning
    ]
    ui_warnings = [
        str(warning)[:500]
        for warning in job.review_ui_warnings
        if isinstance(warning, str) and warning
    ]
    prior_plan = _previous_implementation_plan(job, state)
    return {
        "previous_bundle_touched_files": touched,
        "previous_proposal_likely_files": likely,
        "previous_warnings": warnings,
        "previous_ui_review_warnings": ui_warnings[:MAX_MEMORY_FILES],
        "previous_repo_map_review_warnings": [
            str(warning)[:500]
            for warning in job.review_repo_map_warnings
            if isinstance(warning, str) and warning
        ][:MAX_MEMORY_FILES],
        "previous_implementation_plan": prior_plan,
        "previous_task_type": prior_plan.get("task_type") if prior_plan else None,
        "previous_review_plan_warnings": [
            str(warning)[:500]
            for warning in job.review_plan_warnings
            if isinstance(warning, str) and warning
        ][:MAX_MEMORY_FILES],
        "repo_map_connected_surfaces": _repo_map_connected_surfaces(
            job, state, root, touched + likely
        ),
    }


def _previous_implementation_plan(job: FreestyleJob, state: Path) -> dict[str, Any]:
    """Bounded summary of the previous attempt's implementation plan, if any.

    Read from the previous generation artifact's embedded plan summary so the
    next plan can detect "the prior plan recommended X but the bundle did Y" and
    so the refinement prompt can say what strategy not to repeat. Fail-open: an
    unreadable or malformed artifact yields an empty mapping.
    """
    artifact = _read_state_artifact(job.edit_generation_artifact_path, state)
    if artifact is None:
        return {}
    summary = artifact.get("implementation_plan")
    if not isinstance(summary, dict):
        return {}
    return {
        "task_type": summary.get("task_type"),
        "primary_feature": summary.get("primary_feature"),
        "recommended_strategy": str(summary.get("recommended_strategy", ""))[:500],
        "failed_strategy_patterns": [
            str(pattern)[:100]
            for pattern in (summary.get("failed_strategy_patterns") or [])
            if isinstance(pattern, str)
        ][:MAX_MEMORY_FILES],
        "expected_files": [
            str(path)[:200]
            for path in (summary.get("expected_files") or [])
            if isinstance(path, str)
        ][:MAX_MEMORY_FILES],
    }


def _repo_map_connected_surfaces(
    job: FreestyleJob,
    state: Path,
    root: Path | None,
    seed_paths: list[str],
) -> dict[str, list[str]]:
    """Repo-map connected files for the previously changed/likely files.

    Best-effort and fail-open: lets a refinement retrieve connected
    implementation surfaces (route/page/CSS/JS/tests) rather than only the files
    named directly in the prompt. Returns an empty mapping on any problem.
    """
    if root is None or not seed_paths:
        return {}
    try:
        from app.dev.repo_map_context import (
            connected_files_for_paths,
            resolve_freestyle_index,
        )

        worktree = state / "worktrees" / job.job_id
        index, _ = resolve_freestyle_index(worktree, root, state, refresh=False)
        if index is None:
            return {}
        connected = connected_files_for_paths(index, seed_paths)
        return {
            path: related[:MAX_MEMORY_FILES]
            for path, related in list(connected.items())[:MAX_MEMORY_FILES]
        }
    except Exception:
        return {}


def _previous_bundle_touched_files(job: FreestyleJob, state: Path) -> list[str]:
    artifact = _read_state_artifact(job.edit_generation_artifact_path, state)
    if artifact is None:
        return []
    validation = artifact.get("validation")
    if not isinstance(validation, dict):
        return []
    operations = validation.get("operations")
    if not isinstance(operations, list):
        return []
    paths: list[str] = []
    for operation in operations:
        if isinstance(operation, dict) and isinstance(operation.get("path"), str):
            path = operation["path"]
            if path and path not in paths:
                paths.append(path)
    return paths[:MAX_MEMORY_FILES]


def _previous_proposal_likely_files(job: FreestyleJob, state: Path) -> list[str]:
    artifact = _read_state_artifact(job.proposal_artifact_path, state)
    if artifact is None:
        return []
    likely = artifact.get("likely_files_to_edit")
    if not isinstance(likely, list):
        return []
    return [item for item in likely if isinstance(item, str) and item][
        :MAX_MEMORY_FILES
    ]


def _read_state_artifact(reference: str | None, state: Path) -> dict[str, Any] | None:
    relative = safe_artifact_reference(reference)
    if relative is None:
        return None
    path = state / relative
    try:
        if path.is_symlink() or not path.is_file():
            return None
        path.resolve(strict=True).relative_to(state.resolve())
        if path.stat().st_size > MAX_MEMORY_ARTIFACT_BYTES:
            return None
        payload = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


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
