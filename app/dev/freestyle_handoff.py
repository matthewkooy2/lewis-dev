"""Read-only human handoff/status summaries for developer-only freestyle jobs."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from app.config import ROOT_DIR
from app.dev.freestyle_artifacts import safe_artifact_reference
from app.dev.freestyle_coding import (
    FreestyleJob,
    FreestyleJobRefusedError,
    _bounded_error,
    _validated_job_worktree,
    get_freestyle_job,
)
from app.dev.freestyle_context_plan import BUDGET_OMISSION_REASONS
from app.dev.freestyle_policy import NO_INTEGRATION_STATEMENT_HANDOFF
from app.dev.freestyle_review import _read_changed_paths
from app.dev.freestyle_state import resolved_roots

MAX_HANDOFF_ARTIFACT_BYTES = 1_000_000
MAX_HANDOFF_ERRORS = 20
MAX_HANDOFF_RISKS = 20
_NO_INTEGRATION_STATEMENT = NO_INTEGRATION_STATEMENT_HANDOFF
_DEEPSEEK_CONFIG_MARKERS = (
    "deepseek proposal configuration",
    "deepseek_api_key",
    "deepseek_code_model",
    "deepseek_base_url",
)


def get_freestyle_handoff(
    job_id: str,
    *,
    project_root: Path | None = None,
    state_root: Path | None = None,
) -> dict[str, Any]:
    """Build one read-only human handoff from local dev state and worktree status."""
    root, state = resolved_roots(project_root or ROOT_DIR, state_root)
    job = get_freestyle_job(job_id, project_root=root, state_root=state)

    errors = _job_errors(job)
    artifacts = _artifact_references(job)
    proposal = _read_optional_artifact(
        artifacts["proposal"], state, "proposal", errors
    )
    proposal_failure = _read_optional_artifact(
        artifacts["proposal_failure"], state, "proposal failure", errors
    )
    generation = _read_optional_artifact(
        artifacts["edit_generation"], state, "edit generation", errors
    )
    review = _read_optional_artifact(artifacts["review"], state, "review", errors)
    checks = _read_optional_artifact(artifacts["checks"], state, "checks", errors)
    changed_files = _changed_file_summary(job, root, state, review, errors)
    check_summary = _check_summary(job, checks)
    blocked = _blocked_summary(job, errors)
    risks = _risk_summary(job, proposal, generation, check_summary, blocked, errors)

    return {
        "job": {
            "job_id": job.job_id,
            "title": job.title,
            "goal": job.goal,
            "approval_state": job.approval_state,
            "base_commit": job.base_commit,
            "base_source": "current clean repo HEAD at job creation",
            "branch": job.branch,
            "worktree_path": job.worktree_path,
        },
        "goal_fit_warnings": list(job.edit_bundle_goal_fit_warnings),
        "semantic_warnings": list(job.edit_bundle_semantic_warnings),
        "ui_review_warnings": list(job.review_ui_warnings),
        "refinement": _refinement_summary(job),
        "stages": {
            "materialize": job.materialization_status,
            "propose": job.proposal_status,
            "generate-edits": job.edit_generation_status,
            "apply-edits": job.edit_apply_status,
            "review": job.review_status,
            "run-checks": job.checks_status,
            "run": job.latest_run_status,
            "refine": job.latest_refinement_status,
        },
        "artifacts": artifacts,
        "proposal_failure_diagnostics": _proposal_failure_summary(
            artifacts["proposal_failure"], proposal_failure
        ),
        "changed_files": changed_files,
        "checks": check_summary,
        "bounded_errors": errors[:MAX_HANDOFF_ERRORS],
        "risks": risks[:MAX_HANDOFF_RISKS],
        "blocked": blocked,
        "human_next_steps": _human_next_steps(job, blocked),
        "integration_statement": _NO_INTEGRATION_STATEMENT,
    }


def render_freestyle_handoff_markdown(handoff: dict[str, Any]) -> str:
    """Render a concise handoff without persisting or mutating any job state."""
    job = handoff["job"]
    blocked = handoff["blocked"]
    lines = [
        f"# Freestyle Job {job['job_id']}",
        "",
        f"**Title:** {job['title']}",
        f"**Goal:** {job['goal']}",
        f"**Approval:** {job['approval_state']}",
        f"**Base:** current clean repo `HEAD` at job creation (`{job['base_commit']}`)",
        f"**Branch:** `{job['branch']}`",
        f"**Worktree:** `{job['worktree_path']}`",
        "",
        "## Stages",
        "",
    ]
    lines.extend(
        f"- **{stage}:** {status}" for stage, status in handoff["stages"].items()
    )
    lines.extend(["", "## Artifacts", ""])
    lines.extend(
        f"- **{name}:** {f'`{path}`' if path else 'none'}"
        for name, path in handoff["artifacts"].items()
    )
    refinement = handoff.get("refinement", {})
    if refinement.get("count"):
        lines.extend(["", "## Refinements", ""])
        lines.append(f"- **Count:** {refinement['count']}")
        lines.append(f"- **Latest ID:** `{refinement['latest_id']}`")
        lines.append(f"- **Latest goal:** {refinement['latest_goal']}")
        lines.append(f"- **Latest status:** {refinement['latest_status']}")
    diagnostics = handoff.get("proposal_failure_diagnostics")
    if diagnostics:
        lines.extend(["", "## Proposal Failure Diagnostics", ""])
        lines.append(f"- **Artifact:** `{diagnostics['artifact_path']}`")
        lines.append(f"- **Summary:** {diagnostics['summary']}")
        lines.extend(
            f"- **Attempt {item['attempt']}:** {item['status']}"
            for item in diagnostics["attempts"]
        )

    changed = handoff["changed_files"]
    lines.extend(
        [
            "",
            "## Changed Files",
            "",
            f"- **Source:** {changed['source']}",
            f"- **Count:** {changed['count'] if changed['count'] is not None else 'unknown'}",
        ]
    )
    lines.extend(
        f"- `{item['path']}` ({item['status']})" for item in changed["files"]
    )

    checks = handoff["checks"]
    lines.extend(
        [
            "",
            "## Checks",
            "",
            f"- **Status:** {checks['overall_status']}",
            f"- **Selected:** {', '.join(checks['selected_ids']) or 'none'}",
            f"- **Passed:** {checks['passed']}",
            f"- **Failed:** {checks['failed']}",
            f"- **Timed out:** {checks['timeout']}",
            f"- **Infrastructure errors:** {checks['infra_error']}",
            f"- **Modified worktree:** {str(checks['modified_worktree']).lower()}",
            "",
            "## Blocking State",
            "",
            f"- **Blocked:** {str(blocked['is_blocked']).lower()}",
        ]
    )
    lines.extend(f"- {reason}" for reason in blocked["reasons"])
    if not blocked["reasons"]:
        lines.append("- none")

    goal_fit_warnings = handoff.get("goal_fit_warnings", [])
    if goal_fit_warnings:
        lines.extend(["", "## Goal-Fit Warnings (Advisory)", ""])
        lines.extend(f"- {warning}" for warning in goal_fit_warnings)

    semantic_warnings = handoff.get("semantic_warnings", [])
    if semantic_warnings:
        lines.extend(["", "## Bundle Goal-Fit Critique (Advisory)", ""])
        lines.extend(f"- {warning}" for warning in semantic_warnings)

    ui_review_warnings = handoff.get("ui_review_warnings", [])
    if ui_review_warnings:
        lines.extend(["", "## Rendered UI Review (Advisory)", ""])
        lines.extend(f"- {warning}" for warning in ui_review_warnings)

    lines.extend(["", "## Risks And Errors", ""])
    lines.extend(f"- {risk}" for risk in handoff["risks"])
    lines.extend(f"- Error: {error}" for error in handoff["bounded_errors"])
    if not handoff["risks"] and not handoff["bounded_errors"]:
        lines.append("- none recorded")

    lines.extend(["", "## Human Next Steps", ""])
    lines.extend(f"{index}. {step}" for index, step in enumerate(
        handoff["human_next_steps"], start=1
    ))
    lines.extend(["", f"**Integration safety:** {handoff['integration_statement']}", ""])
    return "\n".join(lines)


def _artifact_references(job: FreestyleJob) -> dict[str, str | None]:
    return {
        "proposal": safe_artifact_reference(job.proposal_artifact_path),
        "proposal_failure": safe_artifact_reference(
            job.proposal_failure_artifact_path
        ),
        "generated_bundle": safe_artifact_reference(job.edit_bundle_path),
        "edit_generation": safe_artifact_reference(job.edit_generation_artifact_path),
        "applied_edit": safe_artifact_reference(job.edit_artifact_path),
        "review": safe_artifact_reference(job.review_artifact_path),
        "checks": safe_artifact_reference(job.checks_artifact_path),
        "orchestration_run": safe_artifact_reference(job.latest_run_artifact_path),
        "latest_refinement": safe_artifact_reference(
            job.latest_refinement_artifact_path
        ),
    }


def _refinement_summary(job: FreestyleJob) -> dict[str, Any]:
    original = job.original_artifacts if isinstance(job.original_artifacts, dict) else {}
    original_stages = (
        job.original_stage_statuses
        if isinstance(job.original_stage_statuses, dict)
        else {}
    )
    return {
        "count": job.refinement_count,
        "latest_id": job.latest_refinement_id,
        "latest_goal": job.latest_refinement_goal,
        "latest_status": job.latest_refinement_status,
        "latest_artifacts": {
            "proposal": safe_artifact_reference(
                job.latest_refinement_proposal_artifact_path
            ),
            "edit_generation": safe_artifact_reference(
                job.latest_refinement_edit_generation_artifact_path
            ),
            "generated_bundle": safe_artifact_reference(
                job.latest_refinement_edit_bundle_path
            ),
            "applied_edit": safe_artifact_reference(
                job.latest_refinement_edit_artifact_path
            ),
            "review": safe_artifact_reference(
                job.latest_refinement_review_artifact_path
            ),
            "checks": safe_artifact_reference(
                job.latest_refinement_checks_artifact_path
            ),
            "manifest": safe_artifact_reference(job.latest_refinement_artifact_path),
        },
        "original_artifacts": {
            str(name): safe_artifact_reference(reference)
            for name, reference in original.items()
            if isinstance(name, str)
        },
        "original_stages": {
            str(name): str(status)[:50]
            for name, status in original_stages.items()
            if isinstance(name, str) and isinstance(status, str)
        },
        "history": [
            {
                "refinement_id": item.get("refinement_id"),
                "iteration": item.get("iteration"),
                "goal": item.get("refinement_goal"),
                "status": item.get("status"),
                "artifact_path": safe_artifact_reference(item.get("artifact_path")),
            }
            for item in job.refinement_history
        ],
    }


def _proposal_failure_summary(
    reference: str | None, artifact: dict[str, Any] | None
) -> dict[str, Any] | None:
    if reference is None:
        return None
    attempts: list[dict[str, Any]] = []
    raw_attempts = artifact.get("proposal_attempts") if artifact is not None else None
    if isinstance(raw_attempts, list):
        for item in raw_attempts[:2]:
            if not isinstance(item, dict):
                continue
            attempt = item.get("attempt")
            status = item.get("status")
            if isinstance(attempt, int) and isinstance(status, str):
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": status[:50],
                        "validator_category": str(
                            item.get("validator_category", "")
                        )[:50],
                    }
                )
    statuses = ", ".join(
        f"attempt {item['attempt']}: {item['status']}" for item in attempts
    )
    return {
        "artifact_path": reference,
        "summary": (
            f"Proposal generation failed ({statuses})."
            if statuses
            else "Proposal generation failed; inspect the bounded diagnostics artifact."
        )[:500],
        "attempts": attempts,
    }


def _read_optional_artifact(
    reference: str | None, state: Path, label: str, errors: list[str]
) -> dict[str, Any] | None:
    if reference is None:
        return None
    path = state.joinpath(*PurePosixPath(reference).parts)
    try:
        if path.is_symlink() or not path.is_file():
            raise FreestyleJobRefusedError(f"Latest {label} artifact is missing or unsafe.")
        path.resolve(strict=True).relative_to(state.resolve(strict=True))
        if path.stat().st_size > MAX_HANDOFF_ARTIFACT_BYTES:
            raise FreestyleJobRefusedError(f"Latest {label} artifact exceeds read limit.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise FreestyleJobRefusedError(f"Latest {label} artifact is not an object.")
        return payload
    except (OSError, ValueError, json.JSONDecodeError, FreestyleJobRefusedError) as exc:
        errors.append(_bounded_error(str(exc), f"Could not read latest {label} artifact."))
        return None


def _changed_file_summary(
    job: FreestyleJob,
    root: Path,
    state: Path,
    review: dict[str, Any] | None,
    errors: list[str],
) -> dict[str, Any]:
    if review is not None:
        files = review.get("files")
        if isinstance(files, list):
            summarized = []
            for item in files:
                if isinstance(item, dict) and isinstance(item.get("path"), str):
                    summarized.append(
                        {
                            "path": item["path"][:500],
                            "status": str(item.get("status", "unknown"))[:50],
                            "git_status": str(item.get("git_status", ""))[:10],
                        }
                    )
            return {
                "source": "latest_review_artifact",
                "count": len(summarized),
                "files": summarized,
            }
    if job.status != "materialized" or job.materialization_status != "materialized":
        return {"source": "not_materialized", "count": None, "files": []}
    try:
        worktree = _validated_job_worktree(job, root, state)
        changed = _read_changed_paths(worktree)
        return {
            "source": "current_isolated_worktree_status",
            "count": len(changed),
            "files": [
                {
                    "path": item.path,
                    "status": item.status,
                    "git_status": item.git_status,
                }
                for item in changed
            ],
        }
    except Exception as exc:
        errors.append(
            _bounded_error(str(exc), "Could not read current isolated worktree status.")
        )
        return {"source": "worktree_status_error", "count": None, "files": []}


def _check_summary(job: FreestyleJob, checks: dict[str, Any] | None) -> dict[str, Any]:
    summary = checks.get("summary") if checks is not None else None
    if not isinstance(summary, dict):
        summary = {}
    return {
        "overall_status": str(summary.get("overall_status", job.checks_status))[:50],
        "selected_ids": list(job.checks_selected_ids),
        "passed": _nonnegative_int(summary.get("passed"), job.checks_passed_count),
        "failed": _nonnegative_int(summary.get("failed"), job.checks_failed_count),
        "timeout": _nonnegative_int(summary.get("timeout"), job.checks_timeout_count),
        "infra_error": _nonnegative_int(
            summary.get("infra_error"), job.checks_infra_error_count
        ),
        "modified_worktree": job.checks_modified_worktree,
    }


def _nonnegative_int(value: Any, fallback: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return fallback


def _job_errors(job: FreestyleJob) -> list[str]:
    values = (
        *job.errors,
        *job.proposal_errors,
        *job.edit_generation_errors,
        *job.edit_errors,
        *job.review_errors,
        *job.checks_errors,
        *job.latest_run_errors,
    )
    return [_bounded_error(str(value), "Freestyle job error.") for value in values][
        :MAX_HANDOFF_ERRORS
    ]


def _blocked_summary(job: FreestyleJob, errors: list[str]) -> dict[str, Any]:
    lowered_errors = " ".join(errors).casefold()
    needs_model_stage = job.edit_apply_status != "applied" and (
        job.proposal_status != "proposed"
        or job.edit_generation_status != "generated"
        or job.edit_bundle_validation_status != "valid"
    )
    flags = {
        "missing_deepseek_config": needs_model_stage
        and any(marker in lowered_errors for marker in _DEEPSEEK_CONFIG_MARKERS),
        "inconsistent_edit_state": job.edit_apply_status == "inconsistent",
        "running_lock": any(
            status in {"preparing", "materializing", "proposing", "generating",
                       "applying", "reviewing", "running"}
            for status in (
                job.status,
                job.materialization_status,
                job.proposal_status,
                job.edit_generation_status,
                job.edit_apply_status,
                job.review_status,
                job.checks_status,
                job.latest_run_status,
                job.latest_refinement_status,
            )
        ),
        "failed_checks": job.checks_status == "failed",
        "missing_review": (
            job.edit_apply_status == "applied" and job.review_status != "reviewed"
        ),
    }
    reasons = [
        label
        for key, label in (
            ("missing_deepseek_config", "Missing or invalid DeepSeek configuration."),
            ("inconsistent_edit_state", "Edit application state is inconsistent."),
            ("running_lock", "A stage is marked running or in progress."),
            ("failed_checks", "One or more allowlisted checks failed."),
            ("missing_review", "Applied edits do not have a completed review."),
        )
        if flags[key]
    ]
    return {**flags, "is_blocked": bool(reasons), "reasons": reasons}


def _risk_summary(
    job: FreestyleJob,
    proposal: dict[str, Any] | None,
    generation: dict[str, Any] | None,
    checks: dict[str, Any],
    blocked: dict[str, Any],
    errors: list[str],
) -> list[str]:
    risks: list[str] = []
    proposal_risks = proposal.get("risks_unknowns") if proposal is not None else None
    if isinstance(proposal_risks, list):
        risks.extend(
            str(item)[:500] for item in proposal_risks if isinstance(item, str)
        )
    check_warnings = (
        proposal.get("suggested_check_warnings") if proposal is not None else None
    )
    if isinstance(check_warnings, list):
        risks.extend(
            str(item)[:500] for item in check_warnings if isinstance(item, str)
        )
    risks.extend(
        str(warning)[:500] for warning in job.edit_bundle_goal_fit_warnings
    )
    risks.extend(
        str(warning)[:500] for warning in job.edit_bundle_semantic_warnings
    )
    risks.extend(str(warning)[:500] for warning in job.review_ui_warnings)
    risks.extend(_context_plan_risks("proposal", proposal))
    risks.extend(_context_plan_risks("edit-generation", generation))
    risks.extend(_anchor_mismatch_risks(generation))
    risks.extend(blocked["reasons"])
    if checks["modified_worktree"]:
        risks.append("The latest check run modified the isolated worktree.")
    if job.edit_bundle_validation_status == "invalid":
        risks.append("The latest generated edit bundle is invalid.")
    if errors:
        risks.append("The job has bounded recorded errors that require inspection.")
    return list(dict.fromkeys(risks))


def _anchor_mismatch_risks(generation: dict[str, Any] | None) -> list[str]:
    """Surface one concise warning when edit generation failed on an exact
    replace_text anchor mismatch, noting whether exact worktree snippets and
    the bounded repair retry were already used."""
    if generation is None:
        return []
    anchor = generation.get("anchor_snippets")
    validation = generation.get("validation")
    if not isinstance(anchor, dict) or not isinstance(validation, dict):
        return []
    if validation.get("status") == "valid":
        return []
    retry = anchor.get("retry")
    if not isinstance(retry, dict) or retry.get("failed_path") is None:
        return []
    path = str(retry.get("failed_path"))[:240]
    if retry.get("used"):
        message = (
            f"Edit generation failed on an exact replace_text anchor mismatch "
            f"in {path}; exact worktree snippets and one bounded repair retry "
            "were used but did not produce a valid bundle. Inspect the "
            "edit-generation artifact."
        )
    else:
        skipped = str(retry.get("skipped_reason") or "unknown reason")[:50]
        message = (
            f"Edit generation failed on an exact replace_text anchor mismatch "
            f"in {path}; the bounded repair retry was not attempted "
            f"({skipped}). Inspect the edit-generation artifact."
        )
    return [message[:500]]


def _context_plan_risks(label: str, artifact: dict[str, Any] | None) -> list[str]:
    """Surface context-plan warnings and budget omissions from one artifact."""
    if artifact is None:
        return []
    plan = artifact.get("context_plan")
    if not isinstance(plan, dict):
        return []
    risks = [
        f"The {label} context plan warned: {warning}"[:500]
        for warning in plan.get("warnings", [])
        if isinstance(warning, str)
    ]
    omitted = plan.get("omitted_priority_files")
    if plan.get("priority_files_omitted_due_to_budget") and isinstance(omitted, list):
        budget_paths = [
            str(record.get("path", ""))
            for record in omitted
            if isinstance(record, dict)
            and record.get("reason") in BUDGET_OMISSION_REASONS
        ]
        sample = ", ".join(budget_paths[:5]) or "see the artifact context plan"
        risks.append(
            f"Likely relevant files were omitted from {label} model context "
            f"due to budget: {sample}."[:500]
        )
    return risks


def _human_next_steps(job: FreestyleJob, blocked: dict[str, Any]) -> list[str]:
    diagnostic_steps = (
        ["Inspect proposal diagnostics before retrying."]
        if job.proposal_failure_artifact_path
        else []
    )
    if job.approval_state != "approved":
        return [
            *diagnostic_steps,
            "Explicitly reapprove the local job before advancing it.",
            _NO_INTEGRATION_STATEMENT,
        ]
    if blocked["running_lock"] or blocked["inconsistent_edit_state"]:
        return [
            *diagnostic_steps,
            "Recover or verify the isolated worktree and job metadata before retrying.",
            "Inspect the bounded errors and latest artifacts.",
            _NO_INTEGRATION_STATEMENT,
        ]
    steps: list[str] = list(diagnostic_steps)
    if job.materialization_status != "materialized":
        steps.append("Materialize the isolated job worktree.")
    elif job.edit_apply_status == "applied":
        if job.review_status != "reviewed":
            steps.append(
                "Create and inspect a read-only review artifact for the applied edits."
            )
        elif job.checks_status in {"not_run", "error"}:
            steps.extend(
                [
                    f"Inspect the review artifact at {job.review_artifact_path!r}.",
                    "Run the appropriate allowlisted checks.",
                ]
            )
        elif job.checks_status == "failed":
            steps.append(f"Inspect failed checks at {job.checks_artifact_path!r}.")
        else:
            steps.extend(
                [
                    f"Inspect the review artifact at {job.review_artifact_path!r}.",
                    f"Inspect the check artifact at {job.checks_artifact_path!r}.",
                    "Manually merge outside Lewis only if the reviewed changes are satisfactory.",
                ]
            )
    elif job.proposal_status != "proposed":
        steps.append("Configure DeepSeek if needed, then generate the proposal.")
    elif (
        job.edit_generation_status != "generated"
        or job.edit_bundle_validation_status != "valid"
    ):
        steps.append("Generate and validate an edit bundle.")
    elif job.edit_apply_status != "applied":
        steps.append(f"Review the generated bundle at {job.edit_bundle_path!r}.")
        if job.edit_bundle_goal_fit_warnings or job.edit_bundle_semantic_warnings:
            steps.append(
                "Inspect the recorded goal-fit warnings and bundle critique "
                "before applying; the bundle may not actually implement the goal."
            )
        steps.append(
            "Apply the reviewed bundle only with explicit developer confirmation."
        )
    steps.append(_NO_INTEGRATION_STATEMENT)
    return steps
