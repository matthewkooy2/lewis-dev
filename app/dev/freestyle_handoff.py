"""Read-only human handoff/status summaries for developer-only freestyle jobs."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from app.config import ROOT_DIR
from app.dev.freestyle_coding import (
    DEV_STATE_DIR_NAME,
    FreestyleJob,
    FreestyleJobRefusedError,
    _bounded_error,
    _require_state_inside_project,
    _validated_job_worktree,
    get_freestyle_job,
)
from app.dev.freestyle_review import _read_changed_paths

MAX_HANDOFF_ARTIFACT_BYTES = 1_000_000
MAX_HANDOFF_ERRORS = 20
MAX_HANDOFF_RISKS = 20
_NO_INTEGRATION_STATEMENT = (
    "Lewis has not committed, merged, pushed, pulled, fetched, opened a PR, "
    "or auto-merged this job."
)
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
    root = (project_root or ROOT_DIR).resolve()
    state = (state_root or root / DEV_STATE_DIR_NAME).resolve()
    _require_state_inside_project(state, root)
    job = get_freestyle_job(job_id, project_root=root, state_root=state)

    errors = _job_errors(job)
    artifacts = _artifact_references(job)
    proposal = _read_optional_artifact(
        artifacts["proposal"], state, "proposal", errors
    )
    review = _read_optional_artifact(artifacts["review"], state, "review", errors)
    checks = _read_optional_artifact(artifacts["checks"], state, "checks", errors)
    changed_files = _changed_file_summary(job, root, state, review, errors)
    check_summary = _check_summary(job, checks)
    blocked = _blocked_summary(job, errors)
    risks = _risk_summary(job, proposal, check_summary, blocked, errors)

    return {
        "job": {
            "job_id": job.job_id,
            "title": job.title,
            "goal": job.goal,
            "approval_state": job.approval_state,
            "branch": job.branch,
            "worktree_path": job.worktree_path,
        },
        "stages": {
            "materialize": job.materialization_status,
            "propose": job.proposal_status,
            "generate-edits": job.edit_generation_status,
            "apply-edits": job.edit_apply_status,
            "review": job.review_status,
            "run-checks": job.checks_status,
            "run": job.latest_run_status,
        },
        "artifacts": artifacts,
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
        "proposal": _safe_artifact_reference(job.proposal_artifact_path),
        "generated_bundle": _safe_artifact_reference(job.edit_bundle_path),
        "edit_generation": _safe_artifact_reference(job.edit_generation_artifact_path),
        "applied_edit": _safe_artifact_reference(job.edit_artifact_path),
        "review": _safe_artifact_reference(job.review_artifact_path),
        "checks": _safe_artifact_reference(job.checks_artifact_path),
        "orchestration_run": _safe_artifact_reference(job.latest_run_artifact_path),
    }


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
    risks.extend(blocked["reasons"])
    if checks["modified_worktree"]:
        risks.append("The latest check run modified the isolated worktree.")
    if job.edit_bundle_validation_status == "invalid":
        risks.append("The latest generated edit bundle is invalid.")
    if errors:
        risks.append("The job has bounded recorded errors that require inspection.")
    return list(dict.fromkeys(risks))


def _human_next_steps(job: FreestyleJob, blocked: dict[str, Any]) -> list[str]:
    if job.approval_state != "approved":
        return [
            "Explicitly reapprove the local job before advancing it.",
            _NO_INTEGRATION_STATEMENT,
        ]
    if blocked["running_lock"] or blocked["inconsistent_edit_state"]:
        return [
            "Recover or verify the isolated worktree and job metadata before retrying.",
            "Inspect the bounded errors and latest artifacts.",
            _NO_INTEGRATION_STATEMENT,
        ]
    steps: list[str] = []
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
        steps.extend(
            [
                f"Review the generated bundle at {job.edit_bundle_path!r}.",
                "Apply the reviewed bundle only with explicit developer confirmation.",
            ]
        )
    steps.append(_NO_INTEGRATION_STATEMENT)
    return steps
