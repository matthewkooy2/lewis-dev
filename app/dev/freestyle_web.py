"""Server-rendered, developer-only views for local freestyle jobs."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, urlencode

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.config import ROOT_DIR
from app.dev.freestyle_artifacts import safe_artifact_reference
from app.dev.freestyle_coding import (
    DEV_STATE_DIR_NAME,
    FreestyleJob,
    _bounded_error,
    apply_freestyle_edit_bundle,
    cleanup_freestyle_job,
    create_freestyle_job,
    get_freestyle_handoff,
    get_freestyle_job,
    list_freestyle_jobs,
    materialize_freestyle_job,
    refine_freestyle_job,
    recover_freestyle_job,
    review_freestyle_job,
    run_freestyle_checks,
    run_freestyle_job,
)
from app.dev.freestyle_checks import CHECK_REGISTRY, CHECKS_BY_ID
from app.dev.freestyle_policy import is_job_id
from app.dev.freestyle_proposal import DeepSeekProposalConfig
from app.dev.freestyle_provider import (
    describe_provider_config,
    provider_config_from_settings,
)
from app.dev.freestyle_run import _required_bundle_path
from app.web_components import (
    empty_state_html,
    escape,
    field,
    flash,
    page_html,
    page_header,
    status_badge,
)

MAX_WEB_ARTIFACT_BYTES = 100_000
MAX_WEB_PREVIEW_CHARS = 24_000
MAX_WEB_REQUEST_BYTES = 20_000
SAFETY_STATEMENT = (
    "Developer-only and local-first. Lewis does not commit, merge, push, pull, "
    "fetch, switch branches, open PRs, or auto-merge freestyle jobs. Human review "
    "and merge remain required."
)


def register_freestyle_web_routes(app: FastAPI) -> None:
    """Register fixed developer routes only on an explicitly debug-enabled app."""

    @app.get("/dev/freestyle", response_class=HTMLResponse)
    def freestyle_command_center(
        request: Request, message: str = "", error: str = ""
    ) -> HTMLResponse:
        if set(request.query_params) - {"message", "error"}:
            return _page_response(
                request,
                error="Freestyle views only accept status-message parameters.",
                status_code=400,
            )
        return _page_response(request, message=message, error=error)

    @app.get("/dev/freestyle/{job_id}", response_class=HTMLResponse)
    def freestyle_job_detail(
        request: Request, job_id: str, message: str = "", error: str = ""
    ) -> HTMLResponse:
        if set(request.query_params) - {"message", "error"}:
            return _page_response(
                request,
                job_id=job_id,
                error="Freestyle views only accept status-message parameters.",
                status_code=400,
            )
        return _page_response(request, job_id=job_id, message=message, error=error)

    @app.post("/dev/freestyle/create")
    async def create_freestyle_job_web(request: Request) -> RedirectResponse:
        fields = await _request_fields(request)
        if set(fields) - {"goal", "title", "developer_approved"}:
            return _redirect(
                error="Create only accepts title, goal, and explicit approval."
            )
        try:
            job = create_freestyle_job(
                str(fields.get("goal", "")),
                title=str(fields.get("title", "")).strip() or None,
                developer_approved=fields.get("developer_approved") == "approved",
                project_root=_project_root(request),
            )
        except Exception as exc:
            return _redirect(error=_web_error(request, exc))
        return _redirect(
            job.job_id, message=f"Created approved freestyle job {job.job_id[:12]}."
        )

    @app.post("/dev/freestyle/{job_id}/materialize")
    async def materialize_freestyle_job_web(
        request: Request, job_id: str
    ) -> RedirectResponse:
        if await _request_fields(request):
            return _redirect(job_id, error="Materialize accepts no input fields.")
        try:
            materialize_freestyle_job(job_id, project_root=_project_root(request))
        except Exception as exc:
            return _redirect(job_id, error=_web_error(request, exc))
        return _redirect(job_id, message="Materialized the isolated worktree.")

    @app.post("/dev/freestyle/{job_id}/run")
    async def run_freestyle_job_web(request: Request, job_id: str) -> RedirectResponse:
        if await _request_fields(request):
            return _redirect(job_id, error="Default run accepts no input fields.")
        try:
            run_freestyle_job(
                job_id,
                apply_latest_generated=False,
                confirm_apply=False,
                run_checks=False,
                check_ids=None,
                project_root=_project_root(request),
                deepseek_config=_deepseek_config(request),
            )
        except Exception as exc:
            return _redirect(job_id, error=_web_error(request, exc))
        return _redirect(
            job_id, message="Default run stopped at the generated bundle boundary."
        )

    @app.post("/dev/freestyle/{job_id}/apply")
    async def apply_latest_generated_web(
        request: Request, job_id: str
    ) -> RedirectResponse:
        fields = await _request_fields(request)
        if set(fields) - {"confirm_apply"}:
            return _redirect(
                job_id,
                error="Apply only accepts explicit confirmation for the latest generated bundle.",
            )
        if fields.get("confirm_apply") != "apply-latest-generated":
            return _redirect(
                job_id,
                error="Explicit confirmation is required to apply the latest generated bundle.",
            )
        try:
            root = _project_root(request)
            job = get_freestyle_job(job_id, project_root=root)
            bundle_path = _required_bundle_path(job, root / DEV_STATE_DIR_NAME)
            handoff = get_freestyle_handoff(job_id, project_root=root)
            if handoff["result"]["bundle_state"] == "applied":
                return _redirect(
                    job_id,
                    message="This bundle is already applied. Review the diff and check results.",
                )
            apply_freestyle_edit_bundle(
                job_id,
                str(bundle_path),
                project_root=root,
            )
        except Exception as exc:
            return _redirect(job_id, error=_web_error(request, exc))
        return _redirect(
            job_id,
            message="Applied the latest validated generated bundle to the isolated worktree.",
        )

    @app.post("/dev/freestyle/{job_id}/refine")
    async def refine_freestyle_job_web(
        request: Request, job_id: str
    ) -> RedirectResponse:
        fields = await _request_fields(request)
        if set(fields) - {"goal", "ui_feedback", "developer_approved"}:
            return _redirect(
                job_id,
                error=(
                    "Refine only accepts a refinement goal, optional observed UI "
                    "feedback, and explicit approval."
                ),
            )
        if fields.get("developer_approved") != "approved":
            return _redirect(
                job_id,
                error="Explicit developer approval is required to refine this job.",
            )
        ui_feedback = str(fields.get("ui_feedback", "")).strip() or None
        try:
            refine_freestyle_job(
                job_id,
                str(fields.get("goal", "")),
                developer_approved=True,
                observed_ui_feedback=ui_feedback,
                project_root=_project_root(request),
                deepseek_config=_deepseek_config(request),
            )
        except Exception as exc:
            return _redirect(job_id, error=_web_error(request, exc))
        return _redirect(
            job_id,
            message="Generated a refinement bundle from the current isolated worktree.",
        )

    @app.post("/dev/freestyle/{job_id}/review")
    async def review_freestyle_job_web(
        request: Request, job_id: str
    ) -> RedirectResponse:
        if await _request_fields(request):
            return _redirect(job_id, error="Review accepts no input fields.")
        try:
            review_freestyle_job(job_id, project_root=_project_root(request))
        except Exception as exc:
            return _redirect(job_id, error=_web_error(request, exc))
        return _redirect(
            job_id, message="Created a deterministic review of the isolated worktree."
        )

    @app.post("/dev/freestyle/{job_id}/run-checks")
    async def run_freestyle_checks_web(
        request: Request, job_id: str
    ) -> RedirectResponse:
        fields = await _request_fields(request)
        if set(fields) - {"check_id"}:
            return _redirect(
                job_id, error="Run checks only accepts static backend check IDs."
            )
        try:
            check_ids = _selected_check_ids(fields.get("check_id"))
            run_freestyle_checks(
                job_id,
                check_ids=check_ids,
                project_root=_project_root(request),
            )
        except Exception as exc:
            return _redirect(job_id, error=_web_error(request, exc))
        return _redirect(
            job_id,
            message=f"Completed allowlisted checks: {', '.join(check_ids)}.",
        )

    @app.post("/dev/freestyle/{job_id}/cleanup")
    async def cleanup_freestyle_job_web(
        request: Request, job_id: str
    ) -> RedirectResponse:
        fields = await _request_fields(request)
        if set(fields) - {"confirm_cleanup", "discard_worktree_changes"}:
            return _redirect(
                job_id,
                error="Cleanup only accepts cleanup and separate discard confirmations.",
            )
        if fields.get("confirm_cleanup") != "cleanup":
            return _redirect(
                job_id, error="Explicit confirmation is required to clean this job."
            )
        discard = fields.get("discard_worktree_changes")
        if discard is not None and discard != "discard":
            return _redirect(
                job_id,
                error="Dirty-worktree discard requires its separate explicit confirmation.",
            )
        try:
            result = cleanup_freestyle_job(
                job_id,
                confirm=True,
                discard_worktree_changes=discard == "discard",
                project_root=_project_root(request),
            )
        except Exception as exc:
            return _redirect(job_id, error=_web_error(request, exc))
        return _redirect(
            message=(
                "Cleaned the selected local freestyle job. Result artifact: "
                f"{result.get('artifact_path', 'not available')}."
            )
        )

    @app.post("/dev/freestyle/{job_id}/recover")
    async def recover_freestyle_job_web(
        request: Request, job_id: str
    ) -> RedirectResponse:
        fields = await _request_fields(request)
        if set(fields) - {"confirm_recover", "confirm_inconsistent"}:
            return _redirect(
                job_id,
                error="Recovery only accepts recovery and stronger inconsistent-state confirmations.",
            )
        if fields.get("confirm_recover") != "recover":
            return _redirect(
                job_id,
                error="Explicit confirmation is required to recover stale state.",
            )
        inconsistent = fields.get("confirm_inconsistent")
        if inconsistent is not None and inconsistent != "verified-inconsistent":
            return _redirect(
                job_id,
                error="Inconsistent-state recovery requires its stronger explicit confirmation.",
            )
        try:
            job = get_freestyle_job(job_id, project_root=_project_root(request))
            if (
                job.edit_apply_status == "inconsistent"
                and inconsistent != "verified-inconsistent"
            ):
                return _redirect(
                    job_id,
                    error=(
                        "Inconsistent-state recovery requires confirmation that the "
                        "isolated worktree was manually verified."
                    ),
                )
            recover_freestyle_job(
                job_id,
                confirm=True,
                confirm_inconsistent=inconsistent == "verified-inconsistent",
                project_root=_project_root(request),
            )
        except Exception as exc:
            return _redirect(job_id, error=_web_error(request, exc))
        return _redirect(
            job_id,
            message="Recovered stale lock state to failed/error status; no work was marked successful.",
        )


def _page_response(
    request: Request,
    *,
    job_id: str | None = None,
    message: str = "",
    error: str = "",
    status_code: int = 200,
) -> HTMLResponse:
    root = _project_root(request)
    selected = None
    handoff = None
    previews = {}
    try:
        jobs = list_freestyle_jobs(project_root=root)
        if job_id is not None:
            selected = get_freestyle_job(job_id, project_root=root)
            handoff = get_freestyle_handoff(job_id, project_root=root)
            previews = read_artifact_previews(handoff, root / DEV_STATE_DIR_NAME)
    except Exception as exc:
        jobs = []
        error = _web_error(request, exc)
    content = freestyle_page(
        jobs,
        selected_job=selected,
        handoff=handoff,
        previews=previews,
        message=message,
        error=error,
        provider_description=describe_provider_config(_deepseek_config(request)),
    )
    return HTMLResponse(
        page_html(
            "Freestyle Command Center",
            request.url.path,
            content,
            show_dev_navigation=True,
            show_projects_navigation=request.app.state.settings.projects_ui_enabled,
            show_coding_agents_navigation=request.app.state.settings.coding_agents_ui_enabled,
        ),
        status_code=status_code,
    )


def _project_root(request: Request) -> Path:
    return Path(
        getattr(request.app.state, "freestyle_project_root", ROOT_DIR)
    ).resolve()


def _deepseek_config(request: Request) -> DeepSeekProposalConfig:
    """Effective provider config (DeepSeek or local) from loaded settings."""
    return provider_config_from_settings(request.app.state.settings)


def _web_error(request: Request, exc: Exception) -> str:
    text = str(exc)
    settings = request.app.state.settings
    for api_key in (settings.deepseek_api_key, settings.local_llm_api_key):
        if api_key:
            text = text.replace(api_key, "[REDACTED]")
    return _bounded_error(text, "Freestyle action could not be completed.")


def _redirect(
    job_id: str | None = None, *, message: str = "", error: str = ""
) -> RedirectResponse:
    path = (
        f"/dev/freestyle/{job_id}"
        if job_id is not None and is_job_id(job_id)
        else "/dev/freestyle"
    )
    values = {"message": message} if message else {"error": error}
    return RedirectResponse(url=f"{path}?{urlencode(values)}", status_code=303)


async def _request_fields(request: Request) -> dict[str, object]:
    raw_body = await request.body()
    if len(raw_body) > MAX_WEB_REQUEST_BYTES:
        return {"invalid_request_body": "too_large"}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            data = json.loads(raw_body)
        except (UnicodeError, json.JSONDecodeError):
            return {"invalid_request_body": "malformed_json"}
        return (
            data if isinstance(data, dict) else {"invalid_request_body": "not_object"}
        )
    decoded_body = raw_body.decode("utf-8", errors="ignore")
    return {
        key: values[0] if len(values) == 1 else tuple(values)
        for key, values in parse_qs(decoded_body, keep_blank_values=True).items()
    }


def _selected_check_ids(value: object) -> tuple[str, ...]:
    values = value if isinstance(value, (list, tuple)) else (value,)
    allowed = ", ".join(definition.check_id for definition in CHECK_REGISTRY)
    if not values or values == (None,):
        raise ValueError(f"Select at least one static backend check ID: {allowed}.")
    if any(
        not isinstance(item, str) or item not in CHECKS_BY_ID for item in values
    ) or len(set(values)) != len(values):
        raise ValueError(
            f"Run checks accepts only static backend check IDs: {allowed}."
        )
    return tuple(values)


def freestyle_page(
    jobs: list[FreestyleJob],
    *,
    selected_job: FreestyleJob | None = None,
    handoff: dict[str, Any] | None = None,
    previews: dict[str, str] | None = None,
    message: str = "",
    error: str = "",
    provider_description: str = "",
) -> str:
    return "".join(
        [
            page_header(
                "Freestyle Command Center",
                "Inspect and lightly operate approved, isolated-worktree coding jobs.",
                "<a class='button' href='/dev/freestyle'>Refresh</a>",
            ),
            _safety_notice()
            if selected_job is None
            else "<p class='muted'>Changes stay in the job worktree. Human review and manual integration are required.</p>",
            _provider_notice(provider_description) if selected_job is None else "",
            flash(message),
            _error_notice(error),
            "<section class='freestyle-layout'><div class='freestyle-rail'>",
            _job_list(jobs, selected_job.job_id if selected_job else None),
            (
                "<details class='band'><summary>Create another job</summary>"
                + _create_form()
                + "</details>"
            )
            if selected_job
            else _create_form(),
            "</div><div class='freestyle-detail'>",
            _job_detail(selected_job, handoff, previews or {}, provider_description),
            "</div></section>",
        ]
    )


def read_artifact_previews(handoff: dict[str, Any], state_root: Path) -> dict[str, str]:
    previews: dict[str, str] = {}
    for label, reference in handoff.get("artifacts", {}).items():
        if label == "proposal_failure":
            continue
        safe_reference = safe_artifact_reference(reference)
        if safe_reference is not None:
            previews[label] = _read_artifact_preview(state_root, safe_reference)
    return previews


def _read_artifact_preview(state_root: Path, reference: str) -> str:
    path = state_root.joinpath(*PurePosixPath(reference).parts)
    try:
        if path.is_symlink() or not path.is_file():
            return "Artifact is missing or unsafe."
        path.resolve(strict=True).relative_to(state_root.resolve(strict=True))
        if path.stat().st_size > MAX_WEB_ARTIFACT_BYTES:
            return (
                f"Preview omitted: artifact exceeds the {MAX_WEB_ARTIFACT_BYTES:,}-byte "
                "web preview limit."
            )
        text = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            preview = text
        else:
            preview = json.dumps(payload, indent=2, sort_keys=True)
        if len(preview) > MAX_WEB_PREVIEW_CHARS:
            return preview[:MAX_WEB_PREVIEW_CHARS] + "\n\n[preview truncated]"
        return preview
    except (OSError, UnicodeError, ValueError):
        return "Artifact could not be read safely."


def _safety_notice() -> str:
    return (
        "<section class='safety-notice'><strong>Human merge gate</strong>"
        f"<p>{escape(SAFETY_STATEMENT)}</p></section>"
    )


def _provider_notice(provider_description: str) -> str:
    if not provider_description:
        return ""
    return (
        "<section class='safety-notice'><strong>Effective model provider</strong>"
        f"<p>{escape(provider_description)}</p></section>"
    )


def _error_notice(error: str) -> str:
    if not error:
        return ""
    return (
        "<div class='inline-error' role='alert'><strong>Freestyle action stopped</strong>"
        f"<span>{escape(error)}</span><details><summary>Raw backend error</summary>"
        f"<pre class='patch-preview'>{escape(error)}</pre></details></div>"
    )


def _create_form() -> str:
    return (
        "<section class='band freestyle-create'><h2>Create Approved Job</h2>"
        "<form method='post' action='/dev/freestyle/create'>"
        "<label class='control'><span>Title (optional)</span>"
        "<input name='title' maxlength='120' placeholder='Short job title'></label>"
        "<label class='control'><span>Goal</span>"
        "<textarea name='goal' rows='5' maxlength='10000' required "
        "placeholder='Describe the bounded coding goal'></textarea></label>"
        "<label class='approval-check'>"
        "<input type='checkbox' name='developer_approved' value='approved' required>"
        "<span>I explicitly approve creating this local isolated-worktree job "
        "from the current clean repo HEAD.</span>"
        "</label><button type='submit'>Create Approved Job</button></form></section>"
    )


def _job_list(jobs: list[FreestyleJob], selected_id: str | None) -> str:
    if not jobs:
        return empty_state_html("No freestyle jobs yet.")
    cards = []
    for job in jobs:
        selected = " selected" if job.job_id == selected_id else ""
        cards.append(
            f"<a class='job-row{selected}' href='/dev/freestyle/{escape(job.job_id)}'>"
            "<span>"
            f"<strong>{escape(job.title)}</strong>"
            f"<small>{escape(job.job_id[:12])} · {escape(job.updated_at)}</small>"
            f"</span>{status_badge(job.status)}</a>"
        )
    return (
        "<section class='band freestyle-jobs'><h2>Jobs</h2>"
        f"<div class='job-list'>{''.join(cards)}</div></section>"
    )


def _job_detail(
    job: FreestyleJob | None,
    handoff: dict[str, Any] | None,
    previews: dict[str, str],
    provider_description: str,
) -> str:
    if job is None or handoff is None:
        return empty_state_html("Select a job to inspect its status and handoff.")
    result = handoff.get("result", {})
    return "".join(
        [
            "<section class='band freestyle-summary'><div class='card-head'>",
            f"<h2>{escape(job.title)}</h2></div>",
            f"<p class='job-outcome'>{escape(result.get('label', job.status))}</p>",
            f"<p>{escape(job.goal)}</p>" if job.goal != job.title else "",
            _job_actions(job, handoff),
            "</section>",
            _check_summary(handoff),
            _review_diff(handoff),
            _changed_files(handoff),
            _handoff_lists(handoff),
            "<details class='band'><summary>Refine or rerun checks</summary>",
            _provider_notice(provider_description),
            _refine_form(job, bool(handoff.get("blocked", {}).get("running_lock"))),
            _review_and_checks_actions(job, handoff),
            "</details>",
            "<details class='band'><summary>Job details and artifacts</summary><div class='fields'>",
            field("job id", job.job_id),
            field("base commit (current clean repo HEAD at creation)", job.base_commit),
            field("isolated branch", job.branch),
            field("isolated worktree", job.worktree_path),
            field("updated", job.updated_at),
            "</div>",
            _safety_notice(),
            _stage_status(handoff),
            _implementation_plan_block(handoff),
            _refinement_status(handoff),
            _proposal_failure_diagnostics(handoff),
            _artifact_paths_and_previews(handoff, previews),
            _danger_actions(job),
            "</details>",
        ]
    )


def _proposal_failure_diagnostics(handoff: dict[str, Any]) -> str:
    diagnostics = handoff.get("proposal_failure_diagnostics")
    if not isinstance(diagnostics, dict):
        return ""
    attempts = diagnostics.get("attempts", [])
    items = "".join(
        f"<li>Attempt {escape(item.get('attempt', ''))}: "
        f"{escape(item.get('status', 'error'))}</li>"
        for item in attempts
        if isinstance(item, dict)
    )
    return (
        "<section class='band'><h2>Proposal Failure Diagnostics</h2>"
        f"<p>{escape(diagnostics.get('summary', 'Proposal generation failed.'))}</p>"
        f"<div class='fields'>{field('artifact path', diagnostics.get('artifact_path', 'none'))}</div>"
        f"{f'<ul>{items}</ul>' if items else ''}"
        "<p class='muted'>Inspect the bounded diagnostics artifact before retrying. "
        "Full model output is not displayed here.</p></section>"
    )


def _job_actions(job: FreestyleJob, handoff: dict[str, Any]) -> str:
    result = handoff.get("result", {})
    action = result.get("action", "inspect")
    url = f"/dev/freestyle/{escape(job.job_id)}"
    if action == "apply":
        body = (
            "<p>Inspect the latest generated bundle in Job details and artifacts before applying it.</p>"
            + _goal_fit_warning_block(handoff)
            + f"<form method='post' action='{url}/apply'>"
            + _confirmation(
                "confirm_apply",
                "apply-latest-generated",
                "I explicitly confirm applying the latest generated bundle to the isolated worktree.",
            )
            + "<button type='submit'>Apply Latest Generated Bundle</button></form>"
        )
    elif action in {"materialize", "generate", "review"}:
        suffix, label = {
            "materialize": ("materialize", "Materialize Isolated Worktree"),
            "generate": ("run", "Run To Generated Bundle"),
            "review": ("review", "Review Current Worktree"),
        }[action]
        body = f"<form method='post' action='{url}/{suffix}'><button type='submit'>{label}</button></form>"
    elif action == "checks":
        body = (
            "<p>Inspect any failures below, then run the appropriate checks. Checks execute trusted repository code.</p>"
            + _checks_form(job)
        )
    elif action == "handoff":
        body = "<p>Review the recorded diff and check results below. Integrate manually outside Lewis only when satisfied.</p><a class='button' href='#review-diff'>Review diff</a>"
    elif action == "wait":
        body = "<p>A stage is running. Refresh its status before taking another action.</p>"
    elif action == "recover":
        body = "<p>Inspect the errors and verify the worktree before using recovery in Job details and artifacts.</p>"
    elif action == "refine":
        body = "<p>Inspect the recorded failure, then use Refine or rerun checks to generate a corrected bundle.</p>"
    else:
        body = "<p>Inspect the recorded errors and job details before continuing. No edit action is available.</p>"
    return (
        "<section class='job-next-action'><h3>Next action</h3>"
        + body
        + f"<a href='{url}'>Refresh Status</a></section>"
        + _ui_review_warning_block(handoff)
        + _repo_map_warning_block(handoff)
    )


def _review_and_checks_actions(job: FreestyleJob, handoff: dict[str, Any]) -> str:
    if (
        job.materialization_status != "materialized"
        or job.approval_state != "approved"
        or handoff.get("blocked", {}).get("running_lock")
        or job.edit_apply_status == "inconsistent"
    ):
        return ""
    return (
        f"<form method='post' action='/dev/freestyle/{escape(job.job_id)}/review'>"
        "<button class='secondary' type='submit'>Review Current Worktree</button></form>"
        + _checks_form(job)
    )


def _review_diff(handoff: dict[str, Any]) -> str:
    review = handoff.get("review_snapshot", {})
    result = handoff.get("result", {})
    parts = ["<section class='band' id='review-diff'><h2>Recorded diff</h2>"]
    parts.append(
        "<p class='muted'>Snapshot from the last review, not a live diff. Rerun review and checks after manual edits.</p>"
    )
    if not result.get("review_current") and review:
        parts.append(
            "<p class='warning'>This review predates the latest edits or generated bundle. Review again after applying the latest bundle.</p>"
        )
    if review.get("generated_at"):
        parts.append(f"<p>Reviewed {escape(review['generated_at'])}</p>")
    remaining = MAX_WEB_PREVIEW_CHARS
    snippets = review.get("diff_snippets", [])
    for snippet in snippets if isinstance(snippets, list) else []:
        if not isinstance(snippet, dict) or not isinstance(
            snippet.get("unified_diff"), str
        ):
            continue
        text = snippet["unified_diff"]
        shown = text[:remaining]
        remaining -= len(shown)
        parts.append(
            f"<h3>{escape(snippet.get('path', ''))}</h3><pre class='patch-preview job-diff'>"
        )
        for line in shown.splitlines(keepends=True):
            style = (
                "diff-addition"
                if line.startswith("+")
                else "diff-deletion"
                if line.startswith("-")
                else "diff-context"
            )
            parts.append(f"<span class='{style}'>{escape(line)}</span>")
        parts.append("</pre>")
        if snippet.get("truncated") or len(shown) < len(text):
            parts.append(
                "<p class='muted'>Diff truncated. Inspect the full change locally.</p>"
            )
        if remaining <= 0:
            parts.append(
                "<p class='muted'>Web diff limit reached; remaining snippets omitted.</p>"
            )
            break
    if not snippets:
        parts.append(
            "<p>No recorded diff is available. Run Review Current Worktree after applying edits.</p>"
        )
    omissions = review.get("files_omitted_from_diff_snippets", [])
    if omissions:
        parts.append(
            "<p class='warning'>Some files were omitted from the recorded diff for safety or size limits. Inspect the review artifact before integration.</p>"
        )
    parts.append("</section>")
    return "".join(parts)


def _refine_form(job: FreestyleJob, running: bool) -> str:
    disabled = (
        job.approval_state != "approved"
        or job.materialization_status != "materialized"
        or job.edit_apply_status == "inconsistent"
        or running
    )
    if disabled:
        return ""
    return (
        "<section class='action-zone primary-zone'><h3>Refine This Job</h3>"
        "<p>This will inspect and modify only this job's isolated worktree. "
        "It will not touch the main checkout, commit, merge, push, pull, fetch, "
        "or use any provider except the one explicitly configured. The default action stops at a "
        "new validated generated bundle.</p>"
        f"<form method='post' action='/dev/freestyle/{escape(job.job_id)}/refine'>"
        "<label class='control'><span>Refinement goal</span>"
        "<textarea name='goal' rows='4' maxlength='10000' required "
        "placeholder='Describe the minimal targeted refinement'></textarea></label>"
        "<label class='control'><span>Observed runtime/UI feedback (optional)</span>"
        "<textarea name='ui_feedback' rows='3' maxlength='4000' "
        "placeholder='Describe any broken rendered layout or UI behavior you saw'>"
        "</textarea></label>"
        f"{_confirmation('developer_approved', 'approved', 'I explicitly approve refining this job in its current isolated worktree.')}"
        "<button type='submit'>Generate Refinement Bundle</button></form></section>"
    )


def _refinement_status(handoff: dict[str, Any]) -> str:
    refinement = handoff.get("refinement", {})
    if not isinstance(refinement, dict) or not refinement.get("count"):
        return ""
    latest = refinement.get("latest_artifacts", {})
    original = refinement.get("original_artifacts", {})
    original_stages = refinement.get("original_stages", {})
    history = refinement.get("history", [])
    rows = "".join(
        field(label.replace("_", " "), path or "none") for label, path in latest.items()
    )
    original_rows = "".join(
        field(f"original {label.replace('_', ' ')}", path or "none")
        for label, path in original.items()
    )
    original_stage_rows = "".join(
        field(f"original {label.replace('_', ' ')} status", status)
        for label, status in original_stages.items()
    )
    items = "".join(
        f"<li>Iteration {escape(item.get('iteration', ''))}: "
        f"{escape(item.get('goal', ''))} ({escape(item.get('status', 'unknown'))})</li>"
        for item in history
        if isinstance(item, dict)
    )
    return (
        "<section class='band'><h2>Refinement History</h2>"
        f"<p>Latest refinement: {escape(refinement.get('latest_goal', ''))}</p>"
        f"<div class='meta'>{status_badge(refinement.get('latest_status', 'unknown'))}"
        f"<span>{escape(refinement.get('count', 0))} iteration(s)</span></div>"
        f"<div class='fields'>{original_stage_rows}{original_rows}{rows}</div>"
        f"{f'<ul>{items}</ul>' if items else ''}</section>"
    )


def _goal_fit_warning_block(handoff: dict[str, Any]) -> str:
    warnings = [
        warning
        for warning in (
            *handoff.get("goal_fit_warnings", []),
            *handoff.get("semantic_warnings", []),
        )
        if isinstance(warning, str) and warning
    ]
    if not warnings:
        return ""
    items = "".join(f"<li>{escape(warning)}</li>" for warning in warnings)
    return (
        "<div class='goal-fit-warnings' role='alert'>"
        "<strong>Goal-fit warnings</strong>"
        f"<ul>{items}</ul>"
        "<p>Advisory only: the bundle passed structural and safety validation, "
        "but it may not actually implement the goal. Review it before applying.</p>"
        "</div>"
    )


def _ui_review_warning_block(handoff: dict[str, Any]) -> str:
    warnings = [
        warning
        for warning in handoff.get("ui_review_warnings", [])
        if isinstance(warning, str) and warning
    ]
    if not warnings:
        return ""
    items = "".join(f"<li>{escape(warning)}</li>" for warning in warnings)
    return (
        "<div class='goal-fit-warnings' role='alert'>"
        "<strong>Rendered UI review warnings</strong>"
        f"<ul>{items}</ul>"
        "<p>Advisory only: the deterministic review flagged likely rendered-layout "
        "or selector problems in the changed web files. Refine with observed UI "
        "feedback to repair them.</p>"
        "</div>"
    )


def _repo_map_warning_block(handoff: dict[str, Any]) -> str:
    warnings = [
        warning
        for warning in handoff.get("repo_map_review_warnings", [])
        if isinstance(warning, str) and warning
    ]
    if not warnings:
        return ""
    items = "".join(f"<li>{escape(warning)}</li>" for warning in warnings)
    return (
        "<div class='goal-fit-warnings' role='alert'>"
        "<strong>Repo map review warnings</strong>"
        f"<ul>{items}</ul>"
        "<p>Advisory only: the persistent repo map links the changed files to "
        "related routes, pages, stylesheets, scripts, or tests that this change "
        "did not consider. The repo map never bypasses validation, approval, or "
        "human merge authority.</p>"
        "</div>"
    )


def _implementation_plan_block(handoff: dict[str, Any]) -> str:
    plan = handoff.get("implementation_plan") or {}
    deviations = [
        warning
        for warning in handoff.get("implementation_plan_warnings", [])
        if isinstance(warning, str) and warning
    ]
    if not plan and not deviations:
        return ""
    parts: list[str] = ["<section class='action-zone'><h3>Implementation Plan</h3>"]
    if plan:
        rows: list[str] = []
        for label, key in (
            ("Task type", "task_type"),
            ("Primary feature", "primary_feature"),
            ("Strategy", "recommended_strategy"),
        ):
            value = plan.get(key)
            if value:
                rows.append(f"<li><strong>{label}:</strong> {escape(str(value))}</li>")
        for label, key in (
            ("Expected files", "expected_files"),
            ("Forbidden files", "forbidden_files"),
            ("Failed prior strategies", "failed_strategy_patterns"),
        ):
            values = plan.get(key) or []
            if values:
                joined = ", ".join(str(item) for item in values)
                rows.append(f"<li><strong>{label}:</strong> {escape(joined)}</li>")
        tests_to_run = (plan.get("validation_plan") or {}).get("tests_to_run") or []
        if tests_to_run:
            joined = ", ".join(str(item) for item in tests_to_run)
            rows.append(f"<li><strong>Validation tests:</strong> {escape(joined)}</li>")
        for warning in plan.get("warnings") or []:
            if isinstance(warning, str) and warning:
                rows.append(
                    f"<li><strong>Plan warning:</strong> {escape(warning)}</li>"
                )
        if rows:
            parts.append("<ul class='plan-summary'>" + "".join(rows) + "</ul>")
        parts.append(
            "<p class='muted'>Deterministic plan built before edit generation. "
            "Advisory: it guides context and the model but never bypasses "
            "validation, approval, or human merge authority.</p>"
        )
    if deviations:
        items = "".join(f"<li>{escape(warning)}</li>" for warning in deviations)
        parts.append(
            "<div class='goal-fit-warnings' role='alert'>"
            "<strong>Implementation plan deviations</strong>"
            f"<ul>{items}</ul>"
            "<p>Advisory only: the applied change appears to contradict the plan "
            "(for example a CSS-only change where DOM/JS wiring was required).</p>"
            "</div>"
        )
    parts.append("</section>")
    return "".join(parts)


def _confirmation(name: str, value: str, label: str, *, required: bool = True) -> str:
    required_attribute = " required" if required else ""
    return (
        "<label class='approval-check action-confirm'>"
        f"<input type='checkbox' name='{escape(name)}' value='{escape(value)}'"
        f"{required_attribute}>"
        f"<span>{escape(label)}</span></label>"
    )


def _checks_form(job: FreestyleJob) -> str:
    choices = "".join(
        "<label class='check-choice'>"
        f"<input type='checkbox' name='check_id' value='{escape(definition.check_id)}'>"
        "<span>"
        f"<strong>{escape(definition.check_id)}</strong>"
        f"<small>{escape(definition.description)}</small>"
        "</span></label>"
        for definition in CHECK_REGISTRY
    )
    return (
        f"<form method='post' action='/dev/freestyle/{escape(job.job_id)}/run-checks'>"
        "<p>Run only selected static backend checks in the isolated worktree. "
        "No command, arguments, executable, cwd, environment, or shell input is accepted.</p>"
        f"<fieldset><legend>Allowlisted checks</legend>{choices}</fieldset>"
        "<button class='secondary' type='submit'>Run Selected Checks</button></form>"
    )


def _danger_actions(job: FreestyleJob) -> str:
    job_id = escape(job.job_id)
    return (
        "<details class='danger-zone'><summary>Advanced Maintenance And Danger Actions</summary>"
        "<div class='danger-grid'>"
        f"<form method='post' action='/dev/freestyle/{job_id}/recover'>"
        "<h3>Recover Stale Lock State</h3>"
        "<p class='warning'>Recovery clears only stale/inconsistent lock metadata "
        "to failed or error states. It never marks interrupted work successful.</p>"
        f"{_confirmation('confirm_recover', 'recover', 'I confirm this operation is abandoned and should be recovered to failed/error state.')}"
        f"{_confirmation('confirm_inconsistent', 'verified-inconsistent', 'Stronger confirmation: I manually verified the isolated worktree before recovering inconsistent apply state.', required=False)}"
        "<button class='warn' type='submit'>Recover Stale State</button></form>"
        f"<form method='post' action='/dev/freestyle/{job_id}/cleanup'>"
        "<h3>Cleanup Selected Job</h3>"
        "<p class='warning'>Cleanup removes this validated isolated worktree, its "
        "matching local freestyle branch, local artifacts, and job metadata. "
        "It does not touch the main checkout or any remote.</p>"
        f"{_confirmation('confirm_cleanup', 'cleanup', 'I explicitly confirm cleanup of this selected local freestyle job.')}"
        f"{_confirmation('discard_worktree_changes', 'discard', 'Separate confirmation: discard unreviewed isolated-worktree changes or local commits if present.', required=False)}"
        "<button class='warn' type='submit'>Cleanup Selected Job</button></form>"
        "</div></details>"
    )


def _stage_status(handoff: dict[str, Any]) -> str:
    return (
        "<section class='band'><h2>Stage Status</h2><div class='stage-grid'>"
        + "".join(
            "<div class='stage-item'>"
            f"<span>{escape(label)}</span>{status_badge(status)}</div>"
            for label, status in handoff.get("stages", {}).items()
        )
        + "</div></section>"
    )


def _artifact_paths_and_previews(
    handoff: dict[str, Any], previews: dict[str, str]
) -> str:
    artifacts = handoff.get("artifacts", {})
    if not any(artifacts.values()):
        return (
            "<section class='band'><h2>Latest Artifacts</h2>"
            "<p class='muted'>No artifacts yet.</p></section>"
        )
    rows = "".join(
        field(label.replace("_", " "), path or "none")
        for label, path in artifacts.items()
    )
    details = "".join(
        "<details class='artifact-preview'>"
        f"<summary>{escape(label.replace('_', ' '))} preview</summary>"
        f"<pre class='patch-preview'>{escape(preview)}</pre></details>"
        for label, preview in previews.items()
    )
    return (
        "<section class='band'><h2>Latest Artifacts</h2>"
        f"<div class='fields'>{rows}</div>{details}</section>"
    )


def _changed_files(handoff: dict[str, Any]) -> str:
    changed = handoff.get("changed_files", {})
    files = changed.get("files", [])
    rows = "".join(
        "<tr>"
        f"<td>{escape(item.get('path', ''))}</td>"
        f"<td>{escape(item.get('status', 'unknown'))}</td>"
        f"<td>{escape(item.get('git_status', ''))}</td></tr>"
        for item in files
    )
    table = (
        "<div class='table-scroll'><table><thead><tr><th>Path</th><th>Status</th>"
        f"<th>Git</th></tr></thead><tbody>{rows}</tbody></table></div>"
        if files
        else "<p class='muted'>No changed files available.</p>"
    )
    return (
        "<section class='band'><h2>Changed Files</h2>"
        f"<div class='meta'>{status_badge(changed.get('source', 'unknown'))}"
        f"<span>{escape(changed.get('count', 'unknown'))} files</span></div>{table}</section>"
    )


def _check_summary(handoff: dict[str, Any]) -> str:
    checks = handoff.get("checks", {})
    result = handoff.get("result", {})
    parts = [
        "<section class='band' id='check-results'><h2>Check results</h2>",
        f"<p><strong>{escape(checks.get('overall_status', 'not_run'))}</strong> · {escape(checks.get('passed', 0))} passed · {escape(checks.get('failed', 0))} failed · {escape(checks.get('timeout', 0))} timed out · {escape(checks.get('infra_error', 0))} infrastructure errors</p>",
    ]
    if result.get("check_attempt_status") != checks.get("overall_status"):
        parts.append(
            f"<p class='warning'>Latest check attempt: {escape(result.get('check_attempt_status', 'unknown'))}. The saved results below are from an earlier attempt.</p>"
        )
    if result.get("checked_at"):
        parts.append(
            f"<p class='muted'>Recorded {escape(result['checked_at'])}. These are saved results, not a live verification.</p>"
        )
    if not result.get("checks_current"):
        parts.append(
            "<p class='warning'>No check results cover the latest edits or generated bundle. Run checks after applying it.</p>"
        )
    if checks.get("modified_worktree"):
        parts.append(
            "<p class='warning'>Checks changed files in the worktree. Review those changes before proceeding.</p>"
        )
    results = handoff.get("check_results", [])
    for check in results[: len(CHECK_REGISTRY)] if isinstance(results, list) else []:
        if not isinstance(check, dict):
            continue
        parts.append(
            f"<details><summary>{escape(check.get('check_id', 'Check'))}: {escape(check.get('status', 'unknown'))}</summary>"
        )
        parts.append(
            f"<p>Exit code: {escape(check.get('exit_code'))} · Duration: {escape(check.get('duration_seconds'))} seconds</p>"
        )
        if check.get("note"):
            parts.append(f"<p>{escape(check['note'])}</p>")
        for stream in ("stdout", "stderr"):
            text = str(check.get(stream, ""))
            if text:
                parts.append(
                    f"<h3>{stream}</h3><pre class='patch-preview'>{escape(text[:MAX_WEB_PREVIEW_CHARS])}</pre>"
                )
            if check.get(f"{stream}_truncated") or len(text) > MAX_WEB_PREVIEW_CHARS:
                parts.append("<p>Output truncated.</p>")
        parts.append("</details>")
    parts.append("</section>")
    return "".join(parts)


def _handoff_lists(handoff: dict[str, Any]) -> str:
    return (
        "<section class='grid freestyle-handoff'>"
        f"{_list_card('Blockers', handoff.get('blocked', {}).get('reasons', []), 'No blockers recorded.')}"
        f"{_list_card('Errors And Risks', [*handoff.get('bounded_errors', []), *handoff.get('risks', [])], 'No errors or risks recorded.')}"
        f"{_list_card('Next Manual Steps', handoff.get('human_next_steps', []), 'No next steps recorded.', ordered=True)}"
        "</section>"
    )


def _list_card(
    title: str, items: list[Any], empty: str, *, ordered: bool = False
) -> str:
    if items:
        tag = "ol" if ordered else "ul"
        body = (
            f"<{tag}>"
            + "".join(f"<li>{escape(item)}</li>" for item in items)
            + f"</{tag}>"
        )
    else:
        body = f"<p class='muted'>{escape(empty)}</p>"
    return f"<article class='card'><h3>{escape(title)}</h3>{body}</article>"
