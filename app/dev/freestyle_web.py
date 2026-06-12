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
    create_freestyle_job,
    get_freestyle_handoff,
    get_freestyle_job,
    list_freestyle_jobs,
    materialize_freestyle_job,
    run_freestyle_job,
)
from app.dev.freestyle_policy import is_job_id
from app.dev.freestyle_proposal import DeepSeekProposalConfig
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
    async def run_freestyle_job_web(
        request: Request, job_id: str
    ) -> RedirectResponse:
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
    )
    return HTMLResponse(
        page_html(
            "Freestyle Command Center",
            request.url.path,
            content,
            show_dev_navigation=True,
        ),
        status_code=status_code,
    )


def _project_root(request: Request) -> Path:
    return Path(getattr(request.app.state, "freestyle_project_root", ROOT_DIR)).resolve()


def _deepseek_config(request: Request) -> DeepSeekProposalConfig:
    settings = request.app.state.settings
    return DeepSeekProposalConfig(
        api_key=settings.deepseek_api_key or "",
        model=settings.deepseek_code_model or "",
        base_url=settings.deepseek_base_url,
    )


def _web_error(request: Request, exc: Exception) -> str:
    text = str(exc)
    api_key = request.app.state.settings.deepseek_api_key
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
        return data if isinstance(data, dict) else {"invalid_request_body": "not_object"}
    decoded_body = raw_body.decode("utf-8", errors="ignore")
    return {
        key: values[-1] if values else ""
        for key, values in parse_qs(decoded_body, keep_blank_values=True).items()
    }


def freestyle_page(
    jobs: list[FreestyleJob],
    *,
    selected_job: FreestyleJob | None = None,
    handoff: dict[str, Any] | None = None,
    previews: dict[str, str] | None = None,
    message: str = "",
    error: str = "",
) -> str:
    return "".join(
        [
            page_header(
                "Freestyle Command Center",
                "Inspect and lightly operate approved, isolated-worktree coding jobs.",
                "<a class='button' href='/dev/freestyle'>Refresh</a>",
            ),
            _safety_notice(),
            flash(message),
            _error_notice(error),
            "<section class='freestyle-layout'><div class='freestyle-rail'>",
            _create_form(),
            _job_list(jobs, selected_job.job_id if selected_job else None),
            "</div><div class='freestyle-detail'>",
            _job_detail(selected_job, handoff, previews or {}),
            "</div></section>",
        ]
    )


def read_artifact_previews(
    handoff: dict[str, Any], state_root: Path
) -> dict[str, str]:
    previews: dict[str, str] = {}
    for label, reference in handoff.get("artifacts", {}).items():
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


def _error_notice(error: str) -> str:
    if not error:
        return ""
    return (
        "<div class='inline-error' role='alert'><strong>Freestyle action stopped</strong>"
        f"<span>{escape(error)}</span></div>"
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
        "<span>I explicitly approve creating this local isolated-worktree job.</span>"
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
) -> str:
    if job is None or handoff is None:
        return empty_state_html("Select a job to inspect its status and handoff.")
    return "".join(
        [
            "<section class='band freestyle-summary'><div class='card-head'>",
            f"<h2>{escape(job.title)}</h2>",
            f"<div class='meta'>{status_badge(job.status)}"
            f"{status_badge(job.approval_state)}</div></div>",
            f"<p>{escape(job.goal)}</p><div class='fields'>",
            field("job id", job.job_id),
            field("base commit", job.base_commit),
            field("isolated branch", job.branch),
            field("isolated worktree", job.worktree_path),
            field("updated", job.updated_at),
            "</div>",
            _allowed_actions(job, handoff),
            "</section>",
            _stage_status(handoff),
            _artifact_paths_and_previews(handoff, previews),
            _changed_files(handoff),
            _check_summary(handoff),
            _handoff_lists(handoff),
        ]
    )


def _allowed_actions(job: FreestyleJob, handoff: dict[str, Any]) -> str:
    running = bool(handoff.get("blocked", {}).get("running_lock"))
    actions = [
        f"<a class='button secondary' href='/dev/freestyle/{escape(job.job_id)}'>Refresh Status</a>"
    ]
    if (
        job.approval_state == "approved"
        and job.status == "ready"
        and job.materialization_status == "not_materialized"
        and not running
    ):
        actions.append(
            f"<form method='post' action='/dev/freestyle/{escape(job.job_id)}/materialize'>"
            "<button class='secondary' type='submit'>Materialize Isolated Worktree</button>"
            "</form>"
        )
    if (
        job.approval_state == "approved"
        and job.edit_apply_status != "applied"
        and job.status in {"ready", "materialized"}
        and not running
    ):
        actions.append(
            f"<form method='post' action='/dev/freestyle/{escape(job.job_id)}/run'>"
            "<button type='submit'>Run To Generated Bundle</button></form>"
        )
    return (
        "<div class='actions'>" + "".join(actions) + "</div>"
        "<p class='muted'>Run stops before apply, review, and checks. Apply, checks, "
        "cleanup, and recovery are intentionally unavailable here.</p>"
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
    return (
        "<section class='band'><h2>Check Summary</h2>"
        f"<div class='meta'>{status_badge(checks.get('overall_status', 'not_run'))}</div>"
        "<div class='fields'>"
        f"{field('selected', ', '.join(checks.get('selected_ids', [])) or 'none')}"
        f"{field('passed', checks.get('passed', 0))}"
        f"{field('failed', checks.get('failed', 0))}"
        f"{field('timed out', checks.get('timeout', 0))}"
        f"{field('infrastructure errors', checks.get('infra_error', 0))}"
        f"{field('modified isolated worktree', checks.get('modified_worktree', False))}"
        "</div></section>"
    )


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
