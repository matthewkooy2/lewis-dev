"""Developer-only web UI for bounded read-only coding-agent jobs."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs, urlencode

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.dev.coding_agents import (
    CLAUDE_EFFORTS,
    KNOWN_PROVIDERS,
    CodingAgentError,
    CodingAgentEventRecord,
    CodingAgentJob,
    CodingAgentService,
    LaunchRequest,
    ProviderStatus,
)
from app.web_components import (
    escape,
    field,
    flash,
    page_html,
    page_header,
    status_badge,
)

MAX_WEB_REQUEST_BYTES = 20_000
READ_ONLY_LABEL = "Read-only analysis and proposal jobs. File modifications and Git integration are disabled."


def register_coding_agent_web_routes(app: FastAPI) -> None:
    @app.get("/dev/coding-agents", response_class=HTMLResponse)
    def coding_agents_home(
        request: Request, message: str = "", error: str = "", project_id: str = ""
    ) -> HTMLResponse:
        if set(request.query_params) - {"message", "error", "project_id"}:
            return _page_response(
                request,
                error="Coding Agents views only accept status-message parameters.",
                status_code=400,
            )
        return _page_response(
            request, message=message, error=error, project_id=project_id
        )

    @app.get("/dev/coding-agents/{job_id}", response_class=HTMLResponse)
    def coding_agent_detail(
        request: Request, job_id: str, message: str = "", error: str = ""
    ) -> HTMLResponse:
        if set(request.query_params) - {"message", "error"}:
            return _page_response(
                request,
                selected_job_id=job_id,
                error="Coding Agents views only accept status-message parameters.",
                status_code=400,
            )
        return _page_response(
            request, selected_job_id=job_id, message=message, error=error
        )

    @app.post("/dev/coding-agents/status")
    async def refresh_status(request: Request) -> RedirectResponse:
        fields = await _request_fields(request)
        if set(fields) - {"confirm_status"}:
            return _redirect(
                error="Status refresh accepts only its confirmation field."
            )
        if fields.get("confirm_status") != "refresh-provider-status":
            return _redirect(error="Explicit status-refresh confirmation is required.")
        try:
            _service(request).refresh_provider_statuses()
        except Exception as exc:
            return _redirect(error=_web_error(exc))
        return _redirect(
            message="Provider status refreshed with fixed local status commands."
        )

    @app.post("/dev/coding-agents/launch")
    async def launch_agent(request: Request) -> RedirectResponse:
        fields = await _request_fields(request)
        allowed = {
            "repository_id",
            "provider",
            "goal",
            "model",
            "effort",
            "max_turns",
            "timeout_seconds",
            "launch_confirm",
            "project_id",
        }
        if set(fields) - allowed:
            return _redirect(
                error="Launch accepts only the bounded Coding Agents form fields."
            )
        if fields.get("launch_confirm") != "read-only-analysis":
            return _redirect(
                error="Explicit confirmation is required before launching a read-only job."
            )
        try:
            job = _service(request).launch(
                LaunchRequest(
                    repository_id=str(fields.get("repository_id", "")),
                    provider=str(fields.get("provider", "")),
                    goal=str(fields.get("goal", "")),
                    model=str(fields.get("model", "")),
                    effort=str(fields.get("effort", "")),
                    max_turns=_int_field(fields.get("max_turns"), 6),
                    timeout_seconds=_int_field(fields.get("timeout_seconds"), 600),
                    project_id=str(fields.get("project_id", "")) or None,
                )
            )
        except Exception as exc:
            return _redirect(error=_web_error(exc))
        return _redirect(
            job.id, message=f"Launched read-only {job.provider} job {job.id[:12]}."
        )

    @app.post("/dev/coding-agents/{job_id}/stop")
    async def stop_agent(request: Request, job_id: str) -> RedirectResponse:
        fields = await _request_fields(request)
        if set(fields) - {"confirm_stop"}:
            return _redirect(
                job_id, error="Stop accepts only explicit stop confirmation."
            )
        if fields.get("confirm_stop") != "stop-running-job":
            return _redirect(job_id, error="Explicit stop confirmation is required.")
        try:
            _service(request).cancel(job_id)
        except Exception as exc:
            return _redirect(job_id, error=_web_error(exc))
        return _redirect(job_id, message="Stop requested for the selected job.")


def _service(request: Request) -> CodingAgentService:
    service = getattr(request.app.state, "coding_agent_service", None)
    if service is None:
        service = CodingAgentService(request.app.state.settings)
        request.app.state.coding_agent_service = service
    return service


def _page_response(
    request: Request,
    *,
    selected_job_id: str | None = None,
    message: str = "",
    error: str = "",
    project_id: str = "",
    status_code: int = 200,
) -> HTMLResponse:
    service = _service(request)
    jobs = service.store.list_jobs()
    selected = None
    events: list[CodingAgentEventRecord] = []
    if selected_job_id:
        selected = service.store.get_job(selected_job_id)
        if selected is None:
            error = "Selected Coding Agents job was not found."
        else:
            events = service.store.list_events(selected.id)
    repositories = service.repositories()
    preselected_repository_id = ""
    preselected_project_id = ""
    if project_id:
        try:
            from app.projects import ProjectStore

            project = ProjectStore(
                request.app.state.settings.database_path
            ).require_project(project_id)
            for repo in repositories:
                if str(repo.root).casefold() == str(project.repository_root).casefold():
                    preselected_repository_id = repo.repo_id
                    preselected_project_id = project.id
                    break
            if not preselected_repository_id:
                error = "Project repository is not in the Coding Agents approved repository list."
        except Exception as exc:
            error = _web_error(exc)
    content = coding_agents_page(
        repositories=repositories,
        statuses=service.provider_statuses(),
        jobs=jobs,
        selected_job=selected,
        events=events,
        message=message,
        error=error,
        preselected_repository_id=preselected_repository_id,
        preselected_project_id=preselected_project_id,
    )
    return HTMLResponse(
        page_html(
            "Coding Agents",
            request.url.path,
            content,
            show_dev_navigation=request.app.state.settings.freestyle_ui_enabled,
            show_projects_navigation=request.app.state.settings.projects_ui_enabled,
            show_coding_agents_navigation=True,
        ),
        status_code=status_code,
    )


def coding_agents_page(
    *,
    repositories: list[Any],
    statuses: dict[str, ProviderStatus],
    jobs: list[CodingAgentJob],
    selected_job: CodingAgentJob | None,
    events: list[CodingAgentEventRecord],
    message: str = "",
    error: str = "",
    preselected_repository_id: str = "",
    preselected_project_id: str = "",
) -> str:
    return "".join(
        [
            page_header(
                "Coding Agents",
                READ_ONLY_LABEL,
                "<a class='button' href='/dev/coding-agents'>Refresh</a>",
            ),
            _safety_notice(),
            flash(message),
            _error_notice(error),
            _status_panel(statuses),
            "<section class='freestyle-layout'><div class='freestyle-rail'>",
            _launch_form(
                repositories, preselected_repository_id, preselected_project_id
            ),
            _jobs_list(jobs, selected_job.id if selected_job else None),
            "</div><div class='freestyle-detail'>",
            _job_detail(selected_job, events),
            "</div></section>",
        ]
    )


def _safety_notice() -> str:
    return (
        "<section class='safety-notice'><strong>Phase 1 boundary</strong>"
        f"<p>{escape(READ_ONLY_LABEL)} Lewis never accepts shell commands, provider argv, cwd, environment variables, Git actions, or executable paths from this page.</p>"
        "</section>"
    )


def _error_notice(error: str) -> str:
    if not error:
        return ""
    return (
        "<div class='inline-error' role='alert'><strong>Coding Agents action stopped</strong>"
        f"<span>{escape(error)}</span></div>"
    )


def _status_panel(statuses: dict[str, ProviderStatus]) -> str:
    cards = []
    for provider in sorted(KNOWN_PROVIDERS):
        status = statuses[provider]
        auth = (
            "not checked"
            if status.authenticated is None
            else ("yes" if status.authenticated else "no")
        )
        cards.append(
            "<article class='card'>"
            f"<h3>{escape(provider.title())}</h3>"
            f"<div class='fields'>{field('installed', 'yes' if status.installed else 'no')}"
            f"{field('executable', status.executable_path or 'unavailable')}"
            f"{field('version', status.version or 'not checked')}"
            f"{field('authenticated', auth)}"
            f"{field('auth status', status.auth_status)}"
            f"{field('usage status', status.usage_status)}"
            f"{field('checked', status.checked_at or 'not checked')}</div>"
            "</article>"
        )
    return (
        "<section class='band'><div class='card-head'><h2>Provider Status</h2>"
        "<form method='post' action='/dev/coding-agents/status'>"
        "<input type='hidden' name='confirm_status' value='refresh-provider-status'>"
        "<button class='secondary' type='submit'>Refresh Provider Status</button></form></div>"
        "<p class='muted'>Opening this page does not launch provider status commands; refresh runs fixed local status probes only.</p>"
        "<section class='grid'>" + "".join(cards) + "</section></section>"
    )


def _launch_form(
    repositories: list[Any],
    preselected_repository_id: str = "",
    preselected_project_id: str = "",
) -> str:
    repo_options = "".join(
        f"<option value='{escape(repo.repo_id)}'{' selected' if repo.repo_id == preselected_repository_id else ''}>{escape(repo.label)} - {escape(repo.root)}</option>"
        for repo in repositories
    )
    if not repo_options:
        repo_options = "<option value=''>No approved repositories configured</option>"
    effort_options = "<option value=''>Codex or default</option>" + "".join(
        f"<option value='{escape(effort)}'>{escape(effort)}</option>"
        for effort in sorted(CLAUDE_EFFORTS)
    )
    return (
        "<section class='band freestyle-create'><h2>Launch Read-Only Job</h2>"
        "<form method='post' action='/dev/coding-agents/launch'>"
        f"<input type='hidden' name='project_id' value='{escape(preselected_project_id)}'>"
        "<label class='control'><span>Approved repository</span>"
        f"<select name='repository_id' required>{repo_options}</select></label>"
        "<label class='control'><span>Provider</span><select name='provider' required>"
        "<option value='codex'>Codex CLI</option><option value='claude'>Claude Code CLI</option>"
        "</select></label>"
        "<label class='control'><span>Model</span><input name='model' maxlength='120' placeholder='optional-provider-model'></label>"
        "<label class='control'><span>Claude effort</span>"
        f"<select name='effort'>{effort_options}</select></label>"
        "<div class='form-grid'>"
        "<label class='control'><span>Max turns</span><input name='max_turns' type='number' min='1' max='20' value='6'></label>"
        "<label class='control'><span>Timeout seconds</span><input name='timeout_seconds' type='number' min='30' max='3600' value='600'></label>"
        "</div>"
        "<label class='control'><span>Read-only goal</span>"
        "<textarea name='goal' rows='6' maxlength='10000' required></textarea></label>"
        "<label class='approval-check action-confirm'><input type='checkbox' name='launch_confirm' value='read-only-analysis' required>"
        "<span>I confirm this is a read-only analysis/proposal job and must not modify files or Git state.</span></label>"
        "<button type='submit'>Launch Read-Only Job</button>"
        "</form></section>"
    )


def _jobs_list(jobs: list[CodingAgentJob], selected_id: str | None) -> str:
    if not jobs:
        return "<section class='band'><h2>Active And Recent Jobs</h2><p class='muted'>No coding-agent jobs yet.</p></section>"
    rows = []
    for job in jobs:
        selected = " selected" if job.id == selected_id else ""
        rows.append(
            f"<a class='job-row{selected}' href='/dev/coding-agents/{escape(job.id)}'>"
            "<span>"
            f"<strong>{escape(job.provider)} - {escape(job.repository_label)}</strong>"
            f"<small>{escape(job.goal[:140])}</small>"
            f"<small>{escape(job.updated_at)}</small>"
            "</span>"
            f"{status_badge(job.state)}</a>"
        )
    return (
        "<section class='band'><h2>Active And Recent Jobs</h2><div class='job-list'>"
        + "".join(rows)
        + "</div></section>"
    )


def _job_detail(
    job: CodingAgentJob | None, events: list[CodingAgentEventRecord]
) -> str:
    if job is None:
        return "<section class='empty-state'><p>Select a job to inspect its bounded event timeline.</p></section>"
    return (
        "<section class='band freestyle-summary'><div class='card-head'>"
        f"<h2>{escape(job.provider)} job</h2><div class='meta'>{status_badge(job.state)}</div></div>"
        f"<p>{escape(job.goal)}</p><div class='fields'>"
        f"{field('job id', job.id)}{field('repository', job.repository_root)}"
        f"{field('model', job.model or '')}{field('effort', job.effort or '')}"
        f"{field('external session/thread', job.external_session_id or '')}"
        f"{field('duration seconds', job.duration_seconds if job.duration_seconds is not None else '')}"
        f"{field('failure category', job.failure_category or '')}</div>"
        f"{_stop_form(job)}"
        "</section>" + _usage_summary(job) + _event_timeline(events)
    )


def _stop_form(job: CodingAgentJob) -> str:
    if job.state not in {"queued", "running"}:
        return ""
    return (
        f"<form class='actions' method='post' action='/dev/coding-agents/{escape(job.id)}/stop'>"
        "<input type='hidden' name='confirm_stop' value='stop-running-job'>"
        "<button class='warn' type='submit'>Stop Job</button></form>"
    )


def _usage_summary(job: CodingAgentJob) -> str:
    return (
        "<section class='band'><h2>Usage Summary</h2><div class='fields'>"
        f"{field('input tokens', job.input_tokens if job.input_tokens is not None else 'unavailable')}"
        f"{field('cached input tokens', job.cached_input_tokens if job.cached_input_tokens is not None else 'unavailable')}"
        f"{field('output tokens', job.output_tokens if job.output_tokens is not None else 'unavailable')}"
        f"{field('reasoning tokens', job.reasoning_tokens if job.reasoning_tokens is not None else 'unavailable')}"
        f"{field('token source', job.token_usage_source or 'unavailable')}"
        f"{field('cost', job.estimated_cost_usd if job.estimated_cost_usd is not None else 'unavailable')}"
        f"{field('cost source', job.cost_source or 'unavailable')}</div></section>"
    )


def _event_timeline(events: list[CodingAgentEventRecord]) -> str:
    if not events:
        return "<section class='band'><h2>Event Timeline</h2><p class='muted'>No events recorded yet.</p></section>"
    rows = "".join(
        "<tr>"
        f"<td>{escape(event.sequence)}</td>"
        f"<td>{escape(event.created_at)}</td>"
        f"<td>{escape(event.event_type)}</td>"
        f"<td>{escape(event.summary)}</td>"
        f"<td><code>{escape(json.dumps(event.content, sort_keys=True))}</code></td>"
        "</tr>"
        for event in events
    )
    return (
        "<section class='band'><h2>Bounded Event Timeline</h2>"
        "<div class='table-scroll'><table><thead><tr><th>#</th><th>Time</th>"
        "<th>Type</th><th>Summary</th><th>Content</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div></section>"
    )


def _redirect(
    job_id: str | None = None, *, message: str = "", error: str = ""
) -> RedirectResponse:
    path = f"/dev/coding-agents/{job_id}" if job_id else "/dev/coding-agents"
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


def _int_field(value: object, default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _web_error(exc: Exception) -> str:
    if isinstance(exc, CodingAgentError):
        return str(exc)[:500]
    return f"Coding Agents action could not be completed: {str(exc)[:400]}"
