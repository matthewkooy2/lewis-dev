"""Feature-gated Projects web workspace."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs, urlencode

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.projects import (
    ProjectError,
    ProjectMapService,
    ProjectStore,
    approved_project_repositories,
    configured_project_repository_options,
    validate_project_repository,
)
from app.web_components import (
    empty_state_html,
    escape,
    field,
    flash,
    page_header,
    page_html,
    status_badge,
    tag_list_html,
)

MAX_PROJECT_WEB_REQUEST_BYTES = 260_000
PROJECT_TABS = ("overview", "notes", "repo-map", "diagrams", "activity", "trace")


def register_project_web_routes(app: FastAPI) -> None:
    @app.get("/projects", response_class=HTMLResponse)
    def projects_home(
        request: Request,
        q: str = "",
        status: str = "active",
        tag: str = "",
        message: str = "",
        error: str = "",
    ) -> HTMLResponse:
        if set(request.query_params) - {"q", "status", "tag", "message", "error"}:
            return _page(
                request,
                "Projects",
                projects_page(
                    request, error="Projects only accepts search and filter parameters."
                ),
                status_code=400,
            )
        store = _store(request)
        projects = store.list_projects(
            query=q, status=status, tag=tag, include_archived=status == "all"
        )
        return _page(
            request,
            "Projects",
            projects_page(
                request,
                projects=projects,
                query=q,
                status=status,
                tag=tag,
                message=message,
                error=error,
            ),
        )

    @app.get("/projects/new", response_class=HTMLResponse)
    def new_project(
        request: Request, message: str = "", error: str = ""
    ) -> HTMLResponse:
        return _page(
            request,
            "New Project",
            new_project_page(request, message=message, error=error),
        )

    @app.post("/projects/validate")
    async def validate_project(request: Request) -> HTMLResponse:
        fields = await _request_fields(request)
        if set(fields) - {"repository_id", "repository_path"}:
            return _page(
                request,
                "New Project",
                new_project_page(
                    request,
                    error="Repository validation accepts only repository fields.",
                ),
                status_code=400,
            )
        try:
            repo = validate_project_repository(
                request.app.state.settings,
                repository_id=str(fields.get("repository_id", "")),
                repository_path=str(fields.get("repository_path", "")),
            )
            message = f"Validated approved Git repository: {repo.root}"
            return _page(
                request,
                "New Project",
                new_project_page(
                    request,
                    message=message,
                    selected_repo_id=repo.repo_id,
                    repository_path=str(repo.root),
                ),
            )
        except Exception as exc:
            return _page(
                request,
                "New Project",
                new_project_page(request, error=_web_error(exc)),
                status_code=400,
            )

    @app.post("/projects")
    async def create_project(request: Request) -> RedirectResponse:
        fields = await _request_fields(request)
        allowed = {
            "display_name",
            "repository_id",
            "repository_path",
            "description",
            "tags",
            "status",
            "color",
            "icon",
        }
        if set(fields) - allowed:
            return _redirect(
                "/projects/new",
                error="New Project accepts only bounded project fields.",
            )
        try:
            repo = validate_project_repository(
                request.app.state.settings,
                repository_id=str(fields.get("repository_id", "")),
                repository_path=str(fields.get("repository_path", "")),
            )
            project = _store(request).create_project(
                display_name=str(fields.get("display_name", "")),
                repository=repo,
                description=str(fields.get("description", "")),
                tags=_tags_field(fields.get("tags", "")),
                status=str(fields.get("status", "active")),
                color=str(fields.get("color", "")),
                icon=str(fields.get("icon", "")),
            )
        except Exception as exc:
            return _redirect("/projects/new", error=_web_error(exc))
        return _redirect(f"/projects/{project.id}", message="Project created.")

    @app.get("/projects/{project_id}", response_class=HTMLResponse)
    def project_detail(
        request: Request,
        project_id: str,
        tab: str = "overview",
        q: str = "",
        language: str = "",
        role: str = "",
        symbol_type: str = "",
        file: str = "",
        category: str = "",
        sort: str = "updated",
        include_archived: str = "",
        message: str = "",
        error: str = "",
    ) -> HTMLResponse:
        if tab not in PROJECT_TABS:
            tab = "overview"
        try:
            content = project_detail_page(
                request,
                project_id,
                tab=tab,
                query=q,
                language=language,
                role=role,
                symbol_type=symbol_type,
                selected_file=file,
                category=category,
                sort=sort,
                include_archived=include_archived == "1",
                message=message,
                error=error,
            )
        except Exception as exc:
            return _page(
                request,
                "Projects",
                projects_page(request, error=_web_error(exc)),
                status_code=404,
            )
        return _page(request, "Project", content)

    @app.post("/projects/{project_id}/metadata")
    async def update_metadata(request: Request, project_id: str) -> RedirectResponse:
        fields = await _request_fields(request)
        allowed = {"display_name", "description", "tags", "status", "color", "icon"}
        if set(fields) - allowed:
            return _project_redirect(
                project_id,
                error="Metadata update accepts only project metadata fields.",
            )
        try:
            _store(request).update_project(
                project_id,
                display_name=str(fields.get("display_name", "")),
                description=str(fields.get("description", "")),
                tags=_tags_field(fields.get("tags", "")),
                status=str(fields.get("status", "active")),
                color=str(fields.get("color", "")),
                icon=str(fields.get("icon", "")),
            )
        except Exception as exc:
            return _project_redirect(project_id, error=_web_error(exc))
        return _project_redirect(project_id, message="Project metadata updated.")

    @app.post("/projects/{project_id}/archive")
    async def archive_project(request: Request, project_id: str) -> RedirectResponse:
        fields = await _request_fields(request)
        try:
            _store(request).archive_project(
                project_id, confirm=str(fields.get("confirm_archive", ""))
            )
        except Exception as exc:
            return _project_redirect(project_id, error=_web_error(exc))
        return _redirect(
            "/projects?status=archived",
            message="Project archived. Notes and map records were preserved.",
        )

    @app.post("/projects/{project_id}/restore")
    async def restore_project(request: Request, project_id: str) -> RedirectResponse:
        try:
            _store(request).restore_project(project_id)
        except Exception as exc:
            return _project_redirect(project_id, error=_web_error(exc))
        return _project_redirect(project_id, message="Project restored.")

    @app.post("/projects/{project_id}/notes")
    async def create_note(request: Request, project_id: str) -> RedirectResponse:
        fields = await _request_fields(request)
        try:
            _store(request).create_note(
                project_id,
                title=str(fields.get("title", "")),
                content_markdown=str(fields.get("content_markdown", "")),
                category=str(fields.get("category", "")),
                tags=_tags_field(fields.get("tags", "")),
                is_pinned=str(fields.get("is_pinned", "")) == "1",
            )
        except Exception as exc:
            return _project_redirect(project_id, tab="notes", error=_web_error(exc))
        return _project_redirect(project_id, tab="notes", message="Note created.")

    @app.post("/projects/{project_id}/notes/{note_id}/update")
    async def update_note(
        request: Request, project_id: str, note_id: str
    ) -> RedirectResponse:
        fields = await _request_fields(request)
        try:
            _store(request).update_note(
                project_id,
                note_id,
                title=str(fields.get("title", "")),
                content_markdown=str(fields.get("content_markdown", "")),
                category=str(fields.get("category", "")),
                tags=_tags_field(fields.get("tags", "")),
                is_pinned=str(fields.get("is_pinned", "")) == "1",
            )
        except Exception as exc:
            return _project_redirect(project_id, tab="notes", error=_web_error(exc))
        return _project_redirect(project_id, tab="notes", message="Note updated.")

    @app.post("/projects/{project_id}/notes/{note_id}/{action}")
    async def note_action(
        request: Request, project_id: str, note_id: str, action: str
    ) -> RedirectResponse:
        fields = await _request_fields(request)
        store = _store(request)
        try:
            if action == "duplicate":
                store.duplicate_note(project_id, note_id)
                message = "Note duplicated."
            elif action == "archive":
                store.archive_note(project_id, note_id)
                message = "Note archived."
            elif action == "restore":
                store.restore_note(project_id, note_id)
                message = "Note restored."
            elif action == "delete":
                store.delete_note_permanently(
                    project_id, note_id, confirm=str(fields.get("confirm_delete", ""))
                )
                message = "Note permanently deleted."
            else:
                raise ProjectError("Unknown note action.")
        except Exception as exc:
            return _project_redirect(project_id, tab="notes", error=_web_error(exc))
        return _project_redirect(project_id, tab="notes", message=message)

    @app.post("/projects/{project_id}/map/refresh")
    async def refresh_map(request: Request, project_id: str) -> RedirectResponse:
        fields = await _request_fields(request)
        if str(fields.get("confirm_refresh", "")) != "refresh-map":
            return _project_redirect(
                project_id,
                tab="repo-map",
                error="Explicit map refresh confirmation is required.",
            )
        try:
            run = _map_service(request).start_refresh(
                project_id, full_rebuild=str(fields.get("full_rebuild", "")) == "1"
            )
        except Exception as exc:
            return _project_redirect(project_id, tab="repo-map", error=_web_error(exc))
        return _project_redirect(
            project_id,
            tab="repo-map",
            message=f"Repository map run queued: {run.id[:12]}.",
        )

    @app.post("/projects/{project_id}/map/{run_id}/cancel")
    async def cancel_map(
        request: Request, project_id: str, run_id: str
    ) -> RedirectResponse:
        fields = await _request_fields(request)
        if str(fields.get("confirm_cancel", "")) != "cancel-map":
            return _project_redirect(
                project_id,
                tab="repo-map",
                error="Explicit cancellation confirmation is required.",
            )
        try:
            _map_service(request).cancel(run_id)
        except Exception as exc:
            return _project_redirect(project_id, tab="repo-map", error=_web_error(exc))
        return _project_redirect(
            project_id, tab="repo-map", message="Map cancellation requested."
        )

    @app.post("/projects/{project_id}/map/{run_id}/resume")
    async def resume_map(
        request: Request, project_id: str, run_id: str
    ) -> RedirectResponse:
        fields = await _request_fields(request)
        if str(fields.get("confirm_resume", "")) != "resume-map":
            return _project_redirect(
                project_id,
                tab="repo-map",
                error="Explicit resume confirmation is required.",
            )
        try:
            run = _map_service(request).start_refresh(project_id, resume_run_id=run_id)
        except Exception as exc:
            return _project_redirect(project_id, tab="repo-map", error=_web_error(exc))
        return _project_redirect(
            project_id, tab="repo-map", message=f"Resume map run queued: {run.id[:12]}."
        )

    @app.post("/projects/{project_id}/diagrams/generate")
    async def generate_diagram(request: Request, project_id: str) -> RedirectResponse:
        fields = await _request_fields(request)
        try:
            _store(request).generate_diagram(
                project_id,
                title=str(fields.get("title", "")),
                diagram_type=str(fields.get("diagram_type", "module_dependency")),
            )
        except Exception as exc:
            return _project_redirect(project_id, tab="diagrams", error=_web_error(exc))
        return _project_redirect(
            project_id,
            tab="diagrams",
            message="Diagram generated from current deterministic map data.",
        )

    @app.post("/projects/{project_id}/diagrams/{diagram_id}/archive")
    async def archive_diagram(
        request: Request, project_id: str, diagram_id: str
    ) -> RedirectResponse:
        fields = await _request_fields(request)
        try:
            _store(request).archive_diagram(
                project_id, diagram_id, confirm=str(fields.get("confirm_archive", ""))
            )
        except Exception as exc:
            return _project_redirect(project_id, tab="diagrams", error=_web_error(exc))
        return _project_redirect(
            project_id, tab="diagrams", message="Diagram archived."
        )

    @app.post("/projects/{project_id}/trace")
    async def trace_project(request: Request, project_id: str) -> HTMLResponse:
        fields = await _request_fields(request)
        try:
            content = project_detail_page(
                request,
                project_id,
                tab="trace",
                trace_result=_store(request).trace_symbol(
                    project_id,
                    symbol=str(fields.get("symbol", "")),
                    mode=str(fields.get("mode", "quick")),
                    file=str(fields.get("file", "")) or None,
                ),
            )
            return _page(request, "Project Trace", content)
        except Exception as exc:
            return _page(
                request,
                "Project Trace",
                project_detail_page(
                    request, project_id, tab="trace", error=_web_error(exc)
                ),
                status_code=400,
            )


def projects_page(
    request: Request,
    *,
    projects: list[Any] | None = None,
    query: str = "",
    status: str = "active",
    tag: str = "",
    message: str = "",
    error: str = "",
) -> str:
    store = _store(request)
    projects = projects if projects is not None else []
    tag_options = "".join(
        f"<option value='{escape(item)}'{' selected' if item == tag else ''}>{escape(item)}</option>"
        for item in store.list_project_tags()
    )
    cards = "".join(_project_card(project) for project in projects)
    if not cards:
        cards = empty_state_html(
            "No projects match these filters. Add an approved repository to create a durable project workspace."
        )
    return "".join(
        [
            page_header(
                "Projects",
                "Durable software project workspaces for notes, repository maps, diagrams, indexing history, and related Lewis tools.",
                "<a class='button' href='/projects/new'>New Project</a>",
            ),
            flash(message),
            _error(error),
            "<section class='band project-filters'><form class='project-filter-form' method='get' action='/projects'>",
            f"<input name='q' value='{escape(query)}' placeholder='Search projects'>",
            "<select name='status'>"
            + _option("active", "Active", status)
            + _option("paused", "Paused", status)
            + _option("archived", "Archived", status)
            + _option("all", "All", status)
            + "</select>",
            "<select name='tag'><option value=''>All tags</option>"
            + tag_options
            + "</select>",
            "<button type='submit'>Search</button></form></section>",
            "<section class='project-grid'>",
            cards,
            "</section>",
        ]
    )


def new_project_page(
    request: Request,
    *,
    message: str = "",
    error: str = "",
    selected_repo_id: str = "",
    repository_path: str = "",
) -> str:
    repos = configured_project_repository_options(request.app.state.settings)
    repo_options = "<option value=''>Select approved repository</option>" + "".join(
        f"<option value='{escape(repo.repo_id)}'{' selected' if repo.repo_id == selected_repo_id else ''}>{escape(repo.label)} - {escape(repo.root)}</option>"
        for repo in repos
    )
    approved = approved_project_repositories(request.app.state.settings)
    repo_rows = (
        "".join(
            "<tr>"
            f"<td>{escape(repo.label)}</td><td><code>{escape(repo.root)}</code></td><td>{'valid' if repo.valid else escape(repo.error)}</td>"
            "</tr>"
            for repo in approved
        )
        or "<tr><td colspan='3'>No approved project repositories configured.</td></tr>"
    )
    return "".join(
        [
            page_header(
                "New Project",
                "Create a workspace from one configured, validated Git repository root.",
            ),
            flash(message),
            _error(error),
            "<section class='band'><h2>Project Details</h2><form class='project-form' method='post' action='/projects'>",
            "<label class='control'><span>Display name</span><input name='display_name' maxlength='120' required></label>",
            "<label class='control'><span>Approved repository</span><select name='repository_id'>"
            + repo_options
            + "</select></label>",
            f"<label class='control'><span>Or exact approved path</span><input name='repository_path' value='{escape(repository_path)}' placeholder='Must match an approved root exactly'></label>",
            "<label class='control'><span>Description</span><textarea name='description' rows='3'></textarea></label>",
            "<div class='form-grid'><label class='control'><span>Tags</span><input name='tags' placeholder='api, frontend'></label>",
            "<label class='control'><span>Status</span><select name='status'><option value='active'>active</option><option value='paused'>paused</option><option value='archived'>archived</option></select></label></div>",
            "<div class='form-grid'><label class='control'><span>Color</span><input name='color' placeholder='#0f6b5a'></label><label class='control'><span>Icon</span><input name='icon' placeholder='optional label'></label></div>",
            "<div class='form-actions'><button type='submit'>Save Project</button></div></form>",
            "<form class='project-validate-form' method='post' action='/projects/validate'>",
            "<label class='control'><span>Validate approved repository</span><select name='repository_id'>"
            + repo_options
            + "</select></label>",
            f"<label class='control'><span>Or exact approved path</span><input name='repository_path' value='{escape(repository_path)}'></label>",
            "<button class='secondary' type='submit'>Validate Repository</button></form></section>",
            "<section class='band'><h2>Approved Roots</h2><p class='muted'>Projects never auto-discover repositories and cannot browse arbitrary filesystem paths.</p>",
            "<div class='table-scroll'><table><thead><tr><th>Label</th><th>Root</th><th>Status</th></tr></thead><tbody>",
            repo_rows,
            "</tbody></table></div></section>",
        ]
    )


def project_detail_page(
    request: Request,
    project_id: str,
    *,
    tab: str = "overview",
    query: str = "",
    language: str = "",
    role: str = "",
    symbol_type: str = "",
    selected_file: str = "",
    category: str = "",
    sort: str = "updated",
    include_archived: bool = False,
    message: str = "",
    error: str = "",
    trace_result: dict[str, Any] | None = None,
) -> str:
    store = _store(request)
    project = store.require_project(project_id)
    tabs = "".join(
        f"<a class='filter{' active' if tab == key else ''}' href='/projects/{escape(project.id)}?tab={key}'>{escape(label)}</a>"
        for key, label in (
            ("overview", "Overview"),
            ("notes", "Notes"),
            ("repo-map", "Repo Map"),
            ("diagrams", "Diagrams"),
            ("activity", "Activity"),
            ("trace", "Trace"),
        )
    )
    body = {
        "overview": _overview_tab,
        "notes": _notes_tab,
        "repo-map": _repo_map_tab,
        "diagrams": _diagrams_tab,
        "activity": _activity_tab,
        "trace": _trace_tab,
    }[tab](
        request,
        project,
        query=query,
        language=language,
        role=role,
        symbol_type=symbol_type,
        selected_file=selected_file,
        category=category,
        sort=sort,
        include_archived=include_archived,
        trace_result=trace_result,
    )
    return "".join(
        [
            page_header(
                project.display_name,
                project.description or "Project workspace",
                "<a class='button' href='/projects'>All Projects</a>",
            ),
            flash(message),
            _error(error),
            "<section class='filters project-tabs'>",
            tabs,
            "</section>",
            body,
        ]
    )


def _overview_tab(request: Request, project: Any, **_: Any) -> str:
    store = _store(request)
    latest = store.latest_map_run(project.id)
    jobs = store.related_coding_agent_jobs(project.id)
    repo_summary = store.repository_summary(project.id)
    job_rows = (
        "".join(
            f"<a class='job-row' href='/dev/coding-agents/{escape(job['id'])}'><span><strong>{escape(job['provider'])} - {escape(job['state'])}</strong><small>{escape(job['goal'])}</small></span></a>"
            for job in jobs
        )
        or "<p class='muted'>No related Coding Agents jobs yet.</p>"
    )
    coding_link = f"/dev/coding-agents?project_id={escape(project.id)}"
    return "".join(
        [
            "<section class='project-summary-grid'>",
            "<article class='band'><h2>Project</h2><div class='fields'>",
            field("status", project.status),
            field("repository", project.repository_root),
            field("branch", project.current_branch or "unknown"),
            field("HEAD", (project.head_commit or "")[:12]),
            field("last indexed", project.last_map_refreshed_at or "never"),
            field("mapping", project.mapping_state),
            "</div>",
            tag_list_html(list(project.tags)),
            "</article>",
            "<article class='band'><h2>Map Freshness</h2>",
            f"<p>{escape(project.mapping_detail)}</p>",
            _run_progress(latest),
            "</article>",
            "</section>",
            "<section class='band'><h2>Quick Actions</h2><div class='actions'>",
            _refresh_form(project.id),
            f"<a class='button secondary' href='/projects/{escape(project.id)}?tab=notes'>Create Note</a>",
            f"<a class='button secondary' href='{coding_link}'>Open Coding Agents</a>",
            f"<a class='button secondary' href='/projects/{escape(project.id)}?tab=trace'>Trace Symbol</a>",
            f"<a class='button secondary' href='/projects/{escape(project.id)}?tab=repo-map'>Browse Repo Map</a>",
            "</div></section>",
            "<section class='band'><h2>Repository Summary</h2>",
            f"<p>{escape(repo_summary or 'Generate a repository map to create deterministic summaries.')}</p>",
            "</section>",
            "<section class='band'><h2>Edit Metadata</h2>",
            _metadata_form(project),
            "</section>",
            "<section class='band'><h2>Related Coding Agent Jobs</h2><div class='job-list'>",
            job_rows,
            "</div></section>",
            _archive_project_controls(project),
        ]
    )


def _notes_tab(
    request: Request,
    project: Any,
    *,
    query: str = "",
    category: str = "",
    sort: str = "updated",
    include_archived: bool = False,
    **_: Any,
) -> str:
    store = _store(request)
    notes = store.list_notes(
        project.id,
        query=query,
        category=category,
        sort=sort,
        include_archived=include_archived,
    )
    categories = tuple(store.list_categories(project.id))
    category_options = "<option value=''>All categories</option>" + "".join(
        f"<option value='{escape(item)}'{' selected' if item == category else ''}>{escape(item)}</option>"
        for item in categories
    )
    notes_html = "".join(
        _note_card(project.id, note, categories) for note in notes
    ) or empty_state_html("No notes match this view.")
    return "".join(
        [
            "<section class='band'><h2>Create Note</h2>",
            _note_form(project.id, categories),
            "</section>",
            "<section class='band'><h2>Search Notes</h2><form class='project-filter-form' method='get'>",
            "<input type='hidden' name='tab' value='notes'>",
            f"<input name='q' value='{escape(query)}' placeholder='Search notes'>",
            f"<select name='category'>{category_options}</select>",
            "<select name='sort'>"
            + _option("updated", "Recently updated", sort)
            + _option("title", "Title", sort)
            + _option("created", "Created", sort)
            + "</select>",
            f"<label class='approval-check'><input type='checkbox' name='include_archived' value='1'{' checked' if include_archived else ''}><span>Archived</span></label>",
            "<button type='submit'>Search</button></form></section>",
            "<section class='note-list'>",
            notes_html,
            "</section>",
        ]
    )


def _repo_map_tab(
    request: Request,
    project: Any,
    *,
    query: str = "",
    language: str = "",
    role: str = "",
    symbol_type: str = "",
    selected_file: str = "",
    **_: Any,
) -> str:
    store = _store(request)
    latest = store.latest_map_run(project.id)
    search = store.search_map(
        project.id, query=query, language=language, role=role, symbol_type=symbol_type
    )
    language_options = "<option value=''>All languages</option>" + "".join(
        _option(item, item, language) for item in store.language_options(project.id)
    )
    role_options = "<option value=''>All roles</option>" + "".join(
        _option(item, item, role) for item in store.role_options(project.id)
    )
    symbol_options = "<option value=''>All symbols</option>" + "".join(
        _option(item, item, symbol_type)
        for item in store.symbol_kind_options(project.id)
    )
    important = (
        "".join(
            _file_link(project.id, item) for item in store.important_files(project.id)
        )
        or "<p class='muted'>No indexed files yet.</p>"
    )
    file_rows = "".join(_file_link(project.id, item) for item in search["files"])
    symbol_rows = "".join(_symbol_row(project.id, item) for item in search["symbols"])
    relation_rows = "".join(_relationship_row(item) for item in search["relationships"])
    details = (
        _file_detail(store, project.id, selected_file)
        if selected_file
        else "<section class='empty-state'><p>Select a file to inspect symbols, imports, relationships, tests, and deterministic summaries.</p></section>"
    )
    return "".join(
        [
            "<section class='band'><div class='card-head'><h2>Repository Map</h2>",
            _refresh_form(project.id),
            "</div>",
            _run_progress(latest),
            _run_controls(project.id, latest),
            "</section>",
            "<section class='band'><h2>Search Map</h2><form class='project-filter-form' method='get'>",
            "<input type='hidden' name='tab' value='repo-map'>",
            f"<input name='q' value='{escape(query)}' placeholder='Files, symbols, routes, summaries'>",
            f"<select name='language'>{language_options}</select><select name='role'>{role_options}</select><select name='symbol_type'>{symbol_options}</select>",
            "<button type='submit'>Search</button></form></section>",
            "<section class='project-map-layout'><div class='project-map-rail'>",
            "<section class='band'><h2>Important Files</h2>",
            important,
            "</section>",
            "<section class='band'><h2>Files</h2><div class='job-list'>",
            file_rows or "<p class='muted'>No file matches.</p>",
            "</div></section>",
            "<section class='band'><h2>Symbols</h2><div class='job-list'>",
            symbol_rows or "<p class='muted'>No symbol matches.</p>",
            "</div></section>",
            "</div><div class='project-map-detail'>",
            details,
            "<section class='band'><h2>Relationships</h2><div class='table-scroll'><table><thead><tr><th>Type</th><th>Source</th><th>Target</th><th>Resolution</th></tr></thead><tbody>",
            relation_rows or "<tr><td colspan='4'>No relationship matches.</td></tr>",
            "</tbody></table></div><p class='muted'>Inferred relationships are static-analysis hints, not guaranteed runtime facts.</p></section>",
            "</div></section>",
        ]
    )


def _diagrams_tab(request: Request, project: Any, **_: Any) -> str:
    diagrams = _store(request).list_diagrams(project.id)
    cards = "".join(
        _diagram_card(project.id, diagram) for diagram in diagrams
    ) or empty_state_html(
        "No saved diagrams yet. Generate one from the current repository map."
    )
    return "".join(
        [
            "<section class='band'><h2>Generate Diagram</h2><form class='project-form' method='post' action='/projects/",
            escape(project.id),
            "/diagrams/generate'>",
            "<label class='control'><span>Title</span><input name='title' value='Module dependency map' required></label>",
            "<label class='control'><span>Type</span><select name='diagram_type'><option value='module_dependency'>Module dependency</option><option value='architecture'>High-level architecture</option><option value='route_map'>Route map</option></select></label>",
            "<div class='form-actions'><button type='submit'>Generate From Current Map</button></div></form></section>",
            "<section class='grid'>",
            cards,
            "</section>",
        ]
    )


def _activity_tab(request: Request, project: Any, **_: Any) -> str:
    rows = (
        "".join(
            f"<tr><td>{escape(item['created_at'])}</td><td>{escape(item['event_type'])}</td><td>{escape(item['summary'])}</td><td><code>{escape(json.dumps(item['details'], sort_keys=True))}</code></td></tr>"
            for item in _store(request).recent_activity(project.id)
        )
        or "<tr><td colspan='4'>No project activity yet.</td></tr>"
    )
    return (
        "<section class='band'><h2>Project Activity</h2><div class='table-scroll'><table><thead><tr><th>Time</th><th>Type</th><th>Summary</th><th>Details</th></tr></thead><tbody>"
        + rows
        + "</tbody></table></div></section>"
    )


def _trace_tab(
    request: Request,
    project: Any,
    *,
    trace_result: dict[str, Any] | None = None,
    **_: Any,
) -> str:
    report = ""
    if trace_result:
        result = trace_result["result"]
        report = (
            "<section class='band'><h2>Trace Result</h2>"
            + status_badge("completed" if result.ok else "failed")
            + f"<pre class='patch-preview'>{escape(trace_result['report'])}</pre></section>"
        )
    return "".join(
        [
            "<section class='band'><h2>Trace Symbol</h2><p class='muted'>Uses the existing read-only Lewis trace engine for this project's repository root.</p>",
            f"<form class='project-form' method='post' action='/projects/{escape(project.id)}/trace'>",
            "<label class='control'><span>Symbol</span><input name='symbol' placeholder='Class.method or function_name' required></label>",
            "<div class='form-grid'><label class='control'><span>Mode</span><select name='mode'><option value='quick'>quick</option><option value='deep'>deep</option><option value='review'>review</option></select></label>",
            "<label class='control'><span>File hint</span><input name='file' placeholder='optional/repo/path.py'></label></div>",
            "<div class='form-actions'><button type='submit'>Run Trace</button></div></form></section>",
            report,
        ]
    )


def _project_card(project: Any) -> str:
    stale = (
        "<span class='badge status-execution-previewed'>stale or failed</span>"
        if project.mapping_state in {"failed", "paused"}
        else ""
    )
    return "".join(
        [
            "<article class='card project-card'><div class='card-head'>",
            f"<h3><a href='/projects/{escape(project.id)}'>{escape(project.display_name)}</a></h3>",
            "<div class='meta'>",
            status_badge(project.status),
            status_badge(project.mapping_state),
            stale,
            "</div></div>",
            f"<p>{escape(project.description or 'No description yet.')}</p>",
            tag_list_html(list(project.tags)),
            "<div class='fields'>",
            field("repository", project.repository_root),
            field("branch", project.current_branch or "unknown"),
            field("last indexed", project.last_map_refreshed_at or "never"),
            field("notes", project.active_notes_count),
            "</div>",
            "</article>",
        ]
    )


def _metadata_form(project: Any) -> str:
    tag_text = ", ".join(project.tags)
    return "".join(
        [
            f"<form class='project-form' method='post' action='/projects/{escape(project.id)}/metadata'>",
            f"<label class='control'><span>Name</span><input name='display_name' value='{escape(project.display_name)}' required></label>",
            f"<label class='control'><span>Description</span><textarea name='description' rows='3'>{escape(project.description)}</textarea></label>",
            f"<div class='form-grid'><label class='control'><span>Tags</span><input name='tags' value='{escape(tag_text)}'></label>",
            "<label class='control'><span>Status</span><select name='status'>"
            + _option("active", "active", project.status)
            + _option("paused", "paused", project.status)
            + _option("archived", "archived", project.status)
            + "</select></label></div>",
            f"<div class='form-grid'><label class='control'><span>Color</span><input name='color' value='{escape(project.color)}'></label><label class='control'><span>Icon</span><input name='icon' value='{escape(project.icon)}'></label></div>",
            "<div class='form-actions'><button type='submit'>Save Metadata</button></div></form>",
        ]
    )


def _archive_project_controls(project: Any) -> str:
    if project.status == "archived":
        return f"<section class='band'><h2>Restore</h2><form method='post' action='/projects/{escape(project.id)}/restore'><button type='submit'>Restore Project</button></form></section>"
    return "".join(
        [
            "<section class='danger-zone'><h3>Archive Project</h3><p>Archiving keeps notes, maps, diagrams, and history.</p>",
            f"<form method='post' action='/projects/{escape(project.id)}/archive'><label class='approval-check'><input type='checkbox' name='confirm_archive' value='archive-project' required><span>Archive this project, preserving its records.</span></label><button class='warn' type='submit'>Archive Project</button></form></section>",
        ]
    )


def _refresh_form(project_id: str) -> str:
    return "".join(
        [
            f"<form class='inline-form' method='post' action='/projects/{escape(project_id)}/map/refresh'>",
            "<input type='hidden' name='confirm_refresh' value='refresh-map'>",
            "<label class='approval-check compact-check'><input type='checkbox' name='full_rebuild' value='1'><span>Full rebuild</span></label>",
            "<button type='submit'>Refresh Repository Map</button></form>",
        ]
    )


def _run_progress(run: Any | None) -> str:
    if run is None:
        return "<p class='muted'>No repository map run yet.</p>"
    return (
        "<div class='fields'>"
        + field("run", run.id[:12])
        + field("status", run.status)
        + field("stage", run.current_stage or "not started")
        + field("progress", f"{run.progress_percent}%")
        + field("files scanned", run.files_scanned)
        + field("changed/reused", f"{run.files_changed}/{run.files_reused}")
        + field("skipped", run.files_skipped)
        + field("revision", (run.revision_commit or "")[:12])
        + "</div>"
    )


def _run_controls(project_id: str, run: Any | None) -> str:
    if run is None:
        return ""
    controls = []
    if run.status in {"queued", "running"}:
        controls.append(
            f"<form method='post' action='/projects/{escape(project_id)}/map/{escape(run.id)}/cancel'><input type='hidden' name='confirm_cancel' value='cancel-map'><button class='warn' type='submit'>Cancel Map Run</button></form>"
        )
    if run.status in {"failed", "cancelled", "paused"}:
        controls.append(
            f"<form method='post' action='/projects/{escape(project_id)}/map/{escape(run.id)}/resume'><input type='hidden' name='confirm_resume' value='resume-map'><button type='submit'>Resume Map Run</button></form>"
        )
    if not controls:
        return ""
    return "<div class='actions'>" + "".join(controls) + "</div>"


def _note_form(
    project_id: str, categories: tuple[str, ...], note: Any | None = None
) -> str:
    action = (
        f"/projects/{escape(project_id)}/notes"
        if note is None
        else f"/projects/{escape(project_id)}/notes/{escape(note.id)}/update"
    )
    title = "" if note is None else note.title
    content = "" if note is None else note.content_markdown
    category = "" if note is None else note.category_name or ""
    tags = "" if note is None else ", ".join(note.tags)
    category_hint = ", ".join(categories[:6])
    return "".join(
        [
            f"<form class='project-form' method='post' action='{action}'>",
            f"<label class='control'><span>Title</span><input name='title' value='{escape(title)}' required></label>",
            f"<label class='control'><span>Category</span><input name='category' value='{escape(category)}' placeholder='{escape(category_hint or 'Optional category')}'></label>",
            f"<label class='control'><span>Tags</span><input name='tags' value='{escape(tags)}'></label>",
            f"<label class='control'><span>Markdown</span><textarea name='content_markdown' rows='8'>{escape(content)}</textarea></label>",
            f"<label class='approval-check'><input type='checkbox' name='is_pinned' value='1'{' checked' if note and note.is_pinned else ''}><span>Pin note</span></label>",
            "<div class='form-actions'><button type='submit'>Save Note</button></div></form>",
        ]
    )


def _note_card(project_id: str, note: Any, categories: tuple[str, ...]) -> str:
    status = status_badge(note.status)
    archive_action = "restore" if note.status == "archived" else "archive"
    archive_label = "Restore" if note.status == "archived" else "Archive"
    pinned = "<span class='badge badge-soft'>pinned</span>" if note.is_pinned else ""
    return "".join(
        [
            "<article class='card project-note'><div class='card-head'>",
            f"<h3>{escape(note.title)}</h3><div class='meta'>{status}{pinned}</div></div>",
            tag_list_html(list(note.tags)),
            f"<div class='memory-content'>{_markdown_preview(note.content_markdown)}</div>",
            "<details class='card-details'><summary>Edit note</summary>",
            _note_form(project_id, categories, note),
            "</details>",
            "<div class='actions'>",
            f"<form method='post' action='/projects/{escape(project_id)}/notes/{escape(note.id)}/duplicate'><button class='secondary' type='submit'>Duplicate</button></form>",
            f"<form method='post' action='/projects/{escape(project_id)}/notes/{escape(note.id)}/{archive_action}'><button class='secondary' type='submit'>{archive_label}</button></form>",
            f"<form method='post' action='/projects/{escape(project_id)}/notes/{escape(note.id)}/delete'><label class='approval-check'><input type='checkbox' name='confirm_delete' value='delete-note' required><span>Confirm permanent deletion.</span></label><button class='warn' type='submit'>Delete Permanently</button></form>",
            "</div></article>",
        ]
    )


def _file_link(project_id: str, item: Any) -> str:
    href = f"/projects/{escape(project_id)}?tab=repo-map&file={escape(item.path)}"
    return f"<a class='job-row' href='{href}'><span><strong>{escape(item.path)}</strong><small>{escape(item.language)} - {escape(item.role)} - {escape(item.summary)}</small></span>{status_badge(item.git_status or 'tracked')}</a>"


def _symbol_row(project_id: str, item: Any) -> str:
    href = f"/projects/{escape(project_id)}?tab=repo-map&file={escape(item.file_path)}"
    return f"<a class='job-row' href='{href}'><span><strong>{escape(item.name)}</strong><small>{escape(item.kind)} in {escape(item.file_path)}:{escape(item.line)}</small></span></a>"


def _relationship_row(item: Any) -> str:
    source = item.source_path + (
        (":" + item.source_symbol) if item.source_symbol else ""
    )
    target = (item.target_path or "") + (
        (":" + item.target_symbol) if item.target_symbol else ""
    )
    return f"<tr><td>{escape(item.relationship_type)}</td><td><code>{escape(source)}</code></td><td><code>{escape(target)}</code></td><td>{escape(item.resolution)}</td></tr>"


def _file_detail(store: ProjectStore, project_id: str, path: str) -> str:
    item = store.get_indexed_file(project_id, path)
    if item is None:
        return empty_state_html("Selected file was not found in the current map.")
    symbols = (
        "".join(
            f"<tr><td>{escape(sym.kind)}</td><td>{escape(sym.name)}</td><td>{escape(sym.line)}</td><td>{escape(sym.summary)}</td></tr>"
            for sym in store.symbols_for_file(project_id, path)
        )
        or "<tr><td colspan='4'>No symbols recorded.</td></tr>"
    )
    relationships = (
        "".join(
            _relationship_row(rel)
            for rel in store.relationships_for_file(project_id, path)
        )
        or "<tr><td colspan='4'>No relationships recorded.</td></tr>"
    )
    imports = ", ".join(item.imports) or "none"
    return "".join(
        [
            "<section class='band'><h2>File Detail</h2><div class='fields'>",
            field("path", item.path),
            field("language", item.language),
            field("role", item.role),
            field("size", item.size_bytes),
            field("hash", item.content_hash[:16]),
            field("git", item.git_status),
            field("imports", imports),
            "</div><p>",
            escape(item.summary),
            "</p><p class='muted'>This panel shows bounded deterministic metadata only; source contents are not exposed.</p></section>",
            "<section class='band'><h2>Symbols</h2><div class='table-scroll'><table><thead><tr><th>Kind</th><th>Name</th><th>Line</th><th>Summary</th></tr></thead><tbody>",
            symbols,
            "</tbody></table></div></section>",
            "<section class='band'><h2>File Relationships</h2><div class='table-scroll'><table><thead><tr><th>Type</th><th>Source</th><th>Target</th><th>Resolution</th></tr></thead><tbody>",
            relationships,
            "</tbody></table></div></section>",
        ]
    )


def _diagram_card(project_id: str, diagram: Any) -> str:
    nodes = (
        diagram.definition.get("nodes", [])
        if isinstance(diagram.definition, dict)
        else []
    )
    edges = (
        diagram.definition.get("edges", [])
        if isinstance(diagram.definition, dict)
        else []
    )
    edge_rows = (
        "".join(
            f"<tr><td>{escape(edge.get('kind', ''))}</td><td><code>{escape(edge.get('source', ''))}</code></td><td><code>{escape(edge.get('target', ''))}</code></td><td>{escape(edge.get('resolution', ''))}</td></tr>"
            for edge in edges[:80]
        )
        or "<tr><td colspan='4'>No edges in this diagram.</td></tr>"
    )
    node_list = "".join(
        f"<span>{escape(node.get('label', node.get('id', '')))}</span>"
        for node in nodes[:80]
    )
    return "".join(
        [
            "<article class='card project-diagram'><div class='card-head'>",
            f"<h3>{escape(diagram.title)}</h3><div class='meta'>{status_badge(diagram.generation_source)}{status_badge(diagram.diagram_type)}</div></div>",
            f"<p class='muted'>Map revision: {escape(diagram.map_run_id or 'unknown')} - {len(nodes)} nodes, {len(edges)} edges.</p>",
            "<div class='tag-list diagram-nodes'>",
            node_list,
            "</div>",
            "<details class='card-details'><summary>Relationship table</summary><div class='table-scroll'><table><thead><tr><th>Kind</th><th>Source</th><th>Target</th><th>Resolution</th></tr></thead><tbody>",
            edge_rows,
            "</tbody></table></div></details>",
            f"<form class='actions' method='post' action='/projects/{escape(project_id)}/diagrams/{escape(diagram.id)}/archive'><input type='hidden' name='confirm_archive' value='archive-diagram'><button class='warn' type='submit'>Archive</button></form>",
            "</article>",
        ]
    )


def _markdown_preview(text: str) -> str:
    safe = escape(text)
    safe = safe.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = []
    for block in safe.split("\n\n")[:20]:
        block = block.strip()
        if not block:
            continue
        if block.startswith("#"):
            block = block.lstrip("# ")
            paragraphs.append(f"<strong>{block}</strong>")
        else:
            paragraphs.append(block.replace("\n", "<br>"))
    return (
        "".join(f"<p>{item}</p>" for item in paragraphs)
        or "<p class='muted'>Empty note.</p>"
    )


def _option(value: str, label: str, selected: str) -> str:
    return f"<option value='{escape(value)}'{' selected' if value == selected else ''}>{escape(label)}</option>"


def _error(error: str) -> str:
    if not error:
        return ""
    return f"<div class='inline-error' role='alert'><strong>Projects action stopped</strong><span>{escape(error)}</span></div>"


def _store(request: Request) -> ProjectStore:
    store = getattr(request.app.state, "project_store", None)
    if store is None:
        store = ProjectStore(request.app.state.settings.database_path)
        request.app.state.project_store = store
    return store


def _map_service(request: Request) -> ProjectMapService:
    service = getattr(request.app.state, "project_map_service", None)
    if service is None:
        service = ProjectMapService(request.app.state.settings, store=_store(request))
        request.app.state.project_map_service = service
    return service


def _page(
    request: Request, title: str, content: str, status_code: int = 200
) -> HTMLResponse:
    return HTMLResponse(
        page_html(
            title,
            request.url.path,
            content,
            show_dev_navigation=request.app.state.settings.freestyle_ui_enabled,
            show_coding_agents_navigation=request.app.state.settings.coding_agents_ui_enabled,
            show_projects_navigation=True,
        ),
        status_code=status_code,
    )


def _redirect(path: str, *, message: str = "", error: str = "") -> RedirectResponse:
    values = {"message": message} if message else {"error": error}
    return RedirectResponse(url=f"{path}?{urlencode(values)}", status_code=303)


def _project_redirect(
    project_id: str, *, tab: str = "overview", message: str = "", error: str = ""
) -> RedirectResponse:
    values = {"tab": tab}
    if message:
        values["message"] = message
    if error:
        values["error"] = error
    return RedirectResponse(
        url=f"/projects/{project_id}?{urlencode(values)}", status_code=303
    )


async def _request_fields(request: Request) -> dict[str, object]:
    raw_body = await request.body()
    if len(raw_body) > MAX_PROJECT_WEB_REQUEST_BYTES:
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
    decoded = raw_body.decode("utf-8", errors="ignore")
    return {
        key: values[-1] if values else ""
        for key, values in parse_qs(decoded, keep_blank_values=True).items()
    }


def _tags_field(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        raw = str(value or "").split(",")
    return tuple(item.strip() for item in raw if str(item).strip())


def _web_error(exc: Exception) -> str:
    return str(exc)[:500]
