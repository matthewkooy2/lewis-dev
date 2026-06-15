from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.approval_service import approve_approval, reject_approval
from app.approvals import ApprovalStore
from app.config import MISSING_API_KEY_MESSAGE, Settings
from app.dashboard import (
    complete_task,
    get_dashboard_summary,
    list_calendar_proposals,
    list_history_items,
    list_pending_approvals,
    list_tasks,
    reopen_task,
)
from app.executors import execute_approved
from app.logging_db import RunLogger
from app.main import build_agent
from app.memory_service import (
    archive_memory,
    list_active_memories,
    list_recent_memory_events,
    propose_memory_update,
)
from app.rag import DocumentSearch, NO_DOCUMENTS_MESSAGE, NO_MATCHES_MESSAGE
from app.web_components import page_html
from app.web_pages import (
    approvals_page,
    calendar_page,
    chat_page,
    dashboard_page,
    document_search_page,
    history_page,
    memories_page,
    tasks_page,
    usage_page,
)
from app.web_rules import normalize_history_filter


STATIC_DIR = Path(__file__).with_name("static")


def register_web_routes(app: FastAPI) -> None:
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def home() -> RedirectResponse:
        return RedirectResponse(url="/dashboard", status_code=303)

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        store = _approval_store(request)
        logger = _run_logger(request)
        data = get_dashboard_summary(store, logger)
        return _page("Dashboard", request, dashboard_page(data))

    @app.get("/approvals", response_class=HTMLResponse)
    def approvals(request: Request, message: str = "") -> HTMLResponse:
        items = list_pending_approvals(_approval_store(request))
        return _page("Approvals", request, approvals_page(items, message))

    @app.post("/approvals/{approval_id}/approve")
    def approve_item(
        request: Request, approval_id: int, return_to: str = "/approvals"
    ) -> RedirectResponse:
        try:
            item = approve_approval(_approval_store(request), approval_id)
            message = f"Approved #{item.id}. No external action was executed."
        except (KeyError, ValueError) as exc:
            message = f"Could not approve item: {exc}"
        return _redirect_with_message(_safe_return_path(return_to), message)

    @app.post("/approvals/{approval_id}/reject")
    def reject_item(
        request: Request, approval_id: int, return_to: str = "/approvals"
    ) -> RedirectResponse:
        try:
            item = reject_approval(_approval_store(request), approval_id)
            message = f"Rejected #{item.id}. No external action was executed."
        except (KeyError, ValueError) as exc:
            message = f"Could not reject item: {exc}"
        return _redirect_with_message(_safe_return_path(return_to), message)

    @app.post("/approvals/{approval_id}/execute-approved")
    def preview_approved_item(
        request: Request, approval_id: int, return_to: str = "/history"
    ) -> RedirectResponse:
        try:
            result = execute_approved(_approval_store(request), approval_id)
            message = (
                f"Execution previewed for #{result.approval_id}. "
                f"{result.message} No external action was taken."
            )
        except (KeyError, ValueError) as exc:
            message = f"Could not preview approved item: {exc}"
        return _redirect_with_message(_safe_return_path(return_to), message)

    @app.get("/tasks", response_class=HTMLResponse)
    def tasks(request: Request, message: str = "") -> HTMLResponse:
        return _page(
            "Tasks",
            request,
            tasks_page(list_tasks(_approval_store(request)), message),
        )

    @app.post("/tasks/{approval_id}/complete")
    def complete_task_item(
        request: Request, approval_id: int, return_to: str = "/tasks"
    ) -> RedirectResponse:
        try:
            item = complete_task(_approval_store(request), approval_id)
            message = (
                f"Completed local task proposal #{item.id}. "
                "No emails, reminders, notifications, or external actions were executed."
            )
        except (KeyError, ValueError) as exc:
            message = f"Could not complete task: {exc}"
        return _redirect_with_message(_safe_return_path(return_to), message)

    @app.post("/tasks/{approval_id}/reopen")
    def reopen_task_item(
        request: Request, approval_id: int, return_to: str = "/history"
    ) -> RedirectResponse:
        try:
            item = reopen_task(_approval_store(request), approval_id)
            message = (
                f"Reopened local task proposal #{item.id} as approved. "
                "No external actions were executed."
            )
        except (KeyError, ValueError) as exc:
            message = f"Could not reopen task: {exc}"
        return _redirect_with_message(_safe_return_path(return_to), message)

    @app.get("/calendar", response_class=HTMLResponse)
    def calendar(request: Request, message: str = "") -> HTMLResponse:
        return _page(
            "Calendar",
            request,
            calendar_page(list_calendar_proposals(_approval_store(request)), message),
        )

    @app.get("/memories", response_class=HTMLResponse)
    def memories(request: Request, message: str = "") -> HTMLResponse:
        settings = _settings(request)
        return _page(
            "Memories",
            request,
            memories_page(
                list_active_memories(settings.database_path),
                list_recent_memory_events(settings.database_path),
                message,
            ),
        )

    @app.post("/memories/{memory_id}/archive")
    async def archive_memory_item(request: Request, memory_id: int) -> RedirectResponse:
        fields = await _request_fields(request)
        reason = str(fields.get("reason", "")).strip()
        try:
            memory = archive_memory(
                _settings(request).database_path,
                memory_id,
                reason=reason,
                actor="user",
            )
            message = f"Archived memory #{memory.id}. The row was preserved."
        except (KeyError, ValueError) as exc:
            message = f"Could not archive memory: {exc}"
        return _redirect_with_message("/memories", message)

    @app.post("/memories/{memory_id}/propose-update")
    async def propose_memory_update_item(
        request: Request, memory_id: int
    ) -> RedirectResponse:
        fields = await _request_fields(request)
        tags = [
            tag.strip() for tag in str(fields.get("tags", "")).split(",") if tag.strip()
        ]
        try:
            item = propose_memory_update(
                memory_id=memory_id,
                memory_type=str(fields.get("memory_type", "")).strip(),
                content=str(fields.get("content", "")).strip(),
                source="web",
                tags=tags,
                reason=str(fields.get("reason", "")).strip(),
                db_path=_settings(request).database_path,
                actor="user",
            )
            message = (
                f"Saved pending memory update approval #{item.id}. "
                "The active memory was not changed."
            )
        except (KeyError, ValueError) as exc:
            message = f"Could not propose memory update: {exc}"
        return _redirect_with_message("/memories", message)

    @app.get("/history", response_class=HTMLResponse)
    def history(request: Request, type: str = "all", message: str = "") -> HTMLResponse:
        selected = normalize_history_filter(type)
        items = list_history_items(_approval_store(request), selected)
        return _page("History", request, history_page(items, selected, message))

    @app.get("/usage", response_class=HTMLResponse)
    def usage(request: Request) -> HTMLResponse:
        logger = _run_logger(request)
        return _page(
            "Usage",
            request,
            usage_page(logger.recent_usage_summary(), logger.recent_runs()),
        )

    @app.get("/documents", response_class=HTMLResponse)
    def documents(request: Request) -> HTMLResponse:
        allowed_params = {"query"}
        if set(request.query_params) - allowed_params:
            return _page(
                "Document Search",
                request,
                document_search_page(
                    "", [], "Document search only accepts a query string."
                ),
                status_code=400,
            )
        query = request.query_params.get("query", "").strip()
        results = []
        message = ""
        if query:
            search = DocumentSearch(_settings(request).documents_dir)
            results = search.search(query)
            if not search.has_documents():
                message = NO_DOCUMENTS_MESSAGE
            elif not results:
                message = NO_MATCHES_MESSAGE
        return _page(
            "Document Search",
            request,
            document_search_page(query, results, message),
        )

    @app.get("/chat", response_class=HTMLResponse)
    def chat(request: Request) -> HTMLResponse:
        return _page("Chat", request, chat_page(request.app.state.chat_messages))

    @app.post("/chat")
    async def chat_message(request: Request) -> JSONResponse:
        fields = await _request_fields(request)
        message = str(fields.get("message", "")).strip()
        if not message:
            return JSONResponse({"ok": False, "answer": "Enter a message first."})
        settings = _settings(request)
        request.app.state.chat_messages.append({"role": "user", "content": message})
        memory_context_count = 0
        if not settings.has_api_key:
            answer = MISSING_API_KEY_MESSAGE
        else:
            try:
                if request.app.state.chat_agent is None:
                    request.app.state.chat_agent = build_agent(settings)
                run = request.app.state.chat_agent.run(message)
                answer = run.answer
                memory_context_count = run.memory_context_count
            except Exception as exc:
                answer = f"Sorry, I hit an error: {exc}"
        request.app.state.chat_messages.append(
            {
                "role": "assistant",
                "content": answer,
                "memory_context_count": memory_context_count,
            }
        )
        return JSONResponse(
            {
                "ok": True,
                "answer": answer,
                "memory_context_count": memory_context_count,
            }
        )


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _approval_store(request: Request) -> ApprovalStore:
    return ApprovalStore(_settings(request).database_path)


def _run_logger(request: Request) -> RunLogger:
    return RunLogger(_settings(request).database_path)


def _page(
    title: str,
    request: Request,
    content: str,
    status_code: int = 200,
) -> HTMLResponse:
    return HTMLResponse(
        page_html(
            title,
            request.url.path,
            content,
            show_dev_navigation=_settings(request).debug,
        ),
        status_code=status_code,
    )


def _redirect_with_message(path: str, message: str) -> RedirectResponse:
    from urllib.parse import urlencode

    return RedirectResponse(
        url=f"{path}?{urlencode({'message': message})}",
        status_code=303,
    )


def _safe_return_path(path: str) -> str:
    if path.startswith("/") and not path.startswith("//"):
        return path
    return "/approvals"


async def _request_fields(request: Request) -> dict[str, object]:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    raw_body = (await request.body()).decode("utf-8", errors="ignore")
    return {
        key: values[-1] if values else ""
        for key, values in parse_qs(raw_body, keep_blank_values=True).items()
    }
