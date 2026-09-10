from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python < 3.9
    ZoneInfo = None  # type: ignore[assignment]

from app.approvals import ApprovalStore
from app.code_proposals import propose_code_change
from app.dashboard import (
    get_dashboard_summary,
    list_calendar_proposals,
    list_pending_approvals,
    list_tasks,
)
from app.logging_db import RunLogger
from app.memory_service import propose_memory
from app.project_search import ProjectFileSearch
from app.project_inspection import ProjectInspector
from app.rag import DocumentSearch, NO_DOCUMENTS_MESSAGE, NO_MATCHES_MESSAGE
from app.trace.model import TRACE_MODES
from app.trace.render import to_compact_dict
from app.trace.service import CodeTracer


ToolFunction = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    function: ToolFunction

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(
        self,
        document_search: DocumentSearch,
        approval_store: ApprovalStore,
        logger: RunLogger | None = None,
        user_name_configured: bool = False,
        project_file_search: ProjectFileSearch | None = None,
        developer_code_workspace: bool = False,
        project_inspector: ProjectInspector | None = None,
        code_tracer: CodeTracer | None = None,
    ):
        self.document_search = document_search
        self.approval_store = approval_store
        self.logger = logger or RunLogger(approval_store.db_path)
        self.user_name_configured = user_name_configured
        self.project_file_search = project_file_search or ProjectFileSearch()
        self.developer_code_workspace = developer_code_workspace
        self.project_inspector = project_inspector or ProjectInspector()
        self.code_tracer = code_tracer or CodeTracer()
        self._tools: dict[str, Tool] = {}
        self._register_defaults()
        if self.developer_code_workspace:
            self._register_project_inspection_tools()

    def schemas(
        self, allowed_tool_names: frozenset[str] | None = None
    ) -> list[dict[str, Any]]:
        return [
            tool.openai_schema()
            for tool in self._tools.values()
            if allowed_tool_names is None or tool.name in allowed_tool_names
        ]

    def registered_tool_names(self) -> frozenset[str]:
        return frozenset(self._tools)

    def execute(self, name: str, arguments: str | dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"Unknown tool: {name}"}
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            if not isinstance(args, dict):
                return {"ok": False, "error": "Tool arguments must be a JSON object."}
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"Malformed tool arguments: {exc.msg}"}
        try:
            return tool.function(args)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def _register_defaults(self) -> None:
        self._add(
            Tool(
                name="search_documents",
                description="Search local notes and documents in the data/documents folder.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                function=self._search_documents,
            )
        )
        self._add(
            Tool(
                name="list_local_files",
                description="List supported local document files available to Lewis.",
                parameters={"type": "object", "properties": {}},
                function=self._list_local_files,
            )
        )
        self._add(
            Tool(
                name="get_current_time",
                description="Get the current local time.",
                parameters={
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "IANA timezone name, defaults to America/New_York.",
                        }
                    },
                },
                function=self._get_current_time,
            )
        )
        self._add(
            Tool(
                name="list_pending_approvals",
                description=(
                    "List real local approval items currently pending approval. "
                    "Read-only; do not invent approvals."
                ),
                parameters={"type": "object", "properties": {}},
                function=self._list_pending_approvals,
            )
        )
        self._add(
            Tool(
                name="list_tasks",
                description=(
                    "List real local task proposals and their statuses. Read-only; "
                    "do not invent tasks."
                ),
                parameters={"type": "object", "properties": {}},
                function=self._list_tasks,
            )
        )
        self._add(
            Tool(
                name="list_calendar_proposals",
                description=(
                    "List real local calendar event proposals and their statuses. "
                    "Read-only; do not invent calendar proposals."
                ),
                parameters={"type": "object", "properties": {}},
                function=self._list_calendar_proposals,
            )
        )
        self._add(
            Tool(
                name="get_dashboard_summary",
                description=(
                    "Get real local dashboard counts, recent usage, and recent "
                    "approval/task items. Read-only."
                ),
                parameters={"type": "object", "properties": {}},
                function=self._get_dashboard_summary,
            )
        )
        self._add(
            Tool(
                name="search_project_files",
                description=(
                    "Search safe Lewis project file paths and text contents. "
                    "Read-only; never reads .env, secrets, credentials, hidden "
                    "files, runtime data, virtualenvs, git data, or files outside "
                    "the project root."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                function=self._search_project_files,
            )
        )
        self._add_proposal_tool(
            "propose_email_draft",
            "Save a pending approval item for an email draft. Do not send email.",
        )
        self._add_calendar_proposal_tool()
        self._add_code_proposal_tool()
        self._add_task_proposal_tool()
        self._add_memory_proposal_tool()

    def _register_project_inspection_tools(self) -> None:
        self._add(
            Tool(
                name="list_project_files",
                description=(
                    "Developer/code workspace only: list safe, read-only text "
                    "project files inside the configured project workspace. "
                    "Does not read file contents."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                function=self._list_project_files,
            )
        )
        self._add(
            Tool(
                name="read_project_file",
                description=(
                    "Developer/code workspace only: read one safe text file from "
                    "the configured project workspace by project-relative path. "
                    "Read-only, bounded, secret-redacted, and blocked for runtime, "
                    "hidden, env, database, binary, and external paths."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Project-relative file path.",
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                function=self._read_project_file,
            )
        )
        self._add(
            Tool(
                name="search_project",
                description=(
                    "Developer/code workspace only: bounded read-only keyword "
                    "search over safe text files in the configured project "
                    "workspace. Separate from document search and never searches "
                    "runtime or hidden files."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                function=self._search_project,
            )
        )
        self._add(
            Tool(
                name="trace_symbol",
                description=(
                    "Developer/code workspace only: static, read-only code trace "
                    "of one named function, class, method, or React component in "
                    "the project. mode 'quick' shows the definition, direct "
                    "callers, direct callees, and direct side effects; 'deep' "
                    "also follows bounded call paths across files, transitive "
                    "side effects, tests, error handling, and cycles; 'review' "
                    "adds deterministic findings with file:line citations. "
                    "Results label every relationship (confirmed, inferred, "
                    "framework-registered, ambiguous, external, dynamic, "
                    "unresolved); if the symbol is ambiguous, candidates are "
                    "returned instead of guessing. Never executes project code."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": (
                                "Symbol to trace: name, Class.method, dotted "
                                "module path (app.llm.LewisAgent.run), or "
                                "file::qualname (app/llm.py::LewisAgent.run)."
                            ),
                        },
                        "file": {
                            "type": "string",
                            "description": (
                                "Optional project-relative source file to "
                                "disambiguate the symbol."
                            ),
                        },
                        "mode": {
                            "type": "string",
                            "enum": list(TRACE_MODES),
                            "description": "quick (default), deep, or review.",
                        },
                    },
                    "required": ["symbol"],
                    "additionalProperties": False,
                },
                function=self._trace_symbol,
            )
        )

    def _add_proposal_tool(self, name: str, description: str) -> None:
        self._add(
            Tool(
                name=name,
                description=description,
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "details": {"type": "object"},
                    },
                    "required": ["title", "details"],
                    "additionalProperties": False,
                },
                function=lambda args, action_type=name: self._proposal(
                    action_type, args
                ),
            )
        )

    def _add_task_proposal_tool(self) -> None:
        self._add(
            Tool(
                name="propose_task",
                description=(
                    "Save a pending approval item for a task, reminder, todo, or "
                    "follow-up. Do not create reminders, notifications, calendar "
                    "events, emails, or external tasks."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "due_date_text": {
                            "type": "string",
                            "description": "Optional natural-language due date.",
                        },
                        "priority": {"type": "string"},
                        "source": {
                            "type": "string",
                            "description": "Optional source or context for the task.",
                        },
                    },
                    "required": ["title", "description"],
                    "additionalProperties": False,
                },
                function=self._task_proposal,
            )
        )

    def _add_calendar_proposal_tool(self) -> None:
        self._add(
            Tool(
                name="propose_calendar_event",
                description=(
                    "Save a pending local approval item for a calendar event, "
                    "meeting, scheduled block, or planned event. Do not create "
                    "Outlook, Google Calendar, OS calendar, reminder, notification, "
                    "email, invite, or external event."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "date_text": {
                            "type": "string",
                            "description": "Natural-language event date.",
                        },
                        "start_time_text": {
                            "type": "string",
                            "description": "Natural-language start time.",
                        },
                        "end_time_text": {
                            "type": "string",
                            "description": "Natural-language end time.",
                        },
                        "location": {"type": "string"},
                        "attendees": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "source": {"type": "string"},
                        "notes": {"type": "string"},
                    },
                    "required": ["title", "description"],
                    "additionalProperties": False,
                },
                function=self._calendar_proposal,
            )
        )

    def _add_code_proposal_tool(self) -> None:
        self._add(
            Tool(
                name="propose_code_change",
                description=(
                    "Save a pending local code proposal for planning only. "
                    "Do not read arbitrary files, run shell commands, generate "
                    "patches, edit files, or create migrations. For UI requests, "
                    "assume the command center web UI unless the user explicitly "
                    "says CLI, terminal, or console. For sensitive configuration, "
                    "propose runtime config support without exposing .env, secrets, "
                    "credentials, tokens, database files, or runtime data. Prefer "
                    "Lewis's existing app/ package paths; do not suggest src/ "
                    "unless that directory exists."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "requested_goal": {"type": "string"},
                        "target_surface": {
                            "type": "string",
                            "description": (
                                "Optional intended target surface such as command "
                                "center web UI, CLI/terminal UI, backend/service "
                                "logic, or database/migrations. Lewis validates "
                                "and infers this from the proposal text."
                            ),
                        },
                        "relevant_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Relative Lewis project paths the user may want "
                                "reviewed later. Do not include secrets, runtime "
                                "files, or paths outside the project. Prefer "
                                "existing app/ paths for Python application code "
                                "and app/config.py for new config support unless "
                                "a matching config module already exists."
                            ),
                        },
                        "proposed_approach": {
                            "type": "string",
                            "description": "Human-readable plan only, not a diff.",
                        },
                        "patch_preview": {
                            "type": "string",
                            "description": (
                                "Optional human-reviewable patch or diff preview "
                                "stored as inert text only. Lewis validates file "
                                "paths mentioned in diff markers and does not "
                                "apply, write, or execute it."
                            ),
                        },
                        "safety_notes": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "suggested_test_commands": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["requested_goal", "proposed_approach"],
                    "additionalProperties": False,
                },
                function=self._code_proposal,
            )
        )

    def _add_memory_proposal_tool(self) -> None:
        self._add(
            Tool(
                name="propose_memory",
                description=(
                    "Create a pending local memory approval only when the user "
                    "explicitly asks Lewis to remember or save something. Does not "
                    "create an active memory."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "memory_type": {
                            "type": "string",
                            "enum": ["fact", "preference", "project_context"],
                        },
                        "content": {"type": "string"},
                        "source": {"type": "string"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["memory_type", "content"],
                    "additionalProperties": False,
                },
                function=self._memory_proposal,
            )
        )

    def _search_documents(self, args: dict[str, Any]) -> dict[str, Any]:
        unexpected = set(args) - {"query"}
        if unexpected:
            return {
                "ok": False,
                "error": "search_documents only accepts a query string.",
            }
        query = str(args.get("query", "")).strip()
        results = self.document_search.search(query=query)
        if not self.document_search.has_documents():
            return {"ok": True, "message": NO_DOCUMENTS_MESSAGE, "results": []}
        if not results:
            return {"ok": True, "message": NO_MATCHES_MESSAGE, "results": []}
        return {
            "ok": True,
            "results": [
                {"path": item.path, "score": item.score, "snippet": item.snippet}
                for item in results
            ],
        }

    def _list_local_files(self, args: dict[str, Any]) -> dict[str, Any]:
        del args
        return {"ok": True, "files": self.document_search.list_files()}

    def _search_project_files(self, args: dict[str, Any]) -> dict[str, Any]:
        unexpected = set(args) - {"query"}
        if unexpected:
            return {
                "ok": False,
                "error": "search_project_files only accepts a query string.",
            }
        results = self.project_file_search.search(str(args.get("query", "")))
        return {
            "ok": True,
            "results": [
                {
                    "path": item.path,
                    "line_number": item.line_number,
                    "snippet": item.snippet,
                }
                for item in results
            ],
        }

    def _list_project_files(self, args: dict[str, Any]) -> dict[str, Any]:
        if args:
            return {
                "ok": False,
                "error": "list_project_files does not accept arguments.",
            }
        return {"ok": True, "files": self.project_inspector.list_files()}

    def _read_project_file(self, args: dict[str, Any]) -> dict[str, Any]:
        unexpected = set(args) - {"path"}
        if unexpected:
            return {
                "ok": False,
                "error": "read_project_file only accepts a project-relative path.",
            }
        read = self.project_inspector.read_file(str(args.get("path", "")))
        return {
            "ok": True,
            "path": read.path,
            "content": read.content,
            "truncated": read.truncated,
            "size_bytes": read.size_bytes,
        }

    def _search_project(self, args: dict[str, Any]) -> dict[str, Any]:
        unexpected = set(args) - {"query"}
        if unexpected:
            return {"ok": False, "error": "search_project only accepts a query."}
        results = self.project_inspector.search(str(args.get("query", "")))
        return {
            "ok": True,
            "results": [
                {"path": item.path, "score": item.score, "snippet": item.snippet}
                for item in results
            ],
        }

    def _trace_symbol(self, args: dict[str, Any]) -> dict[str, Any]:
        unexpected = set(args) - {"symbol", "file", "mode"}
        if unexpected:
            return {
                "ok": False,
                "error": "trace_symbol only accepts symbol, file, and mode.",
            }
        symbol = str(args.get("symbol", "")).strip()
        if not symbol:
            return {"ok": False, "error": "trace_symbol requires a symbol name."}
        raw_file = args.get("file")
        file_hint = str(raw_file).strip() if raw_file is not None else None
        outcome = self.code_tracer.trace(
            symbol,
            mode=str(args.get("mode", "quick") or "quick"),
            file=file_hint or None,
        )
        if not outcome.result.ok:
            return {"ok": False, "error": outcome.result.error}
        return {
            "ok": True,
            "mode": outcome.result.mode,
            "report": outcome.report,
            "data": to_compact_dict(outcome.result),
        }

    def _get_current_time(self, args: dict[str, Any]) -> dict[str, Any]:
        requested_timezone = str(args.get("timezone", "America/New_York"))
        if ZoneInfo is not None:
            try:
                now = datetime.now(ZoneInfo(requested_timezone))
                return {
                    "ok": True,
                    "timezone": requested_timezone,
                    "iso": now.isoformat(),
                }
            except Exception:
                pass

        if requested_timezone.upper() == "UTC":
            now = datetime.now(timezone.utc)
            timezone_name = "UTC"
        else:
            now = datetime.now().astimezone()
            timezone_name = "local"
        return {"ok": True, "timezone": timezone_name, "iso": now.isoformat()}

    def _list_pending_approvals(self, args: dict[str, Any]) -> dict[str, Any]:
        del args
        return {"ok": True, "approvals": list_pending_approvals(self.approval_store)}

    def _list_tasks(self, args: dict[str, Any]) -> dict[str, Any]:
        del args
        return {"ok": True, "tasks": list_tasks(self.approval_store)}

    def _list_calendar_proposals(self, args: dict[str, Any]) -> dict[str, Any]:
        del args
        return {
            "ok": True,
            "calendar_proposals": list_calendar_proposals(self.approval_store),
        }

    def _get_dashboard_summary(self, args: dict[str, Any]) -> dict[str, Any]:
        del args
        return {
            "ok": True,
            "dashboard": get_dashboard_summary(self.approval_store, self.logger),
        }

    def _proposal(self, action_type: str, args: dict[str, Any]) -> dict[str, Any]:
        title = (
            str(args.get("title", "")).strip() or action_type.replace("_", " ").title()
        )
        details = args.get("details")
        if not isinstance(details, dict):
            return {"ok": False, "error": "Proposal details must be an object."}
        if action_type == "propose_email_draft":
            details = self._sanitize_email_details(details)
            existing = self._find_duplicate_pending_email(title, details)
            if existing is not None:
                return {
                    "ok": True,
                    "approval_id": existing.id,
                    "status": existing.status,
                    "message": (
                        f"Reused pending approval item #{existing.id}: {existing.title}"
                    ),
                }
        item = self.approval_store.create(
            action_type=action_type, title=title, payload=details
        )
        return {
            "ok": True,
            "approval_id": item.id,
            "status": item.status,
            "message": f"Saved pending approval item #{item.id}: {item.title}",
        }

    def _task_proposal(self, args: dict[str, Any]) -> dict[str, Any]:
        details = args.get("details")
        if details is not None and not isinstance(details, dict):
            return {"ok": False, "error": "Task details must be an object."}
        details = details if isinstance(details, dict) else {}

        title = str(
            args.get("title") or details.get("title") or details.get("task") or "Task"
        ).strip()
        description = str(
            args.get("description")
            or details.get("description")
            or details.get("task")
            or title
        ).strip()
        if not title or not description:
            return {"ok": False, "error": "Task title and description are required."}

        payload = {"title": title, "description": description}
        for key in ("due_date_text", "priority", "source"):
            value = args.get(key, details.get(key))
            if value is not None and str(value).strip():
                payload[key] = str(value).strip()

        item = self.approval_store.create(
            action_type="propose_task", title=title, payload=payload
        )
        return {
            "ok": True,
            "approval_id": item.id,
            "status": item.status,
            "message": (
                f"Saved pending task proposal #{item.id}: {item.title}. "
                "This is not a real reminder. Approval marks the task proposal "
                "approved; it does not mark the task done or completed."
            ),
        }

    def _calendar_proposal(self, args: dict[str, Any]) -> dict[str, Any]:
        details = args.get("details")
        if details is not None and not isinstance(details, dict):
            return {"ok": False, "error": "Calendar event details must be an object."}
        details = details if isinstance(details, dict) else {}

        title = str(
            args.get("title") or details.get("title") or "Calendar event"
        ).strip()
        description = str(
            args.get("description") or details.get("description") or title
        ).strip()
        if not title or not description:
            return {
                "ok": False,
                "error": "Calendar event title and description are required.",
            }

        payload: dict[str, Any] = {"title": title, "description": description}
        for key in (
            "date_text",
            "start_time_text",
            "end_time_text",
            "location",
            "source",
            "notes",
        ):
            value = args.get(key, details.get(key))
            payload[key] = str(value).strip() if value is not None else ""
        payload["attendees"] = self._normalize_attendees(
            args.get("attendees", details.get("attendees"))
        )

        item = self.approval_store.create(
            action_type="propose_calendar_event", title=title, payload=payload
        )
        return {
            "ok": True,
            "approval_id": item.id,
            "status": item.status,
            "message": (
                f"Saved pending local calendar event proposal #{item.id}: "
                f"{item.title}. This is not a real calendar event. Approval only "
                "marks the proposal approved; it does not create an event, send "
                "invites, notify attendees, or create reminders."
            ),
        }

    def _code_proposal(self, args: dict[str, Any]) -> dict[str, Any]:
        item = propose_code_change(
            store=self.approval_store,
            requested_goal=str(args.get("requested_goal", "")).strip(),
            relevant_files=args.get("relevant_files", []),
            proposed_approach=str(args.get("proposed_approach", "")).strip(),
            patch_preview=args.get("patch_preview"),
            target_surface=args.get("target_surface"),
            safety_notes=args.get("safety_notes", []),
            suggested_test_commands=args.get("suggested_test_commands", []),
        )
        return {
            "ok": True,
            "approval_id": item.id,
            "status": item.status,
            "action_type": item.action_type,
            "message": (
                f"Saved pending code proposal #{item.id}: {item.title.rstrip(' .')}. "
                "This is planning only; no files were read, edited, patched, "
                "or executed."
            ),
        }

    def _memory_proposal(self, args: dict[str, Any]) -> dict[str, Any]:
        tags = args.get("tags")
        if tags is not None and not isinstance(tags, list):
            return {"ok": False, "error": "Memory proposal tags must be a list."}
        item = propose_memory(
            memory_type=str(args.get("memory_type", "")).strip(),
            content=str(args.get("content", "")).strip(),
            source=str(args.get("source", "chat")).strip() or "chat",
            tags=[str(tag).strip() for tag in tags or [] if str(tag).strip()],
            db_path=self.approval_store.db_path,
            actor="assistant",
        )
        return {
            "ok": True,
            "approval_id": item.id,
            "status": item.status,
            "action_type": item.action_type,
            "message": (
                f"Created pending memory approval #{item.id}. Approval is required "
                "before this becomes an active memory."
            ),
        }

    @staticmethod
    def _normalize_attendees(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return [str(value).strip()] if str(value).strip() else []

    def _find_duplicate_pending_email(
        self, title: str, details: dict[str, Any]
    ) -> Any | None:
        for item in self.approval_store.list(status="pending"):
            if (
                item.action_type == "propose_email_draft"
                and item.title == title
                and item.payload == details
            ):
                return item
        return None

    def _sanitize_email_details(self, details: dict[str, Any]) -> dict[str, Any]:
        if self.user_name_configured:
            return details
        return {
            key: self._sanitize_email_value(value) for key, value in details.items()
        }

    def _sanitize_email_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._neutralize_signoff(value)
        if isinstance(value, dict):
            return {
                key: self._sanitize_email_value(inner_value)
                for key, inner_value in value.items()
            }
        if isinstance(value, list):
            return [self._sanitize_email_value(item) for item in value]
        return value

    @staticmethod
    def _neutralize_signoff(text: str) -> str:
        pattern = re.compile(
            r"(?P<signoff>(?:best regards|regards|sincerely|best),?)"
            r"\s*\n\s*[^\n]+$",
            re.IGNORECASE,
        )
        return pattern.sub(
            lambda match: match.group("signoff").rstrip().rstrip(",") + ",", text
        )
