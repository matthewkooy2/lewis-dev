from __future__ import annotations

import argparse
import sys

from app.approval_service import approve_approval, reject_approval
from app.approvals import ApprovalStore
from app.budget import BudgetManager
from app.code_proposals import CODE_PROPOSAL_ACTION_TYPE, propose_code_change
from app.code_proposal_previews import patch_preview_warning
from app.config import MISSING_API_KEY_MESSAGE, Settings, get_settings
from app.dashboard import (
    complete_task,
    format_dashboard_count_lines,
    get_dashboard_summary,
    list_calendar_proposals,
    list_tasks,
    reopen_task,
)
from app.executors import execute_approved
from app.llm import LewisAgent
from app.logging_db import RunLogger
from app.memory_service import (
    MEMORY_TYPES,
    archive_memory,
    list_active_memories,
    list_recent_memory_events,
    propose_memory,
    propose_memory_update,
)
from app.memory_store import Memory, MemoryEvent
from app.rag import DocumentSearch, NO_DOCUMENTS_MESSAGE, NO_MATCHES_MESSAGE
from app.tools import ToolRegistry


def build_agent(settings: Settings | None = None) -> LewisAgent:
    settings = settings or get_settings()
    approval_store = ApprovalStore(settings.database_path)
    document_search = DocumentSearch(settings.documents_dir)
    logger = RunLogger(settings.database_path)
    tools = ToolRegistry(
        document_search=document_search,
        approval_store=approval_store,
        logger=logger,
        user_name_configured=settings.user_name_configured,
    )
    budget = BudgetManager(settings)
    return LewisAgent(settings=settings, tools=tools, budget=budget, logger=logger)


def chat() -> int:
    settings = get_settings()
    if not settings.has_api_key:
        print(MISSING_API_KEY_MESSAGE)
        return 1

    agent = build_agent(settings)
    print("Lewis terminal assistant. Type 'exit' or 'quit' to leave.")
    prompt_name = settings.user_name if settings.user_name_configured else "You"
    while True:
        try:
            user_input = input(f"\n{prompt_name}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return 0
        if not user_input:
            continue
        try:
            run = agent.run(user_input)
        except RuntimeError as exc:
            print(f"Lewis> {exc}")
            return 1
        except Exception as exc:
            print(f"Lewis> Sorry, I hit an error: {exc}")
            continue
        print(f"Lewis> {run.answer}")


def print_approvals(store: ApprovalStore) -> None:
    items = store.list()
    if not items:
        print("No approval items found.")
        return
    for item in items:
        print(f"#{item.id} [{item.status}]")
        print_status_note(item.status)
        print(f"  type: {item.action_type}")
        print(f"  summary: {item.title}")
        print(f"  created: {item.created_at}")
        if item.action_type == "propose_email_draft":
            print_email_payload(item.payload)
        if item.action_type == "propose_task":
            print_task_payload(item.payload)
        if item.action_type == "propose_calendar_event":
            print_calendar_payload(item.payload)
        if item.action_type == "propose_memory":
            print_memory_payload(item.payload)
        if item.action_type == "propose_memory_update":
            print_memory_update_payload(item.payload)
        if item.action_type == CODE_PROPOSAL_ACTION_TYPE:
            print_code_proposal_payload(item.payload)
        print(f"  payload: {item.payload}")


def print_tasks(store: ApprovalStore) -> None:
    tasks = list_tasks(store)
    if not tasks:
        print("No proposed task items found.")
        return
    for task in tasks:
        print(f"#{task['approval_id']} [{task['status']}]")
        print_status_note(str(task["status"]))
        print(f"  approval_id: {task['approval_id']}")
        print(f"  status: {task['status']}")
        print(f"  title: {task['title']}")
        print(f"  description: {task['description']}")
        print(f"  due_date_text: {task['due_date_text']}")
        print(f"  priority: {task['priority']}")
        print(f"  created: {task['created_at']}")


def print_calendar_proposals(store: ApprovalStore) -> None:
    proposals = list_calendar_proposals(store)
    if not proposals:
        print("No local calendar proposals found.")
        return
    for proposal in proposals:
        print(f"#{proposal['approval_id']} [{proposal['status']}]")
        print_status_note(str(proposal["status"]))
        print(f"  approval_id: {proposal['approval_id']}")
        print(f"  status: {proposal['status']}")
        print(f"  title: {proposal['title']}")
        print(f"  date_text: {proposal['date_text']}")
        print(f"  start_time_text: {proposal['start_time_text']}")
        print(f"  end_time_text: {proposal['end_time_text']}")
        print(f"  location: {proposal['location']}")
        print(f"  attendees: {', '.join(proposal['attendees'])}")
        print(f"  description: {proposal['description']}")
        print(f"  created: {proposal['created_at']}")


def print_memories(memories: list[Memory]) -> None:
    if not memories:
        print("No active memories found.")
        return
    for memory in memories:
        print(f"#{memory.id} [{memory.status}]")
        print(f"  approval_id: {memory.approval_id}")
        print(f"  type: {memory.memory_type}")
        print("  content:")
        for line in memory.content.splitlines() or [""]:
            print(f"    {line}")
        print(f"  source: {memory.source}")
        if memory.tags:
            print(f"  tags: {', '.join(memory.tags)}")
        print(f"  created: {memory.created_at}")


def print_memory_events(events: list[MemoryEvent]) -> None:
    if not events:
        print("No memory audit events found.")
        return
    for event in events:
        print(f"#{event.id} {event.event_type}")
        print(f"  memory_id: {event.memory_id if event.memory_id is not None else ''}")
        print(
            f"  approval_id: {event.approval_id if event.approval_id is not None else ''}"
        )
        print(f"  actor: {event.actor}")
        print(f"  created: {event.created_at}")
        print(f"  details: {event.details}")


def print_email_payload(payload: dict[str, object]) -> None:
    to_value = _payload_value(payload, "to", "recipient", "recipients")
    subject = _payload_value(payload, "subject")
    body = _payload_value(payload, "body", "message", "draft")
    print("  email:")
    if to_value:
        print(f"    to: {to_value}")
    if subject:
        print(f"    subject: {subject}")
    if body:
        print("    body:")
        for line in str(body).splitlines():
            print(f"      {line}")


def print_status_note(status: str) -> None:
    if status == "execution_previewed":
        print("  note: execution was previewed locally; no external action was taken.")


def print_task_payload(payload: dict[str, object]) -> None:
    title = _payload_value(payload, "title") or "(untitled)"
    description = _payload_value(payload, "description") or ""
    due_date_text = _payload_value(payload, "due_date_text")
    priority = _payload_value(payload, "priority")
    source = _payload_value(payload, "source")
    print("  task:")
    print(f"    title: {title}")
    print(f"    description: {description}")
    if due_date_text:
        print(f"    due_date_text: {due_date_text}")
    if priority:
        print(f"    priority: {priority}")
    if source:
        print(f"    source: {source}")


def print_calendar_payload(payload: dict[str, object]) -> None:
    title = _payload_value(payload, "title") or "(untitled)"
    description = _payload_value(payload, "description") or ""
    date_text = _payload_value(payload, "date_text") or ""
    start_time_text = _payload_value(payload, "start_time_text") or ""
    end_time_text = _payload_value(payload, "end_time_text") or ""
    location = _payload_value(payload, "location") or ""
    attendees = payload.get("attendees") or []
    if isinstance(attendees, str):
        attendees_text = attendees
    elif isinstance(attendees, list):
        attendees_text = ", ".join(str(item) for item in attendees if str(item))
    else:
        attendees_text = str(attendees) if attendees else ""
    source = _payload_value(payload, "source")
    notes = _payload_value(payload, "notes")
    print("  calendar_event:")
    print(f"    title: {title}")
    print(f"    date_text: {date_text}")
    print(f"    start_time_text: {start_time_text}")
    print(f"    end_time_text: {end_time_text}")
    print(f"    location: {location}")
    print(f"    attendees: {attendees_text}")
    print(f"    description: {description}")
    if source:
        print(f"    source: {source}")
    if notes:
        print(f"    notes: {notes}")


def print_memory_payload(payload: dict[str, object]) -> None:
    memory_type = _payload_value(payload, "memory_type") or ""
    content = _payload_value(payload, "content") or ""
    source = _payload_value(payload, "source") or ""
    tags = payload.get("tags_json", payload.get("tags", []))
    if isinstance(tags, list):
        tags_text = ", ".join(str(tag) for tag in tags if str(tag))
    else:
        tags_text = str(tags) if tags else ""
    print("  memory:")
    print(f"    type: {memory_type}")
    print("    content:")
    for line in str(content).splitlines() or [""]:
        print(f"      {line}")
    if source:
        print(f"    source: {source}")
    if tags_text:
        print(f"    tags: {tags_text}")


def print_memory_update_payload(payload: dict[str, object]) -> None:
    print("  memory_update:")
    print(f"    memory_id: {_payload_value(payload, 'memory_id') or ''}")
    print("    original:")
    print(f"      type: {_payload_value(payload, 'original_memory_type') or ''}")
    original_content = _payload_value(payload, "original_content") or ""
    print("      content:")
    for line in str(original_content).splitlines() or [""]:
        print(f"        {line}")
    print("    replacement:")
    print(f"      type: {_payload_value(payload, 'replacement_memory_type') or ''}")
    replacement_content = _payload_value(payload, "replacement_content") or ""
    print("      content:")
    for line in str(replacement_content).splitlines() or [""]:
        print(f"        {line}")
    replacement_tags = payload.get("replacement_tags_json", [])
    if isinstance(replacement_tags, list) and replacement_tags:
        print(f"      tags: {', '.join(str(tag) for tag in replacement_tags)}")
    reason = _payload_value(payload, "reason")
    if reason:
        print(f"    reason: {reason}")


def print_code_proposal_payload(payload: dict[str, object]) -> None:
    print("  code_proposal:")
    print("    requested_goal:")
    for line in str(_payload_value(payload, "requested_goal") or "").splitlines() or [
        ""
    ]:
        print(f"      {line}")
    print(f"    target_surface: {_payload_value(payload, 'target_surface') or ''}")
    relevant_files = payload.get("relevant_files") or []
    if not isinstance(relevant_files, list):
        relevant_files = []
    print("    relevant_files:")
    if relevant_files:
        for path in relevant_files:
            print(f"      - {path}")
    else:
        print("      -")
    print("    proposed_approach:")
    for line in str(
        _payload_value(payload, "proposed_approach") or ""
    ).splitlines() or [""]:
        print(f"      {line}")
    patch_preview = _payload_value(payload, "patch_preview")
    if patch_preview:
        print("    patch_preview:")
        print(
            "      implementation_ready: "
            f"{_yes_no(_patch_preview_implementation_ready(payload))}"
        )
        warning = _patch_preview_warning(payload)
        if warning:
            print(f"      warning: {warning}")
        for line in str(patch_preview).splitlines() or [""]:
            print(f"      {line}")
    print_list_payload("safety_notes", payload.get("safety_notes"))
    print_list_payload(
        "suggested_test_commands", payload.get("suggested_test_commands")
    )


def print_list_payload(label: str, value: object) -> None:
    items = value if isinstance(value, list) else []
    print(f"    {label}:")
    if not items:
        print("      -")
        return
    for item in items:
        print(f"      - {item}")


def _patch_preview_implementation_ready(payload: dict[str, object]) -> bool:
    value = payload.get("patch_preview_implementation_ready")
    if isinstance(value, bool):
        return value
    return not bool(_patch_preview_warning(payload))


def _patch_preview_warning(payload: dict[str, object]) -> str:
    explicit_warning = _payload_value(payload, "patch_preview_warning")
    if explicit_warning:
        return str(explicit_warning)
    classifications = payload.get("patch_preview_path_classifications")
    if isinstance(classifications, list):
        items = [item for item in classifications if isinstance(item, dict)]
        return patch_preview_warning(items)
    return ""


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _payload_value(payload: dict[str, object], *keys: str) -> object | None:
    for key in keys:
        value = payload.get(key)
        if value:
            return value
    for value in payload.values():
        if isinstance(value, dict):
            nested = _payload_value(value, *keys)
            if nested:
                return nested
    return None


def print_usage(logger: RunLogger, limit: int) -> None:
    summary = logger.recent_usage_summary(limit)
    print(f"Usage summary for latest {summary.run_count} logged run(s):")
    print(f"  total prompt/input tokens: {summary.input_tokens}")
    print(f"  total output tokens: {summary.output_tokens}")
    print(f"  total cache-hit input tokens: {summary.prompt_cache_hit_tokens}")
    print(f"  total cache-miss input tokens: {summary.prompt_cache_miss_tokens}")
    print(f"  estimated cost: ${summary.estimated_cost:.6f}")
    print(f"  API calls: {summary.api_calls}")


def print_dashboard(store: ApprovalStore, logger: RunLogger) -> None:
    dashboard = get_dashboard_summary(store, logger)
    counts = dashboard["counts"]
    usage = dashboard["usage_summary"]
    print("Lewis local dashboard")
    print("Counts:")
    for line in format_dashboard_count_lines(counts, indent="  "):
        print(line)
    print("Recent usage:")
    if usage and usage["run_count"]:
        print(f"  runs: {usage['run_count']}")
        print(f"  input tokens: {usage['input_tokens']}")
        print(f"  output tokens: {usage['output_tokens']}")
        print(f"  estimated cost: ${usage['estimated_cost']:.6f}")
        print(f"  API calls: {usage['api_calls']}")
    else:
        print("  No usage logs yet.")
    print("Recent approval/task items:")
    if not dashboard["recent_items"]:
        print("  No approval or task items found.")
        return
    for item in dashboard["recent_items"]:
        print(
            f"  #{item['approval_id']} [{item['status']}] "
            f"{item['action_type']}: {item['title']}"
        )


def print_document_search(search: DocumentSearch, query: str) -> None:
    results = search.search(query)
    if not search.has_documents():
        print(NO_DOCUMENTS_MESSAGE)
        return
    if not results:
        print(NO_MATCHES_MESSAGE)
        return
    for item in results:
        print(f"{item.path} (score: {item.score})")
        print(f"  {item.snippet}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lewis")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("chat")
    subparsers.add_parser("web")
    subparsers.add_parser("dashboard")
    subparsers.add_parser("approvals")
    subparsers.add_parser("tasks")
    subparsers.add_parser("calendar-proposals")
    usage_parser = subparsers.add_parser("usage")
    usage_parser.add_argument("--limit", type=int, default=50)
    propose_memory_parser = subparsers.add_parser("propose-memory")
    propose_memory_parser.add_argument(
        "--type", required=True, choices=sorted(MEMORY_TYPES)
    )
    propose_memory_parser.add_argument("--content", required=True)
    propose_memory_parser.add_argument("--source", default="cli")
    propose_memory_parser.add_argument("--tag", action="append", default=[])
    propose_memory_update_parser = subparsers.add_parser("propose-memory-update")
    propose_memory_update_parser.add_argument("memory_id", type=int)
    propose_memory_update_parser.add_argument(
        "--type", required=True, choices=sorted(MEMORY_TYPES)
    )
    propose_memory_update_parser.add_argument("--content", required=True)
    propose_memory_update_parser.add_argument("--source", default="cli")
    propose_memory_update_parser.add_argument("--tag", action="append", default=[])
    propose_memory_update_parser.add_argument("--reason", default="")
    propose_code_parser = subparsers.add_parser("propose-code")
    propose_code_parser.add_argument("--goal", required=True)
    propose_code_parser.add_argument("--file", action="append", default=[])
    propose_code_parser.add_argument("--approach", required=True)
    propose_code_parser.add_argument("--patch-preview", default="")
    propose_code_parser.add_argument("--safety-note", action="append", default=[])
    propose_code_parser.add_argument("--test-command", action="append", default=[])
    subparsers.add_parser("memories")
    memory_events_parser = subparsers.add_parser("memory-events")
    memory_events_parser.add_argument("--limit", type=int, default=20)
    archive_memory_parser = subparsers.add_parser("archive-memory")
    archive_memory_parser.add_argument("memory_id", type=int)
    archive_memory_parser.add_argument("--reason", default="")
    search_parser = subparsers.add_parser("search-docs")
    search_parser.add_argument("query")
    approve_parser = subparsers.add_parser("approve")
    approve_parser.add_argument("approval_id", type=int)
    reject_parser = subparsers.add_parser("reject")
    reject_parser.add_argument("approval_id", type=int)
    execute_parser = subparsers.add_parser("execute-approved")
    execute_parser.add_argument("approval_id", type=int)
    complete_parser = subparsers.add_parser("complete-task")
    complete_parser.add_argument("approval_id", type=int)
    reopen_parser = subparsers.add_parser("reopen-task")
    reopen_parser.add_argument("approval_id", type=int)

    args = parser.parse_args(argv)
    settings = get_settings()
    store = ApprovalStore(settings.database_path)
    logger = RunLogger(settings.database_path)

    if args.command == "chat":
        return chat()
    if args.command == "web":
        from app.web import run_web_server

        run_web_server(settings)
        return 0
    if args.command == "approvals":
        print_approvals(store)
        return 0
    if args.command == "dashboard":
        print_dashboard(store, logger)
        return 0
    if args.command == "tasks":
        print_tasks(store)
        return 0
    if args.command == "calendar-proposals":
        print_calendar_proposals(store)
        return 0
    if args.command == "usage":
        print_usage(logger, max(args.limit, 1))
        return 0
    if args.command == "propose-memory":
        try:
            item = propose_memory(
                memory_type=args.type,
                content=args.content,
                source=args.source,
                tags=args.tag,
                db_path=settings.database_path,
                actor="user",
            )
        except ValueError as exc:
            print(f"Could not propose memory: {exc}")
            return 1
        print(f"Saved pending memory approval #{item.id}.")
        print("No active memory was created; approval is required first.")
        return 0
    if args.command == "propose-memory-update":
        try:
            item = propose_memory_update(
                memory_id=args.memory_id,
                memory_type=args.type,
                content=args.content,
                source=args.source,
                tags=args.tag,
                reason=args.reason,
                db_path=settings.database_path,
                actor="user",
            )
        except (KeyError, ValueError) as exc:
            print(f"Could not propose memory update: {exc}")
            return 1
        print(f"Saved pending memory update approval #{item.id}.")
        print("The active memory was not changed; approval is required first.")
        return 0
    if args.command == "propose-code":
        try:
            item = propose_code_change(
                store=store,
                requested_goal=args.goal,
                relevant_files=args.file,
                proposed_approach=args.approach,
                patch_preview=args.patch_preview,
                safety_notes=args.safety_note,
                suggested_test_commands=args.test_command,
            )
        except ValueError as exc:
            print(f"Could not propose code change: {exc}")
            return 1
        print(f"Saved pending code proposal #{item.id}.")
        print("Planning only: no files were read, edited, patched, or executed.")
        return 0
    if args.command == "memories":
        print_memories(list_active_memories(settings.database_path))
        return 0
    if args.command == "memory-events":
        print_memory_events(
            list_recent_memory_events(settings.database_path, max(args.limit, 1))
        )
        return 0
    if args.command == "archive-memory":
        try:
            memory = archive_memory(
                settings.database_path,
                args.memory_id,
                reason=args.reason,
                actor="user",
            )
        except (KeyError, ValueError) as exc:
            print(f"Could not archive memory: {exc}")
            return 1
        print(f"Archived memory #{memory.id}.")
        print("The memory row was preserved and is no longer active.")
        return 0
    if args.command == "search-docs":
        print_document_search(DocumentSearch(settings.documents_dir), args.query)
        return 0
    if args.command == "approve":
        item = approve_approval(store, args.approval_id)
        if item.action_type == "propose_task":
            print(
                f"Approved #{item.id}. This only marks the task proposal approved; "
                "no reminder, notification, calendar event, or external task was created."
            )
            return 0
        if item.action_type == "propose_calendar_event":
            print(
                f"Approved #{item.id}. This only marks the local calendar proposal "
                "approved; no Outlook, Google Calendar, OS calendar event, invite, "
                "notification, reminder, email, or external action was created."
            )
            return 0
        if item.action_type == CODE_PROPOSAL_ACTION_TYPE:
            print(
                f"Approved #{item.id}. This only marks the local code proposal "
                "approved for review; no files were read, edited, patched, or executed."
            )
            return 0
        print(f"Approved #{item.id}. No external action was executed.")
        return 0
    if args.command == "reject":
        item = reject_approval(store, args.approval_id)
        print(f"Rejected #{item.id}.")
        return 0
    if args.command == "execute-approved":
        try:
            result = execute_approved(store, args.approval_id)
        except (KeyError, ValueError) as exc:
            print(f"Could not execute approved item: {exc}")
            return 1
        print(f"Execution previewed for #{result.approval_id} [{result.status}].")
        print(result.message)
        print("No external action was taken.")
        return 0
    if args.command == "complete-task":
        try:
            item = complete_task(store, args.approval_id)
        except (KeyError, ValueError) as exc:
            print(f"Could not complete task: {exc}")
            return 1
        print(
            f"Completed local task proposal #{item.id}. "
            "No emails, reminders, calendar events, notifications, or external "
            "actions were executed."
        )
        return 0
    if args.command == "reopen-task":
        try:
            item = reopen_task(store, args.approval_id)
        except (KeyError, ValueError) as exc:
            print(f"Could not reopen task: {exc}")
            return 1
        print(
            f"Reopened local task proposal #{item.id} as approved. "
            "No external actions were executed."
        )
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
