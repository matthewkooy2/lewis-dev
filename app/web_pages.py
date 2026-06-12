from __future__ import annotations

from typing import Any

from app.dashboard import DASHBOARD_COUNT_LABELS
from app.web_components import (
    approval_cards_html,
    calendar_cards_html,
    chat_bubble,
    empty_state_html,
    escape,
    flash,
    history_filters_html,
    memory_cards_html,
    memory_events_html,
    metric,
    page_header,
    task_cards_html,
    usage_runs_html,
    usage_summary_html,
)


DOCUMENT_SEARCH_NOTICE = (
    "Lewis only searches data/documents/. It cannot search Desktop, Downloads, "
    "Documents, OneDrive, AppData, .ssh, C:\\, or arbitrary paths."
)


def dashboard_page(data: dict[str, Any]) -> str:
    counts = data["counts"]
    usage = data["usage_summary"]
    recent_items = data["recent_items"]
    content = [
        page_header(
            "Dashboard",
            "Local command center. Approvals and execution previews stay on this machine.",
            "<a class='button' href='/history'>View History</a>",
        ),
        "<section class='metrics'>",
        *[metric(label, counts.get(key, 0)) for key, label in DASHBOARD_COUNT_LABELS],
        "</section>",
        "<section class='band'>",
        "<h2>Recent Usage Summary</h2>",
        usage_summary_html(usage),
        "</section>",
        "<section class='band'>",
        "<h2>Recent Approval, Task, And Calendar Items</h2>",
        approval_cards_html(recent_items, "dashboard"),
        "</section>",
    ]
    return "".join(content)


def approvals_page(items: list[dict[str, Any]], message: str = "") -> str:
    content = [
        page_header(
            "Approvals",
            "Pending local proposals only. Approving or rejecting changes local status only.",
        ),
        flash(message),
        approval_cards_html(items, "approvals"),
    ]
    return "".join(content)


def tasks_page(tasks: list[dict[str, Any]], message: str = "") -> str:
    content = [
        page_header(
            "Tasks",
            "Local task proposals only. These are not real reminders, alerts, or external task records.",
        ),
        flash(message),
        task_cards_html(tasks),
    ]
    return "".join(content)


def calendar_page(proposals: list[dict[str, Any]], message: str = "") -> str:
    content = [
        page_header(
            "Calendar Proposals",
            "These are local calendar proposals only. No Outlook, Google Calendar, OS calendar event, invites, reminders, or notifications are created.",
        ),
        flash(message),
        calendar_cards_html(proposals),
    ]
    return "".join(content)


def memories_page(memories: list[Any], events: list[Any], message: str = "") -> str:
    content = [
        page_header(
            "Memories",
            "User-visible active memories and local audit events. Active memories are added to chat only as transient local context.",
        ),
        flash(message),
        "<section class='band'>",
        "<h2>Active Memories</h2>",
        memory_cards_html(memories),
        "</section>",
        "<section class='band'>",
        "<h2>Recent Memory Audit Events</h2>",
        memory_events_html(events),
        "</section>",
    ]
    return "".join(content)


def history_page(items: list[dict[str, Any]], selected: str, message: str = "") -> str:
    content = [
        page_header(
            "History",
            "Processed local records. These entries do not imply that real email, calendar, reminder, notification, or external execution occurred.",
        ),
        flash(message),
        history_filters_html(selected),
        approval_cards_html(items, "history"),
    ]
    return "".join(content)


def usage_page(summary: Any, runs: list[Any]) -> str:
    content = [
        page_header("Usage", "Recent local usage logs from the Lewis SQLite database."),
        "<section class='band'>",
        "<h2>Summary</h2>",
        usage_summary_html(
            {
                "run_count": summary.run_count,
                "input_tokens": summary.input_tokens,
                "output_tokens": summary.output_tokens,
                "prompt_cache_hit_tokens": summary.prompt_cache_hit_tokens,
                "prompt_cache_miss_tokens": summary.prompt_cache_miss_tokens,
                "reasoning_tokens": summary.reasoning_tokens,
                "estimated_cost": summary.estimated_cost,
                "api_calls": summary.api_calls,
            }
        ),
        "</section>",
        usage_runs_html(runs),
    ]
    return "".join(content)


def document_search_page(query: str, results: list[Any], message: str) -> str:
    result_html = ""
    if results:
        cards = []
        for result in results:
            cards.append(
                "<article class='card'>"
                f"<h3>{escape(result.path)}</h3>"
                f"<div class='meta'>score: {escape(result.score)}</div>"
                f"<p>{escape(result.snippet)}</p>"
                "</article>"
            )
        result_html = "<section class='grid'>" + "".join(cards) + "</section>"
    elif message:
        result_html = empty_state_html(message)
    return (
        f"{page_header('Local Document Search', DOCUMENT_SEARCH_NOTICE)}"
        "<section class='band'><form class='search-form' method='get' action='/documents'>"
        f"<input name='query' value='{escape(query)}' placeholder='Search data/documents/'>"
        "<button type='submit'>Search</button></form></section>"
        f"{result_html}"
    )


def chat_page(messages: list[dict[str, str]]) -> str:
    empty_hint = (
        "<p id='chat-empty' class='chat-empty'>No messages yet. Ask Lewis anything; "
        "proposals still go through local approvals.</p>"
        if not messages
        else ""
    )
    content = [
        page_header(
            "Chat",
            "Uses the existing Lewis session behavior when an API key is configured. No external actions are executed; approvals are still required.",
        ),
        "<section class='chat-shell'>",
        "<div id='chat-log' class='chat-log'>",
        empty_hint,
        *[
            chat_bubble(
                item["role"],
                item["content"],
                int(item.get("memory_context_count", 0) or 0),
            )
            for item in messages
        ],
        "</div>",
        "<form id='chat-form' class='chat-form'>",
        "<textarea id='chat-input' name='message' rows='3' placeholder='Ask Lewis...'></textarea>",
        "<button type='submit'>Send</button>",
        "</form>",
        "<p class='muted'>Local web chat does not add browser automation, Gmail, Outlook, Google Calendar, or real execution.</p>",
        "</section>",
        "<script src='/static/chat.js'></script>",
    ]
    return "".join(content)
