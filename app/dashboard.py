from __future__ import annotations

from typing import Any

from app.approvals import ApprovalItem, ApprovalStore
from app.logging_db import RunLogger


TASK_ACTION_TYPE = "propose_task"
CALENDAR_ACTION_TYPE = "propose_calendar_event"
EMAIL_DRAFT_ACTION_TYPE = "propose_email_draft"

DASHBOARD_COUNT_LABELS = (
    ("total_pending", "total pending"),
    ("pending_drafts", "pending drafts"),
    ("pending_tasks", "pending tasks"),
    ("pending_calendar_events", "pending calendar events"),
    ("approved_tasks", "approved tasks"),
    ("completed_tasks", "completed tasks"),
    ("approved_calendar_events", "approved calendar events"),
    ("execution_previewed_items", "execution previewed items"),
    ("total_rejected", "total rejected"),
)

ACTIVE_TASK_STATUSES = {"pending", "approved"}
ACTIVE_CALENDAR_STATUSES = {"pending", "approved"}
HISTORY_STATUSES = {"approved", "rejected", "completed", "execution_previewed"}
HISTORY_TYPE_FILTERS = {
    "all": None,
    "drafts": EMAIL_DRAFT_ACTION_TYPE,
    "tasks": TASK_ACTION_TYPE,
    "calendar": CALENDAR_ACTION_TYPE,
}


def task_payload_value(payload: dict[str, Any], key: str) -> object | None:
    value = payload.get(key)
    return value if value not in (None, "") else None


def serialize_task(item: ApprovalItem) -> dict[str, Any]:
    return {
        "approval_id": item.id,
        "status": item.status,
        "title": task_payload_value(item.payload, "title") or item.title,
        "description": task_payload_value(item.payload, "description") or "",
        "due_date_text": task_payload_value(item.payload, "due_date_text") or "",
        "priority": task_payload_value(item.payload, "priority") or "",
        "created_at": item.created_at,
    }


def calendar_payload_value(payload: dict[str, Any], key: str) -> object | None:
    value = payload.get(key)
    return value if value not in (None, "") else None


def serialize_calendar_proposal(item: ApprovalItem) -> dict[str, Any]:
    attendees = calendar_payload_value(item.payload, "attendees") or []
    if isinstance(attendees, str):
        attendees = [attendees] if attendees else []
    if not isinstance(attendees, list):
        attendees = []
    return {
        "approval_id": item.id,
        "status": item.status,
        "title": calendar_payload_value(item.payload, "title") or item.title,
        "description": calendar_payload_value(item.payload, "description") or "",
        "date_text": calendar_payload_value(item.payload, "date_text") or "",
        "start_time_text": calendar_payload_value(item.payload, "start_time_text")
        or "",
        "end_time_text": calendar_payload_value(item.payload, "end_time_text") or "",
        "location": calendar_payload_value(item.payload, "location") or "",
        "attendees": attendees,
        "source": calendar_payload_value(item.payload, "source") or "",
        "notes": calendar_payload_value(item.payload, "notes") or "",
        "created_at": item.created_at,
    }


def serialize_approval(item: ApprovalItem) -> dict[str, Any]:
    return {
        "approval_id": item.id,
        "action_type": item.action_type,
        "status": item.status,
        "title": item.title,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
        "payload": item.payload,
    }


def list_pending_approvals(store: ApprovalStore) -> list[dict[str, Any]]:
    return [serialize_approval(item) for item in store.list(status="pending")]


def list_tasks(store: ApprovalStore) -> list[dict[str, Any]]:
    return [
        serialize_task(item)
        for item in store.list()
        if item.action_type == TASK_ACTION_TYPE and item.status in ACTIVE_TASK_STATUSES
    ]


def list_calendar_proposals(store: ApprovalStore) -> list[dict[str, Any]]:
    return [
        serialize_calendar_proposal(item)
        for item in store.list()
        if (
            item.action_type == CALENDAR_ACTION_TYPE
            and item.status in ACTIVE_CALENDAR_STATUSES
        )
    ]


def list_history_items(
    store: ApprovalStore, type_filter: str = "all"
) -> list[dict[str, Any]]:
    action_type = HISTORY_TYPE_FILTERS.get(type_filter)
    return [
        serialize_approval(item)
        for item in sorted(
            (
                item
                for item in store.list()
                if item.status in HISTORY_STATUSES
                and (action_type is None or item.action_type == action_type)
            ),
            key=lambda item: (item.updated_at, item.id),
            reverse=True,
        )
    ]


def get_dashboard_summary(
    store: ApprovalStore, logger: RunLogger | None = None, limit: int = 5
) -> dict[str, Any]:
    items = store.list()
    draft_items = [
        item for item in items if item.action_type == EMAIL_DRAFT_ACTION_TYPE
    ]
    task_items = [item for item in items if item.action_type == TASK_ACTION_TYPE]
    calendar_items = [
        item for item in items if item.action_type == CALENDAR_ACTION_TYPE
    ]
    usage = logger.recent_usage_summary() if logger is not None else None

    return {
        "counts": {
            "total_pending": _count_status(items, "pending"),
            "pending_drafts": _count_status(draft_items, "pending"),
            "pending_tasks": _count_status(task_items, "pending"),
            "pending_calendar_events": _count_status(calendar_items, "pending"),
            "approved_tasks": _count_status(task_items, "approved"),
            "completed_tasks": _count_status(task_items, "completed"),
            "execution_previewed_items": _count_status(items, "execution_previewed"),
            "approved_calendar_events": _count_status(calendar_items, "approved"),
            "total_rejected": _count_status(items, "rejected"),
        },
        "usage_summary": (
            {
                "run_count": usage.run_count,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "prompt_cache_hit_tokens": usage.prompt_cache_hit_tokens,
                "prompt_cache_miss_tokens": usage.prompt_cache_miss_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
                "estimated_cost": usage.estimated_cost,
                "api_calls": usage.api_calls,
            }
            if usage is not None
            else None
        ),
        "recent_items": [serialize_approval(item) for item in items[:limit]],
        "recent_tasks": [serialize_task(item) for item in task_items[:limit]],
        "recent_calendar_proposals": [
            serialize_calendar_proposal(item) for item in calendar_items[:limit]
        ],
    }


def complete_task(store: ApprovalStore, approval_id: int) -> ApprovalItem:
    item = _get_task_or_raise(store, approval_id)
    if item.status == "rejected":
        raise ValueError("Rejected task proposals cannot be completed.")
    return store.update_status(approval_id, "completed")


def reopen_task(store: ApprovalStore, approval_id: int) -> ApprovalItem:
    item = _get_task_or_raise(store, approval_id)
    if item.status != "completed":
        raise ValueError("Only completed task proposals can be reopened.")
    return store.update_status(approval_id, "approved")


def _get_task_or_raise(store: ApprovalStore, approval_id: int) -> ApprovalItem:
    item = store.get(approval_id)
    if item is None:
        raise KeyError(f"Approval item {approval_id} not found.")
    if item.action_type != TASK_ACTION_TYPE:
        raise ValueError("Only propose_task items can be managed as tasks.")
    return item


def _count_status(items: list[ApprovalItem], status: str) -> int:
    return sum(1 for item in items if item.status == status)


def format_dashboard_count_lines(counts: dict[str, int], indent: str = "") -> list[str]:
    return [
        f"{indent}{label}: {counts.get(key, 0)}"
        for key, label in DASHBOARD_COUNT_LABELS
    ]
