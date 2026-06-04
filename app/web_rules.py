from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.dashboard import (
    CALENDAR_ACTION_TYPE,
    EMAIL_DRAFT_ACTION_TYPE,
    HISTORY_TYPE_FILTERS,
    TASK_ACTION_TYPE,
)


HISTORY_FILTER_LABELS = (
    ("all", "all"),
    ("drafts", "drafts"),
    ("tasks", "tasks"),
    ("calendar", "calendar events"),
)

ACTION_CONTEXTS = {"approvals", "tasks", "calendar"}
DRY_RUN_ACTION_TYPES = {EMAIL_DRAFT_ACTION_TYPE, CALENDAR_ACTION_TYPE}


@dataclass(frozen=True)
class ActionButton:
    path: str
    label: str
    class_name: str = ""


@dataclass(frozen=True)
class TimestampDisplay:
    meta_label: str
    field_label: str
    value: object


def normalize_history_filter(value: str) -> str:
    if value in HISTORY_TYPE_FILTERS:
        return value
    return "all"


def timestamp_display(item: dict[str, Any], context: str) -> TimestampDisplay:
    if context == "history":
        return TimestampDisplay(
            meta_label="last changed",
            field_label="last changed",
            value=item["updated_at"],
        )
    return TimestampDisplay(
        meta_label="created",
        field_label="created time",
        value=item["created_at"],
    )


def visible_actions(item: dict[str, Any], context: str) -> list[ActionButton]:
    if context == "history":
        return []

    approval_id = item["approval_id"]
    action_type = item["action_type"]
    status = item["status"]
    actions: list[ActionButton] = []

    if status == "pending" and context in ACTION_CONTEXTS:
        actions.append(
            ActionButton(
                path=f"/approvals/{approval_id}/approve?return_to=/{context}",
                label="approve",
            )
        )
        actions.append(
            ActionButton(
                path=f"/approvals/{approval_id}/reject?return_to=/{context}",
                label="reject",
                class_name="secondary",
            )
        )

    if status == "approved" and action_type in DRY_RUN_ACTION_TYPES:
        actions.append(
            ActionButton(
                path=f"/approvals/{approval_id}/execute-approved?return_to=/{context}",
                label="dry-run preview",
                class_name="warn",
            )
        )

    if status == "approved" and action_type == TASK_ACTION_TYPE and context == "tasks":
        actions.append(
            ActionButton(
                path=f"/tasks/{approval_id}/complete",
                label="complete task",
            )
        )

    return actions
