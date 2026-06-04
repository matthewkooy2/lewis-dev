from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.approvals import ApprovalItem, ApprovalStore


EXECUTION_PREVIEWED_STATUS = "execution_previewed"
EXECUTABLE_STATUS = "approved"


@dataclass(frozen=True)
class ExecutionResult:
    approval_id: int
    action_type: str
    status: str
    message: str
    external_action_taken: bool = False


Executor = Callable[[ApprovalItem], ExecutionResult]


def execute_approved(
    store: ApprovalStore,
    approval_id: int,
    registry: dict[str, Executor] | None = None,
) -> ExecutionResult:
    item = store.get(approval_id)
    if item is None:
        raise KeyError(f"Approval item {approval_id} not found.")
    if item.status != EXECUTABLE_STATUS:
        raise ValueError(
            f"Approval item #{item.id} is {item.status}; only approved items can "
            "be execution-previewed."
        )

    executors = registry or build_executor_registry()
    executor = executors.get(item.action_type)
    if executor is None:
        raise ValueError(
            f"No executor is registered for action type {item.action_type}."
        )

    result = executor(item)
    updated = store.update_status(item.id, EXECUTION_PREVIEWED_STATUS)
    return ExecutionResult(
        approval_id=updated.id,
        action_type=updated.action_type,
        status=updated.status,
        message=result.message,
        external_action_taken=False,
    )


def build_executor_registry() -> dict[str, Executor]:
    return {
        "propose_email_draft": execute_email_draft_dry_run,
        "propose_task": execute_task_dry_run,
        "propose_calendar_event": execute_calendar_event_dry_run,
    }


def execute_email_draft_dry_run(item: ApprovalItem) -> ExecutionResult:
    to_value = _payload_value(item.payload, "to", "recipient", "recipients")
    subject = _payload_value(item.payload, "subject")
    target = f" to {to_value}" if to_value else ""
    subject_text = f" with subject '{subject}'" if subject else ""
    return _dry_run_result(
        item,
        f"Dry-run preview only: a future email integration would prepare an email draft{target}{subject_text}. "
        "No email was sent, no draft was created in an external mailbox, and no external system was touched.",
    )


def execute_task_dry_run(item: ApprovalItem) -> ExecutionResult:
    title = _payload_value(item.payload, "title") or item.title
    due_date = _payload_value(item.payload, "due_date_text")
    due_text = f" with due date '{due_date}'" if due_date else ""
    return _dry_run_result(
        item,
        f"Dry-run preview only: a future task integration would prepare task '{title}'{due_text}. "
        "No reminder, notification, file, external task, or external system was created.",
    )


def execute_calendar_event_dry_run(item: ApprovalItem) -> ExecutionResult:
    title = _payload_value(item.payload, "title") or item.title
    date_text = _payload_value(item.payload, "date_text")
    start_time = _payload_value(item.payload, "start_time_text")
    when_parts = [str(part) for part in (date_text, start_time) if part]
    when_text = f" for {' '.join(when_parts)}" if when_parts else ""
    return _dry_run_result(
        item,
        f"Dry-run preview only: a future calendar integration would prepare calendar event '{title}'{when_text}. "
        "No Outlook, Google Calendar, OS calendar event, invite, reminder, notification, email, or external action was created.",
    )


def _dry_run_result(item: ApprovalItem, message: str) -> ExecutionResult:
    return ExecutionResult(
        approval_id=item.id,
        action_type=item.action_type,
        status=EXECUTION_PREVIEWED_STATUS,
        message=message,
        external_action_taken=False,
    )


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
