import json
import sqlite3

import pytest

from app.approvals import ApprovalStore
from app.executors import execute_approved


def _approved_item(store, action_type, title, payload):
    item = store.create(action_type=action_type, title=title, payload=payload)
    return store.update_status(item.id, "approved")


def test_executing_pending_item_fails_safely(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    item = store.create(
        action_type="propose_email_draft",
        title="Email Alex",
        payload={"to": "Alex", "body": "Hello"},
    )

    with pytest.raises(ValueError, match="only approved items"):
        execute_approved(store, item.id)

    assert store.get(item.id).status == "pending"


def test_executing_rejected_item_fails_safely(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    item = store.create(
        action_type="propose_task",
        title="Call dentist",
        payload={"title": "Call dentist", "description": "Schedule a cleaning."},
    )
    store.update_status(item.id, "rejected")

    with pytest.raises(ValueError, match="only approved items"):
        execute_approved(store, item.id)

    assert store.get(item.id).status == "rejected"


def test_executing_completed_task_fails_safely(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    item = store.create(
        action_type="propose_task",
        title="Call dentist",
        payload={"title": "Call dentist", "description": "Schedule a cleaning."},
    )
    store.update_status(item.id, "completed")

    with pytest.raises(ValueError, match="only approved items"):
        execute_approved(store, item.id)

    assert store.get(item.id).status == "completed"


def test_approved_email_draft_only_produces_dry_run_result(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    item = _approved_item(
        store,
        "propose_email_draft",
        "Email Alex",
        {"to": "Alex", "subject": "Hello", "body": "Hi Alex"},
    )

    result = execute_approved(store, item.id)

    assert result.status == "execution_previewed"
    assert result.external_action_taken is False
    assert "Dry-run preview only" in result.message
    assert "No email was sent" in result.message
    assert store.get(item.id).status == "execution_previewed"


def test_approved_calendar_event_only_produces_dry_run_result(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    item = _approved_item(
        store,
        "propose_calendar_event",
        "Project sync",
        {
            "title": "Project sync",
            "description": "Discuss launch.",
            "date_text": "Tuesday",
            "start_time_text": "2:00 PM",
        },
    )

    result = execute_approved(store, item.id)

    assert result.status == "execution_previewed"
    assert result.external_action_taken is False
    assert "Dry-run preview only" in result.message
    assert "No Outlook" in result.message
    assert "external action was created" in result.message
    assert store.get(item.id).status == "execution_previewed"


def test_approved_task_only_produces_dry_run_result(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    item = _approved_item(
        store,
        "propose_task",
        "Call dentist",
        {
            "title": "Call dentist",
            "description": "Schedule a cleaning.",
            "due_date_text": "Friday",
        },
    )

    result = execute_approved(store, item.id)

    assert result.status == "execution_previewed"
    assert result.external_action_taken is False
    assert "Dry-run preview only" in result.message
    assert "No reminder" in result.message
    assert "external system was created" in result.message
    assert store.get(item.id).status == "execution_previewed"


def test_no_executor_sends_email_or_creates_calendar_events(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    items = [
        _approved_item(
            store,
            "propose_email_draft",
            "Email Alex",
            {"to": "Alex", "body": "Hello"},
        ),
        _approved_item(
            store,
            "propose_calendar_event",
            "Project sync",
            {"title": "Project sync", "description": "Discuss launch."},
        ),
        _approved_item(
            store,
            "propose_task",
            "Call dentist",
            {"title": "Call dentist", "description": "Schedule a cleaning."},
        ),
    ]

    results = [execute_approved(store, item.id) for item in items]

    assert all(result.external_action_taken is False for result in results)
    assert all("Dry-run preview only" in result.message for result in results)
    combined = " ".join(result.message for result in results).lower()
    assert "no email was sent" in combined
    assert "no outlook" in combined
    assert "no reminder" in combined


def test_unknown_executor_fails_clearly(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    item = _approved_item(
        store,
        "propose_file_change",
        "Edit file",
        {"path": "README.md", "body": "change"},
    )

    with pytest.raises(ValueError, match="No executor is registered"):
        execute_approved(store, item.id)

    assert store.get(item.id).status == "approved"


def test_code_proposal_has_no_execution_path(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    item = _approved_item(
        store,
        "propose_code_change",
        "Code proposal: Add planning",
        {
            "requested_goal": "Add planning.",
            "relevant_files": ["app/main.py"],
            "proposed_approach": "Plan only.",
            "safety_notes": [],
            "suggested_test_commands": [],
        },
    )

    with pytest.raises(ValueError, match="No executor is registered"):
        execute_approved(store, item.id)

    assert store.get(item.id).status == "approved"


def test_old_approval_rows_remain_readable(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO approvals
                (action_type, title, payload, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "propose_email_draft",
                "Legacy draft",
                json.dumps({"to": "Alex", "body": "Hello"}),
                "approved",
                "2025-01-01T00:00:00+00:00",
                "2025-01-01T00:00:00+00:00",
            ),
        )

    legacy = store.list()[0]

    assert legacy.title == "Legacy draft"
    assert legacy.status == "approved"
    assert legacy.payload == {"to": "Alex", "body": "Hello"}
