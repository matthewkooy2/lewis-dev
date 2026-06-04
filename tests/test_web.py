from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.approval_service import approve_approval
from app.approvals import ApprovalStore
from app.config import Settings
from app.code_proposal_previews import PATCH_PREVIEW_SUSPICIOUS_PATHS_WARNING
from app.code_proposals import propose_code_change
from app.memory_store import MemoryStore
from app.web import LOCAL_WEB_HOST, LOCAL_WEB_PORT, create_app


class FakeChatAgent:
    def __init__(self, memory_context_count: int = 0) -> None:
        self.memory_context_count = memory_context_count

    def run(self, message: str) -> SimpleNamespace:
        return SimpleNamespace(
            answer=f"Reply to {message}",
            memory_context_count=self.memory_context_count,
        )


def _settings(tmp_path: Path) -> Settings:
    docs = tmp_path / "documents"
    docs.mkdir()
    return Settings(
        deepseek_api_key=None,
        deepseek_model="test-model",
        deepseek_base_url="https://example.test",
        debug=False,
        user_name="the user",
        user_name_configured=False,
        database_path=tmp_path / "test.sqlite3",
        documents_dir=docs,
        max_tool_calls_per_request=5,
        max_estimated_daily_cost=1.0,
        price_per_1k_input_tokens=0.0,
        price_per_1k_output_tokens=0.0,
    )


def _chat_settings(tmp_path: Path) -> Settings:
    docs = tmp_path / "documents"
    docs.mkdir(exist_ok=True)
    return Settings(
        deepseek_api_key="test-key",
        deepseek_model="test-model",
        deepseek_base_url="https://example.test",
        debug=False,
        user_name="the user",
        user_name_configured=False,
        database_path=tmp_path / "test.sqlite3",
        documents_dir=docs,
        max_tool_calls_per_request=5,
        max_estimated_daily_cost=1.0,
        price_per_1k_input_tokens=0.0,
        price_per_1k_output_tokens=0.0,
    )


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(_settings(tmp_path)))


def test_web_routes_are_available(tmp_path):
    client = _client(tmp_path)

    for path, label in [
        ("/dashboard", "Dashboard"),
        ("/approvals", "Approvals"),
        ("/tasks", "Tasks"),
        ("/calendar", "Calendar Proposals"),
        ("/memories", "Memories"),
        ("/history", "History"),
        ("/usage", "Usage"),
        ("/documents", "Local Document Search"),
        ("/chat", "Chat"),
    ]:
        response = client.get(path)
        assert response.status_code == 200
        assert label in response.text


def test_dashboard_route_returns_expected_labels(tmp_path):
    settings = _settings(tmp_path)
    store = ApprovalStore(settings.database_path)
    store.create(
        action_type="propose_email_draft",
        title="Email Alex",
        payload={"to": "Alex", "body": "Hello"},
    )
    client = TestClient(create_app(settings))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "total pending" in response.text
    assert "pending drafts" in response.text
    assert "pending tasks" in response.text
    assert "approved calendar events" in response.text
    assert "execution previewed items" in response.text
    assert "Recent Usage Summary" in response.text
    assert "Recent Approval, Task, And Calendar Items" in response.text
    assert "View History" in response.text
    assert '<link rel="stylesheet" href="/static/lewis.css">' in response.text


def test_static_css_is_served(tmp_path):
    client = _client(tmp_path)

    response = client.get("/static/lewis.css")

    assert response.status_code == 200
    assert ".chat-shell" in response.text


def test_chat_response_shows_memory_context_note_without_payload(tmp_path):
    settings = _chat_settings(tmp_path)
    app = create_app(settings)
    app.state.chat_agent = FakeChatAgent(memory_context_count=2)
    client = TestClient(app)

    response = client.post("/chat", json={"message": "How should I write this?"})

    assert response.status_code == 200
    assert response.json()["memory_context_count"] == 2
    page = client.get("/chat")
    assert "Using 2 approved memories for context." in page.text
    assert "How should I write this?" in page.text
    assert "memory_context_count" not in page.text


def test_chat_response_without_memory_context_keeps_ui_unchanged(tmp_path):
    settings = _chat_settings(tmp_path)
    app = create_app(settings)
    app.state.chat_agent = FakeChatAgent(memory_context_count=0)
    client = TestClient(app)

    response = client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 200
    assert response.json()["memory_context_count"] == 0
    assert "approved memories for context" not in client.get("/chat").text


def test_dashboard_recent_cards_keep_existing_action_visibility(tmp_path):
    settings = _settings(tmp_path)
    store = ApprovalStore(settings.database_path)
    pending = store.create(
        action_type="propose_email_draft",
        title="Pending draft",
        payload={"to": "Alex", "body": "Hello"},
    )
    approved = store.create(
        action_type="propose_calendar_event",
        title="Approved meeting",
        payload={"title": "Approved meeting"},
    )
    task = store.create(
        action_type="propose_task",
        title="Approved task",
        payload={"title": "Approved task"},
    )
    store.update_status(approved.id, "approved")
    store.update_status(task.id, "approved")
    client = TestClient(create_app(settings))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert f"/approvals/{pending.id}/approve?return_to=/dashboard" not in response.text
    assert (
        f"/approvals/{approved.id}/execute-approved?return_to=/dashboard"
        in response.text
    )
    assert f"/tasks/{task.id}/complete" not in response.text


def test_approvals_route_renders_pending_items_only(tmp_path):
    settings = _settings(tmp_path)
    store = ApprovalStore(settings.database_path)
    item = store.create(
        action_type="propose_task",
        title="Call dentist",
        payload={
            "title": "Call dentist",
            "description": "Schedule a cleaning.",
            "due_date_text": "Friday",
            "priority": "high",
        },
    )
    approved = store.create(
        action_type="propose_email_draft",
        title="Approved draft",
        payload={"to": "Alex", "body": "Hello"},
    )
    rejected = store.create(
        action_type="propose_task",
        title="Rejected task",
        payload={"title": "Rejected task", "description": "Old."},
    )
    previewed = store.create(
        action_type="propose_calendar_event",
        title="Previewed calendar",
        payload={"title": "Previewed calendar", "description": "Old."},
    )
    store.update_status(approved.id, "approved")
    store.update_status(rejected.id, "rejected")
    store.update_status(previewed.id, "approved")
    from app.executors import execute_approved

    execute_approved(store, previewed.id)
    client = TestClient(create_app(settings))

    response = client.get("/approvals")

    assert response.status_code == 200
    assert f"#{item.id}" in response.text
    assert "propose_task" in response.text
    assert "Schedule a cleaning." in response.text
    assert "type='submit'>approve</button>" in response.text
    assert "type='submit'>reject</button>" in response.text
    assert "Approved draft" not in response.text
    assert "Rejected task" not in response.text
    assert "Previewed calendar" not in response.text


def test_approve_and_reject_form_actions_update_status_only(tmp_path):
    settings = _settings(tmp_path)
    store = ApprovalStore(settings.database_path)
    approve_item = store.create(
        action_type="propose_email_draft",
        title="Email Alex",
        payload={"to": "Alex", "body": "Hello"},
    )
    reject_item = store.create(
        action_type="propose_task",
        title="Call dentist",
        payload={"title": "Call dentist", "description": "Schedule a cleaning."},
    )
    client = TestClient(create_app(settings))

    approve_response = client.post(
        f"/approvals/{approve_item.id}/approve", follow_redirects=False
    )
    reject_response = client.post(
        f"/approvals/{reject_item.id}/reject", follow_redirects=False
    )

    assert approve_response.status_code == 303
    assert reject_response.status_code == 303
    assert store.get(approve_item.id).status == "approved"
    assert store.get(approve_item.id).payload == approve_item.payload
    assert store.get(reject_item.id).status == "rejected"
    assert store.get(reject_item.id).payload == reject_item.payload


def test_code_proposal_approval_displays_patch_preview_path_classification(tmp_path):
    settings = _settings(tmp_path)
    store = ApprovalStore(settings.database_path)
    propose_code_change(
        store=store,
        requested_goal="Review command center UI patch preview.",
        relevant_files=["app/web_components.py"],
        proposed_approach="Keep this as planning-only review text.",
        patch_preview=(
            "diff --git a/app/web_components.py b/app/web_components.py\n"
            "--- a/app/web_components.py\n"
            "+++ b/app/web_components.py\n"
            "diff --git a/app/templates/chat.html b/app/templates/chat.html\n"
            "--- /dev/null\n"
            "+++ b/app/templates/chat.html"
        ),
    )
    client = TestClient(create_app(settings))

    response = client.get("/approvals")

    assert response.status_code == 200
    assert "patch preview files" in response.text
    assert "patch preview implementation ready" in response.text
    assert ">no</span>" in response.text
    assert PATCH_PREVIEW_SUSPICIOUS_PATHS_WARNING in response.text
    assert "app/web_components.py (existing project file)" in response.text
    assert (
        "app/templates/chat.html (suspicious or invalid project structure)"
        in response.text
    )
    assert "Patch Preview" in response.text


def test_execute_approved_route_only_performs_dry_run_preview(tmp_path):
    settings = _settings(tmp_path)
    store = ApprovalStore(settings.database_path)
    item = store.create(
        action_type="propose_email_draft",
        title="Email Alex",
        payload={"to": "Alex", "subject": "Hello", "body": "Hi"},
    )
    store.update_status(item.id, "approved")
    client = TestClient(create_app(settings))

    response = client.post(f"/approvals/{item.id}/execute-approved")

    assert response.status_code == 200
    assert "Dry-run preview only" in response.text
    assert "No external action was taken" in response.text
    assert store.get(item.id).status == "execution_previewed"


def test_tasks_route_renders_active_tasks_only_with_valid_actions(tmp_path):
    settings = _settings(tmp_path)
    store = ApprovalStore(settings.database_path)
    pending = store.create(
        action_type="propose_task",
        title="Pending task",
        payload={"title": "Pending task", "description": "Needs review."},
    )
    approved = store.create(
        action_type="propose_task",
        title="Approved task",
        payload={"title": "Approved task", "description": "Ready."},
    )
    completed = store.create(
        action_type="propose_task",
        title="Completed task",
        payload={"title": "Completed task", "description": "Done."},
    )
    rejected = store.create(
        action_type="propose_task",
        title="Rejected task",
        payload={"title": "Rejected task", "description": "Nope."},
    )
    store.update_status(approved.id, "approved")
    store.update_status(completed.id, "completed")
    store.update_status(rejected.id, "rejected")
    client = TestClient(create_app(settings))

    response = client.get("/tasks")

    assert response.status_code == 200
    assert f"#{pending.id}" in response.text
    assert f"#{approved.id}" in response.text
    assert "type='submit'>approve</button>" in response.text
    assert "type='submit'>reject</button>" in response.text
    assert "type='submit'>complete task</button>" in response.text
    assert "Completed task" not in response.text
    assert "Rejected task" not in response.text
    assert "reopen task" not in response.text


def test_calendar_route_renders_active_calendar_with_valid_actions(tmp_path):
    settings = _settings(tmp_path)
    store = ApprovalStore(settings.database_path)
    pending = store.create(
        action_type="propose_calendar_event",
        title="Pending meeting",
        payload={"title": "Pending meeting", "description": "Review."},
    )
    approved = store.create(
        action_type="propose_calendar_event",
        title="Approved meeting",
        payload={"title": "Approved meeting", "description": "Previewable."},
    )
    rejected = store.create(
        action_type="propose_calendar_event",
        title="Rejected meeting",
        payload={"title": "Rejected meeting", "description": "Old."},
    )
    store.update_status(approved.id, "approved")
    store.update_status(rejected.id, "rejected")
    client = TestClient(create_app(settings))

    response = client.get("/calendar")

    assert response.status_code == 200
    assert "These are local calendar proposals only" in response.text
    assert f"#{pending.id}" in response.text
    assert f"#{approved.id}" in response.text
    assert "type='submit'>approve</button>" in response.text
    assert "type='submit'>reject</button>" in response.text
    assert "type='submit'>dry-run preview</button>" in response.text
    assert "Rejected meeting" not in response.text
    assert "created" in response.text
    assert (
        "OS calendar event, invites, reminders, or notifications are created"
        in response.text
    )


def test_memories_route_shows_active_memories_and_recent_events(tmp_path):
    settings = _settings(tmp_path)
    store = ApprovalStore(settings.database_path)
    active_item = store.create(
        "propose_memory",
        "Remember preference",
        {
            "memory_type": "preference",
            "content": "Blake prefers concise implementation summaries.",
            "source": "test",
            "tags": ["style"],
        },
    )
    archived_item = store.create(
        "propose_memory",
        "Remember old fact",
        {
            "memory_type": "fact",
            "content": "This archived memory should not be active.",
            "source": "test",
        },
    )
    approve_approval(store, active_item.id)
    approve_approval(store, archived_item.id)
    archived = MemoryStore(settings.database_path).get_by_approval_id(archived_item.id)
    from app.memory_service import archive_memory

    archive_memory(settings.database_path, archived.id)
    client = TestClient(create_app(settings))

    response = client.get("/memories")

    active = MemoryStore(settings.database_path).get_by_approval_id(active_item.id)
    assert response.status_code == 200
    assert "Active Memories" in response.text
    assert "Recent Memory Audit Events" in response.text
    assert "Blake prefers concise implementation summaries." in response.text
    assert "This archived memory should not be active." not in response.text
    assert "memory_archived" in response.text
    assert f"/memories/{active.id}/archive" in response.text
    assert "archive</button>" in response.text
    assert "transient local context" in response.text


def test_memories_archive_action_archives_active_memory(tmp_path):
    settings = _settings(tmp_path)
    store = ApprovalStore(settings.database_path)
    item = store.create(
        "propose_memory",
        "Remember preference",
        {
            "memory_type": "preference",
            "content": "Blake prefers concise implementation summaries.",
            "source": "test",
        },
    )
    approve_approval(store, item.id)
    memory = MemoryStore(settings.database_path).get_by_approval_id(item.id)
    client = TestClient(create_app(settings))

    response = client.post(
        f"/memories/{memory.id}/archive",
        data={"reason": "Outdated"},
        follow_redirects=False,
    )

    archived = MemoryStore(settings.database_path).get(memory.id)
    assert response.status_code == 303
    assert response.headers["location"].startswith("/memories?message=")
    assert archived.status == "archived"
    assert archived.archive_reason == "Outdated"
    assert MemoryStore(settings.database_path).list_active() == []

    page = client.get("/memories")
    assert "Blake prefers concise implementation summaries." not in page.text
    assert "memory_archived" in page.text


def test_memories_update_action_creates_pending_approval_without_editing(tmp_path):
    settings = _settings(tmp_path)
    store = ApprovalStore(settings.database_path)
    item = store.create(
        "propose_memory",
        "Remember preference",
        {
            "memory_type": "preference",
            "content": "Blake prefers concise implementation summaries.",
            "source": "test",
            "tags": ["style"],
        },
    )
    approve_approval(store, item.id)
    memory = MemoryStore(settings.database_path).get_by_approval_id(item.id)
    client = TestClient(create_app(settings))

    response = client.post(
        f"/memories/{memory.id}/propose-update",
        data={
            "memory_type": "preference",
            "content": "Blake prefers implementation summaries with tests.",
            "tags": "style, tests",
            "reason": "Preference changed",
        },
        follow_redirects=False,
    )

    approval = ApprovalStore(settings.database_path).list(status="pending")[0]
    active = MemoryStore(settings.database_path).get(memory.id)
    assert response.status_code == 303
    assert response.headers["location"].startswith("/memories?message=")
    assert approval.action_type == "propose_memory_update"
    assert approval.payload["memory_id"] == memory.id
    assert approval.payload["replacement_content"] == (
        "Blake prefers implementation summaries with tests."
    )
    assert approval.payload["replacement_tags_json"] == ["style", "tests"]
    assert active.status == "active"
    assert active.content == "Blake prefers concise implementation summaries."


def test_memory_update_approval_shows_comparison_and_pending_actions(tmp_path):
    settings = _settings(tmp_path)
    store = ApprovalStore(settings.database_path)
    item = store.create(
        "propose_memory",
        "Remember preference",
        {
            "memory_type": "preference",
            "content": "Blake prefers concise implementation summaries.",
            "source": "test",
            "tags": ["style"],
        },
    )
    approve_approval(store, item.id)
    memory = MemoryStore(settings.database_path).get_by_approval_id(item.id)
    client = TestClient(create_app(settings))

    update_response = client.post(
        f"/memories/{memory.id}/propose-update",
        data={
            "memory_type": "preference",
            "content": "Blake prefers implementation summaries with tests.",
            "tags": "style, tests",
            "reason": "Preference changed",
        },
        follow_redirects=False,
    )
    approval = ApprovalStore(settings.database_path).list(status="pending")[0]

    response = client.get("/approvals")

    assert update_response.status_code == 303
    assert response.status_code == 200
    assert "Current Memory" in response.text
    assert "Proposed Replacement" in response.text
    assert "Blake prefers concise implementation summaries." in response.text
    assert "Blake prefers implementation summaries with tests." in response.text
    assert "Preference changed" in response.text
    assert "original_content" not in response.text
    assert "replacement_content" not in response.text
    assert f"/approvals/{approval.id}/approve?return_to=/approvals" in response.text
    assert f"/approvals/{approval.id}/reject?return_to=/approvals" in response.text


def test_history_route_renders_processed_items_without_normal_actions(tmp_path):
    settings = _settings(tmp_path)
    store = ApprovalStore(settings.database_path)
    store.create(
        action_type="propose_email_draft",
        title="Pending draft",
        payload={"to": "Alex", "body": "Hello"},
    )
    completed = store.create(
        action_type="propose_task",
        title="Completed task",
        payload={"title": "Completed task", "description": "Done."},
    )
    rejected = store.create(
        action_type="propose_task",
        title="Rejected task",
        payload={"title": "Rejected task", "description": "Nope."},
    )
    previewed = store.create(
        action_type="propose_calendar_event",
        title="Previewed meeting",
        payload={"title": "Previewed meeting", "description": "Dry run."},
    )
    store.update_status(completed.id, "completed")
    store.update_status(rejected.id, "rejected")
    store.update_status(previewed.id, "approved")
    from app.executors import execute_approved

    execute_approved(store, previewed.id)
    client = TestClient(create_app(settings))

    response = client.get("/history")

    assert response.status_code == 200
    assert "Pending draft" not in response.text
    assert f"#{completed.id}" in response.text
    assert f"#{rejected.id}" in response.text
    assert f"#{previewed.id}" in response.text
    assert (
        "Execution was previewed locally; no external action was taken" in response.text
    )
    assert "last changed" in response.text
    assert "type='submit'>dry-run preview</button>" not in response.text
    assert "type='submit'>approve</button>" not in response.text
    assert "type='submit'>reject</button>" not in response.text
    assert "type='submit'>complete task</button>" not in response.text
    assert "type='submit'>reopen task</button>" not in response.text
    assert "<button" not in response.text
    assert (
        "real email, calendar, reminder, notification, or external execution"
        in response.text
    )


def test_history_route_defaults_invalid_filter_to_all(tmp_path):
    settings = _settings(tmp_path)
    store = ApprovalStore(settings.database_path)
    completed = store.create(
        action_type="propose_task",
        title="Completed task",
        payload={"title": "Completed task"},
    )
    store.update_status(completed.id, "completed")
    client = TestClient(create_app(settings))

    response = client.get("/history?type=not-real")

    assert response.status_code == 200
    assert f"#{completed.id}" in response.text
    assert "class='filter active' href='/history?type=all'" in response.text
    assert "<button" not in response.text


def test_history_route_orders_by_last_changed_then_id(tmp_path):
    settings = _settings(tmp_path)
    store = ApprovalStore(settings.database_path)
    changed_latest = store.create(
        action_type="propose_task",
        title="Changed latest lower id",
        payload={"title": "Changed latest lower id"},
    )
    created_latest_changed_older = store.create(
        action_type="propose_task",
        title="Created latest changed older",
        payload={"title": "Created latest changed older"},
    )
    changed_latest_higher_id = store.create(
        action_type="propose_task",
        title="Changed latest higher id",
        payload={"title": "Changed latest higher id"},
    )
    store.update_status(changed_latest.id, "completed")
    store.update_status(created_latest_changed_older.id, "completed")
    store.update_status(changed_latest_higher_id.id, "completed")
    with sqlite3.connect(settings.database_path) as conn:
        conn.executemany(
            """
            UPDATE approvals
            SET created_at = ?, updated_at = ?
            WHERE id = ?
            """,
            [
                (
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-10T00:00:00+00:00",
                    changed_latest.id,
                ),
                (
                    "2026-01-09T00:00:00+00:00",
                    "2026-01-02T00:00:00+00:00",
                    created_latest_changed_older.id,
                ),
                (
                    "2025-12-31T00:00:00+00:00",
                    "2026-01-10T00:00:00+00:00",
                    changed_latest_higher_id.id,
                ),
            ],
        )
    client = TestClient(create_app(settings))

    response = client.get("/history")

    assert response.status_code == 200
    assert "last changed 2026-01-10T00:00:00+00:00" in response.text
    first = response.text.index(
        f"#{changed_latest_higher_id.id} Changed latest higher id"
    )
    second = response.text.index(f"#{changed_latest.id} Changed latest lower id")
    third = response.text.index(
        f"#{created_latest_changed_older.id} Created latest changed older"
    )
    assert first < second < third


def test_document_search_route_does_not_accept_arbitrary_paths(tmp_path):
    settings = _settings(tmp_path)
    (settings.documents_dir / "note.md").write_text("Project Alpha", encoding="utf-8")
    client = TestClient(create_app(settings))

    response = client.get(
        "/documents",
        params={"query": "Alpha", "path": "C:\\Users\\Example\\Desktop"},
    )
    search_response = client.get("/documents", params={"query": "Alpha"})

    assert response.status_code == 400
    assert "only accepts a query string" in response.text
    assert "Lewis only searches data/documents/" in response.text
    assert search_response.status_code == 200
    assert "note.md" in search_response.text
    assert "Project Alpha" in search_response.text


def test_web_server_config_binds_to_localhost_by_default():
    assert LOCAL_WEB_HOST == "127.0.0.1"
    assert LOCAL_WEB_PORT == 8787
