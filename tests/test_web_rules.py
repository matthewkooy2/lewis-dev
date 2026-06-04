from app.web_rules import (
    ActionButton,
    normalize_history_filter,
    timestamp_display,
    visible_actions,
)


def _item(action_type="propose_email_draft", status="pending"):
    return {
        "approval_id": 7,
        "action_type": action_type,
        "status": status,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-02T00:00:00+00:00",
    }


def test_normalize_history_filter_accepts_known_filters():
    assert normalize_history_filter("all") == "all"
    assert normalize_history_filter("drafts") == "drafts"
    assert normalize_history_filter("tasks") == "tasks"
    assert normalize_history_filter("calendar") == "calendar"


def test_normalize_history_filter_defaults_unknown_values_to_all():
    assert normalize_history_filter("pending") == "all"
    assert normalize_history_filter("") == "all"


def test_history_timestamp_uses_last_changed():
    timestamp = timestamp_display(_item(), "history")

    assert timestamp.meta_label == "last changed"
    assert timestamp.field_label == "last changed"
    assert timestamp.value == "2026-01-02T00:00:00+00:00"


def test_non_history_timestamp_uses_created_time():
    timestamp = timestamp_display(_item(), "approvals")

    assert timestamp.meta_label == "created"
    assert timestamp.field_label == "created time"
    assert timestamp.value == "2026-01-01T00:00:00+00:00"


def test_history_never_shows_actions():
    assert visible_actions(_item(status="approved"), "history") == []
    assert (
        visible_actions(
            _item(action_type="propose_task", status="completed"), "history"
        )
        == []
    )


def test_pending_actions_are_only_visible_on_action_pages():
    assert visible_actions(_item(status="pending"), "approvals") == [
        ActionButton("/approvals/7/approve?return_to=/approvals", "approve"),
        ActionButton(
            "/approvals/7/reject?return_to=/approvals",
            "reject",
            "secondary",
        ),
    ]
    assert visible_actions(_item(status="pending"), "tasks") == [
        ActionButton("/approvals/7/approve?return_to=/tasks", "approve"),
        ActionButton("/approvals/7/reject?return_to=/tasks", "reject", "secondary"),
    ]
    assert visible_actions(_item(status="pending"), "calendar") == [
        ActionButton("/approvals/7/approve?return_to=/calendar", "approve"),
        ActionButton(
            "/approvals/7/reject?return_to=/calendar",
            "reject",
            "secondary",
        ),
    ]
    assert visible_actions(_item(status="pending"), "dashboard") == []


def test_approved_drafts_and_calendar_items_show_dry_run_preview():
    assert visible_actions(_item(status="approved"), "approvals") == [
        ActionButton(
            "/approvals/7/execute-approved?return_to=/approvals",
            "dry-run preview",
            "warn",
        )
    ]
    assert visible_actions(
        _item(action_type="propose_calendar_event", status="approved"),
        "dashboard",
    ) == [
        ActionButton(
            "/approvals/7/execute-approved?return_to=/dashboard",
            "dry-run preview",
            "warn",
        )
    ]


def test_approved_tasks_only_show_complete_task_on_tasks_page():
    task = _item(action_type="propose_task", status="approved")

    assert visible_actions(task, "tasks") == [
        ActionButton("/tasks/7/complete", "complete task")
    ]
    assert visible_actions(task, "dashboard") == []
    assert visible_actions(task, "approvals") == []


def test_reopen_task_is_not_a_visible_ui_action():
    completed_task = _item(action_type="propose_task", status="completed")

    for context in ("dashboard", "approvals", "tasks", "calendar", "history"):
        labels = [action.label for action in visible_actions(completed_task, context)]
        assert "reopen task" not in labels
