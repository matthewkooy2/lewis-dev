from app.approvals import ApprovalStore


def test_create_and_update_approval(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    item = store.create(
        action_type="propose_email_draft",
        title="Email Alex",
        payload={"to": "alex@example.com", "body": "Hello"},
    )

    assert item.id > 0
    assert item.status == "pending"
    assert len(store.list(status="pending")) == 1

    approved = store.update_status(item.id, "approved")
    assert approved.status == "approved"


def test_approving_item_only_changes_status_and_preserves_payload(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    item = store.create(
        action_type="propose_email_draft",
        title="Email Professor Smith",
        payload={"to": "Professor Smith", "body": "Hello"},
    )

    approved = store.update_status(item.id, "approved")

    assert approved.status == "approved"
    assert approved.payload == item.payload


def test_approving_task_only_changes_status_and_preserves_payload(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
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

    approved = store.update_status(item.id, "approved")

    assert approved.status == "approved"
    assert approved.action_type == "propose_task"
    assert approved.payload == item.payload
