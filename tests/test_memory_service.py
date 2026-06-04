import sqlite3

import pytest

from app.approval_service import approve_approval, reject_approval
from app.approvals import ApprovalStore
from app.db import initialize_database
from app.memory_service import MAX_MEMORY_CONTENT_LENGTH, approve_memory_proposal
from app.memory_retrieval import retrieve_active_memories
from app.memory_service import archive_memory
from app.memory_service import propose_memory
from app.memory_service import propose_memory_update
from app.memory_store import MemoryStore


def _memory_payload(**overrides):
    payload = {
        "memory_type": "preference",
        "content": "Blake prefers concise implementation summaries.",
        "source": "chat",
        "tags": ["style", "summary"],
    }
    payload.update(overrides)
    return payload


def _memory_count(db_path):
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]


def test_migration_2_creates_memory_tables(tmp_path):
    db_path = tmp_path / "test.sqlite3"

    initialize_database(db_path)

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        migrations = conn.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version"
        ).fetchall()
    assert {"memories", "memory_events"} <= tables
    assert migrations == [(1, "baseline_schema"), (2, "memory_schema")]


def test_existing_approval_and_run_rows_survive_memory_migration(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE approvals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL,
                title TEXT NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_input TEXT NOT NULL,
                model TEXT NOT NULL,
                tool_calls TEXT NOT NULL,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                estimated_cost REAL NOT NULL DEFAULT 0.0
            )
            """
        )
        conn.execute(
            """
            INSERT INTO approvals
                (action_type, title, payload, status, created_at, updated_at)
            VALUES (
                'propose_email_draft', 'Email Alex',
                '{"body": "Hello", "to": "Alex"}',
                'pending', '2026-01-01T00:00:00+00:00',
                '2026-01-01T00:00:00+00:00'
            )
            """
        )
        conn.execute(
            """
            INSERT INTO runs (
                timestamp, user_input, model, tool_calls, input_tokens,
                output_tokens, total_tokens, estimated_cost
            )
            VALUES (
                '2026-01-01T00:00:00+00:00', 'hello', 'test-model',
                '[]', 1, 2, 3, 0.01
            )
            """
        )

    initialize_database(db_path)

    with sqlite3.connect(db_path) as conn:
        approval = conn.execute(
            "SELECT action_type, title, payload, status FROM approvals WHERE id = 1"
        ).fetchone()
        run = conn.execute(
            "SELECT user_input, input_tokens, output_tokens, total_tokens FROM runs WHERE id = 1"
        ).fetchone()
    assert approval == (
        "propose_email_draft",
        "Email Alex",
        '{"body": "Hello", "to": "Alex"}',
        "pending",
    )
    assert run == ("hello", 1, 2, 3)


def test_pending_memory_approval_alone_creates_no_active_memory(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)

    store.create("propose_memory", "Remember preference", _memory_payload())

    assert _memory_count(db_path) == 0


def test_proposing_memory_creates_pending_approval(tmp_path):
    db_path = tmp_path / "test.sqlite3"

    item = propose_memory(
        memory_type="preference",
        content="Blake prefers concise implementation summaries.",
        source="test",
        tags=["style", "summary"],
        db_path=db_path,
    )

    stored = ApprovalStore(db_path).get(item.id)
    assert stored is not None
    assert stored.action_type == "propose_memory"
    assert stored.status == "pending"
    assert stored.title.startswith("Remember preference:")
    assert stored.payload == {
        "memory_type": "preference",
        "content": "Blake prefers concise implementation summaries.",
        "source": "test",
        "tags_json": ["style", "summary"],
    }


def test_proposing_memory_creates_no_active_memory(tmp_path):
    db_path = tmp_path / "test.sqlite3"

    propose_memory(
        memory_type="fact",
        content="Blake works on Lewis locally.",
        source="test",
        db_path=db_path,
    )

    assert _memory_count(db_path) == 0


def test_proposing_memory_records_proposal_created_event(tmp_path):
    db_path = tmp_path / "test.sqlite3"

    item = propose_memory(
        memory_type="project_context",
        content="Lewis memory proposals require approval before activation.",
        source="test",
        db_path=db_path,
    )

    events = MemoryStore(db_path).list_events()
    assert len(events) == 1
    assert events[0].event_type == "proposal_created"
    assert events[0].approval_id == item.id
    assert events[0].memory_id is None
    assert events[0].details == {
        "memory_type": "project_context",
        "source": "test",
    }


def test_proposing_memory_rejects_invalid_memory_type(tmp_path):
    db_path = tmp_path / "test.sqlite3"

    with pytest.raises(ValueError, match="unsupported memory_type"):
        propose_memory(
            memory_type="secret",
            content="Blake prefers concise implementation summaries.",
            db_path=db_path,
        )

    assert ApprovalStore(db_path).list() == []
    assert _memory_count(db_path) == 0


@pytest.mark.parametrize("content", ["", "   "])
def test_proposing_memory_rejects_empty_content(tmp_path, content):
    db_path = tmp_path / "test.sqlite3"

    with pytest.raises(ValueError, match="content is required"):
        propose_memory(memory_type="fact", content=content, db_path=db_path)

    assert ApprovalStore(db_path).list() == []
    assert _memory_count(db_path) == 0


def test_proposing_memory_rejects_oversized_content(tmp_path):
    db_path = tmp_path / "test.sqlite3"

    with pytest.raises(ValueError, match="content is too long"):
        propose_memory(
            memory_type="fact",
            content="x" * (MAX_MEMORY_CONTENT_LENGTH + 1),
            db_path=db_path,
        )

    assert ApprovalStore(db_path).list() == []
    assert _memory_count(db_path) == 0


@pytest.mark.parametrize(
    "content",
    [
        "My password is correct-horse-battery-staple",
        "My API key is sk-testtesttesttesttesttesttest",
        "My SSN is 123-45-6789",
        "Use card 4111 1111 1111 1111 for billing.",
    ],
)
def test_proposing_memory_rejects_obvious_secrets(tmp_path, content):
    db_path = tmp_path / "test.sqlite3"

    with pytest.raises(ValueError, match="appears to contain a secret"):
        propose_memory(memory_type="fact", content=content, db_path=db_path)

    assert ApprovalStore(db_path).list() == []
    assert _memory_count(db_path) == 0


def test_approving_proposed_memory_creates_exactly_one_active_memory(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    item = propose_memory(
        memory_type="preference",
        content="Blake prefers concise implementation summaries.",
        source="test",
        tags=["style"],
        db_path=db_path,
    )

    approved = approve_approval(store, item.id)
    approve_approval(store, item.id)

    memory = MemoryStore(db_path).get_by_approval_id(item.id)
    assert approved.status == "approved"
    assert _memory_count(db_path) == 1
    assert memory is not None
    assert memory.status == "active"
    assert memory.memory_type == "preference"
    assert memory.content == "Blake prefers concise implementation summaries."
    assert memory.tags == ["style"]


def test_rejecting_proposed_memory_creates_no_active_memory(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    item = propose_memory(
        memory_type="fact",
        content="Blake works on Lewis locally.",
        source="test",
        db_path=db_path,
    )

    rejected = reject_approval(store, item.id)

    assert rejected.status == "rejected"
    assert _memory_count(db_path) == 0


def test_approving_memory_proposal_creates_exactly_one_active_memory(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    item = store.create("propose_memory", "Remember preference", _memory_payload())

    approved = approve_approval(store, item.id)

    memory = MemoryStore(db_path).get_by_approval_id(item.id)
    assert approved.status == "approved"
    assert _memory_count(db_path) == 1
    assert memory is not None
    assert memory.status == "active"
    assert memory.memory_type == "preference"
    assert memory.content == "Blake prefers concise implementation summaries."
    assert memory.tags == ["style", "summary"]


def test_approving_same_memory_proposal_twice_does_not_duplicate(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    item = store.create("propose_memory", "Remember preference", _memory_payload())

    approve_approval(store, item.id)
    approve_approval(store, item.id)

    assert _memory_count(db_path) == 1


def test_rejecting_memory_proposal_creates_no_active_memory(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    item = store.create("propose_memory", "Remember preference", _memory_payload())

    rejected = reject_approval(store, item.id)

    assert rejected.status == "rejected"
    assert _memory_count(db_path) == 0


def test_memory_audit_events_are_recorded_for_approve_and_reject(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    approved_item = store.create(
        "propose_memory", "Remember preference", _memory_payload()
    )
    rejected_item = store.create("propose_memory", "Reject memory", _memory_payload())

    approve_approval(store, approved_item.id)
    reject_approval(store, rejected_item.id)

    events = MemoryStore(db_path).list_events()
    assert [event.event_type for event in events] == ["approved", "rejected"]
    assert events[0].memory_id is not None
    assert events[0].approval_id == approved_item.id
    assert events[1].memory_id is None
    assert events[1].approval_id == rejected_item.id


def test_listing_active_memories_excludes_archived_memories(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    active_item = store.create(
        "propose_memory",
        "Remember active preference",
        _memory_payload(content="Blake likes focused tests."),
    )
    archived_item = store.create(
        "propose_memory",
        "Remember archived preference",
        _memory_payload(content="Blake used an older dashboard label."),
    )
    approve_approval(store, active_item.id)
    approve_approval(store, archived_item.id)
    archived = MemoryStore(db_path).get_by_approval_id(archived_item.id)

    archive_memory(db_path, archived.id, reason="Outdated")

    active_memories = MemoryStore(db_path).list_active()
    assert [memory.approval_id for memory in active_memories] == [active_item.id]
    assert active_memories[0].content == "Blake likes focused tests."


def test_archiving_active_memory_changes_status_and_records_event(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    item = store.create("propose_memory", "Remember preference", _memory_payload())
    approve_approval(store, item.id)
    memory = MemoryStore(db_path).get_by_approval_id(item.id)

    archived = archive_memory(db_path, memory.id, reason="No longer useful")

    assert archived.status == "archived"
    assert archived.archived_at is not None
    assert archived.archive_reason == "No longer useful"
    assert MemoryStore(db_path).get(memory.id).status == "archived"
    events = MemoryStore(db_path).list_events()
    assert events[-1].event_type == "memory_archived"
    assert events[-1].memory_id == memory.id
    assert events[-1].approval_id == item.id
    assert events[-1].details == {"reason": "No longer useful"}


def test_archiving_missing_or_already_archived_memory_fails_safely(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    item = store.create("propose_memory", "Remember preference", _memory_payload())
    approve_approval(store, item.id)
    memory = MemoryStore(db_path).get_by_approval_id(item.id)

    with pytest.raises(KeyError):
        archive_memory(db_path, 999)
    archive_memory(db_path, memory.id)
    with pytest.raises(ValueError, match="not active"):
        archive_memory(db_path, memory.id)

    with sqlite3.connect(db_path) as conn:
        memory_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        archive_events = conn.execute(
            "SELECT COUNT(*) FROM memory_events WHERE event_type = 'memory_archived'"
        ).fetchone()[0]
    assert memory_count == 1
    assert archive_events == 1


def test_proposing_memory_update_creates_pending_approval_without_changing_active_memory(
    tmp_path,
):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    item = propose_memory(
        memory_type="preference",
        content="Blake prefers concise implementation summaries.",
        source="test",
        tags=["style"],
        db_path=db_path,
    )
    approve_approval(store, item.id)
    memory = MemoryStore(db_path).get_by_approval_id(item.id)

    update = propose_memory_update(
        memory_id=memory.id,
        memory_type="preference",
        content="Blake prefers detailed implementation summaries.",
        source="test",
        tags=["style", "detail"],
        reason="Preference changed",
        db_path=db_path,
    )

    active = MemoryStore(db_path).get(memory.id)
    stored = store.get(update.id)
    assert stored.action_type == "propose_memory_update"
    assert stored.status == "pending"
    assert stored.payload == {
        "memory_id": memory.id,
        "original_memory_type": "preference",
        "original_content": "Blake prefers concise implementation summaries.",
        "original_source": "test",
        "original_tags_json": ["style"],
        "replacement_memory_type": "preference",
        "replacement_content": "Blake prefers detailed implementation summaries.",
        "replacement_source": "test",
        "replacement_tags_json": ["style", "detail"],
        "reason": "Preference changed",
    }
    assert active.status == "active"
    assert active.content == "Blake prefers concise implementation summaries."
    assert _memory_count(db_path) == 1


def test_rejecting_memory_update_leaves_original_active_and_unchanged(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    item = propose_memory(
        memory_type="fact",
        content="Blake works on Lewis locally.",
        source="test",
        db_path=db_path,
    )
    approve_approval(store, item.id)
    memory = MemoryStore(db_path).get_by_approval_id(item.id)
    update = propose_memory_update(
        memory_id=memory.id,
        memory_type="fact",
        content="Blake works on Lewis from a local SQLite app.",
        source="test",
        db_path=db_path,
    )

    rejected = reject_approval(store, update.id)

    active = MemoryStore(db_path).get(memory.id)
    assert rejected.status == "rejected"
    assert active.status == "active"
    assert active.content == "Blake works on Lewis locally."
    assert MemoryStore(db_path).list_active() == [active]


def test_approving_memory_update_archives_old_and_creates_one_active_replacement(
    tmp_path,
):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    item = propose_memory(
        memory_type="preference",
        content="Blake prefers concise implementation summaries.",
        source="test",
        tags=["style"],
        db_path=db_path,
    )
    approve_approval(store, item.id)
    original = MemoryStore(db_path).get_by_approval_id(item.id)
    update = propose_memory_update(
        memory_id=original.id,
        memory_type="preference",
        content="Blake prefers implementation summaries with test results.",
        source="test",
        tags=["style", "tests"],
        db_path=db_path,
    )

    approved = approve_approval(store, update.id)

    archived = MemoryStore(db_path).get(original.id)
    replacement = MemoryStore(db_path).get_by_approval_id(update.id)
    active = MemoryStore(db_path).list_active()
    assert approved.status == "approved"
    assert archived.status == "archived"
    assert archived.archive_reason.startswith(f"Replaced by approval #{update.id}")
    assert replacement.status == "active"
    assert (
        replacement.content
        == "Blake prefers implementation summaries with test results."
    )
    assert replacement.tags == ["style", "tests"]
    assert active == [replacement]
    assert _memory_count(db_path) == 2


def test_approved_memory_update_records_clear_audit_events(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    item = propose_memory(
        memory_type="project_context",
        content="Lewis uses memory approvals.",
        source="test",
        db_path=db_path,
    )
    approve_approval(store, item.id)
    original = MemoryStore(db_path).get_by_approval_id(item.id)
    update = propose_memory_update(
        memory_id=original.id,
        memory_type="project_context",
        content="Lewis uses approval-backed memory replacement.",
        source="test",
        reason="More precise",
        db_path=db_path,
    )

    approve_approval(store, update.id)

    update_events = MemoryStore(db_path).list_events(approval_id=update.id)
    assert [event.event_type for event in update_events] == [
        "update_proposal_created",
        "memory_archived_for_replacement",
        "memory_replaced",
    ]
    assert update_events[0].memory_id == original.id
    assert update_events[1].memory_id == original.id
    assert update_events[2].details["archived_memory_id"] == original.id
    assert update_events[2].details["replacement_memory_type"] == "project_context"


def test_approving_same_memory_update_twice_does_not_duplicate_replacement(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    item = propose_memory(
        memory_type="preference",
        content="Blake likes short status updates.",
        source="test",
        db_path=db_path,
    )
    approve_approval(store, item.id)
    original = MemoryStore(db_path).get_by_approval_id(item.id)
    update = propose_memory_update(
        memory_id=original.id,
        memory_type="preference",
        content="Blake likes status updates with next steps.",
        source="test",
        db_path=db_path,
    )

    approve_approval(store, update.id)
    approve_approval(store, update.id)

    active = MemoryStore(db_path).list_active()
    events = MemoryStore(db_path).list_events(approval_id=update.id)
    assert _memory_count(db_path) == 2
    assert len(active) == 1
    assert active[0].approval_id == update.id
    assert [event.event_type for event in events] == [
        "update_proposal_created",
        "memory_archived_for_replacement",
        "memory_replaced",
    ]


def test_memory_update_rejects_invalid_or_archived_memory_id(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    item = propose_memory(
        memory_type="fact",
        content="Blake runs Lewis locally.",
        source="test",
        db_path=db_path,
    )
    approve_approval(store, item.id)
    memory = MemoryStore(db_path).get_by_approval_id(item.id)
    archive_memory(db_path, memory.id)

    with pytest.raises(KeyError):
        propose_memory_update(
            memory_id=999,
            memory_type="fact",
            content="Missing memory cannot be updated.",
            db_path=db_path,
        )
    with pytest.raises(ValueError, match="not active"):
        propose_memory_update(
            memory_id=memory.id,
            memory_type="fact",
            content="Archived memory cannot be updated.",
            db_path=db_path,
        )


@pytest.mark.parametrize(
    ("memory_type", "content", "message"),
    [
        ("secret", "Replacement text.", "unsupported memory_type"),
        ("fact", "", "content is required"),
        ("fact", "My password is correct-horse-battery-staple", "secret"),
    ],
)
def test_memory_update_replacement_uses_existing_validation(
    tmp_path, memory_type, content, message
):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    item = propose_memory(
        memory_type="fact",
        content="Blake runs Lewis locally.",
        source="test",
        db_path=db_path,
    )
    approve_approval(store, item.id)
    memory = MemoryStore(db_path).get_by_approval_id(item.id)

    with pytest.raises(ValueError, match=message):
        propose_memory_update(
            memory_id=memory.id,
            memory_type=memory_type,
            content=content,
            db_path=db_path,
        )

    assert _memory_count(db_path) == 1
    assert ApprovalStore(db_path).list(status="pending") == []


def test_retrieval_uses_replacement_after_memory_update_approval(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    item = propose_memory(
        memory_type="preference",
        content="Blake prefers terse summaries.",
        source="test",
        db_path=db_path,
    )
    approve_approval(store, item.id)
    original = MemoryStore(db_path).get_by_approval_id(item.id)
    update = propose_memory_update(
        memory_id=original.id,
        memory_type="preference",
        content="Blake prefers summaries that include tests run.",
        source="test",
        db_path=db_path,
    )
    approve_approval(store, update.id)

    memories = retrieve_active_memories(
        "What summaries does Blake prefer?", db_path=db_path
    )

    assert [memory.approval_id for memory in memories] == [update.id]
    assert memories[0].content == "Blake prefers summaries that include tests run."
    assert "terse summaries" not in "\n".join(memory.content for memory in memories)


def test_invalid_memory_payload_fails_without_creating_memory(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    item = store.create(
        "propose_memory",
        "Bad memory",
        _memory_payload(memory_type="unsupported"),
    )

    with pytest.raises(ValueError):
        approve_memory_proposal(db_path, item.id)

    assert store.get(item.id).status == "pending"
    assert _memory_count(db_path) == 0


def test_normal_approval_types_still_update_status_only(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    draft = store.create(
        "propose_email_draft", "Email Alex", {"to": "Alex", "body": "Hello"}
    )
    task = store.create(
        "propose_task",
        "Call dentist",
        {"title": "Call dentist", "description": "Schedule a cleaning."},
    )
    calendar = store.create(
        "propose_calendar_event",
        "Project sync",
        {"title": "Project sync", "description": "Discuss launch."},
    )

    approve_approval(store, draft.id)
    approve_approval(store, task.id)
    reject_approval(store, calendar.id)

    assert store.get(draft.id).status == "approved"
    assert store.get(task.id).status == "approved"
    assert store.get(calendar.id).status == "rejected"
    assert _memory_count(db_path) == 0


def test_memory_migration_rerun_is_idempotent(tmp_path):
    db_path = tmp_path / "test.sqlite3"

    initialize_database(db_path)
    initialize_database(db_path)

    with sqlite3.connect(db_path) as conn:
        migrations = conn.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version"
        ).fetchall()
    assert migrations == [(1, "baseline_schema"), (2, "memory_schema")]
