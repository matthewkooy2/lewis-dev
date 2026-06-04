from app.approval_service import approve_approval, reject_approval
from app.approvals import ApprovalStore
from app.memory_retrieval import (
    format_active_memory_context,
    retrieve_active_memories,
)
from app.memory_service import archive_memory, propose_memory
from app.memory_store import MemoryStore


def _approve_memory(db_path, content, *, memory_type="preference", tags=None):
    item = propose_memory(
        memory_type=memory_type,
        content=content,
        source="test",
        tags=tags or [],
        db_path=db_path,
    )
    approve_approval(ApprovalStore(db_path), item.id)
    return MemoryStore(db_path).get_by_approval_id(item.id)


def test_retrieval_returns_active_memories_only(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    active = _approve_memory(
        db_path,
        "Blake prefers concise implementation summaries.",
        tags=["summary"],
    )
    propose_memory(
        memory_type="preference",
        content="Pending memory proposal should stay hidden.",
        db_path=db_path,
    )
    rejected = propose_memory(
        memory_type="fact",
        content="Rejected memory proposal should stay hidden.",
        db_path=db_path,
    )
    reject_approval(ApprovalStore(db_path), rejected.id)
    ApprovalStore(db_path).create(
        "propose_task",
        "Pending task",
        {"title": "Approval payload should stay hidden."},
    )

    memories = retrieve_active_memories(
        "How should implementation summaries sound?", db_path=db_path
    )

    assert [memory.id for memory in memories] == [active.id]
    assert memories[0].content == "Blake prefers concise implementation summaries."
    assert "Pending memory proposal" not in "\n".join(
        memory.content for memory in memories
    )
    assert "Rejected memory proposal" not in "\n".join(
        memory.content for memory in memories
    )
    assert "Approval payload" not in "\n".join(memory.content for memory in memories)


def test_retrieval_excludes_archived_memories(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    active = _approve_memory(db_path, "Blake likes focused tests.")
    archived = _approve_memory(db_path, "Blake used an older dashboard label.")

    archive_memory(db_path, archived.id, reason="Outdated")

    memories = retrieve_active_memories(
        "What testing style does Blake like?", db_path=db_path
    )

    assert [memory.id for memory in memories] == [active.id]
    assert "older dashboard label" not in "\n".join(
        memory.content for memory in memories
    )


def test_retrieval_caps_number_and_memory_length(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    _approve_memory(db_path, "Alpha " + "x" * 80)
    _approve_memory(db_path, "Beta " + "y" * 80)
    _approve_memory(db_path, "Gamma " + "z" * 80)

    memories = retrieve_active_memories(
        "no lexical match",
        db_path=db_path,
        limit=2,
        content_limit=18,
    )

    assert len(memories) == 2
    assert all(len(memory.content) <= 18 for memory in memories)
    assert all(memory.content.endswith("...") for memory in memories)


def test_memory_context_is_short_and_names_active_only_rules(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    _approve_memory(db_path, "Blake prefers concise implementation summaries.")

    context = format_active_memory_context(
        "implementation summaries",
        db_path=db_path,
        limit=3,
        content_limit=50,
        context_limit=500,
    )

    assert "Approved active local memories" in context
    assert "transient context" in context
    assert "archived, rejected, or pending memory proposals" in context
    assert "Blake prefers concise implementation summaries." in context


def test_archived_memory_is_not_used_after_archiving(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    memory = _approve_memory(db_path, "Blake used an older dashboard label.")

    archive_memory(db_path, memory.id, reason="No longer true")

    context = format_active_memory_context("dashboard label", db_path=db_path)

    assert context == ""
