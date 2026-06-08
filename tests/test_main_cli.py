import sqlite3

from app.approval_service import approve_approval
from app.approvals import ApprovalStore
from app.code_proposal_previews import PATCH_PREVIEW_SUSPICIOUS_PATHS_WARNING
from app.dashboard import get_dashboard_summary
from app.logging_db import RunLogger
from app.memory_store import MemoryStore
from app.main import (
    main,
    print_approvals,
    print_calendar_proposals,
    print_dashboard,
    print_memories,
    print_memory_events,
    print_project_file_search,
    print_tasks,
)
from app.project_search import ProjectFileSearch
from app.executors import execute_approved


EXPECTED_CODE_PROPOSAL_TEST_COMMANDS = [
    r".\.venv\Scripts\python.exe -m ruff format .",
    r".\.venv\Scripts\python.exe -m ruff check .",
    r".\.venv\Scripts\python.exe -m pytest",
]


def test_dashboard_counts_pending_approved_rejected_completed_items(tmp_path, capsys):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    logger = RunLogger(db_path)
    store.create(
        action_type="propose_task",
        title="Submit form",
        payload={"title": "Submit form", "description": "Submit the permit form."},
    )
    store.create(
        action_type="propose_email_draft",
        title="Draft Alex",
        payload={"to": "Alex", "body": "Hello"},
    )
    approved_task = store.create(
        action_type="propose_task",
        title="Pack bag",
        payload={"title": "Pack bag", "description": "Pack for the trip."},
    )
    completed_task = store.create(
        action_type="propose_task",
        title="Call dentist",
        payload={"title": "Call dentist", "description": "Schedule a cleaning."},
    )
    rejected_email = store.create(
        action_type="propose_email_draft",
        title="Email Alex",
        payload={"to": "Alex", "body": "Hello"},
    )
    pending_calendar = store.create(
        action_type="propose_calendar_event",
        title="Project sync",
        payload={
            "title": "Project sync",
            "description": "Discuss launch.",
            "date_text": "Tuesday",
        },
    )
    approved_calendar = store.create(
        action_type="propose_calendar_event",
        title="Planning block",
        payload={"title": "Planning block", "description": "Focus time."},
    )
    rejected_calendar = store.create(
        action_type="propose_calendar_event",
        title="Old meeting",
        payload={"title": "Old meeting", "description": "Skip."},
    )
    store.update_status(approved_task.id, "approved")
    store.update_status(completed_task.id, "completed")
    store.update_status(rejected_email.id, "rejected")
    store.update_status(approved_calendar.id, "approved")
    store.update_status(rejected_calendar.id, "rejected")
    logger.log_run(
        user_input="hello",
        model="test-model",
        tool_calls=[],
        input_tokens=10,
        output_tokens=4,
        total_tokens=14,
        api_calls=1,
        estimated_cost=0.01,
    )

    dashboard = get_dashboard_summary(store, logger)
    print_dashboard(store, logger)

    output = capsys.readouterr().out
    assert dashboard["counts"] == {
        "total_pending": 3,
        "pending_drafts": 1,
        "pending_tasks": 1,
        "pending_calendar_events": 1,
        "approved_tasks": 1,
        "completed_tasks": 1,
        "execution_previewed_items": 0,
        "approved_calendar_events": 1,
        "total_rejected": 2,
    }
    assert f"#{pending_calendar.id}" in output
    assert "total pending: 3" in output
    assert "pending drafts: 1" in output
    assert "pending tasks: 1" in output
    assert "pending calendar events: 1" in output
    assert "approved tasks: 1" in output
    assert "completed tasks: 1" in output
    assert "execution previewed items: 0" in output
    assert "approved calendar events: 1" in output
    assert "total rejected: 2" in output
    assert "pending approvals:" not in output
    assert "task proposals:" not in output
    assert "calendar proposals:" not in output
    assert "rejected items:" not in output
    assert "runs: 1" in output


def test_approvals_display_formats_task_proposals(tmp_path, capsys):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    item = store.create(
        action_type="propose_task",
        title="Call dentist",
        payload={
            "title": "Call dentist",
            "description": "Schedule a cleaning.",
            "due_date_text": "next Friday",
            "priority": "high",
        },
    )

    print_approvals(store)

    output = capsys.readouterr().out
    assert f"#{item.id} [pending]" in output
    assert "type: propose_task" in output
    assert "title: Call dentist" in output
    assert "description: Schedule a cleaning." in output
    assert "due_date_text: next Friday" in output
    assert "priority: high" in output


def test_tasks_command_display_is_read_only_and_filters_tasks(tmp_path, capsys):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    pending = store.create(
        action_type="propose_task",
        title="Submit form",
        payload={"title": "Submit form", "description": "Submit the permit form."},
    )
    approved = store.create(
        action_type="propose_task",
        title="Pack bag",
        payload={"title": "Pack bag", "description": "Pack for the trip."},
    )
    rejected = store.create(
        action_type="propose_task",
        title="Old task",
        payload={"title": "Old task", "description": "Ignore this."},
    )
    email = store.create(
        action_type="propose_email_draft",
        title="Email Alex",
        payload={"to": "Alex", "body": "Hello"},
    )
    store.update_status(approved.id, "approved")
    store.update_status(rejected.id, "rejected")

    print_tasks(store)

    output = capsys.readouterr().out
    assert f"#{pending.id} [pending]" in output
    assert f"approval_id: {pending.id}" in output
    assert "status: pending" in output
    assert "title: Submit form" in output
    assert "description: Submit the permit form." in output
    assert "due_date_text:" in output
    assert "priority:" in output
    assert "created:" in output
    assert f"#{approved.id} [approved]" in output
    assert f"#{rejected.id}" not in output
    assert f"#{email.id}" not in output
    assert store.get(pending.id).status == "pending"
    assert store.get(approved.id).status == "approved"


def test_approvals_display_formats_calendar_proposals(tmp_path, capsys):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    item = store.create(
        action_type="propose_calendar_event",
        title="Project sync",
        payload={
            "title": "Project sync",
            "description": "Discuss launch.",
            "date_text": "next Tuesday",
            "start_time_text": "2:00 PM",
            "end_time_text": "2:30 PM",
            "location": "Zoom",
            "attendees": ["Alex", "Jordan"],
        },
    )

    print_approvals(store)

    output = capsys.readouterr().out
    assert f"#{item.id} [pending]" in output
    assert "type: propose_calendar_event" in output
    assert "title: Project sync" in output
    assert "date_text: next Tuesday" in output
    assert "start_time_text: 2:00 PM" in output
    assert "end_time_text: 2:30 PM" in output
    assert "location: Zoom" in output
    assert "attendees: Alex, Jordan" in output
    assert "description: Discuss launch." in output


def test_calendar_proposals_command_is_read_only(tmp_path, capsys):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    pending = store.create(
        action_type="propose_calendar_event",
        title="Project sync",
        payload={
            "title": "Project sync",
            "description": "Discuss launch.",
            "date_text": "Tuesday",
            "start_time_text": "2:00 PM",
            "end_time_text": "2:30 PM",
            "location": "Zoom",
            "attendees": ["Alex"],
        },
    )
    task = store.create(
        action_type="propose_task",
        title="Submit form",
        payload={"title": "Submit form", "description": "Submit it."},
    )

    print_calendar_proposals(store)

    output = capsys.readouterr().out
    assert f"#{pending.id} [pending]" in output
    assert f"approval_id: {pending.id}" in output
    assert "status: pending" in output
    assert "title: Project sync" in output
    assert "date_text: Tuesday" in output
    assert "start_time_text: 2:00 PM" in output
    assert "end_time_text: 2:30 PM" in output
    assert "location: Zoom" in output
    assert "attendees: Alex" in output
    assert "description: Discuss launch." in output
    assert "created:" in output
    assert f"#{task.id}" not in output
    assert store.get(pending.id).status == "pending"


def test_approve_task_cli_explains_status_only_behavior(tmp_path, capsys, monkeypatch):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setenv("LEWIS_DB_PATH", str(db_path))
    store = ApprovalStore(db_path)
    item = store.create(
        action_type="propose_task",
        title="Call dentist",
        payload={"title": "Call dentist", "description": "Schedule a cleaning."},
    )

    exit_code = main(["approve", str(item.id)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "only marks the task proposal approved" in output
    assert (
        "no reminder, notification, calendar event, or external task was created"
        in output
    )
    assert store.get(item.id).status == "approved"


def test_approve_calendar_cli_only_marks_approved(tmp_path, capsys, monkeypatch):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setenv("LEWIS_DB_PATH", str(db_path))
    store = ApprovalStore(db_path)
    item = store.create(
        action_type="propose_calendar_event",
        title="Project sync",
        payload={"title": "Project sync", "description": "Discuss launch."},
    )

    exit_code = main(["approve", str(item.id)])

    output = capsys.readouterr().out
    approved = store.get(item.id)
    assert exit_code == 0
    assert approved.status == "approved"
    assert approved.payload == item.payload
    assert "only marks the local calendar proposal approved" in output
    assert "no Outlook, Google Calendar, OS calendar event, invite" in output


def test_propose_memory_cli_creates_pending_approval(tmp_path, capsys, monkeypatch):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setenv("LEWIS_DB_PATH", str(db_path))

    exit_code = main(
        [
            "propose-memory",
            "--type",
            "preference",
            "--content",
            "Blake prefers concise implementation summaries.",
            "--tag",
            "style",
        ]
    )

    output = capsys.readouterr().out
    item = ApprovalStore(db_path).list()[0]
    assert exit_code == 0
    assert item.action_type == "propose_memory"
    assert item.status == "pending"
    assert item.payload == {
        "memory_type": "preference",
        "content": "Blake prefers concise implementation summaries.",
        "source": "cli",
        "tags_json": ["style"],
    }
    assert MemoryStore(db_path).list_active() == []
    assert "Saved pending memory approval" in output
    assert "No active memory was created" in output


def test_propose_memory_update_cli_creates_pending_approval_without_editing(
    tmp_path, capsys, monkeypatch
):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setenv("LEWIS_DB_PATH", str(db_path))
    store = ApprovalStore(db_path)
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
    memory = MemoryStore(db_path).get_by_approval_id(item.id)

    exit_code = main(
        [
            "propose-memory-update",
            str(memory.id),
            "--type",
            "preference",
            "--content",
            "Blake prefers implementation summaries with test results.",
            "--tag",
            "tests",
            "--reason",
            "Preference changed",
        ]
    )

    output = capsys.readouterr().out
    approval = ApprovalStore(db_path).list(status="pending")[0]
    active = MemoryStore(db_path).get(memory.id)
    assert exit_code == 0
    assert approval.action_type == "propose_memory_update"
    assert approval.payload["memory_id"] == memory.id
    assert approval.payload["replacement_content"] == (
        "Blake prefers implementation summaries with test results."
    )
    assert approval.payload["replacement_tags_json"] == ["tests"]
    assert active.status == "active"
    assert active.content == "Blake prefers concise implementation summaries."
    assert "pending memory update approval" in output
    assert "active memory was not changed" in output


def test_approvals_display_formats_memory_proposals(tmp_path, capsys):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    item = store.create(
        action_type="propose_memory",
        title="Remember preference: Blake prefers concise summaries.",
        payload={
            "memory_type": "preference",
            "content": "Blake prefers concise implementation summaries.",
            "source": "test",
            "tags_json": ["style"],
        },
    )

    print_approvals(store)

    output = capsys.readouterr().out
    assert f"#{item.id} [pending]" in output
    assert "type: propose_memory" in output
    assert "summary: Remember preference" in output
    assert "memory:" in output
    assert "type: preference" in output
    assert "Blake prefers concise implementation summaries." in output
    assert "tags: style" in output


def test_approvals_display_formats_memory_update_proposals(tmp_path, capsys):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    item = store.create(
        action_type="propose_memory_update",
        title="Replace memory #1 with preference",
        payload={
            "memory_id": 1,
            "original_memory_type": "preference",
            "original_content": "Old summary preference.",
            "original_source": "test",
            "original_tags_json": ["style"],
            "replacement_memory_type": "preference",
            "replacement_content": "New summary preference.",
            "replacement_source": "test",
            "replacement_tags_json": ["style", "tests"],
            "reason": "Preference changed",
        },
    )

    print_approvals(store)

    output = capsys.readouterr().out
    assert f"#{item.id} [pending]" in output
    assert "type: propose_memory_update" in output
    assert "memory_update:" in output
    assert "memory_id: 1" in output
    assert "Old summary preference." in output
    assert "New summary preference." in output
    assert "reason: Preference changed" in output


def test_propose_code_cli_creates_pending_review_only_approval(
    tmp_path, capsys, monkeypatch
):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setenv("LEWIS_DB_PATH", str(db_path))

    exit_code = main(
        [
            "propose-code",
            "--goal",
            "Add local code proposal planning.",
            "--file",
            "app/main.py",
            "--file",
            "app/tools.py",
            "--approach",
            "Add a focused validator and save a pending proposal.",
            "--safety-note",
            "Planning only; no project files are changed.",
            "--test-command",
            ".\\.venv\\Scripts\\python.exe -m pytest tests/test_main_cli.py",
        ]
    )

    output = capsys.readouterr().out
    approval = ApprovalStore(db_path).list(status="pending")[0]
    assert exit_code == 0
    assert approval.action_type == "propose_code_change"
    assert approval.payload == {
        "requested_goal": "Add local code proposal planning.",
        "target_surface": "unspecified",
        "relevant_files": ["app/main.py", "app/tools.py"],
        "proposed_approach": "Add a focused validator and save a pending proposal.",
        "safety_notes": ["Planning only; no project files are changed."],
        "suggested_test_commands": EXPECTED_CODE_PROPOSAL_TEST_COMMANDS,
    }
    assert "Saved pending code proposal" in output
    assert "no files were read, edited, patched, or executed" in output


def test_propose_code_cli_rejects_runtime_paths_without_creating_approval(
    tmp_path, capsys, monkeypatch
):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setenv("LEWIS_DB_PATH", str(db_path))

    exit_code = main(
        [
            "propose-code",
            "--goal",
            "Inspect runtime data.",
            "--file",
            "data/lewis.sqlite3",
            "--approach",
            "Keep this as planning only.",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Could not propose code change" in output
    assert ApprovalStore(db_path).list() == []


def test_search_files_cli_reads_only_safe_project_files(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("app.main.ROOT_DIR", tmp_path)
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text(
        "needle in source\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("needle in secret", encoding="utf-8")

    exit_code = main(["search-files", "needle"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "app/main.py:1" in output
    assert "needle in source" in output
    assert ".env" not in output
    assert not (tmp_path / "data" / "lewis.sqlite3").exists()


def test_print_project_file_search_empty_state(tmp_path, capsys):
    print_project_file_search(ProjectFileSearch(tmp_path), "missing")

    assert "No safe project file matches found." in capsys.readouterr().out


def test_approvals_display_formats_code_proposals(tmp_path, capsys):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    item = store.create(
        action_type="propose_code_change",
        title="Code proposal: Add local planning",
        payload={
            "requested_goal": "Add local planning.",
            "relevant_files": ["app/main.py"],
            "proposed_approach": "Add a CLI command and validation helper.",
            "patch_preview": (
                "diff --git a/app/main.py b/app/main.py\n"
                "--- a/app/main.py\n"
                "+++ b/app/main.py"
            ),
            "patch_preview_path_classifications": [
                {
                    "path": "app/main.py",
                    "classification": "existing project file",
                }
            ],
            "patch_preview_implementation_ready": True,
            "safety_notes": ["No patches."],
            "suggested_test_commands": ["python -m pytest tests/test_main_cli.py"],
        },
    )

    print_approvals(store)

    output = capsys.readouterr().out
    assert f"#{item.id} [pending]" in output
    assert "type: propose_code_change" in output
    assert "code_proposal:" in output
    assert "requested_goal:" in output
    assert "Add local planning." in output
    assert "app/main.py" in output
    assert "patch_preview:" in output
    assert "implementation_ready: yes" in output
    assert "diff --git a/app/main.py b/app/main.py" in output
    assert "suggested_test_commands:" in output


def test_approvals_display_warns_for_suspicious_patch_preview_paths(tmp_path, capsys):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    item = store.create(
        action_type="propose_code_change",
        title="Code proposal: Review suspicious patch preview",
        payload={
            "requested_goal": "Review patch preview.",
            "relevant_files": ["app/main.py"],
            "proposed_approach": "Keep this as inert review text.",
            "patch_preview": (
                "diff --git a/app/templates/chat.html b/app/templates/chat.html"
            ),
            "patch_preview_path_classifications": [
                {
                    "path": "app/templates/chat.html",
                    "classification": "suspicious or invalid project structure",
                }
            ],
            "patch_preview_implementation_ready": False,
            "patch_preview_warning": PATCH_PREVIEW_SUSPICIOUS_PATHS_WARNING,
            "safety_notes": [],
            "suggested_test_commands": EXPECTED_CODE_PROPOSAL_TEST_COMMANDS,
        },
    )

    print_approvals(store)

    output = capsys.readouterr().out
    assert f"#{item.id} [pending]" in output
    assert "implementation_ready: no" in output
    assert PATCH_PREVIEW_SUSPICIOUS_PATHS_WARNING in output


def test_memories_command_displays_active_memories(tmp_path, capsys, monkeypatch):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setenv("LEWIS_DB_PATH", str(db_path))
    store = ApprovalStore(db_path)
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
            "content": "This older memory should be archived.",
            "source": "test",
        },
    )
    approve_approval(store, active_item.id)
    approve_approval(store, archived_item.id)
    archived = MemoryStore(db_path).get_by_approval_id(archived_item.id)
    from app.memory_service import archive_memory

    archive_memory(db_path, archived.id)

    exit_code = main(["memories"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Blake prefers concise implementation summaries." in output
    assert "tags: style" in output
    assert "This older memory should be archived." not in output


def test_print_memories_empty_state(tmp_path, capsys):
    print_memories(MemoryStore(tmp_path / "test.sqlite3").list_active())

    assert "No active memories found." in capsys.readouterr().out


def test_memory_events_command_displays_audit_events(tmp_path, capsys, monkeypatch):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setenv("LEWIS_DB_PATH", str(db_path))
    store = ApprovalStore(db_path)
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

    exit_code = main(["memory-events"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "approved" in output
    assert f"approval_id: {item.id}" in output
    assert "memory_id:" in output


def test_print_memory_events_empty_state(tmp_path, capsys):
    print_memory_events(MemoryStore(tmp_path / "test.sqlite3").list_recent_events())

    assert "No memory audit events found." in capsys.readouterr().out


def test_archive_memory_cli_archives_without_deleting_rows(
    tmp_path, capsys, monkeypatch
):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setenv("LEWIS_DB_PATH", str(db_path))
    store = ApprovalStore(db_path)
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
    memory = MemoryStore(db_path).get_by_approval_id(item.id)

    exit_code = main(["archive-memory", str(memory.id), "--reason", "Outdated"])

    output = capsys.readouterr().out
    archived = MemoryStore(db_path).get(memory.id)
    with sqlite3.connect(db_path) as conn:
        row_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert exit_code == 0
    assert archived.status == "archived"
    assert archived.archive_reason == "Outdated"
    assert row_count == 1
    assert MemoryStore(db_path).list_active() == []
    assert "Archived memory" in output
    assert "row was preserved" in output


def test_calendar_proposals_cli_command(tmp_path, capsys, monkeypatch):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setenv("LEWIS_DB_PATH", str(db_path))
    store = ApprovalStore(db_path)
    item = store.create(
        action_type="propose_calendar_event",
        title="Project sync",
        payload={"title": "Project sync", "description": "Discuss launch."},
    )

    exit_code = main(["calendar-proposals"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert f"#{item.id} [pending]" in output
    assert "title: Project sync" in output
    assert store.get(item.id).status == "pending"


def test_older_calendar_approval_rows_with_missing_fields_display(tmp_path, capsys):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    item = store.create(
        action_type="propose_calendar_event",
        title="Sparse calendar row",
        payload={"description": "Older payload shape."},
    )

    print_calendar_proposals(store)

    output = capsys.readouterr().out
    assert f"approval_id: {item.id}" in output
    assert "title: Sparse calendar row" in output
    assert "date_text:" in output
    assert "attendees:" in output


def test_complete_task_cli_only_completes_propose_task_items(
    tmp_path, capsys, monkeypatch
):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setenv("LEWIS_DB_PATH", str(db_path))
    store = ApprovalStore(db_path)
    task = store.create(
        action_type="propose_task",
        title="Submit form",
        payload={"title": "Submit form", "description": "Submit the permit form."},
    )
    email = store.create(
        action_type="propose_email_draft",
        title="Email Alex",
        payload={"to": "Alex", "body": "Hello"},
    )

    task_exit_code = main(["complete-task", str(task.id)])
    email_exit_code = main(["complete-task", str(email.id)])

    output = capsys.readouterr().out
    assert task_exit_code == 0
    assert email_exit_code == 1
    assert store.get(task.id).status == "completed"
    assert store.get(email.id).status == "pending"
    assert "Only propose_task items" in output


def test_complete_task_cli_is_local_only(tmp_path, capsys, monkeypatch):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setenv("LEWIS_DB_PATH", str(db_path))
    store = ApprovalStore(db_path)
    item = store.create(
        action_type="propose_task",
        title="Call dentist",
        payload={
            "title": "Call dentist",
            "description": "Schedule a cleaning.",
            "due_date_text": "Friday",
        },
    )

    exit_code = main(["complete-task", str(item.id)])

    output = capsys.readouterr().out
    completed = store.get(item.id)
    assert exit_code == 0
    assert completed.status == "completed"
    assert completed.payload == item.payload
    assert "No emails, reminders, calendar events, notifications, or external" in output


def test_reopen_task_cli_marks_completed_task_approved(tmp_path, capsys, monkeypatch):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setenv("LEWIS_DB_PATH", str(db_path))
    store = ApprovalStore(db_path)
    item = store.create(
        action_type="propose_task",
        title="Call dentist",
        payload={"title": "Call dentist", "description": "Schedule a cleaning."},
    )
    store.update_status(item.id, "completed")

    exit_code = main(["reopen-task", str(item.id)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert store.get(item.id).status == "approved"
    assert "No external actions were executed" in output


def test_execute_approved_cli_previews_approved_item_only(
    tmp_path, capsys, monkeypatch
):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setenv("LEWIS_DB_PATH", str(db_path))
    store = ApprovalStore(db_path)
    item = store.create(
        action_type="propose_email_draft",
        title="Email Alex",
        payload={"to": "Alex", "subject": "Hello", "body": "Hi"},
    )
    store.update_status(item.id, "approved")

    exit_code = main(["execute-approved", str(item.id)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert store.get(item.id).status == "execution_previewed"
    assert "Execution previewed" in output
    assert "Dry-run preview only" in output
    assert "No email was sent" in output
    assert "No external action was taken" in output


def test_execute_approved_cli_rejects_pending_item(tmp_path, capsys, monkeypatch):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setenv("LEWIS_DB_PATH", str(db_path))
    store = ApprovalStore(db_path)
    item = store.create(
        action_type="propose_task",
        title="Call dentist",
        payload={"title": "Call dentist", "description": "Schedule a cleaning."},
    )

    exit_code = main(["execute-approved", str(item.id)])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert store.get(item.id).status == "pending"
    assert "only approved items can be execution-previewed" in output


def test_dashboard_counts_execution_previewed_items(tmp_path, capsys):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    logger = RunLogger(db_path)
    item = store.create(
        action_type="propose_calendar_event",
        title="Project sync",
        payload={"title": "Project sync", "description": "Discuss launch."},
    )
    store.update_status(item.id, "approved")
    execute_approved(store, item.id)

    dashboard = get_dashboard_summary(store, logger)
    print_dashboard(store, logger)

    output = capsys.readouterr().out
    assert dashboard["counts"]["execution_previewed_items"] == 1
    assert "execution previewed items: 1" in output
