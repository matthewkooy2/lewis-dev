from app.approvals import ApprovalStore
from app.code_proposal_previews import PATCH_PREVIEW_SUSPICIOUS_PATHS_WARNING
from app.logging_db import RunLogger
from app.memory_store import MemoryStore
from app.project_search import ProjectFileSearch
from app.rag import DocumentSearch, NO_DOCUMENTS_MESSAGE, NO_MATCHES_MESSAGE
from app.tools import ToolRegistry


EXPECTED_CODE_PROPOSAL_TEST_COMMANDS = [
    r".\.venv\Scripts\python.exe -m ruff format .",
    r".\.venv\Scripts\python.exe -m ruff check .",
    r".\.venv\Scripts\python.exe -m pytest",
]


def test_search_documents_and_list_files(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "note.md").write_text("Project Alpha has a design note.", encoding="utf-8")

    tools = ToolRegistry(DocumentSearch(docs), ApprovalStore(tmp_path / "test.sqlite3"))

    listed = tools.execute("list_local_files", {})
    assert listed["ok"] is True
    assert listed["files"] == ["note.md"]

    result = tools.execute("search_documents", {"query": "Alpha"})
    assert result["ok"] is True
    assert result["results"][0]["path"] == "note.md"


def test_search_documents_schema_accepts_query_only(tmp_path):
    tools = ToolRegistry(
        DocumentSearch(tmp_path / "documents"),
        ApprovalStore(tmp_path / "test.sqlite3"),
    )

    schema = next(
        item["function"]["parameters"]
        for item in tools.schemas()
        if item["function"]["name"] == "search_documents"
    )

    assert schema["required"] == ["query"]
    assert set(schema["properties"]) == {"query"}
    assert schema["additionalProperties"] is False


def test_search_documents_reports_missing_documents_folder(tmp_path):
    tools = ToolRegistry(
        DocumentSearch(tmp_path / "missing-documents"),
        ApprovalStore(tmp_path / "test.sqlite3"),
    )

    result = tools.execute("search_documents", {"query": "anything"})

    assert result == {"ok": True, "message": NO_DOCUMENTS_MESSAGE, "results": []}


def test_search_documents_reports_no_clear_match(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "note.md").write_text("Project Alpha only.", encoding="utf-8")
    tools = ToolRegistry(DocumentSearch(docs), ApprovalStore(tmp_path / "test.sqlite3"))

    result = tools.execute("search_documents", {"query": "Zanzibar"})

    assert result == {"ok": True, "message": NO_MATCHES_MESSAGE, "results": []}


def test_search_documents_rejects_arbitrary_path_argument(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("escapephrase", encoding="utf-8")
    tools = ToolRegistry(DocumentSearch(docs), ApprovalStore(tmp_path / "test.sqlite3"))

    result = tools.execute(
        "search_documents",
        {"query": "escapephrase", "path": str(outside)},
    )

    assert result["ok"] is False
    assert "only accepts a query" in result["error"]


def test_search_project_files_tool_is_read_only_and_query_only(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "service.py").write_text(
        "Project Alpha helper",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("Alpha secret", encoding="utf-8")
    tools = ToolRegistry(
        DocumentSearch(tmp_path / "documents"),
        ApprovalStore(tmp_path / "test.sqlite3"),
        project_file_search=ProjectFileSearch(tmp_path),
    )

    result = tools.execute("search_project_files", {"query": "Alpha"})
    rejected = tools.execute(
        "search_project_files",
        {"query": "Alpha", "path": ".env"},
    )

    assert result["ok"] is True
    assert result["results"] == [
        {
            "path": "app/service.py",
            "line_number": 1,
            "snippet": "Project Alpha helper",
        }
    ]
    assert rejected["ok"] is False
    assert "only accepts a query" in rejected["error"]


def test_proposal_tool_creates_pending_approval(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    tools = ToolRegistry(DocumentSearch(tmp_path / "documents"), store)

    result = tools.execute(
        "propose_task",
        {
            "title": "Buy stamps",
            "description": "Buy stamps for thank-you notes.",
            "due_date_text": "tomorrow",
            "priority": "medium",
            "source": "chat",
        },
    )

    assert result["ok"] is True
    assert result["status"] == "pending"
    item = store.list(status="pending")[0]
    assert item.action_type == "propose_task"
    assert item.payload == {
        "title": "Buy stamps",
        "description": "Buy stamps for thank-you notes.",
        "due_date_text": "tomorrow",
        "priority": "medium",
        "source": "chat",
    }


def test_task_proposal_tool_schema_is_task_specific(tmp_path):
    tools = ToolRegistry(
        DocumentSearch(tmp_path / "documents"),
        ApprovalStore(tmp_path / "test.sqlite3"),
    )

    schema = next(
        item["function"]["parameters"]
        for item in tools.schemas()
        if item["function"]["name"] == "propose_task"
    )

    assert schema["required"] == ["title", "description"]
    assert set(schema["properties"]) == {
        "title",
        "description",
        "due_date_text",
        "priority",
        "source",
    }
    assert schema["additionalProperties"] is False


def test_calendar_proposal_tool_creates_pending_approval(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    tools = ToolRegistry(DocumentSearch(tmp_path / "documents"), store)

    result = tools.execute(
        "propose_calendar_event",
        {
            "title": "Project sync",
            "description": "Discuss the launch checklist.",
            "date_text": "next Tuesday",
            "start_time_text": "2:00 PM",
            "end_time_text": "2:30 PM",
            "location": "Zoom",
            "attendees": ["Alex", "Jordan"],
            "source": "chat",
            "notes": "Local proposal only.",
        },
    )

    assert result["ok"] is True
    assert result["status"] == "pending"
    assert "not a real calendar event" in result["message"]
    item = store.get(result["approval_id"])
    assert item is not None
    assert item.action_type == "propose_calendar_event"
    assert item.payload == {
        "title": "Project sync",
        "description": "Discuss the launch checklist.",
        "date_text": "next Tuesday",
        "start_time_text": "2:00 PM",
        "end_time_text": "2:30 PM",
        "location": "Zoom",
        "attendees": ["Alex", "Jordan"],
        "source": "chat",
        "notes": "Local proposal only.",
    }


def test_calendar_proposal_tool_schema_is_calendar_specific(tmp_path):
    tools = ToolRegistry(
        DocumentSearch(tmp_path / "documents"),
        ApprovalStore(tmp_path / "test.sqlite3"),
    )

    schema = next(
        item["function"]["parameters"]
        for item in tools.schemas()
        if item["function"]["name"] == "propose_calendar_event"
    )

    assert schema["required"] == ["title", "description"]
    assert set(schema["properties"]) == {
        "title",
        "description",
        "date_text",
        "start_time_text",
        "end_time_text",
        "location",
        "attendees",
        "source",
        "notes",
    }
    assert schema["additionalProperties"] is False


def test_code_proposal_tool_creates_pending_approval_only(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    tools = ToolRegistry(DocumentSearch(tmp_path / "documents"), store)

    result = tools.execute(
        "propose_code_change",
        {
            "requested_goal": "Add a safer planning mode.",
            "relevant_files": ["app/main.py", "app/tools.py"],
            "proposed_approach": "Add a validation helper and store a proposal.",
            "patch_preview": (
                "diff --git a/app/tools.py b/app/tools.py\n"
                "--- a/app/tools.py\n"
                "+++ b/app/tools.py\n"
                "@@ -1 +1 @@\n"
                "-old\n"
                "+new"
            ),
            "safety_notes": ["Planning only; no patches."],
            "suggested_test_commands": ["python -m pytest tests/test_tools.py"],
        },
    )

    assert result["ok"] is True
    assert result["status"] == "pending"
    assert "planning only" in result["message"].lower()
    assert "no files were read, edited, patched, or executed" in result["message"]
    item = store.get(result["approval_id"])
    assert item is not None
    assert item.action_type == "propose_code_change"
    assert item.payload == {
        "requested_goal": "Add a safer planning mode.",
        "target_surface": "unspecified",
        "relevant_files": ["app/main.py", "app/tools.py"],
        "proposed_approach": "Add a validation helper and store a proposal.",
        "patch_preview": (
            "diff --git a/app/tools.py b/app/tools.py\n"
            "--- a/app/tools.py\n"
            "+++ b/app/tools.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new"
        ),
        "patch_preview_path_classifications": [
            {
                "path": "app/tools.py",
                "classification": "existing project file",
            }
        ],
        "patch_preview_implementation_ready": True,
        "safety_notes": ["Planning only; no patches."],
        "suggested_test_commands": EXPECTED_CODE_PROPOSAL_TEST_COMMANDS,
    }


def test_code_proposal_tool_schema_is_planning_specific(tmp_path):
    tools = ToolRegistry(
        DocumentSearch(tmp_path / "documents"),
        ApprovalStore(tmp_path / "test.sqlite3"),
    )

    schema = next(
        item["function"]["parameters"]
        for item in tools.schemas()
        if item["function"]["name"] == "propose_code_change"
    )

    assert schema["required"] == ["requested_goal", "proposed_approach"]
    assert set(schema["properties"]) == {
        "requested_goal",
        "target_surface",
        "relevant_files",
        "proposed_approach",
        "patch_preview",
        "safety_notes",
        "suggested_test_commands",
    }
    assert schema["additionalProperties"] is False


def test_code_proposal_tool_rejects_unsafe_paths_and_patch_text(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    tools = ToolRegistry(DocumentSearch(tmp_path / "documents"), store)

    outside = tools.execute(
        "propose_code_change",
        {
            "requested_goal": "Inspect outside path.",
            "relevant_files": ["../outside.py"],
            "proposed_approach": "Keep this as planning.",
        },
    )
    secret = tools.execute(
        "propose_code_change",
        {
            "requested_goal": "Inspect env.",
            "relevant_files": [".env"],
            "proposed_approach": "Keep this as planning.",
        },
    )
    patch = tools.execute(
        "propose_code_change",
        {
            "requested_goal": "Patch code.",
            "relevant_files": ["app/main.py"],
            "proposed_approach": "diff --git a/app/main.py b/app/main.py",
        },
    )
    sqlite_file = tools.execute(
        "propose_code_change",
        {
            "requested_goal": "Inspect runtime data.",
            "relevant_files": ["data/lewis.sqlite3"],
            "proposed_approach": "Keep this as planning.",
        },
    )
    venv_file = tools.execute(
        "propose_code_change",
        {
            "requested_goal": "Inspect installed dependency.",
            "relevant_files": [".venv/Lib/site-packages/example.py"],
            "proposed_approach": "Keep this as planning.",
        },
    )

    assert outside["ok"] is False
    assert "dot segments" in outside["error"]
    assert secret["ok"] is False
    assert "blocked file name" in secret["error"]
    assert patch["ok"] is False
    assert "not a patch or diff" in patch["error"]
    assert sqlite_file["ok"] is False
    assert "blocked runtime path" in sqlite_file["error"]
    assert venv_file["ok"] is False
    assert "blocked runtime path" in venv_file["error"]
    assert store.list(status="pending") == []


def test_code_proposal_patch_preview_rejects_unsafe_paths(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    tools = ToolRegistry(DocumentSearch(tmp_path / "documents"), store)

    secret = tools.execute(
        "propose_code_change",
        {
            "requested_goal": "Preview env edit.",
            "relevant_files": ["app/main.py"],
            "proposed_approach": "Keep this as planning.",
            "patch_preview": "diff --git a/.env b/.env\n--- a/.env\n+++ b/.env",
        },
    )
    sqlite_file = tools.execute(
        "propose_code_change",
        {
            "requested_goal": "Preview runtime data edit.",
            "relevant_files": ["app/main.py"],
            "proposed_approach": "Keep this as planning.",
            "patch_preview": (
                "diff --git a/data/lewis.sqlite3 b/data/lewis.sqlite3\n"
                "--- a/data/lewis.sqlite3\n"
                "+++ b/data/lewis.sqlite3"
            ),
        },
    )
    outside = tools.execute(
        "propose_code_change",
        {
            "requested_goal": "Preview outside edit.",
            "relevant_files": ["app/main.py"],
            "proposed_approach": "Keep this as planning.",
            "patch_preview": "diff --git a/../outside.py b/../outside.py",
        },
    )
    venv_file = tools.execute(
        "propose_code_change",
        {
            "requested_goal": "Preview dependency edit.",
            "relevant_files": ["app/main.py"],
            "proposed_approach": "Keep this as planning.",
            "patch_preview": ("diff --git a/.venv/Lib/site.py b/.venv/Lib/site.py"),
        },
    )
    git_file = tools.execute(
        "propose_code_change",
        {
            "requested_goal": "Preview git edit.",
            "relevant_files": ["app/main.py"],
            "proposed_approach": "Keep this as planning.",
            "patch_preview": "diff --git a/.git/config b/.git/config",
        },
    )
    absolute = tools.execute(
        "propose_code_change",
        {
            "requested_goal": "Preview absolute edit.",
            "relevant_files": ["app/main.py"],
            "proposed_approach": "Keep this as planning.",
            "patch_preview": "diff --git a/C:/tmp/file.py b/C:/tmp/file.py",
        },
    )

    assert secret["ok"] is False
    assert "blocked file name" in secret["error"]
    assert sqlite_file["ok"] is False
    assert "blocked runtime path" in sqlite_file["error"]
    assert outside["ok"] is False
    assert "dot segments" in outside["error"]
    assert venv_file["ok"] is False
    assert "blocked runtime path" in venv_file["error"]
    assert git_file["ok"] is False
    assert "blocked runtime path" in git_file["error"]
    assert absolute["ok"] is False
    assert "relative paths" in absolute["error"]
    assert store.list(status="pending") == []


def test_code_proposal_patch_preview_classifies_project_structure(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    tools = ToolRegistry(DocumentSearch(tmp_path / "documents"), store)

    result = tools.execute(
        "propose_code_change",
        {
            "requested_goal": "Preview a command center UI refinement.",
            "relevant_files": ["app/web_components.py"],
            "proposed_approach": "Keep the patch as inert review text.",
            "patch_preview": (
                "diff --git a/app/web_components.py b/app/web_components.py\n"
                "--- a/app/web_components.py\n"
                "+++ b/app/web_components.py\n"
                "diff --git a/app/new_preview_helper.py b/app/new_preview_helper.py\n"
                "--- /dev/null\n"
                "+++ b/app/new_preview_helper.py\n"
                "diff --git a/app/templates/chat.html b/app/templates/chat.html\n"
                "--- /dev/null\n"
                "+++ b/app/templates/chat.html"
            ),
        },
    )

    assert result["ok"] is True
    item = store.get(result["approval_id"])
    assert item is not None
    assert item.payload["patch_preview_path_classifications"] == [
        {
            "path": "app/web_components.py",
            "classification": "existing project file",
        },
        {
            "path": "app/new_preview_helper.py",
            "classification": "new file under an existing safe project directory",
        },
        {
            "path": "app/templates/chat.html",
            "classification": "suspicious or invalid project structure",
        },
    ]
    assert item.payload["patch_preview_implementation_ready"] is False
    assert item.payload["patch_preview_warning"] == (
        PATCH_PREVIEW_SUSPICIOUS_PATHS_WARNING
    )


def test_code_proposal_command_center_ui_defaults_to_web_surface(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    tools = ToolRegistry(DocumentSearch(tmp_path / "documents"), store)

    result = tools.execute(
        "propose_code_change",
        {
            "requested_goal": "Update the command center avatar UI animation.",
            "relevant_files": ["app/web_components.py", "app/static/lewis.css"],
            "proposed_approach": "Plan a visual avatar animation for the command center screen.",
        },
    )

    assert result["ok"] is True
    item = store.get(result["approval_id"])
    assert item is not None
    assert item.payload["target_surface"] == "command center web UI"
    assert (
        "Assumption: this targets the command center web UI."
        in item.payload["safety_notes"]
    )
    assert (
        "Do not reinterpret this as a CLI/terminal ASCII UI unless explicitly requested."
        in item.payload["safety_notes"]
    )


def test_code_proposal_command_center_ui_prefers_existing_web_paths(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    tools = ToolRegistry(DocumentSearch(tmp_path / "documents"), store)

    result = tools.execute(
        "propose_code_change",
        {
            "requested_goal": "Improve the command center UI controls.",
            "relevant_files": ["app/main.py"],
            "proposed_approach": "Plan web page and stylesheet updates for the command center interface.",
        },
    )

    assert result["ok"] is True
    item = store.get(result["approval_id"])
    assert item is not None
    assert item.payload["target_surface"] == "command center web UI"
    assert "app/main.py" not in item.payload["relevant_files"]
    assert "app/web_components.py" in item.payload["relevant_files"]
    assert "app/static/lewis.css" in item.payload["relevant_files"]


def test_code_proposal_command_center_ui_avoids_invented_web_directories(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    tools = ToolRegistry(DocumentSearch(tmp_path / "documents"), store)

    result = tools.execute(
        "propose_code_change",
        {
            "requested_goal": "Improve the command center chat UI.",
            "relevant_files": ["app/templates/chat.html", "app/static/css/chat.css"],
            "proposed_approach": "Plan web page and stylesheet updates for the command center interface.",
        },
    )

    assert result["ok"] is True
    item = store.get(result["approval_id"])
    assert item is not None
    assert item.payload["target_surface"] == "command center web UI"
    assert "app/templates/chat.html" not in item.payload["relevant_files"]
    assert "app/static/css/chat.css" not in item.payload["relevant_files"]
    assert "app/web_components.py" in item.payload["relevant_files"]
    assert "app/static/lewis.css" in item.payload["relevant_files"]


def test_code_proposal_explicit_terminal_ui_stays_terminal(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    tools = ToolRegistry(DocumentSearch(tmp_path / "documents"), store)

    result = tools.execute(
        "propose_code_change",
        {
            "requested_goal": "Add a terminal avatar animation to the CLI.",
            "relevant_files": ["app/main.py"],
            "proposed_approach": "Plan an ASCII animation for terminal chat output.",
        },
    )

    assert result["ok"] is True
    item = store.get(result["approval_id"])
    assert item is not None
    assert item.payload["target_surface"] == "CLI/terminal UI"
    assert item.payload["relevant_files"] == ["app/main.py"]
    assert (
        "Assumption: this targets the command center web UI."
        not in item.payload["safety_notes"]
    )


def test_code_proposal_config_prefers_existing_app_config_over_missing_src(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    tools = ToolRegistry(DocumentSearch(tmp_path / "documents"), store)

    result = tools.execute(
        "propose_code_change",
        {
            "requested_goal": "Add runtime configuration for a feature flag.",
            "relevant_files": ["src/config.py"],
            "proposed_approach": "Use the existing config module to read an environment variable.",
        },
    )

    assert result["ok"] is True
    item = store.get(result["approval_id"])
    assert item is not None
    assert item.payload["target_surface"] == "backend/service logic"
    assert item.payload["relevant_files"] == ["app/config.py"]


def test_code_proposal_sensitive_config_adds_safety_notes_without_values(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    tools = ToolRegistry(DocumentSearch(tmp_path / "documents"), store)

    result = tools.execute(
        "propose_code_change",
        {
            "requested_goal": "Support a .env.example API token setting where token is sk-testtesttesttesttesttesttest.",
            "relevant_files": ["app/config.py"],
            "proposed_approach": "Read an environment variable at runtime and document it in .env.example.",
            "safety_notes": ["Avoid storing password: correct-horse-battery-staple."],
        },
    )

    assert result["ok"] is True
    item = store.get(result["approval_id"])
    assert item is not None
    combined = " ".join(
        [
            item.payload["requested_goal"],
            item.payload["proposed_approach"],
            *item.payload["safety_notes"],
        ]
    )
    assert "sk-testtesttesttesttesttesttest" not in combined
    assert "correct-horse-battery-staple" not in combined
    assert "[redacted]" in combined
    assert item.payload["target_surface"] == "backend/service logic"
    assert (
        "Do not inspect, expose, or request .env, secret, credential, token, database, or runtime data contents."
        in item.payload["safety_notes"]
    )
    assert (
        "Prefer .env.example documentation and environment-variable based runtime configuration."
        in item.payload["safety_notes"]
    )
    assert (
        "Frame this as runtime config support, not as exposing .env contents or secret values."
        in item.payload["safety_notes"]
    )


def test_memory_proposal_tool_creates_pending_approval_only(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    tools = ToolRegistry(DocumentSearch(tmp_path / "documents"), store)

    result = tools.execute(
        "propose_memory",
        {
            "memory_type": "preference",
            "content": "Blake prefers concise implementation summaries.",
            "source": "chat",
            "tags": ["summary"],
        },
    )

    assert result["ok"] is True
    assert result["status"] == "pending"
    assert "Approval is required" in result["message"]
    item = store.get(result["approval_id"])
    assert item is not None
    assert item.action_type == "propose_memory"
    assert item.payload == {
        "memory_type": "preference",
        "content": "Blake prefers concise implementation summaries.",
        "source": "chat",
        "tags_json": ["summary"],
    }
    assert MemoryStore(db_path).list_active() == []


def test_memory_proposal_tool_uses_memory_service_validation(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    tools = ToolRegistry(DocumentSearch(tmp_path / "documents"), store)

    invalid_type = tools.execute(
        "propose_memory",
        {"memory_type": "secret", "content": "Blake prefers concise summaries."},
    )
    empty_content = tools.execute(
        "propose_memory",
        {"memory_type": "preference", "content": "  "},
    )
    secret_content = tools.execute(
        "propose_memory",
        {
            "memory_type": "fact",
            "content": "The API key is sk-testtesttesttesttesttesttest.",
        },
    )

    assert invalid_type["ok"] is False
    assert "unsupported memory_type" in invalid_type["error"]
    assert empty_content["ok"] is False
    assert "content is required" in empty_content["error"]
    assert secret_content["ok"] is False
    assert "appears to contain a secret" in secret_content["error"]
    assert store.list(status="pending") == []


def test_email_draft_uses_neutral_signoff_without_configured_user_name(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    tools = ToolRegistry(DocumentSearch(tmp_path / "documents"), store)

    result = tools.execute(
        "propose_email_draft",
        {
            "title": "Draft professor email",
            "details": {
                "to": "Professor Smith",
                "subject": "Question",
                "body": "Hello Professor Smith,\n\nThanks.\n\nBest regards,\nAvery",
            },
        },
    )

    assert result["ok"] is True
    item = store.get(result["approval_id"])
    assert item is not None
    assert item.payload["body"].endswith("Best regards,")
    assert "Avery" not in item.payload["body"]


def test_duplicate_email_draft_reuses_pending_approval(tmp_path):
    store = ApprovalStore(tmp_path / "test.sqlite3")
    tools = ToolRegistry(DocumentSearch(tmp_path / "documents"), store)
    payload = {
        "title": "Draft professor email",
        "details": {
            "to": "Professor Smith",
            "subject": "Question",
            "body": "Hello.\n\nBest regards,",
        },
    }

    first = tools.execute("propose_email_draft", payload)
    second = tools.execute("propose_email_draft", payload)

    assert first["approval_id"] == second["approval_id"]
    assert len(store.list(status="pending")) == 1


def test_malformed_tool_arguments_are_handled(tmp_path):
    tools = ToolRegistry(
        DocumentSearch(tmp_path / "documents"), ApprovalStore(tmp_path / "test.sqlite3")
    )

    result = tools.execute("get_current_time", "{bad json")

    assert result["ok"] is False
    assert "Malformed" in result["error"]


def test_read_only_dashboard_tools_return_real_local_data(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    logger = RunLogger(db_path)
    task = store.create(
        action_type="propose_task",
        title="Submit form",
        payload={
            "title": "Submit form",
            "description": "Submit the permit form.",
            "due_date_text": "Friday",
            "priority": "high",
        },
    )
    email = store.create(
        action_type="propose_email_draft",
        title="Email Alex",
        payload={"to": "Alex", "body": "Hello"},
    )
    logger.log_run(
        user_input="hello",
        model="test-model",
        tool_calls=[],
        input_tokens=5,
        output_tokens=2,
        total_tokens=7,
        api_calls=1,
    )
    tools = ToolRegistry(DocumentSearch(tmp_path / "documents"), store, logger=logger)

    pending = tools.execute("list_pending_approvals", {})
    tasks = tools.execute("list_tasks", {})
    dashboard = tools.execute("get_dashboard_summary", {})
    calendar = tools.execute("list_calendar_proposals", {})

    assert pending["ok"] is True
    assert [item["approval_id"] for item in pending["approvals"]] == [
        email.id,
        task.id,
    ]
    assert tasks["ok"] is True
    assert tasks["tasks"] == [
        {
            "approval_id": task.id,
            "status": "pending",
            "title": "Submit form",
            "description": "Submit the permit form.",
            "due_date_text": "Friday",
            "priority": "high",
            "created_at": task.created_at,
        }
    ]
    assert dashboard["ok"] is True
    assert dashboard["dashboard"]["counts"]["total_pending"] == 2
    assert dashboard["dashboard"]["counts"]["pending_drafts"] == 1
    assert dashboard["dashboard"]["counts"]["pending_tasks"] == 1
    assert dashboard["dashboard"]["counts"]["execution_previewed_items"] == 0
    assert dashboard["dashboard"]["usage_summary"]["run_count"] == 1
    assert calendar == {"ok": True, "calendar_proposals": []}
    assert store.get(task.id).status == "pending"
    assert store.get(email.id).status == "pending"


def test_read_only_dashboard_tools_do_not_invent_items(tmp_path):
    tools = ToolRegistry(
        DocumentSearch(tmp_path / "documents"),
        ApprovalStore(tmp_path / "test.sqlite3"),
    )

    pending = tools.execute("list_pending_approvals", {})
    tasks = tools.execute("list_tasks", {})
    calendar = tools.execute("list_calendar_proposals", {})
    dashboard = tools.execute("get_dashboard_summary", {})

    assert pending == {"ok": True, "approvals": []}
    assert tasks == {"ok": True, "tasks": []}
    assert calendar == {"ok": True, "calendar_proposals": []}
    assert dashboard["ok"] is True
    assert dashboard["dashboard"]["counts"] == {
        "total_pending": 0,
        "pending_drafts": 0,
        "pending_tasks": 0,
        "pending_calendar_events": 0,
        "approved_tasks": 0,
        "completed_tasks": 0,
        "execution_previewed_items": 0,
        "approved_calendar_events": 0,
        "total_rejected": 0,
    }
