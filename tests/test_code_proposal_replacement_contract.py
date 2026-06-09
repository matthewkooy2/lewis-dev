from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.approvals import ApprovalItem, ApprovalStore
from app.budget import BudgetManager
from app.code_proposal_previews import extract_patch_preview_paths
from app.config import Settings
from app.llm import LewisAgent
from app.logging_db import RunLogger
from app.project_search import ProjectFileSearch
from app.rag import DocumentSearch
from app.tools import ToolRegistry


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        deepseek_api_key="test-key",
        deepseek_model="test-model",
        deepseek_base_url="https://api.deepseek.com",
        debug=False,
        user_name="the user",
        user_name_configured=False,
        database_path=tmp_path / "test.sqlite3",
        documents_dir=tmp_path / "documents",
        max_tool_calls_per_request=5,
        max_estimated_daily_cost=1.0,
        price_per_1k_input_tokens=0.0,
        price_per_1k_output_tokens=0.0,
        deepseek_code_model="test-code-model",
    )


def _make_agent(tmp_path: Path, project_root: Path) -> LewisAgent:
    settings = _make_settings(tmp_path)
    settings.documents_dir.mkdir()
    approvals = ApprovalStore(settings.database_path)
    tools = ToolRegistry(
        DocumentSearch(settings.documents_dir),
        approvals,
        project_file_search=ProjectFileSearch(project_root),
    )
    agent = LewisAgent(
        settings=settings,
        tools=tools,
        budget=BudgetManager(settings),
        logger=RunLogger(settings.database_path),
    )
    agent.client = SimpleNamespace(
        chat=SimpleNamespace(completions=_NoOpCompletions())
    )
    return agent


class _NoOpCompletions:
    def create(self, **kwargs: object) -> object:
        del kwargs
        message = SimpleNamespace(content="No proposal generated.", tool_calls=None)
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=usage,
        )


def _snapshot(project_root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(project_root).as_posix(): path.read_bytes()
        for path in project_root.rglob("*")
        if path.is_file()
    }


def _assert_real_inert_unified_diff(
    project_root: Path,
    before: dict[str, bytes],
    approval: ApprovalItem,
) -> None:
    assert _snapshot(project_root) == before
    patch_preview = approval.payload.get("patch_preview")
    assert isinstance(patch_preview, str) and patch_preview.startswith("diff --git ")

    checked = subprocess.run(
        ["git", "apply", "--check", "-"],
        cwd=project_root,
        input=f"{patch_preview}\n",
        text=True,
        capture_output=True,
        check=False,
    )

    assert checked.returncode == 0, checked.stderr
    assert _snapshot(project_root) == before


@pytest.mark.parametrize(
    ("prompt", "files"),
    [
        (
            "Create a code proposal that makes the command center chat text blue.",
            {
                "app/static/lewis.css": (
                    ".chat-log .bubble {\n"
                    "    color: var(--ink);\n"
                    "}\n"
                ),
                "app/web_pages.py": (
                    "def chat_page():\n"
                    "    return \"<section class='chat-shell'>chat</section>\"\n"
                ),
            },
        ),
        (
            "Create a code proposal that adds a thinking animation to chat.",
            {
                "app/static/chat.js": (
                    "function showThinkingBubble() {\n"
                    "    return 'thinking';\n"
                    "}\n"
                ),
                "app/static/lewis.css": (
                    ".bubble.thinking {\n"
                    "    opacity: 0.7;\n"
                    "}\n"
                ),
            },
        ),
    ],
)
def test_code_proposal_persists_real_inert_unified_diff(
    tmp_path: Path,
    prompt: str,
    files: dict[str, str],
) -> None:
    project_root = tmp_path / "project"
    for relative_path, content in files.items():
        target = project_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    before = _snapshot(project_root)
    agent = _make_agent(tmp_path, project_root)

    run = agent.run(prompt)

    assert run.tool_calls[-1]["name"] == "propose_code_change"
    approval = ApprovalStore(agent.settings.database_path).list(status="pending")[0]
    _assert_real_inert_unified_diff(project_root, before, approval)


def test_tool_use_limit_proposal_uses_search_backed_integer_setting_diff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    (project_root / "app").mkdir(parents=True)
    (project_root / "app" / "config.py").write_text(
        "def get_settings():\n"
        "    return Settings(\n"
        '        max_tool_calls_per_request=_env_int("LEWIS_MAX_TOOL_CALLS", 5),\n'
        "    )\n",
        encoding="utf-8",
    )
    (project_root / "app" / "tool_call_limits.py").write_text(
        "def effective_tool_call_limit(settings):\n"
        "    return max(0, settings.max_tool_calls_per_request)\n",
        encoding="utf-8",
    )
    before = _snapshot(project_root)
    agent = _make_agent(tmp_path, project_root)

    def fail_runtime_action(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Lewis runtime attempted shell or process execution.")

    with monkeypatch.context() as context:
        context.setattr(subprocess, "run", fail_runtime_action)
        context.setattr(subprocess, "Popen", fail_runtime_action)
        run = agent.run("make a code proposal to increase the tool use limit")

    approvals = ApprovalStore(agent.settings.database_path).list(status="pending")
    assert len(approvals) == 1
    approval = approvals[0]
    assert approval.payload["requested_goal"] == "Increase the tool use limit."

    patch_preview = approval.payload["patch_preview"]
    assert patch_preview.startswith("diff --git ")
    assert all(marker in patch_preview for marker in ("\n--- ", "\n+++ ", "\n@@ "))
    assert 'LEWIS_MAX_TOOL_CALLS", 5' in patch_preview
    assert 'LEWIS_MAX_TOOL_CALLS", 8' in patch_preview

    searched_paths = {
        item["path"]
        for record in run.tool_calls
        if record["name"] == "search_project_files"
        for item in record["result"]["results"]
    }
    touched_paths = set(extract_patch_preview_paths(patch_preview))
    assert set(approval.payload["relevant_files"]) <= searched_paths
    assert touched_paths <= set(approval.payload["relevant_files"])
    assert run.tool_calls[-1]["name"] == "propose_code_change"
    assert run.tool_calls[-1]["result"]["approval_id"] == approval.id
    assert f"#{approval.id}" in run.answer
    assert not any(
        marker in run.answer for marker in ("<function_calls>", "<invoke", "DSML")
    )
    _assert_real_inert_unified_diff(project_root, before, approval)


def test_tool_use_limit_grounding_failure_explains_why_and_saves_no_approval(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    (project_root / "app").mkdir(parents=True)
    target = project_root / "app" / "tool_call_limits.py"
    target.write_text(
        "def effective_tool_call_limit(settings):\n"
        "    return settings.max_tool_calls_per_request\n",
        encoding="utf-8",
    )
    before = _snapshot(project_root)
    agent = _make_agent(tmp_path, project_root)

    run = agent.run("make a code proposal to increase the tool use limit")

    assert all(record["name"] == "search_project_files" for record in run.tool_calls)
    assert ApprovalStore(agent.settings.database_path).list(status="pending") == []
    assert "Cleaned goal: Increase the tool use limit." in run.answer
    assert "Search terms tried:" in run.answer
    assert "Candidate files found: yes" in run.answer
    assert "app/tool_call_limits.py" in run.answer
    assert "Reason no grounded diff could be created:" in run.answer
    assert "No pending code proposal was saved." in run.answer
    assert _snapshot(project_root) == before


def test_clear_history_feature_discovers_sources_and_fails_safely_with_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    (project_root / "app").mkdir(parents=True)
    (project_root / "app" / "web_pages.py").write_text(
        "def history_page(items, selected, message=''):\n"
        "    return page_header('History') + history_filters_html(selected)\n",
        encoding="utf-8",
    )
    (project_root / "app" / "web_components.py").write_text(
        "def history_filters_html(selected):\n"
        "    return \"<a href='/history?type=all'>All</a>\"\n",
        encoding="utf-8",
    )
    (project_root / "app" / "web_routes.py").write_text(
        "def history(request, type='all'):\n"
        "    items = list_history_items(request.app.state.approval_store, type)\n"
        "    return history_page(items, type)\n",
        encoding="utf-8",
    )
    (project_root / "README.md").write_text(
        r"Manual retest: .\.venv\Scripts\python.exe -m pytest tests/test_web.py",
        encoding="utf-8",
    )
    before = _snapshot(project_root)
    agent = _make_agent(tmp_path, project_root)
    store = ApprovalStore(agent.settings.database_path)
    existing = store.create(
        action_type="propose_task",
        title="Existing processed row",
        payload={"title": "Existing processed row"},
    )
    existing = store.update_status(existing.id, "approved")

    def fail_runtime_action(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Lewis runtime attempted shell or process execution.")

    with monkeypatch.context() as context:
        context.setattr(subprocess, "run", fail_runtime_action)
        context.setattr(subprocess, "Popen", fail_runtime_action)
        context.setattr(os, "system", fail_runtime_action)
        run = agent.run(
            "I would like for you to make a code proposal adding a new feature "
            "to clear all of the history in the history tab"
        )

    assert all(record["name"] == "search_project_files" for record in run.tool_calls)
    assert 1 <= len(run.tool_calls) <= 2
    assert ApprovalStore(agent.settings.database_path).list(status="pending") == []
    assert store.get(existing.id) == existing
    assert _snapshot(project_root) == before
    assert (
        "Cleaned goal: Add a new feature to clear all of the history in the history tab."
        in run.answer
    )
    assert "Search terms tried:" in run.answer
    assert "history tab" in run.answer
    assert "history_tab" in run.answer
    assert "clear_history" in run.answer
    assert "Candidate files found: yes" in run.answer
    assert "app/web_pages.py" in run.answer
    assert "app/web_components.py" in run.answer
    assert "app/web_routes.py" in run.answer
    assert "potentially destructive" in run.answer
    assert "explicit confirmation" in run.answer
    assert "without a patch_preview is prohibited" in run.answer
    assert "README.md" not in run.answer


def test_ungroundable_feature_returns_specific_no_approval_diagnostics(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    agent = _make_agent(tmp_path, project_root)

    run = agent.run(
        "Make a code proposal to add quantum telemetry controls to the app."
    )

    assert ApprovalStore(agent.settings.database_path).list(status="pending") == []
    assert "Cleaned goal: Add quantum telemetry controls to the app." in run.answer
    assert "Search terms tried:" in run.answer
    assert "Candidate files found: no" in run.answer
    assert "Reason no grounded diff could be created:" in run.answer
    assert "README, documentation, and test/manual noise" in run.answer
    assert "No pending code proposal was saved." in run.answer


def test_file_discovery_proposal_does_not_return_readme_or_manual_command_noise(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    (project_root / "app").mkdir(parents=True)
    (project_root / "app" / "project_search.py").write_text(
        "def search_project_files(relevant_files):\n"
        "    return rank_relevant_files(relevant_files)\n",
        encoding="utf-8",
    )
    (project_root / "app" / "code_proposal_engine.py").write_text(
        "def discover_relevant_files(project_search):\n"
        "    return project_search\n",
        encoding="utf-8",
    )
    noisy_command = r".\.venv\Scripts\python.exe -m pytest tests/manual_retest.py"
    (project_root / "README.md").write_text(
        f"Manual retest command: {noisy_command}\n",
        encoding="utf-8",
    )
    agent = _make_agent(tmp_path, project_root)

    run = agent.run(
        "make a proposal to help you find relevant files so that this stops happening"
    )

    assert ApprovalStore(agent.settings.database_path).list(status="pending") == []
    assert "Cleaned goal: Help you find relevant files so that this stops happening." in run.answer
    assert "app/project_search.py" in run.answer
    assert "README.md" not in run.answer
    assert noisy_command not in run.answer
    assert "Project search found this relevant evidence:" not in run.answer
    assert "<function_calls>" not in run.answer
    assert "DSML" not in run.answer


def test_non_code_dinner_proposal_still_does_not_create_code_approval(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    agent = _make_agent(tmp_path, project_root)

    run = agent.run("create a proposal for dinner")

    assert run.tool_calls == []
    assert ApprovalStore(agent.settings.database_path).list(status="pending") == []


def _write_strategy_collision_project(project_root: Path) -> None:
    files = {
        "app/static/lewis.css": (
            ".chat-log .bubble {\n"
            "    color: var(--ink);\n"
            "}\n"
            ".bubble.thinking {\n"
            "    opacity: 0.7;\n"
            "}\n"
        ),
        "app/static/chat.js": "function showThinkingBubble() { return 'thinking'; }\n",
        "app/web_components.py": (
            "def command_center_chat():\n"
            "    return 'chat ui'\n"
            "\n"
            "def sidebar_navigation():\n"
            "    return 'dashboard sidebar chat icon status badge'\n"
        ),
        "app/config.py": (
            "def get_settings():\n"
            "    return Settings(\n"
            '        max_tool_calls_per_request=_env_int("LEWIS_MAX_TOOL_CALLS", 5),\n'
            "    )\n"
        ),
    }
    for relative_path, content in files.items():
        target = project_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def test_dashboard_chat_icon_completion_status_does_not_use_thinking_bubble_strategy(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _write_strategy_collision_project(project_root)
    before = _snapshot(project_root)
    agent = _make_agent(tmp_path, project_root)

    run = agent.run(
        'make a code proposal to add a little status dot to the "chat" icon on '
        "the left side dashboard, that changes to a blue color when you are done "
        "thinking and have produced your response"
    )

    assert ApprovalStore(agent.settings.database_path).list(status="pending") == []
    assert "Cleaned goal:" in run.answer
    assert "Candidate files found:" in run.answer
    assert "chat thinking-animation strategy" in run.answer
    assert "dashboard/sidebar/navigation link or icon" in run.answer
    assert "status indicator" in run.answer
    assert "chat content/message/bubble" in run.answer
    assert "No pending code proposal was saved." in run.answer
    assert ".bubble.thinking" not in run.answer
    assert all(record["name"] == "search_project_files" for record in run.tool_calls)
    assert _snapshot(project_root) == before
    # Diagnostic must not use the "no files found" lead-in when candidates exist.
    assert not run.answer.startswith(
        "Project search did not find enough relevant existing file content"
    )
    assert "Lewis found relevant project files" in run.answer
    assert "no conservative proposal strategy matched" in run.answer


@pytest.mark.parametrize(
    ("prompt", "expected_diagnostic"),
    [
        (
            "make a code proposal for an animated thinking status indicator in "
            "the sidebar navigation",
            "chat thinking-animation strategy",
        ),
        (
            "make a code proposal to make chat text blue while Lewis is thinking",
            "static chat-text-color strategy",
        ),
        (
            "make a code proposal to increase the maximum dashboard status indicator",
            "backend setting-limit strategy",
        ),
        (
            "make a code proposal to add a dynamic status badge to the navigation UI",
            "generic static-label strategy",
        ),
    ],
)
def test_strategy_collision_matrix_saves_no_wrong_approval(
    tmp_path: Path,
    prompt: str,
    expected_diagnostic: str,
) -> None:
    project_root = tmp_path / "project"
    _write_strategy_collision_project(project_root)
    before = _snapshot(project_root)
    agent = _make_agent(tmp_path, project_root)

    run = agent.run(prompt)

    assert ApprovalStore(agent.settings.database_path).list(status="pending") == []
    assert expected_diagnostic in run.answer
    assert "No pending code proposal was saved." in run.answer
    assert all(record["name"] == "search_project_files" for record in run.tool_calls)
    assert _snapshot(project_root) == before


def test_chat_text_blue_after_done_thinking_saves_no_approval_with_strategy_diagnostics(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _write_strategy_collision_project(project_root)
    before = _snapshot(project_root)
    agent = _make_agent(tmp_path, project_root)

    run = agent.run(
        "make a code proposal to make the chat text blue only after you are done thinking"
    )

    assert ApprovalStore(agent.settings.database_path).list(status="pending") == []
    assert "No pending code proposal was saved." in run.answer
    assert "Candidate files found:" in run.answer
    # Must NOT claim no files were found when candidates exist.
    assert not run.answer.startswith(
        "Project search did not find enough relevant existing file content"
    )
    # Must mention strategy incompatibility, not apply a static color strategy.
    assert "Lewis found relevant project files" in run.answer
    assert "static chat-text-color strategy" in run.answer
    assert "after-response/completion-ready state" in run.answer
    assert all(record["name"] == "search_project_files" for record in run.tool_calls)
    assert _snapshot(project_root) == before


def test_maximum_status_badge_count_sidebar_saves_no_approval_with_target_mismatch(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _write_strategy_collision_project(project_root)
    before = _snapshot(project_root)
    agent = _make_agent(tmp_path, project_root)

    run = agent.run(
        "make a code proposal to increase the maximum status badge count in the sidebar"
    )

    assert ApprovalStore(agent.settings.database_path).list(status="pending") == []
    assert "No pending code proposal was saved." in run.answer
    assert "Candidate files found:" in run.answer
    # Must NOT claim no files were found when candidates exist.
    assert not run.answer.startswith(
        "Project search did not find enough relevant existing file content"
    )
    # Must mention the strategy considered and target surface mismatch.
    assert "Lewis found relevant project files" in run.answer
    assert "backend setting-limit strategy" in run.answer
    assert "sidebar" in run.answer.lower() or "dashboard" in run.answer.lower()
    assert all(record["name"] == "search_project_files" for record in run.tool_calls)
    assert _snapshot(project_root) == before


def test_clear_all_approved_memories_destructive_request_gets_destructive_diagnostics(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _write_strategy_collision_project(project_root)
    before = _snapshot(project_root)
    agent = _make_agent(tmp_path, project_root)

    run = agent.run("make a code proposal to clear all approved memories")

    assert ApprovalStore(agent.settings.database_path).list(status="pending") == []
    assert "No pending code proposal was saved." in run.answer
    assert "destructive" in run.answer.lower() or "storage-sensitive" in run.answer.lower()
    assert _snapshot(project_root) == before


def test_teleportation_mode_ungroundable_returns_no_files_diagnostics(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _write_strategy_collision_project(project_root)
    before = _snapshot(project_root)
    agent = _make_agent(tmp_path, project_root)

    run = agent.run("make a code proposal to add teleportation mode")

    assert ApprovalStore(agent.settings.database_path).list(status="pending") == []
    assert "No pending code proposal was saved." in run.answer
    assert "Cleaned goal:" in run.answer
    assert _snapshot(project_root) == before


def test_exact_valid_thinking_animation_prompt_still_creates_one_approval(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _write_strategy_collision_project(project_root)
    before = _snapshot(project_root)
    agent = _make_agent(tmp_path, project_root)

    run = agent.run(
        "make a code proposal to make a new animation for when youre thinking"
    )

    approvals = ApprovalStore(agent.settings.database_path).list(status="pending")
    assert len(approvals) == 1
    approval = approvals[0]
    assert approval.payload["requested_goal"] == (
        "Make a new animation for when Lewis is thinking."
    )
    assert ".bubble.thinking" in approval.payload["patch_preview"]
    assert approval.payload["patch_preview"].startswith("diff --git ")
    assert run.tool_calls[-1]["name"] == "propose_code_change"
    _assert_real_inert_unified_diff(project_root, before, approval)
