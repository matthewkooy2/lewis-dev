"""Tests for LEWIS_DEVELOPER_WORKSPACE_ROOT: developer code tools (CLI trace,
chat trace_symbol, project inspection/search) must consistently operate on
the configured external workspace, never silently on the Lewis repo.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import (
    DeveloperWorkspaceRootError,
    Settings,
    get_settings,
    resolve_developer_workspace_root,
)
from app.llm import build_system_prompt
from app.main import build_agent, chat
from app.main import main as cli_main


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _external_project(tmp_path: Path) -> Path:
    root = tmp_path / "external_repo"
    _write(
        root,
        "src/util.py",
        "def greet(name):\n    return f'hello {name}'\n",
    )
    _write(root, "notes.md", "external project notes")
    return root


def _base_settings(**overrides) -> Settings:
    fields = dict(
        deepseek_api_key="test-key",
        deepseek_model="deepseek-v4-flash",
        deepseek_base_url="https://api.deepseek.com",
        debug=False,
        user_name="the user",
        user_name_configured=False,
        database_path=Path("unused.sqlite3"),
        documents_dir=Path("unused-documents"),
        max_tool_calls_per_request=8,
        max_estimated_daily_cost=1.0,
        price_per_1k_input_tokens=0.0,
        price_per_1k_output_tokens=0.0,
    )
    fields.update(overrides)
    return Settings(**fields)


def _clear_workspace_env(monkeypatch) -> None:
    monkeypatch.delenv("LEWIS_DEVELOPER_WORKSPACE_ROOT", raising=False)
    monkeypatch.setenv("LEWIS_DEVELOPER_CODE_WORKSPACE", "false")


# ---- resolve_developer_workspace_root --------------------------------------


def test_resolve_developer_workspace_root_defaults_when_unset(tmp_path):
    settings = _base_settings(developer_workspace_root=None)
    fallback = tmp_path / "lewis"
    fallback.mkdir()

    assert resolve_developer_workspace_root(settings, default=fallback) == fallback


def test_resolve_developer_workspace_root_uses_configured_external_path(tmp_path):
    external = _external_project(tmp_path)
    settings = _base_settings(developer_workspace_root=external)
    fallback = tmp_path / "lewis"
    fallback.mkdir()

    resolved = resolve_developer_workspace_root(settings, default=fallback)

    assert resolved == external.resolve()
    assert resolved != fallback


def test_resolve_developer_workspace_root_rejects_missing_path(tmp_path):
    missing = tmp_path / "does-not-exist"
    settings = _base_settings(developer_workspace_root=missing)

    with pytest.raises(DeveloperWorkspaceRootError, match=str(missing)):
        resolve_developer_workspace_root(settings, default=tmp_path)


def test_resolve_developer_workspace_root_rejects_file_path(tmp_path):
    a_file = tmp_path / "not_a_dir.txt"
    a_file.write_text("x", encoding="utf-8")
    settings = _base_settings(developer_workspace_root=a_file)

    with pytest.raises(DeveloperWorkspaceRootError, match="directory"):
        resolve_developer_workspace_root(settings, default=tmp_path)


def test_get_settings_parses_workspace_root_env(monkeypatch, tmp_path):
    _clear_workspace_env(monkeypatch)
    external = _external_project(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("LEWIS_DEVELOPER_WORKSPACE_ROOT", str(external))

    settings = get_settings()

    assert settings.developer_workspace_root == external


# ---- build_agent: chat tools (trace, inspection, search) -------------------


def test_build_agent_wires_external_workspace_into_trace_and_inspection(
    monkeypatch, tmp_path
):
    _clear_workspace_env(monkeypatch)
    external = _external_project(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("LEWIS_DEVELOPER_CODE_WORKSPACE", "true")
    monkeypatch.setenv("LEWIS_DEVELOPER_WORKSPACE_ROOT", str(external))
    monkeypatch.setenv("LEWIS_DB_PATH", str(tmp_path / "test.sqlite3"))

    agent = build_agent()

    trace_result = agent.tools.execute("trace_symbol", {"symbol": "greet"})
    assert trace_result["ok"] is True
    assert "src/util.py" in trace_result["report"]

    files = agent.tools.execute("list_project_files", {})
    assert files["ok"] is True
    assert "src/util.py" in files["files"]
    assert "notes.md" in files["files"]

    search = agent.tools.execute("search_project", {"query": "external project"})
    assert search["ok"] is True
    assert any(item["path"] == "notes.md" for item in search["results"])


def test_build_agent_rejects_missing_external_workspace(monkeypatch, tmp_path):
    _clear_workspace_env(monkeypatch)
    missing = tmp_path / "not-there"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("LEWIS_DEVELOPER_CODE_WORKSPACE", "true")
    monkeypatch.setenv("LEWIS_DEVELOPER_WORKSPACE_ROOT", str(missing))
    monkeypatch.setenv("LEWIS_DB_PATH", str(tmp_path / "test.sqlite3"))

    with pytest.raises(DeveloperWorkspaceRootError, match=str(missing)):
        build_agent()


def test_build_agent_disabled_workspace_ignores_bad_root_and_hides_tools(
    monkeypatch, tmp_path
):
    _clear_workspace_env(monkeypatch)
    missing = tmp_path / "not-there"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("LEWIS_DEVELOPER_CODE_WORKSPACE", "false")
    monkeypatch.setenv("LEWIS_DEVELOPER_WORKSPACE_ROOT", str(missing))
    monkeypatch.setenv("LEWIS_DB_PATH", str(tmp_path / "test.sqlite3"))

    agent = build_agent()

    names = {schema["function"]["name"] for schema in agent.tools.schemas()}
    assert "trace_symbol" not in names
    assert "list_project_files" not in names
    assert "read_project_file" not in names
    assert "search_project" not in names


def test_system_prompt_shows_configured_workspace_label(tmp_path):
    settings = _base_settings(developer_code_workspace=True)
    external = tmp_path / "external_repo"

    prompt = build_system_prompt(settings, external)

    assert str(external) in prompt
    assert "trace_symbol" in prompt


def test_system_prompt_defaults_to_lewis_label_without_workspace_root():
    settings = _base_settings(developer_code_workspace=True)

    prompt = build_system_prompt(settings, None)

    assert "Lewis" in prompt


# ---- CLI --------------------------------------------------------------------


def test_cli_trace_uses_configured_external_workspace(monkeypatch, tmp_path, capsys):
    _clear_workspace_env(monkeypatch)
    external = _external_project(tmp_path)
    monkeypatch.setenv("LEWIS_DEVELOPER_WORKSPACE_ROOT", str(external))

    exit_code = cli_main(["trace", "greet"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "QUICK TRACE: greet" in output
    assert "src/util.py" in output


def test_cli_trace_reports_clear_error_for_invalid_external_workspace(
    monkeypatch, tmp_path, capsys
):
    _clear_workspace_env(monkeypatch)
    missing = tmp_path / "not-there"
    monkeypatch.setenv("LEWIS_DEVELOPER_WORKSPACE_ROOT", str(missing))

    exit_code = cli_main(["trace", "anything"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "TRACE ERROR" in output
    assert str(missing) in output


def test_cli_chat_reports_clear_error_for_invalid_external_workspace(
    monkeypatch, tmp_path, capsys
):
    _clear_workspace_env(monkeypatch)
    missing = tmp_path / "not-there"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("LEWIS_DEVELOPER_CODE_WORKSPACE", "true")
    monkeypatch.setenv("LEWIS_DEVELOPER_WORKSPACE_ROOT", str(missing))
    monkeypatch.setenv("LEWIS_DB_PATH", str(tmp_path / "test.sqlite3"))

    exit_code = chat()
    output = capsys.readouterr().out

    assert exit_code == 1
    assert str(missing) in output


def test_cli_trace_json_reports_clear_error_for_invalid_external_workspace(
    monkeypatch, tmp_path, capsys
):
    _clear_workspace_env(monkeypatch)
    missing = tmp_path / "not-there"
    monkeypatch.setenv("LEWIS_DEVELOPER_WORKSPACE_ROOT", str(missing))

    exit_code = cli_main(["trace", "anything", "--json"])
    output = capsys.readouterr().out

    assert exit_code == 1
    payload = __import__("json").loads(output)
    assert payload["ok"] is False
    assert str(missing) in payload["error"]
