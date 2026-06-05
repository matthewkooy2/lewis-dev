from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.approvals import ApprovalStore
from app.executors import build_executor_registry
from app.project_inspection import (
    MAX_PROJECT_FILE_BYTES,
    MAX_RETURNED_CHARS,
    ProjectInspector,
)
from app.rag import DocumentSearch
from app.tools import ToolRegistry


def _make_project(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "app").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "README.md").write_text("Lewis repo docs", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'lewis'\n", encoding="utf-8"
    )
    (root / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    (root / "app" / "__init__.py").write_text("", encoding="utf-8")
    (root / "app" / "tools.py").write_text(
        "def tool():\n    return 'developer workspace'\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_tools.py").write_text(
        "def test_tool():\n    assert True\n",
        encoding="utf-8",
    )
    return root


def _developer_tools(tmp_path: Path, root: Path) -> ToolRegistry:
    return ToolRegistry(
        document_search=DocumentSearch(tmp_path / "documents"),
        approval_store=ApprovalStore(tmp_path / "test.sqlite3"),
        developer_code_workspace=True,
        project_inspector=ProjectInspector(root),
    )


def test_project_inspection_tools_are_gated_from_normal_mode(tmp_path):
    root = _make_project(tmp_path)
    normal_tools = ToolRegistry(
        DocumentSearch(tmp_path / "documents"),
        ApprovalStore(tmp_path / "test.sqlite3"),
        project_inspector=ProjectInspector(root),
    )

    names = {schema["function"]["name"] for schema in normal_tools.schemas()}

    assert "list_project_files" not in names
    assert "read_project_file" not in names
    assert "search_project" not in names
    assert normal_tools.execute("read_project_file", {"path": "app/tools.py"}) == {
        "ok": False,
        "error": "Unknown tool: read_project_file",
    }


def test_project_inspection_tools_are_available_in_developer_mode(tmp_path):
    root = _make_project(tmp_path)
    tools = _developer_tools(tmp_path, root)

    names = {schema["function"]["name"] for schema in tools.schemas()}

    assert "list_project_files" in names
    assert "read_project_file" in names
    assert "search_project" in names


def test_list_project_files_returns_only_safe_project_files(tmp_path):
    root = _make_project(tmp_path)
    (root / ".env").write_text("DEEPSEEK_API_KEY=secret", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("private", encoding="utf-8")
    (root / ".venv").mkdir()
    (root / ".venv" / "site.py").write_text("private", encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "lewis.sqlite3").write_bytes(b"sqlite")
    (root / "app" / "__pycache__").mkdir()
    (root / "app" / "__pycache__" / "tools.pyc").write_bytes(b"\x00binary")
    (root / "app" / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    inspector = ProjectInspector(root)

    files = inspector.list_files()

    assert "app/tools.py" in files
    assert "tests/test_tools.py" in files
    assert "README.md" in files
    assert ".env" not in files
    assert ".git/config" not in files
    assert ".venv/site.py" not in files
    assert "data/lewis.sqlite3" not in files
    assert "app/__pycache__/tools.pyc" not in files
    assert "app/image.png" not in files


def test_read_project_file_returns_bounded_redacted_text(tmp_path):
    root = _make_project(tmp_path)
    (root / "app" / "config.py").write_text(
        "API token is sk-testtesttesttesttesttesttest\nvisible line\n",
        encoding="utf-8",
    )
    inspector = ProjectInspector(root)

    result = inspector.read_file("app/config.py")

    assert result.path == "app/config.py"
    assert "visible line" in result.content
    assert "sk-testtesttesttesttesttesttest" not in result.content
    assert "[redacted]" in result.content
    assert result.truncated is False


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("/tmp/outside.py", "relative paths"),
        ("../outside.py", "dot segments"),
        (".env", "blocked file name"),
        (".env.example", "blocked file name"),
        (".git/config", "blocked runtime path"),
        (".venv/site.py", "blocked runtime path"),
        ("app/.hidden.py", "blocked file name"),
        ("data/lewis.sqlite3", "blocked file name"),
        ("app/__pycache__/tools.pyc", "blocked runtime path"),
        ("data/documents/note.md", "runtime files"),
        ("app/image.png", "text files"),
        ("secrets.py", "sensitive path name"),
    ],
)
def test_read_project_file_blocks_unsafe_paths(
    tmp_path: Path, path: str, message: str
) -> None:
    root = _make_project(tmp_path)
    (root / "app" / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    inspector = ProjectInspector(root)

    with pytest.raises(ValueError, match=message):
        inspector.read_file(path)


def test_read_project_file_blocks_binary_files(tmp_path):
    root = _make_project(tmp_path)
    (root / "app" / "binary.py").write_bytes(b"print('x')\x00")
    inspector = ProjectInspector(root)

    with pytest.raises(ValueError, match="binary"):
        inspector.read_file("app/binary.py")


def test_read_project_file_blocks_oversized_files(tmp_path):
    root = _make_project(tmp_path)
    (root / "app" / "large.py").write_text(
        "x" * (MAX_PROJECT_FILE_BYTES + 1),
        encoding="utf-8",
    )
    inspector = ProjectInspector(root)

    with pytest.raises(ValueError, match="too large"):
        inspector.read_file("app/large.py")


def test_read_project_file_truncates_returned_content(tmp_path, monkeypatch):
    root = _make_project(tmp_path)
    (root / "app" / "long.py").write_text("x" * 200, encoding="utf-8")
    monkeypatch.setattr("app.project_inspection.MAX_RETURNED_CHARS", 50)
    inspector = ProjectInspector(root)

    result = inspector.read_file("app/long.py")

    assert len(result.content) == 50
    assert result.truncated is True
    assert MAX_RETURNED_CHARS > 50


def test_search_project_is_bounded_read_only_and_separate_from_documents(tmp_path):
    root = _make_project(tmp_path)
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "note.md").write_text("developer workspace", encoding="utf-8")
    store = ApprovalStore(tmp_path / "test.sqlite3")
    tools = ToolRegistry(
        DocumentSearch(docs),
        store,
        developer_code_workspace=True,
        project_inspector=ProjectInspector(root),
    )
    before = (root / "app" / "tools.py").read_text(encoding="utf-8")

    result = tools.execute("search_project", {"query": "developer workspace"})

    assert result["ok"] is True
    assert result["results"][0]["path"] == "app/tools.py"
    assert "note.md" not in {item["path"] for item in result["results"]}
    assert (root / "app" / "tools.py").read_text(encoding="utf-8") == before
    assert store.list() == []


def test_project_inspection_refuses_unexpected_tool_arguments(tmp_path):
    root = _make_project(tmp_path)
    tools = _developer_tools(tmp_path, root)

    assert tools.execute("list_project_files", {"path": "app/tools.py"})["ok"] is False
    assert (
        tools.execute(
            "read_project_file",
            {"path": "app/tools.py", "write": "new content"},
        )["ok"]
        is False
    )
    assert (
        tools.execute("search_project", {"query": "tool", "path": "app"})["ok"] is False
    )


def test_project_inspection_blocks_symlink_escape(tmp_path):
    if os.name == "nt":
        pytest.skip("Symlink creation is not consistently available on Windows.")
    root = _make_project(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("escape", encoding="utf-8")
    try:
        (root / "app" / "outside_link.py").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Symlink creation unavailable: {exc}")
    inspector = ProjectInspector(root)

    with pytest.raises(ValueError, match="project root"):
        inspector.read_file("app/outside_link.py")
    assert "app/outside_link.py" not in inspector.list_files()


def test_project_inspection_adds_no_code_proposal_executor():
    assert "propose_code_change" not in build_executor_registry()
