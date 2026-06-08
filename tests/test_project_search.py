from __future__ import annotations

from pathlib import Path

from app.project_search import (
    MAX_EXCERPT_CHARS,
    MAX_EXCERPT_LINES,
    MAX_FILE_BYTES,
    ProjectFileSearch,
)


def test_project_file_search_lists_only_safe_text_files(tmp_path: Path) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "service.py").write_text(
        "def run():\n    return 'Project Alpha'\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("Project overview", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=escapephrase", encoding="utf-8")
    (tmp_path / ".env.example").write_text("SAFE=but_hidden", encoding="utf-8")
    (tmp_path / ".hidden.py").write_text("escapephrase", encoding="utf-8")
    (tmp_path / "credentials.txt").write_text("escapephrase", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "runtime.txt").write_text(
        "escapephrase",
        encoding="utf-8",
    )
    (tmp_path / "data" / "lewis.sqlite3").write_text(
        "escapephrase",
        encoding="utf-8",
    )
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("escapephrase", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("escapephrase", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "service.pyc").write_bytes(b"escapephrase")
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".pytest_cache" / "state").write_text(
        "escapephrase",
        encoding="utf-8",
    )
    (tmp_path / "image.bin").write_bytes(b"\x00escapephrase")
    (tmp_path / "large.txt").write_text(
        "a" * (MAX_FILE_BYTES + 1),
        encoding="utf-8",
    )

    search = ProjectFileSearch(tmp_path)

    assert search.list_files() == ["README.md", "app/service.py"]
    assert search.search("escapephrase") == []


def test_project_file_search_matches_paths_and_text(tmp_path: Path) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "search_helper.py").write_text(
        "first line\nneedle appears here\nneedle appears again\n",
        encoding="utf-8",
    )

    results = ProjectFileSearch(tmp_path).search("needle")

    assert results[0].path == "app/search_helper.py"
    assert results[0].line_number == 2
    assert results[0].snippet == "needle appears here"

    path_results = ProjectFileSearch(tmp_path).search("search_helper")
    assert path_results[0].path == "app/search_helper.py"
    assert path_results[0].line_number is None
    assert path_results[0].snippet == "path match"


def test_project_file_search_ranks_path_matches_before_broad_content_matches(
    tmp_path: Path,
) -> None:
    (tmp_path / "app" / "static").mkdir(parents=True)
    for index in range(12):
        (tmp_path / "app" / f"distractor_{index:02}.py").write_text(
            "command center chat message rendering\n",
            encoding="utf-8",
        )
    (tmp_path / "app" / "static" / "chat.js").write_text(
        "function render() {}\n",
        encoding="utf-8",
    )
    (tmp_path / "app" / "static" / "lewis.css").write_text(
        ".message-label {}\n",
        encoding="utf-8",
    )
    (tmp_path / "app" / "web_components.py").write_text(
        "def render():\n    return ''\n",
        encoding="utf-8",
    )

    results = ProjectFileSearch(tmp_path).search(
        "web_components lewis.css chat.js chat message rendering"
    )

    assert [item.path for item in results[:3]] == [
        "app/static/chat.js",
        "app/static/lewis.css",
        "app/web_components.py",
    ]
    assert all(item.line_number is None for item in results[:3])


def test_project_file_search_rejects_unsafe_relative_paths(tmp_path: Path) -> None:
    search = ProjectFileSearch(tmp_path)

    assert search.is_safe_relative_path("app/main.py") is True
    assert search.is_safe_relative_path("../outside.py") is False
    assert search.is_safe_relative_path("app/../outside.py") is False
    assert search.is_safe_relative_path(str(tmp_path / "app" / "main.py")) is False
    assert search.is_safe_relative_path(".env") is False
    assert search.is_safe_relative_path(".git/config") is False
    assert search.is_safe_relative_path(".venv/Lib/site.py") is False
    assert search.is_safe_relative_path("__pycache__/main.pyc") is False
    assert search.is_safe_relative_path(".pytest_cache/state") is False
    assert search.is_safe_relative_path("data/lewis.sqlite3") is False
    assert search.is_safe_relative_path("credentials/api_token.txt") is False


def test_project_file_search_reads_bounded_excerpt_from_safe_existing_file(
    tmp_path: Path,
) -> None:
    (tmp_path / "app").mkdir()
    target = tmp_path / "app" / "web_components.py"
    target.write_text(
        "\n".join(
            ["header"] * 100
            + ["def chat():", "    return 'chat ui'"]
            + ["footer"] * 100
        ),
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("SECRET=value", encoding="utf-8")

    search = ProjectFileSearch(tmp_path)
    excerpt = search.read_excerpt(
        "app/web_components.py",
        focus_terms=("def chat",),
        max_lines=1000,
        max_chars=100_000,
    )

    assert search.is_safe_file_path("app/web_components.py") is True
    assert excerpt is not None
    assert excerpt.path == "app/web_components.py"
    assert excerpt.start_line == 81
    assert "def chat():" in excerpt.text
    assert len(excerpt.text.splitlines()) <= MAX_EXCERPT_LINES
    assert len(excerpt.text) <= MAX_EXCERPT_CHARS
    assert search.is_safe_file_path("app/missing.py") is False
    assert search.read_excerpt("app/missing.py") is None
    assert search.is_safe_file_path(".env") is False
    assert search.read_excerpt(".env") is None


def test_project_file_search_blocks_unsafe_excerpts(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.py"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "site.py").write_text("secret", encoding="utf-8")
    (tmp_path / "runtime.sqlite3").write_text("secret", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "runtime.txt").write_text("secret", encoding="utf-8")
    (tmp_path / ".hidden.py").write_text("secret", encoding="utf-8")

    search = ProjectFileSearch(tmp_path)

    blocked = [
        ".env",
        ".git/config",
        ".venv/site.py",
        "runtime.sqlite3",
        "data/runtime.txt",
        ".hidden.py",
        "../outside.py",
        "app/../outside.py",
        str(outside),
    ]
    assert all(search.read_excerpt(path) is None for path in blocked)
