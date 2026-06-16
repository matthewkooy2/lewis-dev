"""Tests for the persistent repo-map indexer and pure extractors.

Covers safe-file exclusion, approval-sensitive metadata-only handling, hash
incremental refresh, deleted-file invalidation, descriptor/symbol/route/CSS/JS/
test/chunk/relationship extraction, source-immutability, the approved-root
restriction, and module purity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.dev import repo_map, repo_map_extract
from app.dev.freestyle_coding import FreestyleJobRefusedError

JOB_ID = "a" * 32


def _write(root: Path, rel: str, content: str | bytes) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def _populate(root: Path) -> None:
    _write(
        root,
        "app/web_routes.py",
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n\n"
        'CALENDAR_TITLE = "Calendar"\n\n'
        '@app.get("/calendar")\n'
        "def calendar(request):\n"
        "    return render_calendar()\n",
    )
    _write(
        root,
        "app/web_pages.py",
        "def calendar_page():\n"
        '    return ("<link href=\\"/static/lewis.css\\">"\n'
        '            "<script src=\\"/static/calendar.js\\"></script>"\n'
        "            \"<div class='calendar-grid'>\"\n"
        "            \"<ul class='calendar-day'></ul></div>\")\n",
    )
    _write(
        root,
        "app/static/lewis.css",
        ".calendar-grid {\n"
        "  display: grid;\n"
        "  grid-template-columns: repeat(7, 1fr);\n"
        "}\n"
        ".calendar-day {\n"
        "  display: flex;\n"
        "  flex-direction: column;\n"
        "}\n"
        "#agenda { color: red; }\n",
    )
    _write(
        root,
        "app/static/calendar.js",
        'const grid = document.getElementById("agenda");\n'
        'document.querySelectorAll(".calendar-day").forEach((el) => {\n'
        '  el.classList.add("active");\n'
        "});\n"
        'function refresh() { grid.addEventListener("click", refresh); }\n',
    )
    _write(
        root,
        "tests/test_web_routes.py",
        "import app.web_routes\n\n"
        "def test_calendar_route(client):\n"
        '    assert client.get("/calendar").status_code == 200\n',
    )
    _write(
        root,
        "app/approvals.py",
        "class ApprovalStore:\n"
        "    def approve_approval(self, ident):\n"
        "        return ident\n",
    )
    # Excluded files.
    _write(root, ".env", "DEEPSEEK_API_KEY=sk-not-a-real-key-1234567890\n")
    _write(root, "data/lewis.sqlite3", b"SQLite format 3\x00binarystuff")
    _write(root, "assets/logo.png", b"\x89PNG\r\n\x00\x00binary")
    _write(root, "big.py", "x = 1\n" + ("# pad\n" * 120_000))
    _write(root, "__pycache__/cached.pyc", b"\x00\x01compiled")
    _write(
        root,
        "app/config_values.py",
        'password = "hunter2hunter2pass"\n',
    )


def _worktree_index(tmp_path: Path) -> tuple[repo_map.RepoIndex, Path, Path]:
    state = (tmp_path / ".lewis-dev").resolve()
    worktree = state / "worktrees" / JOB_ID
    worktree.mkdir(parents=True)
    _populate(worktree)
    index = repo_map.build_or_refresh_index(worktree, state=state)
    return index, state, worktree


# --------------------------------------------------------------------------- #
# Approved-root restriction.
# --------------------------------------------------------------------------- #
def test_arbitrary_root_is_refused(tmp_path: Path) -> None:
    state = (tmp_path / ".lewis-dev").resolve()
    state.mkdir(parents=True)
    rogue = tmp_path / "elsewhere"
    rogue.mkdir()
    with pytest.raises(FreestyleJobRefusedError):
        repo_map.build_or_refresh_index(rogue, state=state)


def test_worktree_root_must_be_valid_job_id(tmp_path: Path) -> None:
    state = (tmp_path / ".lewis-dev").resolve()
    bad = state / "worktrees" / "not-a-job-id"
    bad.mkdir(parents=True)
    with pytest.raises(FreestyleJobRefusedError):
        repo_map.build_or_refresh_index(bad, state=state)


def test_host_scope_resolves_against_root_dir(tmp_path: Path, monkeypatch) -> None:
    root = (tmp_path / "host").resolve()
    root.mkdir()
    _populate(root)
    monkeypatch.setattr(repo_map, "ROOT_DIR", root)
    state = (tmp_path / "host" / ".lewis-dev").resolve()
    index = repo_map.build_or_refresh_index(root, state=state)
    assert index.manifest["scope"] == "host"
    assert index.manifest["scope_id"] == repo_map.HOST_SCOPE_ID


# --------------------------------------------------------------------------- #
# Exclusion behaviour.
# --------------------------------------------------------------------------- #
def test_unsafe_files_are_excluded(tmp_path: Path) -> None:
    index, _, _ = _worktree_index(tmp_path)
    indexed = {f["path"] for f in index.files}
    assert ".env" not in indexed
    assert "data/lewis.sqlite3" not in indexed
    assert "assets/logo.png" not in indexed
    assert "big.py" not in indexed
    assert "app/config_values.py" not in indexed
    assert not any("__pycache__" in path for path in indexed)
    skipped = index.manifest["skipped_counts"]
    assert skipped.get("secret_file", 0) >= 1
    assert skipped.get("database_file", 0) >= 1
    assert skipped.get("oversized", 0) >= 1
    assert skipped.get("sensitive_content", 0) >= 1
    # A binary png is rejected as an unsupported (non-source) type.
    assert skipped.get("unsupported_file_type", 0) >= 1


def test_approval_files_are_metadata_only(tmp_path: Path) -> None:
    index, _, _ = _worktree_index(tmp_path)
    descriptor = index.descriptor_for("app/approvals.py")
    assert descriptor is not None
    assert descriptor["role"] == repo_map_extract.ROLE_APPROVAL_SENSITIVE
    assert descriptor["safety_classification"] == "approval_sensitive"
    assert descriptor["imports"] == []
    # Never parsed for symbols, chunks, selectors, or relationships.
    for collection in (
        index.symbols,
        index.chunks,
        index.css_selectors,
        index.js_assets,
        index.relationships,
    ):
        assert all(
            "approvals.py" not in str(record.get("path", ""))
            and "approvals.py" not in str(record.get("source", ""))
            for record in collection
        )
    assert index.manifest["approval_sensitive_file_count"] >= 1


# --------------------------------------------------------------------------- #
# Incremental refresh + deletion.
# --------------------------------------------------------------------------- #
def test_unchanged_files_are_reused(tmp_path: Path) -> None:
    index, state, worktree = _worktree_index(tmp_path)
    again = repo_map.build_or_refresh_index(worktree, state=state)
    assert again.manifest["reused_file_count"] == index.manifest["file_count"]
    assert again.manifest["changed_file_count"] == 0
    assert again.manifest["new_file_count"] == 0


def test_changed_file_is_reindexed(tmp_path: Path) -> None:
    index, state, worktree = _worktree_index(tmp_path)
    _write(
        worktree,
        "app/web_routes.py",
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n\n"
        '@app.get("/calendar")\n'
        "def calendar(request):\n"
        "    return render_calendar()\n\n"
        '@app.get("/tasks")\n'
        "def tasks(request):\n"
        "    return render_tasks()\n",
    )
    refreshed = repo_map.build_or_refresh_index(worktree, state=state)
    assert refreshed.manifest["changed_file_count"] == 1
    routes = {r["route"] for r in refreshed.routes if r["path"] == "app/web_routes.py"}
    assert {"/calendar", "/tasks"} <= routes


def test_deleted_file_is_invalidated(tmp_path: Path) -> None:
    index, state, worktree = _worktree_index(tmp_path)
    (worktree / "app" / "static" / "calendar.js").unlink()
    refreshed = repo_map.build_or_refresh_index(worktree, state=state)
    assert refreshed.manifest["deleted_file_count"] == 1
    assert "app/static/calendar.js" in refreshed.manifest["deleted_paths"]
    assert all(f["path"] != "app/static/calendar.js" for f in refreshed.files)
    assert all(
        c["path"] != "app/static/calendar.js" for c in refreshed.chunks
    )


def test_indexing_does_not_mutate_source(tmp_path: Path) -> None:
    state = (tmp_path / ".lewis-dev").resolve()
    worktree = state / "worktrees" / JOB_ID
    worktree.mkdir(parents=True)
    _populate(worktree)
    before = {
        path.relative_to(worktree).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in worktree.rglob("*")
        if path.is_file()
    }
    repo_map.build_or_refresh_index(worktree, state=state)
    after = {
        path.relative_to(worktree).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in worktree.rglob("*")
        if path.is_file()
    }
    assert before == after


# --------------------------------------------------------------------------- #
# Extraction correctness.
# --------------------------------------------------------------------------- #
def test_descriptors_and_roles(tmp_path: Path) -> None:
    index, _, _ = _worktree_index(tmp_path)
    by_path = {f["path"]: f for f in index.files}
    assert by_path["app/web_routes.py"]["role"] == repo_map_extract.ROLE_WEB_ROUTE
    assert by_path["app/web_pages.py"]["role"] == repo_map_extract.ROLE_WEB_PAGE
    assert by_path["app/static/lewis.css"]["role"] == repo_map_extract.ROLE_STATIC_CSS
    assert by_path["app/static/calendar.js"]["role"] == repo_map_extract.ROLE_STATIC_JS
    assert by_path["tests/test_web_routes.py"]["role"] == repo_map_extract.ROLE_TEST
    summary = by_path["app/static/lewis.css"]["responsibility_summary"]
    assert "stylesheet" in summary.lower()


def test_python_symbol_extraction(tmp_path: Path) -> None:
    index, _, _ = _worktree_index(tmp_path)
    symbols = [s for s in index.symbols if s["path"] == "app/web_routes.py"]
    names = {(s["name"], s["kind"]) for s in symbols}
    assert ("calendar", "route_handler") in names
    assert ("CALENDAR_TITLE", "constant") in names


def test_route_extraction(tmp_path: Path) -> None:
    index, _, _ = _worktree_index(tmp_path)
    routes = [r for r in index.routes if r["path"] == "app/web_routes.py"]
    assert len(routes) == 1
    assert routes[0]["route"] == "/calendar"
    assert routes[0]["method"] == "GET"
    assert routes[0]["handler"] == "calendar"


def test_css_selector_extraction_with_layout(tmp_path: Path) -> None:
    index, _, _ = _worktree_index(tmp_path)
    selectors = {
        s["selector"]: s
        for s in index.css_selectors
        if s["path"] == "app/static/lewis.css"
    }
    assert "display:grid" in selectors[".calendar-grid"]["layout"]
    assert "grid-template-columns" in selectors[".calendar-grid"]["layout"]
    assert "display:flex" in selectors[".calendar-day"]["layout"]
    assert "flex-direction" in selectors[".calendar-day"]["layout"]
    assert selectors["#agenda"]["ids"] == ["agenda"]


def test_js_asset_extraction(tmp_path: Path) -> None:
    index, _, _ = _worktree_index(tmp_path)
    js = next(j for j in index.js_assets if j["path"] == "app/static/calendar.js")
    assert "agenda" in js["lookup_ids"]
    assert "calendar-day" in js["lookup_classes"]
    assert "active" in js["managed_classes"]
    assert "click" in js["event_types"]


def test_test_map_extraction(tmp_path: Path) -> None:
    index, _, _ = _worktree_index(tmp_path)
    record = next(
        t for t in index.tests if t["path"] == "tests/test_web_routes.py"
    )
    assert "test_calendar_route" in record["test_functions"]
    assert "app.web_routes" in record["referenced_modules"]


def test_chunks_only_for_safe_files(tmp_path: Path) -> None:
    index, _, _ = _worktree_index(tmp_path)
    chunk_paths = {c["path"] for c in index.chunks}
    assert "app/web_routes.py" in chunk_paths
    assert "app/approvals.py" not in chunk_paths
    assert ".env" not in chunk_paths
    for chunk in index.chunks:
        assert chunk["start_line"] >= 1
        assert chunk["end_line"] >= chunk["start_line"]
        assert chunk["hash"]


def test_relationship_map_connects_surfaces(tmp_path: Path) -> None:
    index, _, _ = _worktree_index(tmp_path)
    kinds = {r["kind"] for r in index.relationships}
    assert "route_to_handler" in kinds
    assert "page_to_asset" in kinds
    # Page references the calendar stylesheet/script.
    targets = {
        r["target"]
        for r in index.relationships
        if r["kind"] == "page_to_asset"
    }
    assert any("lewis.css" in t for t in targets)
    assert any("calendar.js" in t for t in targets)


def test_persisted_artifacts_are_inspectable_jsonl(tmp_path: Path) -> None:
    index, state, _ = _worktree_index(tmp_path)
    scope_dir = repo_map.repo_index_root(state) / JOB_ID
    manifest = json.loads((scope_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == repo_map_extract.SCHEMA_VERSION
    assert manifest["root_id"]
    for name in ("files.jsonl", "routes.jsonl", "css_selectors.jsonl"):
        for line in (scope_dir / name).read_text().splitlines():
            json.loads(line)  # each line parses as one JSON object


def test_extract_module_is_pure() -> None:
    source = Path(repo_map_extract.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "import subprocess",
        "import socket",
        "import requests",
        "import httpx",
        "from app.",
        "import app",
        "open(",
        "deepseek",
        "testclient",
    ):
        assert forbidden not in source, forbidden
