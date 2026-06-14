"""Tests for deterministic exact-anchor snippet harvesting."""

from __future__ import annotations

from pathlib import Path

import app.dev.freestyle_anchor_snippets as anchor_snippets
from app.dev.freestyle_anchor_snippets import (
    MAX_SNIPPET_CHARS,
    MAX_SNIPPETS_PER_FILE,
    harvest_anchor_snippets,
)
from app.dev.freestyle_policy import (
    APPROVAL_STORE_CLASS_NAME,
    approval_sensitive_region_labels,
)

_TETRIS_GOAL = (
    "Wire the existing shenanigans_page into app/web_routes.py so GET "
    "/shenanigans returns the Shenanigans page using the same _page(...) "
    "wrapper pattern as /dashboard, /usage, and /chat."
)

# Mirrors the real app/web_routes.py shape: a multi-line web_pages import
# block, marker-free page routes, and approval routes in the same file.
# Assembled with the concatenated class name so this test file never
# contains the literal text.
_ROUTES_TEXT = f'''from __future__ import annotations

from app.approvals import {APPROVAL_STORE_CLASS_NAME}
from app.approval_service import approve_approval, reject_approval
from app.web_pages import (
    approvals_page,
    chat_page,
    dashboard_page,
    usage_page,
)
from app.web_rules import normalize_history_filter


def register_web_routes(app):
    @app.get("/dashboard")
    def dashboard(request):
        store = _approval_store(request)
        return _page("Dashboard", request, dashboard_page(store))

    @app.get("/usage")
    def usage(request):
        return _page("Usage", request, usage_page())

    @app.post("/approvals/{{approval_id}}/approve")
    def approve_item(
        request, approval_id
    ) -> str:
        return approve_approval(_approval_store(request), approval_id)

    @app.get("/chat")
    def chat(request):
        return _page("Chat", request, chat_page())
'''


def test_harvester_extracts_exact_import_block_and_route_snippets():
    result = harvest_anchor_snippets("app/web_routes.py", _ROUTES_TEXT, _TETRIS_GOAL)

    by_label = {snippet.label: snippet.exact_text for snippet in result.snippets}
    assert "import-block:app.web_pages" in by_label
    import_block = by_label["import-block:app.web_pages"]
    assert import_block.startswith("from app.web_pages import (")
    assert import_block.endswith(")")
    assert "usage_page," in import_block
    assert "route:/usage" in by_label
    assert "route:/chat" in by_label
    assert 'def usage(request):' in by_label["route:/usage"]
    for snippet in result.snippets:
        assert snippet.exact_text in _ROUTES_TEXT
        assert len(snippet.exact_text) <= MAX_SNIPPET_CHARS
    assert len(result.snippets) <= MAX_SNIPPETS_PER_FILE


def test_goal_mentioned_routes_rank_before_other_routes():
    result = harvest_anchor_snippets(
        "app/web_routes.py", _ROUTES_TEXT, "Change only the /chat endpoint."
    )

    route_labels = [
        snippet.label
        for snippet in result.snippets
        if snippet.label.startswith("route:")
    ]
    assert route_labels[0] == "route:/chat"


def test_snippets_never_contain_approval_sensitive_markers():
    result = harvest_anchor_snippets("app/web_routes.py", _ROUTES_TEXT, _TETRIS_GOAL)

    for snippet in result.snippets:
        assert approval_sensitive_region_labels(snippet.exact_text) == ()
    omitted_reasons = {
        record["label"]: record["reason"] for record in result.omitted
    }
    assert omitted_reasons["route:/dashboard"] == "approval_sensitive_region"
    assert (
        omitted_reasons["route:/approvals/{approval_id}/approve"]
        == "approval_sensitive_region"
    )


def test_multiline_signature_routes_capture_their_whole_block():
    marker_free = _ROUTES_TEXT.replace(
        "approve_approval(_approval_store(request), approval_id)",
        "record_item(request, approval_id)",
    )
    result = harvest_anchor_snippets(
        "app/web_routes.py", marker_free, "Adjust the approve endpoint."
    )

    by_label = {snippet.label: snippet.exact_text for snippet in result.snippets}
    block = by_label["route:/approvals/{approval_id}/approve"]
    assert block.endswith("return record_item(request, approval_id)")
    assert ") -> str:" in block
    assert block in marker_free


def test_test_files_harvest_goal_matching_test_functions_first():
    text = (
        "import pytest\n"
        "\n"
        "\n"
        "def test_dashboard_loads():\n"
        "    assert True\n"
        "\n"
        "\n"
        "@pytest.mark.parametrize('value', [1])\n"
        "def test_shenanigans_route(value):\n"
        "    assert value\n"
    )
    result = harvest_anchor_snippets(
        "tests/test_web.py", text, "Add a shenanigans route test."
    )

    assert result.snippets[0].label == "test:test_shenanigans_route"
    assert result.snippets[0].exact_text.startswith(
        "@pytest.mark.parametrize('value', [1])"
    )
    assert result.snippets[0].exact_text.endswith("    assert value")
    for snippet in result.snippets:
        assert snippet.exact_text in text


def test_unrecognized_files_fall_back_to_one_bounded_verbatim_tail():
    text = "\n".join(f".rule-{index} {{ color: red; }}" for index in range(200))
    result = harvest_anchor_snippets("app/static/lewis.css", text, _TETRIS_GOAL)

    assert [snippet.label for snippet in result.snippets] == ["file-tail"]
    tail = result.snippets[0].exact_text
    assert tail in text
    assert len(tail) <= MAX_SNIPPET_CHARS
    assert tail.endswith(".rule-199 { color: red; }")


def test_sensitive_content_and_oversized_blocks_are_bounded_or_dropped():
    text = (
        '@app.get("/tokens")\n'
        "def tokens(request):\n"
        "    secret = \"hunter2-hunter2\"\n"
        "    return secret\n"
    )
    result = harvest_anchor_snippets("app/web_routes.py", text, "Update /tokens.")
    assert all(
        "hunter2" not in snippet.exact_text for snippet in result.snippets
    )
    reasons = {record["reason"] for record in result.omitted}
    assert "sensitive_content" in reasons

    big_route = (
        '@app.get("/big")\n'
        "def big(request):\n"
        + "\n".join(f"    value_{index} = {index}" for index in range(200))
    )
    bounded = harvest_anchor_snippets("app/web_routes.py", big_route, "big")
    assert bounded.snippets
    for snippet in bounded.snippets:
        assert len(snippet.exact_text) <= MAX_SNIPPET_CHARS
        assert snippet.exact_text in big_route


def test_module_is_pure_with_no_io_model_or_runtime_dependencies():
    source = Path(anchor_snippets.__file__).read_text(encoding="utf-8")
    lowered = source.casefold()
    for forbidden in (
        "subprocess",
        "deepseek",
        "anthropic",
        "ollama",
        "sqlite",
        "load_dotenv",
        "os.environ",
        "os.getenv",
        "import os",
        "pathlib",
        "open(",
        "requests",
        "httpx",
    ):
        assert forbidden not in lowered, forbidden
    assert "Approval" "Store" not in source
    app_imports = [
        line
        for line in source.splitlines()
        if line.startswith(("from app", "import app"))
    ]
    assert app_imports == ["from app.dev.freestyle_policy import ("]
