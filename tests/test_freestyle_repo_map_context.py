"""Tests for repo-map Freestyle integration: index resolution, connected-file
selection, context-plan ranking, refinement memory, and review warnings.
"""

from __future__ import annotations

from pathlib import Path

from app.dev import repo_map, repo_map_context as ctx
from app.dev.freestyle_context_plan import plan_context, priority_rank
from app.dev.freestyle_proposal import (
    _ContextCandidate,
    _context_planning_goal,
    build_proposal_context,
)

JOB_ID = "b" * 32


def _write(root: Path, rel: str, content: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _populate(root: Path) -> None:
    _write(
        root,
        "app/web_routes.py",
        "from fastapi import FastAPI\n\napp = FastAPI()\n\n"
        '@app.get("/calendar")\n'
        "def calendar(request):\n    return render_calendar()\n",
    )
    _write(
        root,
        "app/web_pages.py",
        "def chat_page():\n"
        '    return "<script src=\\"/static/chat.js\\"></script><div id=\\"chat\\"></div>"\n\n'
        "def calendar_page():\n"
        '    return ("<link href=\\"/static/lewis.css\\">"\n'
        '            "<script src=\\"/static/calendar.js\\"></script>"\n'
        "            \"<div class='calendar-grid'></div>\")\n",
    )
    _write(
        root,
        "app/static/lewis.css",
        ".calendar-grid { display: grid; grid-template-columns: repeat(7,1fr); }\n"
        ".calendar-day { display: flex; }\n",
    )
    _write(
        root,
        "app/static/calendar.js",
        'document.querySelectorAll(".calendar-day").forEach((e) => '
        'e.classList.add("calendar-grid"));\n',
    )
    _write(
        root,
        "app/static/chat.js",
        "document.getElementById('chat')?.classList.add('chat-ready');\n",
    )
    _write(
        root,
        "app/code_proposal_engine.py",
        "def create_code_proposal_if_supported():\n    return None\n"
        + "# process decoy\n" * 1_000,
    )
    _write(
        root,
        "tests/test_calendar_page.py",
        "from app.web_pages import calendar_page\n\n"
        "def test_calendar_page_renders_grid():\n"
        "    assert 'calendar-grid' in calendar_page()\n",
    )
    _write(
        root,
        "tests/test_web.py",
        "def test_web_smoke(client):\n"
        "    assert client.get('/calendar').status_code in {200, 404}\n",
    )
    _write(
        root,
        "tests/test_web_routes.py",
        "import app.web_routes\n\ndef test_calendar(client):\n    pass\n",
    )
    _write(root, "app/unrelated_thing.py", "VALUE = 1\n")


def _worktree(tmp_path: Path) -> tuple[repo_map.RepoIndex, Path, Path]:
    state = (tmp_path / ".lewis-dev").resolve()
    worktree = state / "worktrees" / JOB_ID
    worktree.mkdir(parents=True)
    _populate(worktree)
    index = repo_map.build_or_refresh_index(worktree, state=state)
    return index, state, worktree


# --------------------------------------------------------------------------- #
# Index resolution (worktree-preferred, stale host fallback).
# --------------------------------------------------------------------------- #
def test_active_worktree_index_is_used(tmp_path: Path) -> None:
    _, state, worktree = _worktree(tmp_path)
    index, provenance = ctx.resolve_freestyle_index(worktree, worktree, state)
    assert index is not None
    assert provenance["scope"] == "worktree"
    assert provenance["job_id"] == JOB_ID
    assert provenance["stale"] is False
    assert provenance["fallback"] is False


def test_host_fallback_is_marked_stale(tmp_path: Path, monkeypatch) -> None:
    host_root = (tmp_path / "host").resolve()
    host_root.mkdir()
    _populate(host_root)
    monkeypatch.setattr(repo_map, "ROOT_DIR", host_root)
    state = (host_root / ".lewis-dev").resolve()
    repo_map.build_or_refresh_index(host_root, state=state)

    # A worktree that does not exist forces the stale host fallback.
    missing_worktree = state / "worktrees" / JOB_ID
    index, provenance = ctx.resolve_freestyle_index(missing_worktree, host_root, state)
    assert index is not None
    assert provenance["scope"] == "host"
    assert provenance["stale"] is True
    assert provenance["fallback"] is True
    summary = ctx.index_provenance_summary(provenance)
    assert "stale host fallback" in summary


def test_no_index_provenance_when_nothing_available(tmp_path: Path) -> None:
    state = (tmp_path / ".lewis-dev").resolve()
    state.mkdir()
    missing = state / "worktrees" / JOB_ID
    index, provenance = ctx.resolve_freestyle_index(missing, tmp_path, state)
    assert index is None
    assert provenance["available"] is False
    assert provenance["fallback"] is True


# --------------------------------------------------------------------------- #
# Connected-file selection.
# --------------------------------------------------------------------------- #
def test_calendar_goal_selects_connected_surface(tmp_path: Path) -> None:
    index, _, _ = _worktree(tmp_path)
    selection = ctx.select_connected_files(
        index,
        goal="Fix the broken /calendar page layout",
        named_targets=("app/static/lewis.css",),
        seed_paths=("app/web_routes.py",),
    )
    connected = set(selection["connected_paths"])
    assert "app/static/lewis.css" in connected
    assert "app/web_routes.py" in connected
    assert "app/web_pages.py" in connected
    assert "app/static/calendar.js" in connected
    assert "tests/test_calendar_page.py" in connected
    assert "app/static/chat.js" not in connected
    assert "app/code_proposal_engine.py" not in connected
    assert "app/unrelated_thing.py" not in connected
    assert (
        "named target from goal"
        in (selection["selection_reasons"]["app/static/lewis.css"])
    )


def test_connected_files_for_paths(tmp_path: Path) -> None:
    index, _, _ = _worktree(tmp_path)
    connected = ctx.connected_files_for_paths(index, ["app/static/lewis.css"])
    assert "app/static/lewis.css" in connected
    related = " ".join(connected["app/static/lewis.css"])
    assert "calendar.js" in related


def test_calendar_goal_with_process_language_stays_calendar_focused(
    tmp_path: Path,
) -> None:
    index, _, worktree = _worktree(tmp_path)
    for number in range(20):
        _write(worktree, f"app/feature_decoy_{number:02d}.py", "VALUE = 1\n")
    planning_goal = (
        "Repair the Calendar UI wiring and /calendar layout regression. The "
        "previous Freestyle proposal failure artifact mentioned repo map "
        "context, review warnings, previous attempt memory, validation, and "
        "DeepSeek, but those are process details. Add or update tests."
    )
    plan = plan_context(planning_goal, title="Calendar refinement")

    context = build_proposal_context(
        worktree,
        planning_goal,
        plan=plan,
        repo_map=index,
        repo_map_provenance={"available": True, "scope": "worktree", "stale": False},
    )

    included = list(context.inspected_paths)
    assert "freestyle_internal" not in context.context_plan["detected_goal_categories"]
    assert (
        "app/code_proposal_engine.py"
        not in context.context_plan["named_target_files"]["named_target_paths"]
    )
    for path in (
        "app/web_pages.py",
        "app/static/lewis.css",
        "app/static/calendar.js",
        "tests/test_calendar_page.py",
    ):
        assert path in included
    assert "app/code_proposal_engine.py" not in included
    repo_section = context.context_plan["repo_map"]
    assert repo_section["provenance"]["scope"] == "worktree"
    assert "app/static/calendar.js" in repo_section["connected_files"]
    assert "tests/test_calendar_page.py" in repo_section["connected_files"]
    assert "app/static/chat.js" not in repo_section["connected_files"]
    assert repo_section["selection_reasons"]["app/static/calendar.js"]


def test_calendar_priority_ranks_specific_assets_and_tests_first() -> None:
    plan = plan_context(
        "Fix the Calendar UI and /calendar page layout. Add regression tests.",
        title="Calendar refinement",
    )

    assert priority_rank("app/static/calendar.js", plan) < priority_rank(
        "app/static/chat.js", plan
    )
    assert priority_rank("tests/test_calendar_page.py", plan) < priority_rank(
        "tests/test_web.py", plan
    )


def test_refinement_planning_goal_ignores_previous_attempt_memory_paths() -> None:
    effective_goal = (
        "Previous attempt memory:\n"
        "- The previous proposal expected to edit: app/code_proposal_engine.py.\n\n"
        "New refinement goal: Repair the Calendar UI wiring."
    )
    metadata = {
        "refinement_goal": "Repair the Calendar UI wiring.",
        "observed_ui_feedback": "The /calendar page renders as a vertical list.",
        "previous_attempt_memory": {
            "previous_proposal_likely_files": ["app/code_proposal_engine.py"]
        },
    }

    planning_goal = _context_planning_goal(effective_goal, metadata)
    plan = plan_context(planning_goal, title="Calendar refinement")

    assert "app/code_proposal_engine.py" not in plan.named_target_paths
    assert "code_proposal_engine.py" not in plan.named_target_basenames


# --------------------------------------------------------------------------- #
# Named / developer-selected context outranks repo-map suggestions.
# --------------------------------------------------------------------------- #
def test_named_target_outranks_repo_map_connection() -> None:
    named = _ContextCandidate(
        path=Path("a"),
        relative_path="app/named.py",
        score=0,
        preferred=False,
        priority_rank=None,
        named_target=True,
        repo_map_connected=False,
    )
    connected = _ContextCandidate(
        path=Path("b"),
        relative_path="app/connected.py",
        score=99_999,
        preferred=False,
        priority_rank=None,
        named_target=False,
        repo_map_connected=True,
    )
    assert named.selection_key < connected.selection_key


def test_preferred_outranks_repo_map_connection() -> None:
    preferred = _ContextCandidate(
        path=Path("a"),
        relative_path="app/preferred.py",
        score=0,
        preferred=True,
        priority_rank=None,
        repo_map_connected=False,
    )
    connected = _ContextCandidate(
        path=Path("b"),
        relative_path="app/connected.py",
        score=99_999,
        preferred=False,
        priority_rank=None,
        repo_map_connected=True,
    )
    assert preferred.selection_key < connected.selection_key


def test_repo_map_connection_beats_plain_score() -> None:
    connected = _ContextCandidate(
        path=Path("a"),
        relative_path="app/connected.py",
        score=0,
        preferred=False,
        priority_rank=None,
        repo_map_connected=True,
    )
    plain = _ContextCandidate(
        path=Path("b"),
        relative_path="app/plain.py",
        score=10_000,
        preferred=False,
        priority_rank=None,
        repo_map_connected=False,
    )
    assert connected.selection_key < plain.selection_key


# --------------------------------------------------------------------------- #
# build_proposal_context records the repo-map selection.
# --------------------------------------------------------------------------- #
def test_build_proposal_context_uses_repo_map(tmp_path: Path) -> None:
    index, _, worktree = _worktree(tmp_path)
    context = build_proposal_context(
        worktree,
        "Fix the broken /calendar page layout",
        preferred_paths=("app/web_routes.py",),
        repo_map=index,
        repo_map_provenance={"available": True, "scope": "worktree", "stale": False},
    )
    section = context.context_plan["repo_map"]
    assert section["provenance"]["scope"] == "worktree"
    assert "app/static/lewis.css" in section["connected_files"]
    included = set(section["included_connected_files"])
    assert {"app/static/lewis.css", "app/static/calendar.js"} <= included
    included_paths = {f["path"] for f in context.files}
    assert "app/static/calendar.js" in included_paths


def test_build_proposal_context_without_repo_map_is_empty_section(
    tmp_path: Path,
) -> None:
    _, _, worktree = _worktree(tmp_path)
    context = build_proposal_context(worktree, "Fix the calendar layout", repo_map=None)
    section = context.context_plan["repo_map"]
    assert section["connected_files"] == []
    assert section["provenance"]["available"] is False


# --------------------------------------------------------------------------- #
# Advisory review warnings.
# --------------------------------------------------------------------------- #
def test_review_warns_about_unconsidered_surfaces(tmp_path: Path) -> None:
    index, _, _ = _worktree(tmp_path)
    warnings = ctx.repo_map_review_warnings(index, ["app/web_pages.py"])
    joined = " ".join(warnings)
    # The page changed but its linked stylesheet/script were not touched.
    assert "lewis.css" in joined or "calendar.js" in joined


def test_review_warns_tests_not_considered(tmp_path: Path) -> None:
    index, _, _ = _worktree(tmp_path)
    warnings = ctx.repo_map_review_warnings(index, ["app/web_routes.py"])
    assert any("test_web_routes.py" in w for w in warnings)


# --------------------------------------------------------------------------- #
# Surfacing: refinement memory + handoff markdown.
# --------------------------------------------------------------------------- #
def test_refinement_memory_renders_repo_map_context() -> None:
    from app.dev.freestyle_refinement import _memory_prompt_section

    rendered = _memory_prompt_section(
        {
            "previous_repo_map_review_warnings": ["Repo-map warning: missing asset"],
            "repo_map_connected_surfaces": {
                "app/static/lewis.css": ["app/static/calendar.js (shared selector)"]
            },
        }
    )
    assert "Repo map context used" in rendered
    assert "app/static/calendar.js" in rendered
    assert "The repo map flagged" in rendered
