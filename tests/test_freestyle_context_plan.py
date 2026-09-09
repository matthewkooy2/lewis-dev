"""Tests for the deterministic goal-aware freestyle context planner."""

from __future__ import annotations

from pathlib import Path

import pytest

import app.dev.freestyle_context_plan as context_plan
from app.dev.freestyle_context_plan import (
    GOAL_FIT_WARNING_CREATE_ONLY,
    GOAL_FIT_WARNING_NO_WEB_SURFACE,
    GOAL_FIT_WARNING_UNWIRED_ENTRY_POINT,
    PLAN_WARNING_NO_UI_FILES,
    UI_CONTEXT_GAP_MESSAGE,
    goal_fit_warnings,
    is_named_target,
    is_test_pattern,
    plan_context,
    priority_rank,
    summarize_context_plan,
    ui_context_gap,
)

_TETRIS_GOAL = (
    "I would like you to add a new Tab under 'overview' called "
    '"Shennanigans" where I can play tetris'
)
_FREESTYLE_RENDERED_UI_REVIEW_GOAL = (
    "Improve Freestyle rendered UI review so it understands JS-generated "
    "DOM/classes instead of falsely treating every CSS selector absent from "
    "server-rendered HTML as an orphan."
)


@pytest.mark.parametrize(
    ("goal", "expected_category"),
    [
        (_TETRIS_GOAL, "web_ui"),
        ("Add a dashboard page with a navigation sidebar.", "web_ui"),
        ("Restyle the form buttons with new CSS.", "web_ui"),
        ("Fix the failing pytest regression suite.", "tests"),
        ("Rewrite the README documentation for setup.", "docs"),
        ("Add a cli flag for verbose output.", "cli_dev_tooling"),
        ("Tighten the settings configuration defaults.", "config"),
        ("Harden freestyle worktree materialization.", "freestyle_internal"),
    ],
)
def test_goal_classification_detects_expected_categories(goal, expected_category):
    assert expected_category in plan_context(goal).goal_categories


def test_goal_without_known_keywords_falls_back_to_core_app():
    plan = plan_context("Update the sample value.")
    assert plan.goal_categories == ("core_app",)
    assert plan.priority_patterns == ()
    assert plan.demote_freestyle_internals is True


def test_title_terms_also_classify_the_goal():
    plan = plan_context("Improve the labels.", title="Navigation sidebar polish")
    assert "web_ui" in plan.goal_categories


def test_classification_is_deterministic_and_canonically_ordered():
    goal = "Add tests for the new dashboard page."
    first = plan_context(goal)
    second = plan_context(goal)
    assert first == second
    assert first.goal_categories == ("web_ui", "tests")
    assert first.priority_patterns == tuple(dict.fromkeys(first.priority_patterns))


def test_freestyle_goals_keep_freestyle_internals_undemoted():
    assert (
        plan_context(
            "Refactor the freestyle proposal slice."
        ).demote_freestyle_internals
        is False
    )
    assert plan_context("Add a settings page tab.").demote_freestyle_internals is True


def test_freestyle_ui_review_goal_uses_subsystem_surface_not_web_ui_globs():
    plan = plan_context(_FREESTYLE_RENDERED_UI_REVIEW_GOAL)

    assert plan.goal_categories == ("freestyle_internal",)
    assert priority_rank("app/dev/freestyle_ui_review.py", plan) == 0
    assert priority_rank("tests/test_freestyle_ui_review.py", plan) == 1
    assert priority_rank("app/web_pages.py", plan) is None
    assert priority_rank("app/static/lewis.css", plan) is None


def test_freestyle_ui_review_goal_does_not_expand_ordinary_prose_words():
    plan = plan_context(_FREESTYLE_RENDERED_UI_REVIEW_GOAL)
    patterns = set(plan.priority_patterns)

    for word in (
        "absent",
        "actually",
        "advisory",
        "orphan",
        "rendered",
        "class",
        "selector",
        "static",
    ):
        assert f"app/static/*{word}*.js" not in patterns
        assert f"app/static/*{word}*.css" not in patterns
        assert f"templates/*{word}*.html" not in patterns
        assert f"tests/test_*{word}*.py" not in patterns


def test_freestyle_implementation_planner_goal_uses_planner_surface():
    plan = plan_context(
        "Improve Freestyle implementation planner context recommendations.",
        title="Implementation planner",
    )

    assert plan.goal_categories == ("freestyle_internal",)
    assert priority_rank("app/dev/freestyle_implementation_plan.py", plan) == 0
    assert priority_rank("tests/test_freestyle_implementation_plan.py", plan) == 1
    assert priority_rank("tests/test_freestyle_planning_improvements.py", plan) == 2


@pytest.mark.parametrize(
    ("goal", "module", "test_file"),
    [
        (
            "route render",
            "app/dev/freestyle_route_render.py",
            "tests/test_freestyle_route_render.py",
        ),
        (
            "repo map",
            "app/dev/repo_map.py",
            "tests/test_repo_map.py",
        ),
        (
            "edit bundle",
            "app/dev/freestyle_edit_bundle.py",
            "tests/test_freestyle_edit_bundle.py",
        ),
    ],
)
def test_freestyle_subsystem_phrases_use_focused_surfaces(goal, module, test_file):
    plan = plan_context(goal)

    assert plan.goal_categories == ("freestyle_internal",)
    assert priority_rank(module, plan) == 0
    assert priority_rank(test_file, plan) is not None
    assert priority_rank("app/web_pages.py", plan) is None


def test_literal_repo_paths_remain_named_targets_with_subsystem_recommendations():
    plan = plan_context(
        "Improve Freestyle rendered UI review and inspect app/static/orphan.js."
    )

    assert is_named_target("app/static/orphan.js", plan)
    assert priority_rank("app/dev/freestyle_ui_review.py", plan) == 0


@pytest.mark.parametrize(
    "path",
    [
        "app/dev/freestyle_web.py",
        "app/static/lewis.css",
        "app/web.py",
        "app/web_routes.py",
        "app/web_pages.py",
        "app/web_components.py",
        "app/static/chat.js",
        "templates/shennanigans.html",
        "tests/test_freestyle_web.py",
        "tests/test_web.py",
        "tests/test_web_rules.py",
    ],
)
def test_web_ui_priority_patterns_cover_the_required_surface(path):
    plan = plan_context(_TETRIS_GOAL)
    assert priority_rank(path, plan) is not None, path


def test_priority_rank_orders_navigation_files_before_assets_and_ignores_internals():
    plan = plan_context(_TETRIS_GOAL)
    assert priority_rank("app/web_pages.py", plan) == 0
    assert priority_rank("app/web_pages.py", plan) < priority_rank(
        "app/static/chat.js", plan
    )
    for unrelated in (
        "app/db.py",
        "app/dev/freestyle_checks.py",
        "app/dev/freestyle_edit_bundle.py",
        "tests/test_freestyle_checks.py",
    ):
        assert priority_rank(unrelated, plan) is None, unrelated


def test_non_matching_categories_add_no_priority_patterns():
    plan = plan_context("Update the sample value.")
    assert priority_rank("app/web_pages.py", plan) is None


def test_web_ui_test_patterns_rank_after_every_implementation_pattern():
    plan = plan_context("Add tests for the new dashboard page.")
    assert plan.goal_categories == ("web_ui", "tests")
    implementation_ranks = [
        priority_rank(path, plan)
        for path in (
            "app/web_pages.py",
            "app/web_components.py",
            "app/web.py",
            "app/web_rules.py",
            "app/static/lewis.css",
            "app/static/chat.js",
            "app/dev/freestyle_web.py",
            "app/static/extra.css",
            "templates/shennanigans.html",
            "app/static/logo.svg",
        )
    ]
    test_ranks = [
        priority_rank(path, plan)
        for path in (
            "tests/test_web.py",
            "tests/test_web_rules.py",
            "tests/test_web_pages.py",
            "tests/test_freestyle_web.py",
            "tests/test_budget.py",
            "conftest.py",
        )
    ]
    assert all(rank is not None for rank in (*implementation_ranks, *test_ranks))
    assert max(implementation_ranks) < min(test_ranks)


def test_pattern_lists_split_into_core_and_test_patterns():
    plan = plan_context("Add tests for the new dashboard page.")
    core = [p for p in plan.priority_patterns if not is_test_pattern(p)]
    tests = [p for p in plan.priority_patterns if is_test_pattern(p)]
    assert core and tests
    assert all(not p.startswith(("tests/", "conftest")) for p in core)
    assert all(p.startswith(("tests/", "conftest")) for p in tests)
    assert is_test_pattern("tests/*.py") is True
    assert is_test_pattern("conftest.py") is True
    assert is_test_pattern("app/web_pages.py") is False


def test_goal_fit_warnings_flag_a_create_only_template_bundle():
    plan = plan_context(_TETRIS_GOAL)
    warnings = goal_fit_warnings(
        plan, (("create_file", "templates/shennanigans.html"),)
    )
    assert GOAL_FIT_WARNING_CREATE_ONLY in warnings
    assert GOAL_FIT_WARNING_UNWIRED_ENTRY_POINT in warnings
    assert GOAL_FIT_WARNING_NO_WEB_SURFACE not in warnings


def test_goal_fit_warning_flags_created_assets_without_an_entry_point_edit():
    plan = plan_context(_TETRIS_GOAL)
    warnings = goal_fit_warnings(
        plan,
        (
            ("create_file", "app/static/tetris.js"),
            ("replace_text", "tests/test_web.py"),
        ),
    )
    assert GOAL_FIT_WARNING_UNWIRED_ENTRY_POINT in warnings
    assert GOAL_FIT_WARNING_CREATE_ONLY not in warnings


def test_goal_fit_warning_absent_when_an_entry_point_is_touched():
    plan = plan_context(_TETRIS_GOAL)
    warnings = goal_fit_warnings(
        plan,
        (
            ("create_file", "app/static/tetris.js"),
            ("replace_text", "app/web_components.py"),
        ),
    )
    assert GOAL_FIT_WARNING_UNWIRED_ENTRY_POINT not in warnings


def test_goal_fit_warnings_flag_a_bundle_missing_every_web_surface():
    plan = plan_context(_TETRIS_GOAL)
    warnings = goal_fit_warnings(plan, (("replace_text", "app/db.py"),))
    assert warnings == (GOAL_FIT_WARNING_NO_WEB_SURFACE,)


def test_goal_fit_warnings_absent_when_existing_web_surface_is_modified():
    plan = plan_context(_TETRIS_GOAL)
    operations = (
        ("replace_text", "app/web_pages.py"),
        ("create_file", "app/static/tetris.js"),
    )
    assert goal_fit_warnings(plan, operations) == ()


def test_goal_fit_warnings_empty_for_non_ui_goals_and_empty_bundles():
    ui_plan = plan_context(_TETRIS_GOAL)
    plain_plan = plan_context("Update the sample value.")
    assert goal_fit_warnings(plain_plan, (("create_file", "notes.txt"),)) == ()
    assert goal_fit_warnings(ui_plan, ()) == ()


def test_summarize_context_plan_reports_inclusions_omissions_and_budget():
    plan = plan_context(_TETRIS_GOAL)
    summary = summarize_context_plan(
        plan,
        ["app/web_pages.py", "README.md"],
        [
            {"path": "app/static/chat.js", "reason": "file_count_limit"},
            {"path": "app/static/lewis.css", "reason": "oversized"},
            {"path": "app/web_pages.py", "reason": "content_truncated"},
            {"path": "app/db.py", "reason": "file_count_limit"},
        ],
        priority_candidate_count=4,
    )
    assert summary["detected_goal_categories"] == ["web_ui"]
    assert "app/web_pages.py" in summary["high_priority_patterns"]
    assert summary["included_priority_files"] == ["app/web_pages.py"]
    assert summary["omitted_priority_files"] == [
        {"path": "app/static/chat.js", "reason": "file_count_limit"},
        {"path": "app/static/lewis.css", "reason": "oversized"},
    ]
    assert summary["truncated_priority_files"] == ["app/web_pages.py"]
    assert summary["priority_files_omitted_due_to_budget"] is True
    assert summary["warnings"] == []


def test_summarize_context_plan_makes_core_before_tests_ordering_explicit():
    plan = plan_context("Add tests for the new dashboard page.")
    summary = summarize_context_plan(plan, [], [], priority_candidate_count=0)
    assert summary["core_priority_patterns"]
    assert summary["test_priority_patterns"]
    assert all(
        not pattern.startswith(("tests/", "conftest"))
        for pattern in summary["core_priority_patterns"]
    )
    assert all(
        pattern.startswith(("tests/", "conftest"))
        for pattern in summary["test_priority_patterns"]
    )
    assert "core implementation" in summary["ordering_policy"]
    assert "tests are dropped" in summary["ordering_policy"]


def test_summarize_context_plan_warns_when_no_ui_candidates_exist():
    plan = plan_context(_TETRIS_GOAL)
    summary = summarize_context_plan(
        plan, ["README.md"], [], priority_candidate_count=0
    )
    assert summary["warnings"] == [PLAN_WARNING_NO_UI_FILES]
    assert summary["priority_files_omitted_due_to_budget"] is False


def test_ui_context_gap_fires_only_when_candidates_exist_but_none_included():
    plan = plan_context(_TETRIS_GOAL)
    gap_summary = summarize_context_plan(
        plan,
        ["app/dev/freestyle_checks.py"],
        [{"path": "app/web_pages.py", "reason": "file_count_limit"}],
        priority_candidate_count=3,
    )
    assert ui_context_gap(gap_summary) == UI_CONTEXT_GAP_MESSAGE

    included_summary = summarize_context_plan(
        plan, ["app/web_pages.py"], [], priority_candidate_count=3
    )
    assert ui_context_gap(included_summary) is None

    no_candidates = summarize_context_plan(
        plan, ["README.md"], [], priority_candidate_count=0
    )
    assert ui_context_gap(no_candidates) is None
    assert ui_context_gap(None) is None

    plain_plan = plan_context("Update the sample value.")
    plain_summary = summarize_context_plan(
        plain_plan, [], [], priority_candidate_count=0
    )
    assert ui_context_gap(plain_summary) is None


def test_planner_module_is_pure_with_no_io_model_or_runtime_dependencies():
    source = Path(context_plan.__file__).read_text(encoding="utf-8")
    lowered = source.casefold()
    for forbidden in (
        "subprocess",
        "deepseek_client",
        "create_deepseek_client",
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
        "import app",
        "from app.",
    ):
        assert forbidden not in lowered, forbidden
    assert "ApprovalStore" not in source
