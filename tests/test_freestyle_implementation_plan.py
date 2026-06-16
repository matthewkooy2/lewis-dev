"""Tests for the deterministic freestyle implementation/repair planner.

Covers task classification, affected-surface mapping, failed-strategy detection,
the Calendar benchmark acceptance case, plan-deviation review warnings, prompt
integration, and module purity.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import app.dev.freestyle_implementation_plan as impl_plan
import app.dev.freestyle_proposal as proposal
import app.dev.freestyle_review as review
from app.dev.freestyle_coding import FreestyleJob
from app.dev.freestyle_context_plan import plan_context
from app.dev.freestyle_implementation_plan import (
    FAILED_CSS_ONLY_PATCH,
    FAILED_FORBIDDEN_OR_GENERIC_TEST_TOUCHED,
    PLAN_DEVIATION_CSS_ONLY_WHEN_WIRING_REQUIRED,
    TASK_BACKEND_FEATURE,
    TASK_REPO_MAP_OR_FREESTYLE_MACHINERY,
    TASK_SAFETY_VALIDATION_FIX,
    TASK_TEST_UPDATE,
    TASK_WEB_UI_FEATURE,
    TASK_WEB_UI_REPAIR,
    build_implementation_plan,
    detect_failed_strategies,
    plan_deviation_warnings,
    summarize_implementation_plan,
)
from app.dev.freestyle_intake import build_intake_summary
from app.dev.freestyle_proposal import DeepSeekProposalConfig, ProposalContext
from app.dev.repo_map import RepoIndex
from app.dev.repo_map_context import affected_surface_map

JOB_ID = "a" * 32


def _job(goal: str, title: str = "Job", job_id: str = JOB_ID) -> FreestyleJob:
    return FreestyleJob(
        job_id=job_id,
        title=title,
        goal=goal,
        branch="freestyle/job",
        worktree_path="/tmp/worktree",
        base_commit="0" * 40,
        status="materialized",
        created_at="2026-06-16T00:00:00Z",
        updated_at="2026-06-16T00:00:00Z",
    )


def _surfaces(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "available": True,
        "primary_feature": None,
        "primary_routes": [],
        "routes": [],
        "pages": [],
        "css": [],
        "js": [],
        "tests": [],
        "backend_services": [],
        "data_surfaces": [],
        "static_assets": [],
        "approval_sensitive": [],
        "connected_files": [],
        "graph_file_count": 0,
        "pipeline": [],
    }
    base.update(overrides)
    return base


def _calendar_surfaces(**overrides: object) -> dict[str, object]:
    surfaces = _surfaces(
        primary_feature="calendar",
        primary_routes=["/calendar"],
        routes=["app/web_routes.py"],
        pages=["app/web_pages.py"],
        js=["app/static/calendar.js"],
        css=["app/static/lewis.css"],
        tests=["tests/test_calendar_page.py"],
        connected_files=[
            "app/web_pages.py",
            "app/static/calendar.js",
            "app/static/lewis.css",
            "tests/test_calendar_page.py",
        ],
        graph_file_count=5,
    )
    surfaces.update(overrides)
    return surfaces


def _plan(
    goal: str,
    *,
    title: str = "Job",
    surfaces: dict[str, object] | None = None,
    memory: dict[str, object] | None = None,
    observed_feedback: str | None = None,
    is_refinement: bool = False,
    provenance: dict[str, object] | None = None,
):
    intake = build_intake_summary(
        goal,
        title,
        refinement_goal=goal if is_refinement else None,
        previous_bundle_touched_files=(memory or {}).get(
            "previous_bundle_touched_files", ()
        ),
        previous_proposal_likely_files=(memory or {}).get(
            "previous_proposal_likely_files", ()
        ),
    )
    return build_implementation_plan(
        job_id=JOB_ID,
        created_at="2026-06-16T00:00:00Z",
        refinement_id="r" * 16 if is_refinement else None,
        goal=goal,
        title=title,
        intake=intake,
        context_plan=plan_context(goal, title),
        surface_facts=surfaces if surfaces is not None else _surfaces(),
        repo_map_provenance=provenance
        or {"available": True, "scope": "worktree", "stale": False, "fallback": False},
        memory=memory or {},
        observed_feedback=observed_feedback,
        is_refinement=is_refinement,
    )


# --------------------------------------------------------------------------- #
# Initial job planning.
# --------------------------------------------------------------------------- #
def test_initial_web_ui_feature_plan():
    plan = _plan(
        "Add a calendar month view to the dashboard page",
        title="Calendar",
        surfaces=_calendar_surfaces(),
    )
    art = plan.to_artifact()
    assert art["task_type"] == TASK_WEB_UI_FEATURE
    for surface in ("app/web_pages.py", "app/static/calendar.js", "app/static/lewis.css"):
        assert surface in art["expected_files"]
    assert art["affected_surfaces"]["pipeline"] == [] or art["primary_routes"] == [
        "/calendar"
    ]
    assert art["validation_plan"]["tests_to_run"]
    assert art["non_goals"]
    assert art["failed_strategy_patterns"] == []


def test_backend_feature_plan_has_no_ui_assumptions():
    plan = _plan(
        "Add a retry budget cap to the task scheduler service",
        title="Scheduler",
        surfaces=_surfaces(
            backend_services=["app/scheduler.py"],
            tests=["tests/test_scheduler.py"],
            connected_files=["app/scheduler.py", "tests/test_scheduler.py"],
        ),
    )
    art = plan.to_artifact()
    assert art["task_type"] == TASK_BACKEND_FEATURE
    assert "app/scheduler.py" in art["expected_files"]
    assert not any("tab" in goal.lower() for goal in art["non_goals"])
    assert not art["affected_surfaces"]["css"]
    assert not art["affected_surfaces"]["js"]


def test_test_update_plan_makes_no_implementation_change():
    plan = _plan(
        "Update the regression tests in tests/test_scheduler.py to cover the new branch",
        title="Tests",
        surfaces=_surfaces(tests=["tests/test_scheduler.py"]),
    )
    art = plan.to_artifact()
    assert art["task_type"] == TASK_TEST_UPDATE
    assert "implementation" in art["recommended_strategy"].lower()


def test_freestyle_runner_framing_is_not_machinery():
    plan = _plan(
        "Add a calendar tab to the dashboard. Freestyle is the runner and the "
        "repo map provides context; do not change previous attempt memory.",
        title="Calendar",
        surfaces=_calendar_surfaces(),
    )
    assert plan.task_type == TASK_WEB_UI_FEATURE


def test_actual_freestyle_goal_is_machinery():
    plan = _plan(
        "Modify app/dev/freestyle_edit_generation.py to change the prompt builder",
        title="Freestyle",
    )
    assert plan.task_type == TASK_REPO_MAP_OR_FREESTYLE_MACHINERY


def test_freestyle_ui_review_plan_recommends_subsystem_module_and_test():
    plan = _plan(
        "Improve Freestyle rendered UI review so it understands JS-generated "
        "DOM/classes instead of falsely treating every CSS selector absent from "
        "server-rendered HTML as an orphan.",
        title="Rendered UI review",
    )
    art = plan.to_artifact()

    assert art["task_type"] == TASK_REPO_MAP_OR_FREESTYLE_MACHINERY
    assert "app/dev/freestyle_ui_review.py" in art["expected_files"]
    assert "tests/test_freestyle_ui_review.py" in art["expected_files"]
    assert "app/dev/freestyle_ui_review.py" in plan.context_recommendation_paths
    assert art["validation_plan"]["tests_to_run"] == [
        "tests/test_freestyle_ui_review.py",
        "tests/test_freestyle_route_render.py",
    ]


def test_freestyle_implementation_planner_plan_recommends_planner_test():
    plan = _plan(
        "Improve Freestyle implementation planner context recommendations.",
        title="Implementation planner",
    )
    art = plan.to_artifact()

    assert art["task_type"] == TASK_REPO_MAP_OR_FREESTYLE_MACHINERY
    assert "app/dev/freestyle_implementation_plan.py" in art["expected_files"]
    assert "tests/test_freestyle_implementation_plan.py" in art["expected_files"]
    assert "tests/test_freestyle_planning_improvements.py" in art["expected_files"]
    assert art["validation_plan"]["tests_to_run"] == [
        "tests/test_freestyle_implementation_plan.py",
        "tests/test_freestyle_planning_improvements.py",
    ]


# --------------------------------------------------------------------------- #
# Refinement planning + failed-strategy detection.
# --------------------------------------------------------------------------- #
def test_css_only_refinement_marks_failed_strategy_and_avoids_repeat():
    memory = {
        "previous_bundle_touched_files": ["app/static/lewis.css"],
        "previous_proposal_likely_files": ["app/static/lewis.css"],
        "previous_ui_review_warnings": [
            "UI_REVIEW_ORPHAN_CSS_SELECTORS .cal-day, .day-number remain unused"
        ],
    }
    plan = _plan(
        "Repair the calendar layout so the days render in a grid",
        title="Calendar",
        surfaces=_calendar_surfaces(),
        memory=memory,
        observed_feedback="The rendered calendar still stacks vertically",
        is_refinement=True,
    )
    art = plan.to_artifact()
    assert art["task_type"] == TASK_WEB_UI_REPAIR
    assert FAILED_CSS_ONLY_PATCH in art["failed_strategy_patterns"]
    strategy = art["recommended_strategy"].lower()
    assert "css-only" in strategy or "do not produce another css-only" in strategy
    assert "app/static/calendar.js" in plan.context_recommendation_paths
    assert "app/web_pages.py" in plan.context_recommendation_paths
    assert any("Do not submit another CSS-only patch." in g for g in art["non_goals"])


def test_touching_generic_test_is_a_failed_strategy():
    failed = detect_failed_strategies(
        intake=build_intake_summary("Add a calendar feature", "Calendar"),
        memory={
            "previous_bundle_touched_files": ["tests/test_web.py"],
            "previous_repo_map_review_warnings": ["repo map warning"],
        },
        observed_feedback=None,
        surface_facts=_calendar_surfaces(),
    )
    assert FAILED_FORBIDDEN_OR_GENERIC_TEST_TOUCHED in failed


def test_forbidden_test_refinement_recommends_focused_test():
    plan = _plan(
        "Add calendar tests but do not modify tests/test_web.py",
        title="Calendar",
        surfaces=_calendar_surfaces(),
        memory={"previous_bundle_touched_files": ["tests/test_web.py"]},
        is_refinement=True,
    )
    art = plan.to_artifact()
    assert "tests/test_web.py" in art["forbidden_files"]
    assert any(
        "tests/test_calendar_page.py" in req for req in art["inspection_requirements"]
    )


def test_safety_validation_fix_keeps_approval_core():
    plan = _plan(
        "Fix the approval validation false positive that blocks a test fixture",
        title="Approval policy",
    )
    art = plan.to_artifact()
    assert art["task_type"] == TASK_SAFETY_VALIDATION_FIX
    assert "narrow" in art["recommended_strategy"].lower()
    assert any("approval" in boundary.lower() for boundary in art["safety_boundaries"])
    assert any(
        "approval" in goal.lower() or "safety" in goal.lower()
        for goal in art["non_goals"]
    )


def test_budget_omitted_surface_is_elevated_into_recommendations():
    plan = _plan(
        "Repair the calendar grid layout",
        title="Calendar",
        surfaces=_calendar_surfaces(),
        memory={
            "previous_bundle_touched_files": ["app/static/lewis.css"],
            "previous_proposal_likely_files": ["app/static/calendar.js"],
            "previous_ui_review_warnings": ["UI_REVIEW_ORPHAN_CSS_SELECTORS .cal-day"],
        },
        observed_feedback="calendar still stacks vertically",
        is_refinement=True,
    )
    # A previously relevant JS surface is elevated into context recommendations.
    assert "app/static/calendar.js" in plan.context_recommendation_paths


# --------------------------------------------------------------------------- #
# Calendar benchmark acceptance fixture.
# --------------------------------------------------------------------------- #
def test_calendar_benchmark_acceptance():
    memory = {
        "previous_bundle_touched_files": ["app/static/lewis.css"],
        "previous_proposal_likely_files": ["app/static/lewis.css"],
        "previous_ui_review_warnings": [
            "UI_REVIEW_ORPHAN_CSS_SELECTORS .cal-day",
            "UI_REVIEW_ORPHAN_CSS_SELECTORS .day-number",
            "UI_REVIEW_ORPHAN_CSS_SELECTORS .event-chip",
            "UI_REVIEW_ORPHAN_CSS_SELECTORS .agenda-event",
        ],
    }
    plan = _plan(
        "Repair the Calendar page so days render in a month grid",
        title="Calendar",
        surfaces=_calendar_surfaces(),
        memory=memory,
        observed_feedback="rendered Calendar still stacks vertically",
        is_refinement=True,
    )
    art = plan.to_artifact()
    assert art["task_type"] == TASK_WEB_UI_REPAIR
    assert FAILED_CSS_ONLY_PATCH in art["failed_strategy_patterns"]
    strategy = art["recommended_strategy"].lower()
    assert "dom" in strategy and "css" in strategy and "wiring" in strategy
    recs = plan.context_recommendation_paths
    for surface in (
        "app/web_pages.py",
        "app/static/calendar.js",
        "app/static/lewis.css",
        "tests/test_calendar_page.py",
    ):
        assert surface in recs
    section = plan.prompt_section()
    assert section["task_type"] == TASK_WEB_UI_REPAIR
    assert FAILED_CSS_ONLY_PATCH in section["failed_strategy_patterns"]


# --------------------------------------------------------------------------- #
# Plan-deviation review warnings.
# --------------------------------------------------------------------------- #
def test_plan_deviation_css_only_when_wiring_required():
    art = _plan(
        "Repair the calendar layout",
        title="Calendar",
        surfaces=_calendar_surfaces(),
        memory={
            "previous_bundle_touched_files": ["app/static/lewis.css"],
            "previous_ui_review_warnings": ["UI_REVIEW_ORPHAN_CSS_SELECTORS .cal-day"],
        },
        observed_feedback="still stacks vertically",
        is_refinement=True,
    ).to_artifact()
    warnings = plan_deviation_warnings(art, ["app/static/lewis.css"])
    assert PLAN_DEVIATION_CSS_ONLY_WHEN_WIRING_REQUIRED in warnings
    # A change that also touches JS is not flagged as a CSS-only deviation.
    mixed = plan_deviation_warnings(
        art, ["app/static/lewis.css", "app/static/calendar.js"]
    )
    assert PLAN_DEVIATION_CSS_ONLY_WHEN_WIRING_REQUIRED not in mixed


def test_plan_deviation_forbidden_touched():
    art = _plan(
        "Add calendar tests but do not modify tests/test_web.py",
        title="Calendar",
        surfaces=_calendar_surfaces(),
        is_refinement=True,
    ).to_artifact()
    warnings = plan_deviation_warnings(art, ["tests/test_web.py"])
    assert any("not to modify" in warning for warning in warnings)


# --------------------------------------------------------------------------- #
# affected_surface_map over a synthetic repo index.
# --------------------------------------------------------------------------- #
def _calendar_index() -> RepoIndex:
    return RepoIndex(
        manifest={
            "scope": "worktree",
            "scope_id": "b" * 32,
            "schema_version": 1,
            "indexed_at": "2026-06-16T00:00:00Z",
            "file_count": 7,
        },
        files=(
            {"path": "app/web_routes.py", "role": "web_route"},
            {"path": "app/web_pages.py", "role": "web_page"},
            {"path": "app/static/calendar.js", "role": "static_js"},
            {"path": "app/static/lewis.css", "role": "static_css"},
            {"path": "tests/test_calendar_page.py", "role": "test"},
            {"path": "app/approvals.py", "role": "approval_sensitive_metadata"},
            {"path": "app/scheduler.py", "role": "normal_source"},
        ),
        routes=(
            {
                "path": "app/web_routes.py",
                "route": "/calendar",
                "handler": "calendar_page",
                "referenced_assets": ["/static/calendar.js"],
            },
        ),
        css_selectors=(
            {"path": "app/static/lewis.css", "classes": ["cal-day"], "ids": []},
        ),
        js_assets=(
            {
                "path": "app/static/calendar.js",
                "lookup_classes": ["cal-day"],
                "managed_classes": [],
                "lookup_ids": [],
            },
        ),
        tests=(
            {
                "path": "tests/test_calendar_page.py",
                "referenced_modules": ["app.web_pages"],
                "test_functions": ["test_calendar"],
                "fixtures": [],
                "referenced_routes": ["/calendar"],
            },
        ),
    )


def test_affected_surface_map_buckets_roles_and_keeps_approval_metadata_only():
    surfaces = affected_surface_map(
        _calendar_index(),
        goal="repair calendar layout",
        named_targets=(
            "app/web_pages.py",
            "app/static/calendar.js",
            "app/static/lewis.css",
            "tests/test_calendar_page.py",
        ),
        seed_paths=("app/web_routes.py",),
        changed_paths=("app/approvals.py",),
    )
    assert "app/web_routes.py" in surfaces["routes"]
    assert "app/web_pages.py" in surfaces["pages"]
    assert "app/static/calendar.js" in surfaces["js"]
    assert "app/static/lewis.css" in surfaces["css"]
    assert "tests/test_calendar_page.py" in surfaces["tests"]
    assert "app/approvals.py" in surfaces["approval_sensitive"]
    assert "app/approvals.py" not in surfaces["backend_services"]
    assert "app/approvals.py" not in surfaces["connected_files"]
    assert surfaces["primary_feature"] == "calendar"
    assert "/calendar" in surfaces["primary_routes"]
    assert surfaces["pipeline"]


def test_affected_surface_map_empty_without_index():
    surfaces = affected_surface_map(None, goal="anything")
    assert surfaces["available"] is False
    assert surfaces["connected_files"] == []


# --------------------------------------------------------------------------- #
# Prompt integration.
# --------------------------------------------------------------------------- #
def _context() -> ProposalContext:
    return ProposalContext(
        files=(
            {
                "path": "app/web_pages.py",
                "content": "def calendar_page():\n    return ''\n",
                "truncated": False,
                "sha256": "x" * 64,
            },
        ),
        omitted_files=(),
        omission_counts={},
        included_char_count=32,
        scanned_file_count=1,
        context_plan={"priority_candidate_count": 1},
    )


def test_edit_generation_prompt_includes_implementation_plan():
    from app.dev.freestyle_edit_generation import _fit_prompt

    job = _job("Repair the calendar layout", "Calendar")
    plan = _plan(
        "Repair the calendar layout",
        title="Calendar",
        surfaces=_calendar_surfaces(),
        memory={
            "previous_bundle_touched_files": ["app/static/lewis.css"],
            "previous_ui_review_warnings": ["UI_REVIEW_ORPHAN_CSS_SELECTORS .cal-day"],
        },
        observed_feedback="still stacks vertically",
        is_refinement=True,
    )
    _fitted, user_prompt = _fit_prompt(
        job,
        {"likely_files_to_edit": ["app/web_pages.py"]},
        _context(),
        plan_context("Repair the calendar layout", "Calendar"),
        [],
        plan,
    )
    assert '"implementation_plan"' in user_prompt
    assert TASK_WEB_UI_REPAIR in user_prompt
    assert FAILED_CSS_ONLY_PATCH in user_prompt


def test_proposal_call_includes_plan_in_messages():
    captured: dict[str, object] = {}

    def _create(**kwargs: object):
        captured["messages"] = kwargs["messages"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"proposed_implementation_steps":["x"],'
                            '"likely_files_to_edit":["app/web_pages.py"],'
                            '"risks_unknowns":[],'
                            '"suggested_allowlisted_checks":["pytest"]}'
                        )
                    )
                )
            ],
            model="deepseek-chat",
            usage=None,
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    goal = "Add a calendar page to the dashboard"
    job = _job(goal, "Calendar")
    config = DeepSeekProposalConfig(api_key="k", model="deepseek-chat")
    plan = _plan(goal, title="Calendar", surfaces=_calendar_surfaces())
    assert plan.task_type == TASK_WEB_UI_FEATURE
    proposal._call_deepseek(
        client,
        config,
        job,
        _context(),
        plan_context(goal, "Calendar"),
        implementation_plan=plan,
    )
    messages = captured["messages"]
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert "implementation_plan" in system
    assert '"implementation_plan"' in user
    assert TASK_WEB_UI_FEATURE in user


# --------------------------------------------------------------------------- #
# Review surfaces an advisory plan-deviation warning (clarification 4).
# --------------------------------------------------------------------------- #
def test_review_surfaces_css_only_plan_deviation(tmp_path):
    state = tmp_path / ".lewis-dev"
    state.mkdir()
    plan = _plan(
        "Repair the calendar layout",
        title="Calendar",
        surfaces=_calendar_surfaces(),
        memory={
            "previous_bundle_touched_files": ["app/static/lewis.css"],
            "previous_ui_review_warnings": ["UI_REVIEW_ORPHAN_CSS_SELECTORS .cal-day"],
        },
        observed_feedback="still stacks vertically",
        is_refinement=True,
    )
    relative = proposal._write_implementation_plan_artifact(
        plan, state, JOB_ID, "gen-implementation-plan.json"
    )
    assert relative is not None
    job = replace(_job("Repair the calendar layout"), implementation_plan_artifact_path=relative)
    changed = (review._ChangedPath(path="app/static/lewis.css", status="M", git_status="M"),)
    section = review._implementation_plan_review(job, changed, state)
    assert section["plan_available"] is True
    assert any("only CSS" in warning for warning in section["warnings"])
    assert review._artifact_plan_warnings(
        {"implementation_plan_review": section}
    )


def test_summarize_plan_carries_repo_map_provenance():
    art = _plan(
        "Add a calendar view",
        title="Calendar",
        surfaces=_calendar_surfaces(),
        provenance={
            "available": True,
            "scope": "worktree",
            "stale": True,
            "fallback": True,
            "indexed_at": "2026-06-16T00:00:00Z",
            "schema_version": 1,
        },
    ).to_artifact()
    summary = summarize_implementation_plan(art)
    assert summary["repo_map"]["scope"] == "worktree"
    assert summary["repo_map"]["stale"] is True
    assert summary["repo_map"]["indexed_at"] == "2026-06-16T00:00:00Z"


# --------------------------------------------------------------------------- #
# Module purity.
# --------------------------------------------------------------------------- #
def test_implementation_plan_module_is_pure():
    source = Path(impl_plan.__file__).read_text(encoding="utf-8").casefold()
    for forbidden in (
        "subprocess",
        "create_deepseek_client",
        "deepseek_client",
        "os.environ",
        "os.getenv",
        "import os",
        "open(",
        "requests",
        "httpx",
        "sqlite",
        "from app.web",
        "testclient",
    ):
        assert forbidden not in source, forbidden
