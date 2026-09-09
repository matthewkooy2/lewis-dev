"""Deterministic implementation/repair planning for freestyle jobs.

Converts a job goal (plus, for refinements, the prior attempt's behaviour,
review warnings, observed runtime feedback, and the repo map's affected
surfaces) into an explicit, inspectable implementation plan *before* any edit
bundle is generated. The plan classifies the task, names the affected
implementation surfaces, recommends a concrete strategy, records which prior
strategies already failed, and lists the files context selection should prefer.

This is the layer that turns "find the right files" into "choose the right
strategy": a CSS-only patch that left orphan-selector warnings becomes a
recorded ``css_only_patch`` failure with a "inspect the DOM/JS/CSS wiring"
strategy instead of another CSS-only patch.

Everything here is pure string and data work: this module performs no file,
process, model, or network I/O, opens no files, runs no commands, makes no
model call, and never relaxes any safety exclusion. It only reads already
derived, safety-cleared inputs (the deterministic intake summary, the model-
free context plan, the repo-map surface facts, and the bounded previous-attempt
memory) and emits advisory guidance. The plan can advise and rank; it never
blocks the deterministic applier and never adds a new rejection path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import re

from app.dev.freestyle_context_plan import ContextPlan
from app.dev.freestyle_intake import (
    INTENT_DOCS_ONLY,
    INTENT_FREESTYLE_INTERNAL,
    INTENT_REFACTOR,
    INTENT_TESTS_ONLY,
    INTENT_UI,
    is_test_path,
)

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "freestyle_implementation_plan"

# Task categories, used by the strategy generator and surfaced in artifacts.
TASK_WEB_UI_FEATURE = "web_ui_feature"
TASK_WEB_UI_REPAIR = "web_ui_repair"
TASK_BACKEND_FEATURE = "backend_feature"
TASK_TEST_UPDATE = "test_update"
TASK_SAFETY_VALIDATION_FIX = "safety_validation_fix"
TASK_REPO_MAP_OR_FREESTYLE_MACHINERY = "repo_map_or_freestyle_machinery"
TASK_APPROVAL_SENSITIVE_REFUSAL = "approval_sensitive_refusal"
TASK_REFACTOR = "refactor"
TASK_UNKNOWN = "unknown"

# Stable failed-strategy pattern identifiers recorded for refinements.
FAILED_CSS_ONLY_PATCH = "css_only_patch"
FAILED_TESTS_ONLY_CHANGE = "tests_only_change"
FAILED_FORBIDDEN_OR_GENERIC_TEST_TOUCHED = "forbidden_or_generic_test_touched"
FAILED_APPROVAL_VALIDATION_LITERAL = "approval_validation_literal"
FAILED_OMITTED_RELEVANT_SURFACE = "omitted_relevant_surface"
FAILED_OVERBROAD_CHANGE = "overbroad_change"

# Advisory plan-deviation warnings surfaced by review when an applied bundle
# contradicts the plan. These never block apply; they only flag for a human.
PLAN_DEVIATION_CSS_ONLY_WHEN_WIRING_REQUIRED = (
    "Plan-deviation warning: the implementation plan recommended inspecting and "
    "fixing the page/JS/CSS wiring, but the change touches only CSS; confirm the "
    "DOM/JS emission was actually corrected, not just the stylesheet."
)
PLAN_DEVIATION_FORBIDDEN_TOUCHED = (
    "Plan-deviation warning: the implementation plan listed {paths} as files not "
    "to modify, but the change touches them anyway."
)
PLAN_DEVIATION_EXPECTED_UNTOUCHED = (
    "Plan-deviation warning: the implementation plan expected to touch {paths}, "
    "but the change touches none of the expected implementation surfaces."
)

MAX_PLAN_LIST = 16
MAX_STRATEGY_STEPS = 8
MAX_FIELD_CHARS = 2000
MAX_FEATURE_CHARS = 60

# Tokens that signal a rendered layout is still broken (drives css_only_patch
# detection and the web_ui_repair classification from human feedback).
_LAYOUT_BROKEN_TOKENS = frozenset(
    {
        "stack",
        "stacks",
        "stacked",
        "stacking",
        "vertical",
        "vertically",
        "overlap",
        "overlapping",
        "misaligned",
        "broken",
        "unstyled",
        "render",
        "rendered",
        "rendering",
        "layout",
        "column",
        "columns",
        "grid",
        "flex",
        "wrap",
        "collapsed",
    }
)
_UI_REPAIR_VERB_TOKENS = frozenset(
    {
        "fix",
        "repair",
        "broken",
        "stack",
        "stacks",
        "stacking",
        "misaligned",
        "overlap",
        "unstyled",
        "realign",
        "restore",
    }
)
_APPROVAL_TOKENS = frozenset(
    {"approval", "approvals", "approve", "approver", "approved"}
)
_VALIDATION_TOKENS = frozenset(
    {"validation", "validate", "validator", "policy", "guard", "guardrail"}
)
_REFUSAL_TOKENS = frozenset({"refuse", "refused", "refusal", "disallow", "disallowed"})
_TOKEN = re.compile(r"[a-z0-9]+")
_FEATURE_STOP_TOKENS = frozenset(
    {
        "add",
        "and",
        "app",
        "change",
        "feature",
        "fix",
        "implement",
        "make",
        "page",
        "rendered",
        "repair",
        "the",
        "update",
        "view",
        "with",
    }
)

# Fixed advisory safety reminders. These restate boundaries the deterministic
# validator already enforces; the plan only repeats them for the model.
_SAFETY_BOUNDARIES_COMMON = (
    "Edits apply only to the isolated worktree; the main checkout is never mutated.",
    "Do not touch approval-store, secrets, .env, database, or generated files.",
    "All edits must pass the deterministic edit-bundle validator; this plan only advises.",
)
_SAFETY_BOUNDARIES_APPROVAL = (
    "Preserve approval/validation core protections; only narrow the specific "
    "false positive, never loosen the approval boundary itself.",
)


@dataclass(frozen=True)
class ImplementationPlan:
    """One deterministic, model-free implementation/repair plan for a job."""

    job_id: str
    created_at: str
    task_type: str
    primary_feature: str | None
    primary_routes: tuple[str, ...] = ()
    refinement_id: str | None = None
    is_refinement: bool = False
    affected_surfaces: Mapping[str, object] = field(default_factory=dict)
    repo_map_context: Mapping[str, object] = field(default_factory=dict)
    expected_files: tuple[str, ...] = ()
    forbidden_files: tuple[str, ...] = ()
    non_goals: tuple[str, ...] = ()
    safety_boundaries: tuple[str, ...] = ()
    risk_flags: tuple[str, ...] = ()
    review_signals: tuple[str, ...] = ()
    observed_feedback: str | None = None
    prior_attempts: Mapping[str, object] = field(default_factory=dict)
    failed_strategy_patterns: tuple[str, ...] = ()
    recommended_strategy: str = ""
    strategy_steps: tuple[str, ...] = ()
    inspection_requirements: tuple[str, ...] = ()
    validation_plan: Mapping[str, object] = field(default_factory=dict)
    context_recommendations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def context_recommendation_paths(self) -> tuple[str, ...]:
        """The safe context paths the planner asks selection to elevate."""
        return self.context_recommendations

    def to_artifact(self) -> dict[str, object]:
        """The full, inspectable, JSON-serializable plan artifact."""
        return {
            "artifact_type": ARTIFACT_TYPE,
            "schema_version": SCHEMA_VERSION,
            "job_id": self.job_id,
            "refinement_id": self.refinement_id,
            "created_at": self.created_at,
            "task_type": self.task_type,
            "primary_feature": self.primary_feature,
            "primary_routes": list(self.primary_routes),
            "is_refinement": self.is_refinement,
            "affected_surfaces": dict(self.affected_surfaces),
            "repo_map_context": dict(self.repo_map_context),
            "expected_files": list(self.expected_files),
            "forbidden_files": list(self.forbidden_files),
            "non_goals": list(self.non_goals),
            "safety_boundaries": list(self.safety_boundaries),
            "risk_flags": list(self.risk_flags),
            "review_signals": list(self.review_signals),
            "observed_feedback": self.observed_feedback,
            "prior_attempts": dict(self.prior_attempts),
            "failed_strategy_patterns": list(self.failed_strategy_patterns),
            "recommended_strategy": self.recommended_strategy,
            "strategy_steps": list(self.strategy_steps),
            "inspection_requirements": list(self.inspection_requirements),
            "validation_plan": dict(self.validation_plan),
            "context_recommendations": list(self.context_recommendations),
            "warnings": list(self.warnings),
        }

    def prompt_section(self) -> dict[str, object]:
        """The compact, bounded plan block embedded in the model prompt."""
        return {
            "task_type": self.task_type,
            "primary_feature": self.primary_feature,
            "primary_routes": list(self.primary_routes),
            "expected_files": list(self.expected_files),
            "forbidden_files": list(self.forbidden_files),
            "non_goals": list(self.non_goals),
            "recommended_strategy": self.recommended_strategy,
            "strategy_steps": list(self.strategy_steps),
            "failed_strategy_patterns": list(self.failed_strategy_patterns),
            "inspection_requirements": list(self.inspection_requirements),
            "validation_plan": dict(self.validation_plan),
            "repo_map_provenance": _provenance_brief(self.repo_map_context),
        }


def build_implementation_plan(
    *,
    job_id: str,
    created_at: str,
    refinement_id: str | None = None,
    goal: str,
    title: str | None = None,
    intake: Mapping[str, object],
    context_plan: ContextPlan | None = None,
    surface_facts: Mapping[str, object] | None = None,
    repo_map_provenance: Mapping[str, object] | None = None,
    memory: Mapping[str, object] | None = None,
    observed_feedback: str | None = None,
    is_refinement: bool = False,
) -> ImplementationPlan:
    """Build one deterministic implementation/repair plan from derived inputs."""
    surfaces = dict(surface_facts or {})
    provenance = dict(repo_map_provenance or {})
    mem = dict(memory or {})
    feedback = observed_feedback.strip() if isinstance(observed_feedback, str) else None
    feedback = feedback or None

    task_type = classify_task_type(
        intake,
        is_refinement=is_refinement,
        memory=mem,
        observed_feedback=feedback,
        goal=goal,
        title=title,
    )
    forbidden = _bounded_paths(intake.get("forbidden_paths"))
    failed = detect_failed_strategies(
        intake=intake,
        memory=mem,
        observed_feedback=feedback,
        surface_facts=surfaces,
    )
    primary_feature = _primary_feature(surfaces, goal, title)
    primary_routes = _bounded_paths(surfaces.get("primary_routes"))
    expected = _expected_files(
        task_type, intake, surfaces, mem, forbidden, context_plan
    )
    inspection = _inspection_requirements(task_type, failed, surfaces, forbidden, mem)
    recommended, steps = _strategy(task_type, failed, surfaces, intake)
    context_recommendations = _context_recommendations(
        expected,
        inspection_paths=_surface_inspection_paths(surfaces),
        forbidden=forbidden,
    )
    review_signals = _review_signals(mem, feedback)
    warnings = _plan_warnings(task_type, failed, provenance, mem)

    return ImplementationPlan(
        job_id=job_id,
        created_at=created_at,
        refinement_id=refinement_id,
        is_refinement=is_refinement,
        task_type=task_type,
        primary_feature=primary_feature,
        primary_routes=primary_routes,
        affected_surfaces=_bounded_surfaces(surfaces),
        repo_map_context=_repo_map_context(surfaces, provenance),
        expected_files=expected,
        forbidden_files=forbidden,
        non_goals=_non_goals(task_type, failed, forbidden),
        safety_boundaries=_safety_boundaries(task_type),
        risk_flags=_risk_flags(task_type, failed, provenance, surfaces, forbidden),
        review_signals=review_signals,
        observed_feedback=_clip(feedback) if feedback else None,
        prior_attempts=_prior_attempts(mem),
        failed_strategy_patterns=failed,
        recommended_strategy=recommended,
        strategy_steps=steps,
        inspection_requirements=inspection,
        validation_plan=_validation_plan(intake, surfaces, expected),
        context_recommendations=context_recommendations,
        warnings=warnings,
    )


def classify_task_type(
    intake: Mapping[str, object],
    *,
    is_refinement: bool,
    memory: Mapping[str, object] | None,
    observed_feedback: str | None,
    goal: str,
    title: str | None = None,
) -> str:
    """Classify one job into a fixed task category without a model.

    A prompt that merely mentions Freestyle/repo-map/review/memory as runner
    context does not classify as machinery: that decision is delegated to the
    intake intent (``_freestyle_internal_requested``), which already ignores
    process/meta clauses and only flags an action that targets the machinery.
    Approval/safety-validation work is the more specific signal, so it is
    checked before the machinery fallback (the intake intent flags "approval"
    and "validation" as machinery terms, which would otherwise mask it).
    """
    intents = _intents(intake)
    text = _text(goal, title)
    if _is_approval_refusal(text):
        return TASK_APPROVAL_SENSITIVE_REFUSAL
    if _is_safety_validation_fix(text):
        return TASK_SAFETY_VALIDATION_FIX
    if INTENT_FREESTYLE_INTERNAL in intents:
        return TASK_REPO_MAP_OR_FREESTYLE_MACHINERY
    if INTENT_UI in intents:
        if _has_repair_signal(memory, observed_feedback, text):
            return TASK_WEB_UI_REPAIR
        return TASK_WEB_UI_FEATURE
    requires_impl = bool(intake.get("requires_implementation_change", True))
    if INTENT_TESTS_ONLY in intents and not requires_impl:
        return TASK_TEST_UPDATE
    if INTENT_REFACTOR in intents:
        return TASK_REFACTOR
    if requires_impl and INTENT_DOCS_ONLY not in intents:
        return TASK_BACKEND_FEATURE
    return TASK_UNKNOWN


def detect_failed_strategies(
    *,
    intake: Mapping[str, object],
    memory: Mapping[str, object] | None,
    observed_feedback: str | None,
    surface_facts: Mapping[str, object] | None,
) -> tuple[str, ...]:
    """Detect which prior strategy classes already failed, from prior memory."""
    mem = dict(memory or {})
    touched = _bounded_paths(mem.get("previous_bundle_touched_files"))
    if not touched:
        return ()
    surfaces = dict(surface_facts or {})
    forbidden = {p.casefold() for p in _bounded_paths(intake.get("forbidden_paths"))}
    ui_warnings = _bounded_strings(mem.get("previous_ui_review_warnings"))
    repo_warnings = _bounded_strings(mem.get("previous_repo_map_review_warnings"))
    prior_warnings = _bounded_strings(mem.get("previous_warnings"))
    layout_still_broken = bool(ui_warnings) or _signals_layout_broken(observed_feedback)
    patterns: list[str] = []

    css_only = touched and all(path.casefold().endswith(".css") for path in touched)
    if css_only and (layout_still_broken or repo_warnings):
        patterns.append(FAILED_CSS_ONLY_PATCH)

    requires_impl = bool(intake.get("requires_implementation_change", True))
    if requires_impl and touched and all(is_test_path(path) for path in touched):
        patterns.append(FAILED_TESTS_ONLY_CHANGE)

    touched_forbidden = any(path.casefold() in forbidden for path in touched)
    touched_generic_test = _touched_generic_test(touched, surfaces)
    if touched_forbidden or touched_generic_test:
        patterns.append(FAILED_FORBIDDEN_OR_GENERIC_TEST_TOUCHED)

    if _mentions_approval_refusal(prior_warnings):
        patterns.append(FAILED_APPROVAL_VALIDATION_LITERAL)

    if repo_warnings and not css_only:
        patterns.append(FAILED_OMITTED_RELEVANT_SURFACE)

    if len(touched) > MAX_PLAN_LIST:
        patterns.append(FAILED_OVERBROAD_CHANGE)

    return tuple(dict.fromkeys(patterns))


def summarize_implementation_plan(
    plan_artifact: Mapping[str, object] | None,
) -> dict[str, object]:
    """Build the bounded plan summary surfaced in handoff and the web UI."""
    if not isinstance(plan_artifact, Mapping):
        return {}
    repo_map = plan_artifact.get("repo_map_context")
    return {
        "task_type": plan_artifact.get("task_type"),
        "primary_feature": plan_artifact.get("primary_feature"),
        "expected_files": _bounded_strings(plan_artifact.get("expected_files")),
        "forbidden_files": _bounded_strings(plan_artifact.get("forbidden_files")),
        "recommended_strategy": _clip(plan_artifact.get("recommended_strategy")),
        "failed_strategy_patterns": _bounded_strings(
            plan_artifact.get("failed_strategy_patterns")
        ),
        "validation_plan": plan_artifact.get("validation_plan") or {},
        "warnings": _bounded_strings(plan_artifact.get("warnings")),
        "repo_map": _provenance_brief(
            repo_map if isinstance(repo_map, Mapping) else {}
        ),
    }


def plan_deviation_warnings(
    plan_artifact: Mapping[str, object] | None,
    changed_paths: Sequence[str],
) -> tuple[str, ...]:
    """Advisory warnings when an applied bundle contradicts the plan.

    Deterministic and advisory only: never blocks apply, never inspects file
    contents. Flags a CSS-only change where the plan called for DOM/JS/CSS
    wiring, a forbidden file touched, and a plan whose expected surfaces were
    all skipped.
    """
    if not isinstance(plan_artifact, Mapping):
        return ()
    changed = {str(path).casefold() for path in changed_paths if str(path).strip()}
    if not changed:
        return ()
    warnings: list[str] = []
    task_type = plan_artifact.get("task_type")
    failed = set(_bounded_strings(plan_artifact.get("failed_strategy_patterns")))
    wiring_required = task_type == TASK_WEB_UI_REPAIR and (
        FAILED_CSS_ONLY_PATCH in failed
        or _strategy_mentions_wiring(plan_artifact.get("recommended_strategy"))
    )
    if wiring_required and all(path.endswith(".css") for path in changed):
        warnings.append(PLAN_DEVIATION_CSS_ONLY_WHEN_WIRING_REQUIRED)

    forbidden = _bounded_strings(plan_artifact.get("forbidden_files"))
    touched_forbidden = [p for p in forbidden if p.casefold() in changed]
    if touched_forbidden:
        warnings.append(
            PLAN_DEVIATION_FORBIDDEN_TOUCHED.format(paths=_join(touched_forbidden))
        )

    expected = _bounded_strings(plan_artifact.get("expected_files"))
    if expected and not any(p.casefold() in changed for p in expected):
        warnings.append(
            PLAN_DEVIATION_EXPECTED_UNTOUCHED.format(paths=_join(expected[:5]))
        )
    return tuple(warnings)


# --------------------------------------------------------------------------- #
# Classification helpers.
# --------------------------------------------------------------------------- #
def _has_repair_signal(
    memory: Mapping[str, object] | None,
    observed_feedback: str | None,
    text: str,
) -> bool:
    if observed_feedback and observed_feedback.strip():
        return True
    mem = dict(memory or {})
    if (
        mem.get("previous_ui_review_warnings")
        or mem.get("previous_bundle_touched_files")
        or mem.get("previous_repo_map_review_warnings")
    ):
        return True
    return not _UI_REPAIR_VERB_TOKENS.isdisjoint(_tokens(text))


def _is_safety_validation_fix(text: str) -> bool:
    terms = _tokens(text)
    if _APPROVAL_TOKENS.isdisjoint(terms):
        return False
    if not _VALIDATION_TOKENS.isdisjoint(terms):
        return True
    return "false positive" in text or "false-positive" in text


def _is_approval_refusal(text: str) -> bool:
    terms = _tokens(text)
    if _APPROVAL_TOKENS.isdisjoint(terms):
        return False
    if not _REFUSAL_TOKENS.isdisjoint(terms):
        return True
    return "should be refused" in text or "must be refused" in text


def _signals_layout_broken(observed_feedback: str | None) -> bool:
    if not observed_feedback:
        return False
    return not _LAYOUT_BROKEN_TOKENS.isdisjoint(_tokens(observed_feedback))


def _mentions_approval_refusal(warnings: Sequence[str]) -> bool:
    joined = " ".join(warnings).casefold()
    return "approval" in joined and (
        "approval-sensitive" in joined
        or "approval sensitive" in joined
        or "approval region" in joined
        or "approval behavior" in joined
        or "approval behaviour" in joined
    )


def _touched_generic_test(
    touched: Sequence[str], surfaces: Mapping[str, object]
) -> bool:
    """A broad shared test file was touched while a focused test surface exists."""
    feature_tests = {p.casefold() for p in _bounded_paths(surfaces.get("tests"))}
    if not feature_tests:
        return False
    for path in touched:
        lowered = path.casefold()
        if not is_test_path(lowered) or lowered in feature_tests:
            continue
        name = lowered.rsplit("/", 1)[-1]
        if name in {"test_web.py", "test_web_pages.py", "conftest.py"}:
            return True
    return False


# --------------------------------------------------------------------------- #
# Strategy + field generation.
# --------------------------------------------------------------------------- #
def _strategy(
    task_type: str,
    failed: Sequence[str],
    surfaces: Mapping[str, object],
    intake: Mapping[str, object],
) -> tuple[str, tuple[str, ...]]:
    pages = _bounded_paths(surfaces.get("pages")) or _bounded_paths(
        surfaces.get("routes")
    )
    js = _bounded_paths(surfaces.get("js"))
    css = _bounded_paths(surfaces.get("css"))
    tests = _bounded_paths(surfaces.get("tests"))
    page_ref = pages[0] if pages else "the page/render function"
    js_ref = js[0] if js else "the static JS asset"
    css_ref = css[0] if css else "the stylesheet"

    if task_type == TASK_WEB_UI_REPAIR:
        if FAILED_CSS_ONLY_PATCH in failed:
            strategy = (
                "The previous CSS-only patch did not resolve the rendered layout. "
                "Do not produce another CSS-only patch. Inspect the page render "
                f"({page_ref}) and the JS that builds the DOM and emits classes "
                f"({js_ref}); confirm the DOM structure and class names the CSS "
                f"targets actually exist, then fix the DOM/JS wiring together with "
                f"the CSS ({css_ref})."
            )
        else:
            strategy = (
                "Repair the rendered UI by inspecting the route/page markup, the "
                "static JS that reads and writes the DOM, and the CSS "
                "selectors/layout together; verify the rendered result instead of "
                "patching one surface in isolation."
            )
        steps = (
            f"Inspect {page_ref} for the emitted HTML structure.",
            f"Inspect {js_ref} for the DOM nodes and class names it creates.",
            f"Confirm {css_ref} selectors match the real emitted classes and the "
            "grid/flex layout rules apply to the right container.",
            "Fix the DOM emission and/or CSS so the rendered layout matches the goal.",
            f"Update the feature test ({tests[0] if tests else 'the calendar/feature test'}).",
        )
    elif task_type == TASK_WEB_UI_FEATURE:
        strategy = (
            "Implement the feature end to end: touch the route/page, the static "
            "JS and CSS, and the feature test together, and wire any new asset "
            "into an existing entry point so it is reachable. Avoid unrelated "
            "tabs, pages, or features."
        )
        steps = (
            f"Add or extend the page/render code ({page_ref}).",
            f"Add the static JS/CSS for the feature ({js_ref}, {css_ref}).",
            "Wire the new surface into an existing route/navigation entry point.",
            f"Add or update the feature test ({tests[0] if tests else 'a feature test'}).",
        )
    elif task_type == TASK_TEST_UPDATE:
        strategy = (
            "Check the implementation's actual expectation first, then update or "
            "add the focused test. Do not change implementation code unless the "
            "behaviour is intentionally changing."
        )
        steps = (
            "Read the implementation the test covers and its current behaviour.",
            "Update the focused test to match intended behaviour.",
        )
    elif task_type == TASK_SAFETY_VALIDATION_FIX:
        strategy = (
            "Fix the validation false positive by narrowing the policy precisely. "
            "Do not weaken or remove approval/safety core protections. Add a "
            "regression test proving the legitimate case passes and the unsafe "
            "case still fails closed."
        )
        steps = (
            "Locate the over-broad validation/policy rule causing the false positive.",
            "Narrow the rule to the specific legitimate case only.",
            "Add a regression test for both the allowed and the still-refused case.",
        )
    elif task_type == TASK_BACKEND_FEATURE:
        strategy = (
            "Implement the change in the service/API and its data/state, then "
            "update the tests and any UI caller. Prefer minimal targeted edits to "
            "the real implementation surface, not test-only changes."
        )
        steps = (
            "Edit the service/API implementation for the new behaviour.",
            "Update the data model/state it depends on if needed.",
            "Update or add the covering tests and any UI caller.",
        )
    elif task_type == TASK_REFACTOR:
        strategy = (
            "Refactor without changing behaviour: keep the public surface and "
            "tests green, and confirm callers still work."
        )
        steps = (
            "Make the structural change in the target module.",
            "Update call sites so behaviour is unchanged.",
            "Run the covering tests to confirm no behaviour change.",
        )
    elif task_type == TASK_REPO_MAP_OR_FREESTYLE_MACHINERY:
        strategy = (
            "Modify the named freestyle/repo-map machinery module(s) only, "
            "preserve every safety boundary and the module purity contracts, and "
            "update the module's own tests."
        )
        steps = (
            "Edit the named machinery module.",
            "Preserve safety/purity contracts and source-scan invariants.",
            "Update the module's tests.",
        )
    elif task_type == TASK_APPROVAL_SENSITIVE_REFUSAL:
        strategy = (
            "This request touches approval-sensitive behaviour. Do not weaken any "
            "approval protection; if the change would loosen the approval "
            "boundary, it should fail closed rather than proceed."
        )
        steps = (
            "Confirm whether the change loosens any approval protection.",
            "Refuse or narrow rather than weaken the approval boundary.",
        )
    else:
        strategy = (
            "Make a minimal, targeted implementation change to the real surface "
            "the goal names, and update its covering test. Avoid unrelated files."
        )
        steps = (
            "Identify the real implementation surface for the goal.",
            "Make the minimal change and update the covering test.",
        )
    return _clip(strategy), tuple(step for step in steps if step)[:MAX_STRATEGY_STEPS]


def _expected_files(
    task_type: str,
    intake: Mapping[str, object],
    surfaces: Mapping[str, object],
    memory: Mapping[str, object],
    forbidden: Sequence[str],
    context_plan: ContextPlan | None,
) -> tuple[str, ...]:
    ordered: list[str] = list(_bounded_paths(intake.get("named_implementation_files")))
    if task_type in {TASK_WEB_UI_FEATURE, TASK_WEB_UI_REPAIR}:
        for key in ("pages", "routes", "js", "css", "tests"):
            ordered.extend(_bounded_paths(surfaces.get(key)))
    elif task_type == TASK_BACKEND_FEATURE:
        for key in ("backend_services", "data_surfaces", "pages", "tests"):
            ordered.extend(_bounded_paths(surfaces.get(key)))
    elif task_type == TASK_TEST_UPDATE:
        ordered.extend(_bounded_paths(intake.get("named_test_files")))
        ordered.extend(_bounded_paths(surfaces.get("tests")))
    elif task_type == TASK_REPO_MAP_OR_FREESTYLE_MACHINERY:
        ordered.extend(_exact_freestyle_priority_paths(context_plan))
        ordered.extend(_bounded_paths(surfaces.get("connected_files")))
    else:
        ordered.extend(_bounded_paths(surfaces.get("connected_files")))
    ordered.extend(_bounded_paths(memory.get("previous_proposal_likely_files")))
    ordered.extend(_bounded_paths(memory.get("previous_bundle_touched_files")))
    ordered.extend(_bounded_paths(surfaces.get("connected_files")))
    forbidden_set = {p.casefold() for p in forbidden}
    return _dedupe_paths(ordered, exclude=forbidden_set)


def _inspection_requirements(
    task_type: str,
    failed: Sequence[str],
    surfaces: Mapping[str, object],
    forbidden: Sequence[str],
    memory: Mapping[str, object],
) -> tuple[str, ...]:
    reqs: list[str] = []
    pages = _bounded_paths(surfaces.get("pages")) or _bounded_paths(
        surfaces.get("routes")
    )
    js = _bounded_paths(surfaces.get("js"))
    if FAILED_CSS_ONLY_PATCH in failed:
        page_ref = pages[0] if pages else "the page/render code"
        js_ref = js[0] if js else "the static JS asset"
        reqs.append(
            f"Confirm {js_ref} actually emits the DOM nodes and class names the "
            f"CSS targets, and that {page_ref} renders that structure, before "
            "editing any stylesheet."
        )
    if FAILED_FORBIDDEN_OR_GENERIC_TEST_TOUCHED in failed:
        tests = _bounded_paths(surfaces.get("tests"))
        focused = tests[0] if tests else "the feature-specific test file"
        reqs.append(
            f"Use the feature-specific test file ({focused}); do not edit a shared "
            "generic test file or a file the goal asked to leave untouched."
        )
    if FAILED_APPROVAL_VALIDATION_LITERAL in failed:
        reqs.append(
            "Inspect the approval/validation policy that refused the prior attempt "
            "and narrow it precisely; do not weaken the approval core."
        )
    if FAILED_OMITTED_RELEVANT_SURFACE in failed:
        connected = _bounded_paths(surfaces.get("connected_files"))
        if connected:
            reqs.append(
                "Consider the connected implementation surfaces the prior attempt "
                f"ignored: {_join(connected[:5])}."
            )
    return tuple(reqs)[:MAX_PLAN_LIST]


def _context_recommendations(
    expected: Sequence[str],
    *,
    inspection_paths: Sequence[str],
    forbidden: Sequence[str],
) -> tuple[str, ...]:
    forbidden_set = {p.casefold() for p in forbidden}
    return _dedupe_paths([*expected, *inspection_paths], exclude=forbidden_set)


def _surface_inspection_paths(surfaces: Mapping[str, object]) -> tuple[str, ...]:
    paths: list[str] = []
    for key in ("pages", "routes", "js", "css", "tests"):
        paths.extend(_bounded_paths(surfaces.get(key)))
    return tuple(paths)


def _non_goals(
    task_type: str, failed: Sequence[str], forbidden: Sequence[str]
) -> tuple[str, ...]:
    goals: list[str] = [f"Do not modify {path}." for path in forbidden]
    if task_type in {TASK_WEB_UI_FEATURE, TASK_WEB_UI_REPAIR}:
        goals.append("Do not modify unrelated pages, tabs, or features.")
    if FAILED_CSS_ONLY_PATCH in failed:
        goals.append("Do not submit another CSS-only patch.")
    if FAILED_TESTS_ONLY_CHANGE in failed:
        goals.append("Do not change only tests for a goal that needs implementation.")
    goals.append(
        "Do not weaken approval, validation, path-safety, or other safety boundaries."
    )
    return tuple(dict.fromkeys(goals))[:MAX_PLAN_LIST]


def _safety_boundaries(task_type: str) -> tuple[str, ...]:
    boundaries = list(_SAFETY_BOUNDARIES_COMMON)
    if task_type in {TASK_SAFETY_VALIDATION_FIX, TASK_APPROVAL_SENSITIVE_REFUSAL}:
        boundaries.extend(_SAFETY_BOUNDARIES_APPROVAL)
    return tuple(boundaries)


def _risk_flags(
    task_type: str,
    failed: Sequence[str],
    provenance: Mapping[str, object],
    surfaces: Mapping[str, object],
    forbidden: Sequence[str],
) -> tuple[str, ...]:
    flags: list[str] = []
    if not provenance.get("available"):
        flags.append("no_repo_map")
    elif provenance.get("stale") or provenance.get("fallback"):
        flags.append("repo_map_stale_fallback")
    if _bounded_paths(surfaces.get("approval_sensitive")):
        flags.append("approval_sensitive_surface_present")
    if FAILED_CSS_ONLY_PATCH in failed:
        flags.append("prior_css_only_failed")
    if forbidden:
        flags.append("human_forbidden_paths_present")
    if task_type == TASK_UNKNOWN:
        flags.append("unclassified_task")
    return tuple(dict.fromkeys(flags))


def _review_signals(
    memory: Mapping[str, object], observed_feedback: str | None
) -> tuple[str, ...]:
    signals: list[str] = []
    signals.extend(_bounded_strings(memory.get("previous_ui_review_warnings")))
    signals.extend(_bounded_strings(memory.get("previous_repo_map_review_warnings")))
    signals.extend(_bounded_strings(memory.get("previous_warnings")))
    if observed_feedback:
        signals.append("observed runtime/UI feedback was reported by a human")
    return tuple(dict.fromkeys(signals))[:MAX_PLAN_LIST]


def _plan_warnings(
    task_type: str,
    failed: Sequence[str],
    provenance: Mapping[str, object],
    memory: Mapping[str, object],
) -> tuple[str, ...]:
    warnings: list[str] = []
    if not provenance.get("available"):
        warnings.append(
            "No repo map was available; the affected-surface map is derived from "
            "the goal and prior attempt only and may be incomplete."
        )
    prior_plan = memory.get("previous_implementation_plan")
    if isinstance(prior_plan, Mapping):
        prior_failed = _bounded_strings(prior_plan.get("failed_strategy_patterns"))
        prior_strategy = _clip(prior_plan.get("recommended_strategy"))
        if (
            FAILED_CSS_ONLY_PATCH in failed
            and prior_strategy
            and (
                "css-only" in prior_strategy.casefold()
                or "wiring" in prior_strategy.casefold()
            )
        ):
            warnings.append(
                "A prior plan already recommended a non-CSS repair but the bundle "
                "changed only CSS; do not repeat the CSS-only strategy."
            )
        warnings.extend(
            f"Prior plan already flagged failed strategy: {pattern}"
            for pattern in prior_failed
            if pattern in failed
        )
    return tuple(dict.fromkeys(warnings))[:MAX_PLAN_LIST]


def _validation_plan(
    intake: Mapping[str, object],
    surfaces: Mapping[str, object],
    expected: Sequence[str],
) -> dict[str, object]:
    tests = _dedupe_paths(
        [
            *_bounded_paths(surfaces.get("tests")),
            *_bounded_paths(intake.get("named_test_files")),
            *(path for path in expected if is_test_path(path)),
        ]
    )
    return {
        "recommended_checks": ["pytest-freestyle", "ruff-check"],
        "tests_to_run": list(tests),
    }


def _prior_attempts(memory: Mapping[str, object]) -> dict[str, object]:
    touched = _bounded_paths(memory.get("previous_bundle_touched_files"))
    likely = _bounded_paths(memory.get("previous_proposal_likely_files"))
    if not touched and not likely:
        return {}
    prior_plan = memory.get("previous_implementation_plan")
    prior_strategy = None
    prior_task = memory.get("previous_task_type")
    if isinstance(prior_plan, Mapping):
        prior_strategy = _clip(prior_plan.get("recommended_strategy"))
        prior_task = prior_task or prior_plan.get("task_type")
    return {
        "touched_files": list(touched),
        "likely_files": list(likely),
        "prior_task_type": prior_task if isinstance(prior_task, str) else None,
        "prior_recommended_strategy": prior_strategy,
    }


def _repo_map_context(
    surfaces: Mapping[str, object], provenance: Mapping[str, object]
) -> dict[str, object]:
    return {
        "available": bool(provenance.get("available")),
        "scope": provenance.get("scope"),
        "stale": bool(provenance.get("stale")),
        "fallback": bool(provenance.get("fallback")),
        "indexed_at": provenance.get("indexed_at"),
        "schema_version": provenance.get("schema_version"),
        "connected_files": list(_bounded_paths(surfaces.get("connected_files"))),
        "graph_file_count": int(surfaces.get("graph_file_count") or 0),
    }


def _bounded_surfaces(surfaces: Mapping[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key in (
        "routes",
        "pages",
        "css",
        "js",
        "tests",
        "backend_services",
        "data_surfaces",
        "static_assets",
        "approval_sensitive",
        "connected_files",
    ):
        out[key] = list(_bounded_paths(surfaces.get(key)))
    pipeline = surfaces.get("pipeline")
    out["pipeline"] = _bounded_strings(pipeline)
    return out


# --------------------------------------------------------------------------- #
# Small pure utilities.
# --------------------------------------------------------------------------- #
def _primary_feature(
    surfaces: Mapping[str, object], goal: str, title: str | None
) -> str | None:
    feature = surfaces.get("primary_feature")
    if isinstance(feature, str) and feature.strip():
        return feature.strip()[:MAX_FEATURE_CHARS]
    for token in _tokens(_text(goal, title)):
        if len(token) >= 4 and token not in _FEATURE_STOP_TOKENS:
            return token[:MAX_FEATURE_CHARS]
    return None


def _provenance_brief(repo_map_context: Mapping[str, object]) -> dict[str, object]:
    return {
        "available": bool(repo_map_context.get("available")),
        "scope": repo_map_context.get("scope"),
        "stale": bool(repo_map_context.get("stale")),
        "indexed_at": repo_map_context.get("indexed_at"),
        "schema_version": repo_map_context.get("schema_version"),
    }


def _strategy_mentions_wiring(strategy: object) -> bool:
    if not isinstance(strategy, str):
        return False
    lowered = strategy.casefold()
    return "wiring" in lowered or ("dom" in lowered and "css" in lowered)


def _exact_freestyle_priority_paths(
    context_plan: ContextPlan | None,
) -> tuple[str, ...]:
    if context_plan is None:
        return ()
    paths = [
        pattern
        for pattern in context_plan.priority_patterns
        if "*" not in pattern
        and pattern.startswith(("app/dev/freestyle_", "app/dev/repo_map", "tests/"))
    ]
    return _dedupe_paths(paths)


def _intents(intake: Mapping[str, object]) -> frozenset[str]:
    raw = intake.get("intent_categories")
    if isinstance(raw, (list, tuple)):
        return frozenset(str(item) for item in raw)
    return frozenset()


def _text(goal: str, title: str | None) -> str:
    parts = [goal if isinstance(goal, str) else ""]
    if isinstance(title, str):
        parts.append(title)
    return " ".join(parts).casefold()


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(text.casefold()))


def _bounded_paths(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, str):
            continue
        candidate = raw.strip().replace("\\", "/")
        key = candidate.casefold()
        if candidate and key not in seen:
            seen.add(key)
            ordered.append(candidate)
        if len(ordered) >= MAX_PLAN_LIST:
            break
    return tuple(ordered)


def _bounded_strings(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for raw in value:
        if isinstance(raw, str) and raw.strip():
            out.append(_clip(raw))
        if len(out) >= MAX_PLAN_LIST:
            break
    return out


def _dedupe_paths(
    paths: Sequence[str], *, exclude: set[str] | None = None
) -> tuple[str, ...]:
    blocked = exclude or set()
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        if not isinstance(raw, str):
            continue
        candidate = raw.strip().replace("\\", "/")
        key = candidate.casefold()
        if not candidate or key in seen or key in blocked:
            continue
        seen.add(key)
        ordered.append(candidate)
        if len(ordered) >= MAX_PLAN_LIST:
            break
    return tuple(ordered)


def _join(paths: Sequence[str]) -> str:
    return ", ".join(str(path) for path in paths)


def _clip(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:MAX_FIELD_CHARS]
