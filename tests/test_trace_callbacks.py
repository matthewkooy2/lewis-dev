"""Regression tests for injected-callback handling and JS date effects.

Covers the issues demonstrated while tracing the TFG Portal:
1. Parameters/destructured option bindings are injected callbacks, never
   matched workspace-wide by name.
2. Caller-supplied callbacks are followed per call path, labeled inferred.
3. Only clock-dependent Date usage (`new Date()`, `Date.now()`) counts as a
   time effect; `new Date(value)` / `new Date(y, m, d)` are deterministic.
4. Review mode includes bounded caller/callee callback context snippets.
"""

from pathlib import Path

from app.trace.js_analysis import analyze_js_file
from app.trace.service import CodeTracer


LOGIC_JS = (
    "export const getToday = (now = new Date()) => now.toISOString();\n"
    "\n"
    "export const normalize = (value) => new Date(value).toISOString();\n"
    "\n"
    "export const fromParts = (y, m, d) => new Date(y, m - 1, d);\n"
    "\n"
    "export const eligibleRow = (row) => row?.status !== \"paid\";\n"
    "\n"
    "export const pickNext = (rows, options = {}) => {\n"
    "  const {\n"
    "    getDue = (row) => row?.due ?? row?.dueDate,\n"
    "    isEligible = () => true,\n"
    "    today = getToday(),\n"
    "  } = options;\n"
    "  let winner = null;\n"
    "  for (const row of rows) {\n"
    "    if (!isEligible(row)) continue;\n"
    "    const due = normalize(getDue(row));\n"
    "    if (due >= today && (!winner || due < winner.due)) winner = row;\n"
    "  }\n"
    "  return winner;\n"
    "};\n"
)

PAGE_A_JS = (
    "import { pickNext, eligibleRow } from \"../utils/logic\";\n"
    "\n"
    "const getDue = (row) => row?.due;\n"
    "\n"
    "export const nextForA = (rows) => pickNext(rows, {\n"
    "  getDue,\n"
    "  isEligible: eligibleRow,\n"
    "});\n"
)

PAGE_B_JS = (
    "import { pickNext } from \"../utils/logic\";\n"
    "\n"
    "const getDue = (row) => row?.dueDate;\n"
    "\n"
    "export const nextForB = (rows) => pickNext(rows, { getDue });\n"
)

# Same-name definitions that must never be matched to the injected bindings.
DECOY_JS = (
    "export const getDue = (row) => \"decoy\";\n"
    "export const isEligible = (row) => \"decoy\";\n"
)


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _callback_project(tmp_path: Path) -> Path:
    root = tmp_path / "cbproj"
    _write(root, "utils/logic.js", LOGIC_JS)
    _write(root, "pages/a.js", PAGE_A_JS)
    _write(root, "pages/b.js", PAGE_B_JS)
    _write(root, "server/decoy.js", DECOY_JS)
    return root


# ---- issue 1: injected and locally bound callbacks -----------------------------


def test_destructured_option_defaults_become_local_bindings():
    index = analyze_js_file("utils/logic.js", LOGIC_JS, "javascript")
    calls = {
        (call.caller_qualname, call.head): call
        for call in index.calls
        if call.caller_qualname == "pickNext"
    }
    is_eligible = calls[("pickNext", "isEligible")]
    get_due = calls[("pickNext", "getDue")]
    assert is_eligible.is_local_binding is True
    assert is_eligible.local_default_line == 12
    assert get_due.is_local_binding is True
    assert get_due.local_default_line == 11
    # Real module-level callees keep resolving normally.
    get_today = calls[("pickNext", "getToday")]
    assert get_today.is_local_binding is False


def test_parameters_are_local_bindings_but_local_arrows_still_resolve(tmp_path):
    root = tmp_path / "params"
    _write(
        root,
        "mod.js",
        "export const outer = (handler) => {\n"
        "  const helper = (x) => x + 1;\n"
        "  handler(1);\n"
        "  return helper(2);\n"
        "};\n"
        "export const handler = () => \"module-level decoy\";\n",
    )
    tracer = CodeTracer(root)
    result = tracer.trace("outer", mode="quick").result
    edges = {edge.callee_text: edge for edge in result.callees}
    # The `handler` parameter shadows the module-level `handler`.
    assert edges["handler"].resolution == "dynamic"
    assert "injected" in edges["handler"].note
    assert edges["helper"].resolution == "confirmed"
    assert edges["helper"].target_qualname == "outer.helper"


def test_injected_callbacks_are_not_matched_to_same_name_workspace_functions(
    tmp_path,
):
    tracer = CodeTracer(_callback_project(tmp_path))
    result = tracer.trace("pickNext", mode="quick").result
    injected = [
        edge for edge in result.callees if edge.callee_text in {"getDue", "isEligible"}
    ]
    assert injected, "injected callback invocations must appear as callees"
    for edge in injected:
        assert edge.resolution in {"dynamic", "inferred"}
        assert edge.resolution != "ambiguous"
        assert "decoy" not in (edge.target_path or "")
    dynamic = [edge for edge in injected if edge.resolution == "dynamic"]
    assert any(
        "default implementation at utils/logic.js:" in edge.note for edge in dynamic
    )


# ---- issue 2: call-path-specific callback propagation ---------------------------


def test_caller_supplied_callbacks_create_path_specific_inferred_edges(tmp_path):
    tracer = CodeTracer(_callback_project(tmp_path))
    result = tracer.trace("pickNext", mode="deep").result

    inferred = [
        edge
        for edge in result.callees
        if edge.resolution == "inferred" and "injected" in edge.note
    ]
    by_target = {
        (edge.callee_text, edge.target_path, edge.target_qualname): edge.note
        for edge in inferred
    }
    assert ("isEligible", "utils/logic.js", "eligibleRow") in by_target
    assert ("getDue", "pages/a.js", "getDue") in by_target
    assert ("getDue", "pages/b.js", "getDue") in by_target
    # Each edge names its specific supplier — never a global claim.
    assert "pages/a.js:5" in by_target[("isEligible", "utils/logic.js", "eligibleRow")]
    assert "pages/b.js:5" in by_target[("getDue", "pages/b.js", "getDue")]

    caller_notes = " ".join(edge.note for edge in result.callers)
    assert "supplies callbacks" in caller_notes


def test_deep_trace_from_caller_follows_only_its_own_callbacks(tmp_path):
    tracer = CodeTracer(_callback_project(tmp_path))
    result = tracer.trace("nextForB", mode="deep").result
    flattened = " ".join(" ".join(path) for path in result.call_paths)
    assert "pickNext" in flattened
    assert "pages/b.js::getDue" in flattened
    # nextForB never supplies isEligible, so eligibleRow must not appear.
    assert "eligibleRow" not in flattened

    result_a = tracer.trace("nextForA", mode="deep").result
    flattened_a = " ".join(" ".join(path) for path in result_a.call_paths)
    assert "eligibleRow" in flattened_a
    assert "pages/a.js::getDue" in flattened_a
    assert "pages/b.js::getDue" not in flattened_a


# ---- issue 3: JS date-effect accuracy --------------------------------------------


def test_only_clock_dependent_date_usage_is_a_time_effect():
    source = (
        "export const stamp = () => new Date();\n"
        "export const nowMs = () => Date.now();\n"
        "export const convert = (value) => new Date(value);\n"
        "export const build = (y, m, d) => new Date(y, m - 1, d);\n"
    )
    index = analyze_js_file("dates.js", source, "javascript")
    effects = {
        (effect.qualname, effect.detail)
        for effect in index.side_effects
        if effect.category == "time_or_randomness"
    }
    assert ("stamp", "new Date()") in effects
    assert ("nowMs", "Date.now()") in effects
    assert not any(qualname in {"convert", "build"} for qualname, _ in effects)


def test_new_date_in_parameter_default_is_attributed_to_the_function():
    index = analyze_js_file("utils/logic.js", LOGIC_JS, "javascript")
    time_effects = [
        effect
        for effect in index.side_effects
        if effect.category == "time_or_randomness"
    ]
    assert [(e.qualname, e.line) for e in time_effects] == [("getToday", 1)]


def test_deterministic_date_construction_leaves_trace_effects_honest(tmp_path):
    tracer = CodeTracer(_callback_project(tmp_path))
    result = tracer.trace("pickNext", mode="deep").result
    transitive = {
        (effect.qualname, effect.detail)
        for effect in result.transitive_side_effects
        if effect.category == "time_or_randomness"
    }
    # getToday genuinely reads the clock; normalize/fromParts do not.
    assert transitive == {("getToday", "new Date()")}


# ---- issue 4: review caller/callee evidence ---------------------------------------


def test_review_includes_bounded_callback_context_snippets(tmp_path):
    tracer = CodeTracer(_callback_project(tmp_path))
    outcome = tracer.trace("pickNext", mode="review")
    evidence = outcome.result.evidence
    paths = {snippet.path for snippet in evidence}
    assert "pages/a.js" in paths, "supplier call-site context expected"
    assert "pages/b.js" in paths, "supplier call-site context expected"
    for snippet in evidence:
        assert snippet.end_line - snippet.start_line + 1 <= 6
    a_snippet = next(s for s in evidence if s.path == "pages/a.js")
    assert "isEligible: eligibleRow" in a_snippet.text
    assert "Evidence:" in outcome.report
    assert "pages/a.js" in outcome.report


def test_review_without_callbacks_keeps_prior_evidence_behavior(tmp_path):
    root = tmp_path / "plain"
    _write(
        root,
        "app/x.py",
        "def f():\n    try:\n        g()\n    except:\n        pass\n",
    )
    tracer = CodeTracer(root)
    result = tracer.trace("f", mode="review").result
    assert result.findings
    assert all(snippet.path == "app/x.py" for snippet in result.evidence)
