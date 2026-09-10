"""Deterministic implementation review built on a deep trace.

Runs mechanical checks over the traced symbol's source, resolved call graph,
side effects, error handling, and test links, and emits findings that cite
file and line. Every finding carries a severity (issue/risk/note) and a
confidence (confirmed/heuristic): ``confirmed`` findings are structurally
certain from static analysis; ``heuristic`` findings are risks or smells the
reviewer should verify. The checks never execute analyzed code — dynamic
behavior, runtime values, and third-party internals are out of scope and the
report says so.
"""

from __future__ import annotations

import ast
import re

from app.trace.engine import TraceEngine
from app.trace.model import (
    EFFECT_GLOBAL_STATE,
    KIND_ASYNC_FUNCTION,
    KIND_ASYNC_METHOD,
    KIND_COMPONENT,
    KIND_HOOK,
    LANGUAGE_PYTHON,
    ModuleIndex,
    ReviewFinding,
    Snippet,
    SymbolDefinition,
    TraceResult,
)
from app.secret_redaction import redact_secret_values


CATEGORY_CORRECTNESS = "correctness"
CATEGORY_ERROR_HANDLING = "error_handling"
CATEGORY_SECURITY = "security"
CATEGORY_SIDE_EFFECTS = "side_effects"
CATEGORY_MAINTAINABILITY = "maintainability"
CATEGORY_CONSISTENCY = "consistency"
CATEGORY_TESTING = "testing"

SEVERITY_ISSUE = "issue"
SEVERITY_RISK = "risk"
SEVERITY_NOTE = "note"

CONFIDENCE_CONFIRMED = "confirmed"
CONFIDENCE_HEURISTIC = "heuristic"

MAX_FUNCTION_LINES = 90
MAX_EVIDENCE_SNIPPETS = 8

_SQL_KEYWORDS = re.compile(r"\b(select|insert|update|delete|create|drop)\b", re.I)
_SQL_DYNAMIC = re.compile(
    r"""execute\w*\s*\(\s*(?:f["']|["'][^"']*["']\s*[%+]|.*\.format\()"""
)
_SHELL_TRUE = re.compile(r"\bshell\s*=\s*True\b")
_LOOSE_EQUALITY = re.compile(r"[^=!<>]==[^=]|[^!]!=[^=]")
_NONE_COMPARISON = re.compile(r"[=!]=\s*None\b")
_TODO_MARKER = re.compile(r"#\s*(TODO|FIXME|XXX|HACK)|//\s*(TODO|FIXME|XXX|HACK)", re.I)
_VAR_DECL = re.compile(r"(^|[^\w.])var\s+[A-Za-z_$]")
_HARDCODED_SECRET = re.compile(
    r"(?:secret|token|password|passwd|api[_-]?key|private[_-]?key)\s*[:=]\s*['\"][^'\"]{6,}",
    re.I,
)
_QUERY_NAME = re.compile(r"^(get_|is_|has_|list_|read_|find_|fetch_|load_)")
_WRITE_EFFECTS = frozenset({"filesystem", "database", "process", "environment"})
_PY_MUTABLE_DEFAULT = (ast.List, ast.Dict, ast.Set)


def review_implementation(engine: TraceEngine, target: SymbolDefinition) -> TraceResult:
    result = engine.deep_trace(target)
    result.mode = "review"
    if not result.ok:
        return result
    module = engine.index.module_for(target.path)
    source = engine.index.workspace.read_source(target.path) or ""
    lines = source.splitlines()
    findings: list[ReviewFinding] = []

    if module is not None:
        _check_error_handling(findings, module, target, result)
        _check_calls_and_security(findings, engine, module, target, lines)
        _check_side_effects(findings, target, result)
        _check_maintainability(findings, target, lines)
        _check_consistency(findings, module, target)
        _check_testing(findings, target, result)
        if module.language == LANGUAGE_PYTHON:
            _check_python_specifics(findings, target, source)
        else:
            _check_js_specifics(findings, engine, module, target, lines)

    ranked = sorted(findings, key=_finding_rank)
    result.findings = tuple(ranked[: engine.limits.max_findings])
    if len(ranked) > engine.limits.max_findings:
        result.truncations = (
            *result.truncations,
            f"Findings truncated at {engine.limits.max_findings}.",
        )
    context = _callback_context_snippets(engine, target)
    finding_evidence = _evidence_snippets(lines, target, result.findings)
    if context:
        finding_evidence = finding_evidence[
            : max(0, MAX_EVIDENCE_SNIPPETS - len(context))
        ]
    result.evidence = (*finding_evidence, *context)
    result.warnings = (
        *result.warnings,
        "Static review only: dynamic behavior, runtime values, and "
        "third-party internals were not executed or verified.",
    )
    return result


def _finding_rank(finding: ReviewFinding) -> tuple[int, int, int]:
    severity_rank = {SEVERITY_ISSUE: 0, SEVERITY_RISK: 1, SEVERITY_NOTE: 2}
    confidence_rank = {CONFIDENCE_CONFIRMED: 0, CONFIDENCE_HEURISTIC: 1}
    return (
        severity_rank.get(finding.severity, 3),
        confidence_rank.get(finding.confidence, 2),
        finding.line,
    )


def _target_lines(target: SymbolDefinition, lines: list[str]) -> list[tuple[int, str]]:
    start = max(1, target.line)
    end = min(len(lines), max(target.end_line, target.line))
    return [(number, lines[number - 1]) for number in range(start, end + 1)]


# ---- checks -----------------------------------------------------------------


def _check_error_handling(
    findings: list[ReviewFinding],
    module: ModuleIndex,
    target: SymbolDefinition,
    result: TraceResult,
) -> None:
    for info in module.error_handling:
        if info.qualname != target.qualname and not info.qualname.startswith(
            target.qualname + "."
        ):
            continue
        for line in info.bare_catch_lines:
            findings.append(
                ReviewFinding(
                    category=CATEGORY_ERROR_HANDLING,
                    severity=SEVERITY_ISSUE,
                    confidence=CONFIDENCE_CONFIRMED,
                    path=target.path,
                    line=line,
                    message=(
                        "Bare `except:` catches everything including "
                        "SystemExit/KeyboardInterrupt; catch specific exceptions."
                    ),
                )
            )
        for line in info.swallowed_catch_lines:
            findings.append(
                ReviewFinding(
                    category=CATEGORY_ERROR_HANDLING,
                    severity=SEVERITY_ISSUE,
                    confidence=CONFIDENCE_CONFIRMED,
                    path=target.path,
                    line=line,
                    message=(
                        "Exception handler swallows the error without logging, "
                        "re-raising, or recovering."
                    ),
                )
            )
    # Raising callees whose exceptions have no matching catch in the target
    # (static name comparison only).
    target_info = next(
        (info for info in result.error_handling if info.qualname == target.qualname),
        None,
    )
    caught_text = " ".join(target_info.catches) if target_info else ""
    reported: set[str] = set()
    for edge in result.callees:
        if edge.resolution not in {"confirmed", "inferred"}:
            continue
        callee_info = next(
            (
                info
                for info in result.error_handling
                if info.qualname == edge.target_qualname and info.raises
            ),
            None,
        )
        if callee_info is None:
            continue
        for raised in callee_info.raises[:3]:
            if raised in {"re-raise"} or raised in reported:
                continue
            reported.add(raised)
            if raised.split(".")[-1] not in caught_text and "except" not in raised:
                findings.append(
                    ReviewFinding(
                        category=CATEGORY_ERROR_HANDLING,
                        severity=SEVERITY_NOTE,
                        confidence=CONFIDENCE_HEURISTIC,
                        path=edge.caller_path,
                        line=edge.line,
                        message=(
                            f"Callee `{edge.callee_text}` can raise {raised}; no "
                            "matching handler in this function (static name "
                            "check only — it may be handled by callers)."
                        ),
                    )
                )
        if len(reported) >= 4:
            break


def _check_calls_and_security(
    findings: list[ReviewFinding],
    engine: TraceEngine,
    module: ModuleIndex,
    target: SymbolDefinition,
    lines: list[str],
) -> None:
    scope = {target.qualname}
    prefix = target.qualname + "."
    for definition in module.definitions:
        if definition.qualname.startswith(prefix):
            scope.add(definition.qualname)
    for call in module.calls:
        if call.caller_qualname not in scope:
            continue
        line_text = lines[call.line - 1] if 0 < call.line <= len(lines) else ""
        head = call.head
        chain_text = ".".join([head, *call.attr_chain]) if head else ""
        if head in {"eval", "exec"} and not call.attr_chain:
            findings.append(
                ReviewFinding(
                    category=CATEGORY_SECURITY,
                    severity=SEVERITY_ISSUE,
                    confidence=CONFIDENCE_CONFIRMED,
                    path=target.path,
                    line=call.line,
                    message=f"`{head}()` executes arbitrary code; avoid or isolate it.",
                )
            )
        if chain_text in {"pickle.load", "pickle.loads"}:
            findings.append(
                ReviewFinding(
                    category=CATEGORY_SECURITY,
                    severity=SEVERITY_RISK,
                    confidence=CONFIDENCE_CONFIRMED,
                    path=target.path,
                    line=call.line,
                    message=(
                        "`pickle.load(s)` deserializes arbitrary objects; unsafe "
                        "on untrusted input."
                    ),
                )
            )
        if chain_text == "yaml.load":
            findings.append(
                ReviewFinding(
                    category=CATEGORY_SECURITY,
                    severity=SEVERITY_RISK,
                    confidence=CONFIDENCE_HEURISTIC,
                    path=target.path,
                    line=call.line,
                    message=(
                        "`yaml.load` without SafeLoader can construct arbitrary "
                        "objects; verify a safe loader is passed."
                    ),
                )
            )
        if chain_text == "os.system" or (
            head == "subprocess" and _SHELL_TRUE.search(line_text)
        ):
            findings.append(
                ReviewFinding(
                    category=CATEGORY_SECURITY,
                    severity=SEVERITY_ISSUE,
                    confidence=CONFIDENCE_CONFIRMED,
                    path=target.path,
                    line=call.line,
                    message=(
                        "Shell execution (`os.system`/`shell=True`) is injectable "
                        "if any argument derives from input."
                    ),
                )
            )
        if (
            call.attr_chain
            and call.attr_chain[-1].startswith("execute")
            and _SQL_KEYWORDS.search(line_text)
            and _SQL_DYNAMIC.search(line_text)
        ):
            findings.append(
                ReviewFinding(
                    category=CATEGORY_SECURITY,
                    severity=SEVERITY_RISK,
                    confidence=CONFIDENCE_HEURISTIC,
                    path=target.path,
                    line=call.line,
                    message=(
                        "SQL appears to be built with f-string/format/concat; "
                        "use parameterized queries."
                    ),
                )
            )
        if head == "Function" and call.is_new:
            findings.append(
                ReviewFinding(
                    category=CATEGORY_SECURITY,
                    severity=SEVERITY_ISSUE,
                    confidence=CONFIDENCE_CONFIRMED,
                    path=target.path,
                    line=call.line,
                    message="`new Function(...)` compiles arbitrary code at runtime.",
                )
            )
        # Await discipline: calls resolved to workspace async definitions.
        resolved = engine.resolve_call(module, call)
        if (
            resolved.definition is not None
            and resolved.definition.kind in {KIND_ASYNC_FUNCTION, KIND_ASYNC_METHOD}
            and not call.is_await
            and not call.jsx_component
            and not call.is_callback_reference
        ):
            findings.append(
                ReviewFinding(
                    category=CATEGORY_CORRECTNESS,
                    severity=SEVERITY_RISK,
                    confidence=CONFIDENCE_HEURISTIC,
                    path=target.path,
                    line=call.line,
                    message=(
                        f"`{call.callee_text}` resolves to async "
                        f"`{resolved.definition.qualname}` but is not awaited here "
                        "(verify the promise/coroutine is handled)."
                    ),
                )
            )
    if _HARDCODED_SECRET.search(
        "\n".join(text for _, text in _target_lines(target, lines))
    ):
        findings.append(
            ReviewFinding(
                category=CATEGORY_SECURITY,
                severity=SEVERITY_RISK,
                confidence=CONFIDENCE_HEURISTIC,
                path=target.path,
                line=target.line,
                message=(
                    "A hardcoded credential-like literal appears in this "
                    "function; move secrets to configuration."
                ),
            )
        )


def _check_side_effects(
    findings: list[ReviewFinding],
    target: SymbolDefinition,
    result: TraceResult,
) -> None:
    direct_categories = {effect.category for effect in result.side_effects}
    if _QUERY_NAME.match(target.name) and direct_categories & _WRITE_EFFECTS:
        effect = next(e for e in result.side_effects if e.category in _WRITE_EFFECTS)
        findings.append(
            ReviewFinding(
                category=CATEGORY_SIDE_EFFECTS,
                severity=SEVERITY_RISK,
                confidence=CONFIDENCE_HEURISTIC,
                path=effect.path,
                line=effect.line,
                message=(
                    f"`{target.name}` is named like a query but performs "
                    f"{effect.category} effects ({effect.detail})."
                ),
            )
        )
    for effect in result.side_effects:
        if effect.qualname is None:
            findings.append(
                ReviewFinding(
                    category=CATEGORY_SIDE_EFFECTS,
                    severity=SEVERITY_RISK,
                    confidence=CONFIDENCE_CONFIRMED,
                    path=effect.path,
                    line=effect.line,
                    message=(
                        f"Module-level {effect.category} effect ({effect.detail}) "
                        "runs at import time."
                    ),
                )
            )
    for effect in result.side_effects:
        if effect.category == EFFECT_GLOBAL_STATE:
            findings.append(
                ReviewFinding(
                    category=CATEGORY_SIDE_EFFECTS,
                    severity=SEVERITY_NOTE,
                    confidence=CONFIDENCE_CONFIRMED,
                    path=effect.path,
                    line=effect.line,
                    message=f"Mutates shared module state: {effect.detail}.",
                )
            )
    hidden = {
        effect.category
        for effect in result.transitive_side_effects
        if effect.category in _WRITE_EFFECTS
    } - direct_categories
    if hidden:
        listed = ", ".join(sorted(hidden))
        findings.append(
            ReviewFinding(
                category=CATEGORY_SIDE_EFFECTS,
                severity=SEVERITY_NOTE,
                confidence=CONFIDENCE_HEURISTIC,
                path=target.path,
                line=target.line,
                message=(
                    f"Transitive callees perform {listed} effects not visible "
                    "in this function's body (see deep trace)."
                ),
            )
        )


def _check_maintainability(
    findings: list[ReviewFinding], target: SymbolDefinition, lines: list[str]
) -> None:
    length = max(0, target.end_line - target.line + 1)
    if length > MAX_FUNCTION_LINES and target.kind != "class":
        findings.append(
            ReviewFinding(
                category=CATEGORY_MAINTAINABILITY,
                severity=SEVERITY_NOTE,
                confidence=CONFIDENCE_CONFIRMED,
                path=target.path,
                line=target.line,
                message=(
                    f"Function body spans {length} lines; consider extracting helpers."
                ),
            )
        )
    if target.signature.count(",") >= 6:
        findings.append(
            ReviewFinding(
                category=CATEGORY_MAINTAINABILITY,
                severity=SEVERITY_NOTE,
                confidence=CONFIDENCE_CONFIRMED,
                path=target.path,
                line=target.line,
                message="Long parameter list; consider a parameter object/dataclass.",
            )
        )
    for number, text in _target_lines(target, lines):
        if _TODO_MARKER.search(text):
            findings.append(
                ReviewFinding(
                    category=CATEGORY_MAINTAINABILITY,
                    severity=SEVERITY_NOTE,
                    confidence=CONFIDENCE_CONFIRMED,
                    path=target.path,
                    line=number,
                    message="Unresolved TODO/FIXME marker in this implementation.",
                )
            )
            break


def _check_consistency(
    findings: list[ReviewFinding], module: ModuleIndex, target: SymbolDefinition
) -> None:
    siblings = [
        definition
        for definition in module.definitions
        if definition.parent == target.parent
        and definition.qualname != target.qualname
        and definition.kind == target.kind
    ]
    if len(siblings) >= 3 and module.language == LANGUAGE_PYTHON:
        documented = sum(1 for sibling in siblings if sibling.doc)
        if documented / len(siblings) >= 0.6 and not target.doc:
            findings.append(
                ReviewFinding(
                    category=CATEGORY_CONSISTENCY,
                    severity=SEVERITY_NOTE,
                    confidence=CONFIDENCE_CONFIRMED,
                    path=target.path,
                    line=target.line,
                    message=(
                        "Most sibling definitions have docstrings; this one does not."
                    ),
                )
            )
        annotated = sum(1 for sibling in siblings if "->" in sibling.signature)
        if (
            annotated / len(siblings) >= 0.6
            and target.signature
            and "->" not in target.signature
        ):
            findings.append(
                ReviewFinding(
                    category=CATEGORY_CONSISTENCY,
                    severity=SEVERITY_NOTE,
                    confidence=CONFIDENCE_CONFIRMED,
                    path=target.path,
                    line=target.line,
                    message=(
                        "Sibling functions annotate return types; this one does not."
                    ),
                )
            )


def _check_testing(
    findings: list[ReviewFinding],
    target: SymbolDefinition,
    result: TraceResult,
) -> None:
    direct = [link for link in result.tests if link.via == "calls target"]
    if not direct:
        severity = SEVERITY_RISK if target.exported else SEVERITY_NOTE
        findings.append(
            ReviewFinding(
                category=CATEGORY_TESTING,
                severity=severity,
                confidence=CONFIDENCE_CONFIRMED,
                path=target.path,
                line=target.line,
                message=(
                    "No test was found calling this symbol directly "
                    "(name-resolution based; parametrized/dynamic test calls "
                    "may be missed)."
                ),
            )
        )
        return
    target_info = next(
        (info for info in result.error_handling if info.qualname == target.qualname),
        None,
    )
    if target_info is not None and target_info.raises:
        findings.append(
            ReviewFinding(
                category=CATEGORY_TESTING,
                severity=SEVERITY_NOTE,
                confidence=CONFIDENCE_HEURISTIC,
                path=target.path,
                line=target_info.raise_lines[0]
                if target_info.raise_lines
                else target.line,
                message=(
                    f"{len(direct)} direct test(s) found; verify the error paths "
                    f"({', '.join(target_info.raises[:3])}) are covered as well."
                ),
            )
        )


def _check_python_specifics(
    findings: list[ReviewFinding], target: SymbolDefinition, source: str
) -> None:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return
    node = _find_def_node(tree, target)
    if node is None:
        return
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for default in node.args.defaults + [
            item for item in node.args.kw_defaults if item is not None
        ]:
            if isinstance(default, _PY_MUTABLE_DEFAULT) or (
                isinstance(default, ast.Call)
                and isinstance(default.func, ast.Name)
                and default.func.id in {"list", "dict", "set"}
            ):
                findings.append(
                    ReviewFinding(
                        category=CATEGORY_CORRECTNESS,
                        severity=SEVERITY_ISSUE,
                        confidence=CONFIDENCE_CONFIRMED,
                        path=target.path,
                        line=default.lineno,
                        message=(
                            "Mutable default argument is shared across calls; "
                            "use None and create inside."
                        ),
                    )
                )
    segment = ast.get_source_segment(source, node) or ""
    for offset, line_text in enumerate(segment.splitlines()):
        if _NONE_COMPARISON.search(line_text):
            findings.append(
                ReviewFinding(
                    category=CATEGORY_CONSISTENCY,
                    severity=SEVERITY_NOTE,
                    confidence=CONFIDENCE_CONFIRMED,
                    path=target.path,
                    line=node.lineno + offset,
                    message="Use `is None` / `is not None` instead of equality.",
                )
            )
            break


def _check_js_specifics(
    findings: list[ReviewFinding],
    engine: TraceEngine,
    module: ModuleIndex,
    target: SymbolDefinition,
    lines: list[str],
) -> None:
    body = _target_lines(target, lines)
    for number, text in body:
        stripped = re.sub(r"(['\"]).*?\1|===|!==", "", text)
        if _LOOSE_EQUALITY.search(stripped):
            findings.append(
                ReviewFinding(
                    category=CATEGORY_CORRECTNESS,
                    severity=SEVERITY_NOTE,
                    confidence=CONFIDENCE_HEURISTIC,
                    path=target.path,
                    line=number,
                    message="Loose equality (`==`/`!=`) coerces types; prefer `===`.",
                )
            )
            break
    for number, text in body:
        if _VAR_DECL.search(text):
            findings.append(
                ReviewFinding(
                    category=CATEGORY_CONSISTENCY,
                    severity=SEVERITY_NOTE,
                    confidence=CONFIDENCE_CONFIRMED,
                    path=target.path,
                    line=number,
                    message="`var` is function-scoped and hoisted; prefer const/let.",
                )
            )
            break
    for number, text in body:
        if "dangerouslySetInnerHTML" in text or "innerHTML" in text:
            findings.append(
                ReviewFinding(
                    category=CATEGORY_SECURITY,
                    severity=SEVERITY_RISK,
                    confidence=CONFIDENCE_CONFIRMED,
                    path=target.path,
                    line=number,
                    message=(
                        "Raw HTML injection sink (innerHTML/dangerouslySetInnerHTML); "
                        "ensure the content is sanitized."
                    ),
                )
            )
            break
    # Rules of hooks: workspace hooks called outside components/custom hooks.
    if target.kind not in {KIND_COMPONENT, KIND_HOOK}:
        for call in module.calls:
            if call.caller_qualname != target.qualname:
                continue
            if (
                len(call.head) > 3
                and call.head.startswith("use")
                and call.head[3].isupper()
            ):
                findings.append(
                    ReviewFinding(
                        category=CATEGORY_CORRECTNESS,
                        severity=SEVERITY_RISK,
                        confidence=CONFIDENCE_HEURISTIC,
                        path=target.path,
                        line=call.line,
                        message=(
                            f"Hook `{call.head}` is called from a function that "
                            "is neither a component nor a custom hook (rules of "
                            "hooks)."
                        ),
                    )
                )
                break


# ---- evidence -----------------------------------------------------------------


def _find_def_node(tree: ast.Module, target: SymbolDefinition) -> ast.AST | None:
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and getattr(node, "lineno", None) == target.line
        ):
            return node
    return None


MAX_CONTEXT_SUPPLIERS = 2
MAX_CONTEXT_CALLBACK_DEFS = 2
CONTEXT_SNIPPET_LINES = 6


def _callback_context_snippets(
    engine: TraceEngine, target: SymbolDefinition
) -> list[Snippet]:
    """Bounded caller/callee context when callbacks shape the target.

    Shows how important callers configure the target's injected callback
    options, plus the head of each supplied callback implementation —
    a few lines each, never whole files.
    """
    supplied = engine.caller_supplied_callbacks(target)
    if not supplied:
        return []
    snippets: list[Snippet] = []
    seen: set[tuple[str, int]] = set()
    suppliers = 0
    for item in supplied:
        if suppliers >= MAX_CONTEXT_SUPPLIERS:
            break
        key = (item.supplier_path, item.supplier_line)
        if key in seen:
            continue
        seen.add(key)
        snippet = _bounded_context_snippet(
            engine,
            item.supplier_path,
            item.supplier_line,
            before=1,
            after=CONTEXT_SNIPPET_LINES - 2,
        )
        if snippet is not None:
            snippets.append(snippet)
            suppliers += 1
    callback_defs = 0
    for item in supplied:
        if callback_defs >= MAX_CONTEXT_CALLBACK_DEFS:
            break
        definition = item.definition
        key = (definition.path, definition.line)
        if key in seen:
            continue
        seen.add(key)
        span = min(CONTEXT_SNIPPET_LINES - 1, definition.end_line - definition.line)
        snippet = _bounded_context_snippet(
            engine, definition.path, definition.line, before=0, after=span
        )
        if snippet is not None:
            snippets.append(snippet)
            callback_defs += 1
    return snippets


def _bounded_context_snippet(
    engine: TraceEngine, path: str, line: int, *, before: int, after: int
) -> Snippet | None:
    text = engine.index.workspace.read_source(path)
    if text is None:
        return None
    lines = text.splitlines()
    if not (0 < line <= len(lines)):
        return None
    start = max(1, line - before)
    end = min(len(lines), line + after)
    return Snippet(
        path=path,
        start_line=start,
        end_line=end,
        text=redact_secret_values("\n".join(lines[start - 1 : end])),
    )


def _evidence_snippets(
    lines: list[str],
    target: SymbolDefinition,
    findings: tuple[ReviewFinding, ...],
) -> tuple[Snippet, ...]:
    snippets: list[Snippet] = []
    seen_lines: set[tuple[str, int]] = set()
    for finding in findings:
        if finding.confidence != CONFIDENCE_CONFIRMED:
            continue
        if finding.path != target.path:
            continue
        key = (finding.path, finding.line)
        if key in seen_lines or not (0 < finding.line <= len(lines)):
            continue
        seen_lines.add(key)
        start = max(1, finding.line - 1)
        end = min(len(lines), finding.line + 1)
        snippets.append(
            Snippet(
                path=finding.path,
                start_line=start,
                end_line=end,
                text=redact_secret_values("\n".join(lines[start - 1 : end])),
            )
        )
        if len(snippets) >= MAX_EVIDENCE_SNIPPETS:
            break
    return tuple(snippets)
