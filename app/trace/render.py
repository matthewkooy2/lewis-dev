"""Compact human-readable rendering of trace and review results.

Output is plain text with one fact per line, always line-numbered, always
labeled with how each relationship was established. The renderer enforces
the report character bound and states truncation explicitly.

``to_compact_dict`` is the bounded structured form handed to the chat model
alongside the text report; the full ``TraceResult.to_dict()`` remains
available for CLI ``--json`` output and future Freestyle reuse.
"""

from __future__ import annotations

from typing import Any

from app.trace.model import (
    CallEdge,
    SideEffect,
    SymbolDefinition,
    TraceLimits,
    TraceResult,
)

COMPACT_LIST_LIMIT = 15


def render_result(result: TraceResult, limits: TraceLimits | None = None) -> str:
    limits = limits or TraceLimits()
    lines: list[str] = []
    if not result.ok:
        lines.append(f"TRACE ERROR: {result.error}")
        if result.candidates:
            lines.append("Candidates:")
            lines.extend(_candidate_lines(result.candidates))
        text = "\n".join(lines)
        return text[: limits.max_report_chars]

    if result.target is None:
        lines.append(
            f"AMBIGUOUS SYMBOL: {len(result.candidates)} definitions match."
        )
        lines.append("Re-run with file::symbol or a qualified name. Candidates:")
        lines.extend(_candidate_lines(result.candidates))
        text = "\n".join(lines)
        return text[: limits.max_report_chars]

    target = result.target
    header = result.mode.upper() + " TRACE" if result.mode != "review" else "REVIEW"
    lines.append(
        f"{header}: {target.qualname} [{target.kind}, {target.language}]"
    )
    location = f"Defined at {target.path}:{target.line}-{target.end_line}"
    if target.signature:
        location += f"  signature: {target.signature}"
    lines.append(location)
    meta: list[str] = []
    if target.parent:
        meta.append(f"member of {target.parent}")
    if target.exported:
        names = ", ".join(target.export_names) if target.export_names else target.name
        meta.append(f"exported as {names}")
    if target.decorators:
        meta.append("decorators: " + ", ".join(target.decorators[:4]))
    if target.doc:
        meta.append(f'doc: "{target.doc}"')
    if meta:
        lines.append("  " + "; ".join(meta))

    if result.mode == "review":
        _render_findings(lines, result)

    caller_cap = 12 if result.mode == "review" else None
    _render_edges(
        lines, "Callers", result.callers, direction="caller", cap=caller_cap
    )
    _render_edges(lines, "Callees", result.callees, direction="callee")
    _render_effects(lines, "Side effects (direct)", result.side_effects)
    if result.transitive_side_effects:
        _render_effects(
            lines, "Side effects (transitive)", result.transitive_side_effects
        )

    uncertain = [
        edge
        for edge in result.unresolved_calls
        if edge.resolution in {"dynamic", "unresolved", "ambiguous"}
    ]
    if uncertain:
        lines.append(f"Uncertain calls ({len(uncertain)}):")
        for edge in uncertain[:12]:
            note = f" — {edge.note}" if edge.note else ""
            lines.append(
                f"  [{edge.resolution}] {edge.caller_path}:{edge.line}  "
                f"{edge.callee_text}{note}"
            )

    if result.error_handling:
        entries = [
            info
            for info in result.error_handling
            if info.raises or info.catches
        ]
        if entries:
            lines.append("Error handling:")
            for info in entries[:12]:
                raises = ", ".join(info.raises[:4]) or "-"
                catches = ", ".join(info.catches[:4]) or "-"
                lines.append(f"  {info.qualname}: raises [{raises}] catches [{catches}]")
                for line in info.bare_catch_lines[:2]:
                    lines.append(f"    bare except at line {line}")
                for line in info.swallowed_catch_lines[:2]:
                    lines.append(f"    swallowed exception at line {line}")

    if result.cycles:
        lines.append("Recursion/cycles:")
        for cycle in result.cycles:
            lines.append("  " + " -> ".join(_short_id(part) for part in cycle))

    if result.call_paths:
        lines.append(f"Representative call paths ({len(result.call_paths)}):")
        for path in result.call_paths[:6]:
            lines.append("  " + " -> ".join(_short_id(part) for part in path))

    if result.tests:
        lines.append(f"Tests ({len(result.tests)}):")
        for link in result.tests:
            lines.append(
                f"  [{link.resolution}] {link.test_path}:{link.line}  "
                f"{link.test_name} ({link.via})"
            )
    elif result.mode in {"deep", "review"}:
        lines.append("Tests: none found that call this symbol directly.")

    if result.definition_snippet is not None and result.mode != "review":
        snippet = result.definition_snippet
        suffix = " (truncated)" if snippet.truncated else ""
        lines.append(
            f"Definition {snippet.path}:{snippet.start_line}-{snippet.end_line}"
            f"{suffix}:"
        )
        for offset, text in enumerate(snippet.text.splitlines()):
            lines.append(f"  {snippet.start_line + offset:>5} | {text}")

    if result.truncations:
        lines.append("Truncation:")
        for item in dict.fromkeys(result.truncations):
            lines.append(f"  - {item}")
    if result.warnings:
        lines.append("Warnings:")
        for item in dict.fromkeys(result.warnings):
            lines.append(f"  - {item}")
    if result.stats:
        stats = ", ".join(f"{key}={value}" for key, value in result.stats.items())
        lines.append(f"Stats: {stats}")

    text = "\n".join(lines)
    if len(text) > limits.max_report_chars:
        text = text[: limits.max_report_chars - 60].rstrip()
        text += "\n[report truncated at the configured output limit]"
    return text


def _candidate_lines(candidates: tuple[SymbolDefinition, ...]) -> list[str]:
    lines = []
    for candidate in candidates:
        lines.append(
            f"  - {candidate.path}::{candidate.qualname} "
            f"[{candidate.kind}] line {candidate.line}"
        )
    return lines


def _render_edges(
    lines: list[str],
    title: str,
    edges: tuple[CallEdge, ...],
    *,
    direction: str,
    cap: int | None = None,
) -> None:
    if not edges:
        lines.append(f"{title}: none found.")
        return
    shown = edges if cap is None else edges[:cap]
    lines.append(f"{title} ({len(edges)}):")
    for edge in shown:
        origin = edge.caller_qualname or "<module>"
        note = f" — {edge.note}" if edge.note else ""
        if direction == "caller":
            lines.append(
                f"  [{edge.resolution}] {edge.caller_path}:{edge.line}  "
                f"{origin}  calls  {edge.callee_text}{note}"
            )
        else:
            resolved = (
                f"{edge.target_path}:{edge.target_qualname}"
                if edge.target_path
                else "?"
            )
            depth = f" d{edge.depth}" if edge.depth > 1 else ""
            lines.append(
                f"  [{edge.resolution}]{depth} line {edge.line}  "
                f"{edge.callee_text}  ->  {resolved}{note}"
            )
    if cap is not None and len(edges) > cap:
        lines.append(f"  … {len(edges) - cap} more (see structured data)")


def _render_effects(
    lines: list[str], title: str, effects: tuple[SideEffect, ...]
) -> None:
    if not effects:
        if title.endswith("(direct)"):
            lines.append(f"{title}: none detected.")
        return
    lines.append(f"{title} ({len(effects)}):")
    for effect in effects[:24]:
        where = effect.qualname or "<module level / import time>"
        lines.append(
            f"  [{effect.resolution}] {effect.category}: {effect.detail}  "
            f"({effect.path}:{effect.line} in {where})"
        )


def _render_findings(lines: list[str], result: TraceResult) -> None:
    if not result.findings:
        lines.append(
            "Findings: no confirmed issues from deterministic checks "
            "(static analysis only — not a proof of correctness)."
        )
        return
    lines.append(f"Findings ({len(result.findings)}):")
    for finding in result.findings:
        lines.append(
            f"  [{finding.severity}/{finding.confidence}] {finding.category} "
            f"{finding.path}:{finding.line} — {finding.message}"
        )
    if result.evidence:
        lines.append("Evidence:")
        for snippet in result.evidence:
            lines.append(f"  {snippet.path}:{snippet.start_line}-{snippet.end_line}:")
            for offset, text in enumerate(snippet.text.splitlines()):
                lines.append(f"    {snippet.start_line + offset:>5} | {text}")


def _short_id(symbol_id: str) -> str:
    path, _, qualname = symbol_id.partition("::")
    name = path.rsplit("/", 1)[-1]
    return f"{qualname}({name})" if qualname else name


def to_compact_dict(result: TraceResult) -> dict[str, Any]:
    """Bounded structured summary for tool results and reuse by Freestyle."""

    def _definition(definition: SymbolDefinition | None) -> dict[str, Any] | None:
        if definition is None:
            return None
        return {
            "path": definition.path,
            "qualname": definition.qualname,
            "kind": definition.kind,
            "language": definition.language,
            "line": definition.line,
            "end_line": definition.end_line,
            "signature": definition.signature,
            "exported": definition.exported,
        }

    def _edge(edge: CallEdge) -> dict[str, Any]:
        return {
            "caller_path": edge.caller_path,
            "caller": edge.caller_qualname,
            "line": edge.line,
            "callee_text": edge.callee_text,
            "target_path": edge.target_path,
            "target": edge.target_qualname,
            "resolution": edge.resolution,
            "note": edge.note,
            "depth": edge.depth,
        }

    def _effect(effect: SideEffect) -> dict[str, Any]:
        return {
            "category": effect.category,
            "detail": effect.detail,
            "path": effect.path,
            "line": effect.line,
            "in": effect.qualname,
            "resolution": effect.resolution,
        }

    return {
        "schema": "lewis-trace-compact/1",
        "mode": result.mode,
        "ok": result.ok,
        "error": result.error,
        "target": _definition(result.target),
        "candidates": [_definition(item) for item in result.candidates],
        "callers": [_edge(edge) for edge in result.callers[:COMPACT_LIST_LIMIT]],
        "callees": [_edge(edge) for edge in result.callees[:COMPACT_LIST_LIMIT]],
        "uncertain_calls": [
            _edge(edge) for edge in result.unresolved_calls[:COMPACT_LIST_LIMIT]
        ],
        "side_effects": [
            _effect(effect) for effect in result.side_effects[:COMPACT_LIST_LIMIT]
        ],
        "transitive_side_effects": [
            _effect(effect)
            for effect in result.transitive_side_effects[:COMPACT_LIST_LIMIT]
        ],
        "tests": [
            {
                "path": link.test_path,
                "name": link.test_name,
                "line": link.line,
                "via": link.via,
                "resolution": link.resolution,
            }
            for link in result.tests[:COMPACT_LIST_LIMIT]
        ],
        "cycles": [list(cycle) for cycle in result.cycles[:5]],
        "findings": [
            {
                "category": finding.category,
                "severity": finding.severity,
                "confidence": finding.confidence,
                "path": finding.path,
                "line": finding.line,
                "message": finding.message,
            }
            for finding in result.findings
        ],
        "warnings": list(result.warnings),
        "truncations": list(result.truncations),
        "stats": dict(result.stats),
    }
