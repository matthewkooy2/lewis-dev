"""Facade used by the Lewis chat tool and CLI to run traces.

Owns the workspace, builds/caches the project index (invalidated when file
sizes/mtimes change), dispatches the three modes, and returns both the
structured ``TraceResult`` and the rendered report text. Read-only: nothing
here writes, executes, installs, or touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import ROOT_DIR
from app.trace.engine import TraceEngine
from app.trace.model import (
    TRACE_MODE_DEEP,
    TRACE_MODE_QUICK,
    TRACE_MODES,
    TraceLimits,
    TraceResult,
)
from app.trace.project_index import ProjectIndex
from app.trace.render import render_result
from app.trace.review import review_implementation
from app.trace.workspace import TraceWorkspace


@dataclass(frozen=True)
class TraceOutcome:
    result: TraceResult
    report: str


class CodeTracer:
    """Runs quick/deep/review traces against one workspace root."""

    def __init__(
        self,
        workspace_root: Path = ROOT_DIR,
        *,
        limits: TraceLimits | None = None,
    ):
        self.limits = limits or TraceLimits()
        self.workspace = TraceWorkspace(workspace_root, limits=self.limits)
        self._index: ProjectIndex | None = None
        self._index_fingerprint: tuple[tuple[str, int], ...] = ()

    def trace(
        self,
        symbol: str,
        *,
        mode: str = TRACE_MODE_QUICK,
        file: str | None = None,
    ) -> TraceOutcome:
        normalized_mode = str(mode or TRACE_MODE_QUICK).strip().lower()
        if normalized_mode not in TRACE_MODES:
            result = TraceResult(
                mode=normalized_mode,
                ok=False,
                error=(
                    f"Unknown mode {mode!r}; expected one of {', '.join(TRACE_MODES)}."
                ),
            )
            return TraceOutcome(result, render_result(result, self.limits))

        engine = TraceEngine(self._current_index(), self.limits)
        lookup = engine.resolve_target(symbol, file)
        if lookup.error:
            message = lookup.error
            if lookup.suggestions:
                message += " Closest names: " + ", ".join(lookup.suggestions) + "."
            result = TraceResult(mode=normalized_mode, ok=False, error=message)
            return TraceOutcome(result, render_result(result, self.limits))
        if lookup.target is None:
            result = TraceResult(
                mode=normalized_mode,
                ok=True,
                target=None,
                candidates=lookup.candidates,
                warnings=(
                    "The symbol is ambiguous; pass file::symbol or a "
                    "qualified name to pick one candidate.",
                ),
            )
            return TraceOutcome(result, render_result(result, self.limits))

        if normalized_mode == TRACE_MODE_QUICK:
            result = engine.quick_trace(lookup.target)
        elif normalized_mode == TRACE_MODE_DEEP:
            result = engine.deep_trace(lookup.target)
        else:
            result = review_implementation(engine, lookup.target)
        return TraceOutcome(result, render_result(result, self.limits))

    def _current_index(self) -> ProjectIndex:
        fingerprint = self._workspace_fingerprint()
        if self._index is None or fingerprint != self._index_fingerprint:
            self._index = ProjectIndex.build(self.workspace)
            self._index_fingerprint = fingerprint
        return self._index

    def _workspace_fingerprint(self) -> tuple[tuple[str, int], ...]:
        entries: list[tuple[str, int]] = []
        for record in self.workspace.list_source_files():
            try:
                stat = (self.workspace.resolved_root / record.path).stat()
            except OSError:
                continue
            entries.append((record.path, stat.st_mtime_ns ^ stat.st_size))
        return tuple(entries)
