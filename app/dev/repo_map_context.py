"""Freestyle-facing retrieval and advisory review over the persistent repo map.

Turns a loaded :class:`~app.dev.repo_map.RepoIndex` into the connected file
groups Freestyle should prefer in model context (route -> page -> CSS -> JS ->
tests), the selection reasons recorded in artifacts, and advisory review
warnings about implementation surfaces a change appears to have ignored.

This module never selects a file the safety layer has not already cleared: the
index it reads is built only from safe files, and the connectivity it computes
can only reorder or annotate already-safe candidates. It also resolves which
index a Freestyle job should use, preferring a fresh index of the job's own
isolated worktree and falling back to a clearly-marked stale host index.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import re
from typing import Any

from app.dev.repo_map import (
    RepoIndex,
    build_or_refresh_index,
    load_host_index,
    load_index,
)

MAX_CONNECTED_FILES = 24
MAX_REASONS_PER_FILE = 4
MAX_WARNINGS = 12

REVIEW_RELATED_SURFACES_UNCONSIDERED = (
    "Repo-map warning: changed route/page file {path} has related "
    "{surfaces} in the repo map that this change did not touch; confirm the "
    "rendered result instead of patching one surface in isolation."
)
REVIEW_TESTS_NOT_CONSIDERED = (
    "Repo-map warning: the repo map links {path} to tests ({tests}) that this "
    "change did not touch or update."
)
REVIEW_ASSET_MISSING = (
    "Repo-map warning: page {path} references static asset {asset} that is "
    "absent from the repo map; the asset may be missing or misnamed."
)
REVIEW_DISCONNECTED_CSS = (
    "Repo-map warning: changed stylesheet {path} defines selectors "
    "({selectors}) that no rendered page or static script in the repo map "
    "appears to use; confirm they are reachable."
)

_TOKEN = re.compile(r"[a-z0-9]+")
# Generic verbs/nouns that carry no feature signal; "calendar", "tasks", etc.
# are deliberately absent so they remain selectable feature tokens.
_STOP_TOKENS = frozenset(
    {
        "add",
        "and",
        "app",
        "approval",
        "artifact",
        "bundle",
        "change",
        "class",
        "code",
        "context",
        "deepseek",
        "engine",
        "file",
        "fix",
        "freestyle",
        "from",
        "generation",
        "html",
        "into",
        "make",
        "memory",
        "only",
        "proposal",
        "refinement",
        "repo",
        "review",
        "test",
        "tests",
        "the",
        "this",
        "update",
        "validation",
        "warning",
        "with",
        "worktree",
    }
)

_STATIC_CSS = "static_css"
_STATIC_JS = "static_js"
_WEB_ROUTE = "web_route"
_WEB_PAGE = "web_page"
_TEST = "test"
_APPROVAL = "approval_sensitive_metadata"


# --------------------------------------------------------------------------- #
# Index resolution (worktree-preferred, host fallback).
# --------------------------------------------------------------------------- #
def resolve_freestyle_index(
    worktree: Path,
    root: Path,
    state: Path,
    *,
    refresh: bool = True,
) -> tuple[RepoIndex | None, dict[str, Any]]:
    """Return the index a Freestyle job should use plus visible provenance.

    Prefers a fresh index of the job's own isolated worktree (which holds the
    diverged refinement state). If the worktree cannot be indexed, falls back to
    the host index and marks the provenance ``fallback``/``stale`` so artifacts
    show the model did not see worktree-local changes. Fail-open: returns
    ``(None, provenance)`` rather than raising into the pipeline.
    """
    if refresh:
        worktree_index = _safe_build(worktree, state)
        if worktree_index is None:
            worktree_index = _safe_load(worktree, state)
    else:
        worktree_index = _safe_load(worktree, state)
    if worktree_index is not None and worktree_index.manifest.get("scope") == "worktree":
        return worktree_index, _provenance(worktree_index, stale=False, fallback=False)

    host_index = load_host_index(state)
    if host_index is not None:
        return host_index, _provenance(host_index, stale=True, fallback=True)
    return None, {
        "available": False,
        "scope": None,
        "stale": True,
        "fallback": True,
        "indexed_at": None,
        "reason": "no_index_available",
    }


def _safe_build(root: Path, state: Path) -> RepoIndex | None:
    try:
        return build_or_refresh_index(root, state=state)
    except Exception:
        return None


def _safe_load(root: Path, state: Path) -> RepoIndex | None:
    try:
        return load_index(root, state=state)
    except Exception:
        return None


def _provenance(index: RepoIndex, *, stale: bool, fallback: bool) -> dict[str, Any]:
    manifest = index.manifest
    return {
        "available": True,
        "scope": manifest.get("scope"),
        "scope_id": manifest.get("scope_id"),
        "job_id": manifest.get("job_id"),
        "root_id": manifest.get("root_id"),
        "schema_version": manifest.get("schema_version"),
        "indexed_at": manifest.get("indexed_at"),
        "file_count": manifest.get("file_count"),
        "stale": stale,
        "fallback": fallback,
    }


def index_provenance_summary(provenance: dict[str, Any]) -> str:
    if not provenance.get("available"):
        return "No repo map was available; context selection used heuristics only."
    scope = provenance.get("scope")
    marker = " (stale host fallback)" if provenance.get("fallback") else ""
    return (
        f"Repo map: {scope} scope indexed at {provenance.get('indexed_at')} "
        f"covering {provenance.get('file_count')} files{marker}."
    )


# --------------------------------------------------------------------------- #
# Connectivity graph + connected-file selection.
# --------------------------------------------------------------------------- #
class _Graph:
    """A bounded, deterministic file-to-file connection graph over one index."""

    def __init__(self, index: RepoIndex) -> None:
        self.index = index
        self.role: dict[str, str] = {
            str(f["path"]): str(f.get("role", "")) for f in index.files
        }
        self.paths: tuple[str, ...] = tuple(self.role)
        self._adjacency: dict[str, dict[str, str]] = {}
        self.feature_tokens: dict[str, frozenset[str]] = {}
        self._route_by_path: dict[str, set[str]] = {}
        self._build()
        self.feature_tokens = {
            path: _file_feature_tokens(index, path) for path in self.paths
        }

    def _connect(self, a: str, b: str, reason: str) -> None:
        if a not in self.role or b not in self.role or a == b:
            return
        if self.role.get(a) == _APPROVAL or self.role.get(b) == _APPROVAL:
            return
        self._adjacency.setdefault(a, {}).setdefault(b, reason)
        self._adjacency.setdefault(b, {}).setdefault(a, reason)

    def neighbors(self, path: str) -> dict[str, str]:
        return self._adjacency.get(path, {})

    def _build(self) -> None:
        css_files = [p for p, r in self.role.items() if r == _STATIC_CSS]
        js_files = [p for p, r in self.role.items() if r == _STATIC_JS]
        page_files = [
            p for p, r in self.role.items() if r in {_WEB_ROUTE, _WEB_PAGE}
        ]
        # Page <-> referenced static asset (and the shared global stylesheet).
        asset_index = {p.casefold(): p for p in (*css_files, *js_files)}
        for route in self.index.routes:
            for asset in route.get("referenced_assets", []) or []:
                resolved = _resolve_asset(str(asset), asset_index)
                if resolved:
                    self._connect(
                        str(route["path"]), resolved, "page references asset"
                    )
        # Page-builder markup that references a static asset (page_to_asset
        # relationship edges carry the source file before any ``:symbol`` part).
        for edge in self.index.relationships:
            if edge.get("kind") != "page_to_asset":
                continue
            source = str(edge.get("source", "")).split(":", 1)[0]
            resolved = _resolve_asset(str(edge.get("target", "")), asset_index)
            if resolved:
                self._connect(source, resolved, "page references asset")
        # Feature-token grouping connects a page to the CSS/JS that share its
        # feature word (calendar.js, calendar styles, ...).
        for page in page_files:
            for token in _path_feature_tokens(page):
                for asset in (*css_files, *js_files):
                    if token in _path_feature_tokens(asset):
                        self._connect(page, asset, f"shared feature '{token}'")
        # CSS <-> JS by shared class/id token (selector ownership).
        css_tokens = {p: _css_tokens(self.index, p) for p in css_files}
        js_tokens = {p: _js_tokens(self.index, p) for p in js_files}
        for css_path, ctokens in css_tokens.items():
            for js_path, jtokens in js_tokens.items():
                if ctokens & jtokens:
                    self._connect(css_path, js_path, "shared selector/class")
        # Test <-> source module (referenced modules + name correspondence).
        for record in self.index.tests:
            test_path = str(record["path"])
            for module in record.get("referenced_modules", []) or []:
                module_path = _module_to_path(str(module))
                if module_path in self.role:
                    self._connect(test_path, module_path, "test references module")
            for route in record.get("referenced_routes", []) or []:
                route_path = str(route)
                for source in self._route_sources(route_path):
                    self._connect(test_path, source, f"test covers route {route_path}")
        for test_path, role in self.role.items():
            if role != _TEST:
                continue
            for source in _source_for_test(test_path):
                if source in self.role:
                    self._connect(test_path, source, "test name matches source")

    def _route_sources(self, route: str) -> tuple[str, ...]:
        if not self._route_by_path:
            by_route: dict[str, set[str]] = {}
            for record in self.index.routes:
                path = str(record.get("path", ""))
                route_path = str(record.get("route", ""))
                if path in self.role and route_path:
                    by_route.setdefault(route_path, set()).add(path)
            self._route_by_path = by_route
        return tuple(sorted(self._route_by_path.get(route, ())))


def select_connected_files(
    index: RepoIndex,
    *,
    goal: str,
    named_targets: Sequence[str] = (),
    seed_paths: Sequence[str] = (),
    limit: int = MAX_CONNECTED_FILES,
) -> dict[str, Any]:
    """Select a connected group of repo files for one goal, with reasons.

    Returns the connected paths (ordered, bounded), the per-file selection
    reasons, and the connected paths that did not fit the bound, so context
    planning can record what it preferred and what it omitted.
    """
    graph = _Graph(index)
    reasons: dict[str, list[str]] = {}

    def add(path: str, reason: str) -> None:
        if path not in graph.role or graph.role.get(path) == _APPROVAL:
            return
        bucket = reasons.setdefault(path, [])
        if reason not in bucket and len(bucket) < MAX_REASONS_PER_FILE:
            bucket.append(reason)

    terms = _goal_feature_tokens(goal)
    for raw in named_targets:
        add(_normalize(raw), "named target from goal")
    for raw in seed_paths:
        add(_normalize(raw), "model-preferred likely file")
    # Routes whose path or file matches a goal term seed the group. Only real
    # route-handler files (role web_route, with a resolved handler) seed, so a
    # ``/calendar`` literal in a test fixture or a docstring example does not
    # drag unrelated files into the group.
    for route in index.routes:
        route_path = str(route.get("route", ""))
        file_path = str(route.get("path", ""))
        if graph.role.get(file_path) != _WEB_ROUTE or not route.get("handler"):
            continue
        if _matches_terms(route_path, terms) or _matches_terms(file_path, terms):
            add(file_path, f"defines route {route_path}")
    # Files whose path matches a goal feature token.
    for path in graph.paths:
        matched = _matched_feature_terms(graph, path, terms)
        if matched:
            add(path, f"repo-map feature matches goal term '{matched[0]}'")
    # Expand one hop through the connection graph from every seed file.
    for seed in list(reasons):
        for neighbor, edge in graph.neighbors(seed).items():
            if not _should_expand_neighbor(graph, seed, neighbor, terms):
                continue
            add(neighbor, f"{edge} ({_short(seed)})")

    ordered = _order_connected(reasons, named_targets, seed_paths, graph, terms)
    included = ordered[:limit]
    omitted = ordered[limit:]
    return {
        "connected_paths": included,
        "selection_reasons": {path: reasons[path] for path in included},
        "omitted_relevant": omitted,
        "graph_file_count": len(graph.role),
    }


def connected_files_for_paths(
    index: RepoIndex, paths: Sequence[str]
) -> dict[str, list[str]]:
    """Return one-hop connected files (with reasons) for a set of changed paths."""
    graph = _Graph(index)
    out: dict[str, list[str]] = {}
    for raw in paths:
        path = _normalize(raw)
        neighbors = graph.neighbors(path)
        if neighbors:
            out[path] = [f"{p} ({edge})" for p, edge in sorted(neighbors.items())]
    return out


# --------------------------------------------------------------------------- #
# Advisory review warnings.
# --------------------------------------------------------------------------- #
def repo_map_review_warnings(
    index: RepoIndex, changed_paths: Sequence[str]
) -> list[str]:
    """Build advisory repo-map review warnings for one set of changed paths."""
    graph = _Graph(index)
    changed = {_normalize(p) for p in changed_paths}
    warnings: list[str] = []
    asset_index = {
        p.casefold(): p
        for p, r in graph.role.items()
        if r in {_STATIC_CSS, _STATIC_JS}
    }
    for path in sorted(changed):
        role = graph.role.get(path)
        if role in {_WEB_ROUTE, _WEB_PAGE}:
            related = {
                p
                for p in graph.neighbors(path)
                if graph.role.get(p) in {_STATIC_CSS, _STATIC_JS}
            }
            unconsidered = sorted(related - changed)
            if unconsidered:
                warnings.append(
                    REVIEW_RELATED_SURFACES_UNCONSIDERED.format(
                        path=path, surfaces=", ".join(unconsidered[:5])
                    )
                )
        tests = sorted(
            p
            for p in graph.neighbors(path)
            if graph.role.get(p) == _TEST and p not in changed
        )
        if tests and role and role != _TEST:
            warnings.append(
                REVIEW_TESTS_NOT_CONSIDERED.format(
                    path=path, tests=", ".join(tests[:5])
                )
            )
    for route in index.routes:
        if _normalize(str(route.get("path", ""))) not in changed:
            continue
        for asset in route.get("referenced_assets", []) or []:
            if not _resolve_asset(str(asset), asset_index):
                warnings.append(
                    REVIEW_ASSET_MISSING.format(path=route["path"], asset=asset)
                )
    return _bounded(warnings, MAX_WARNINGS)


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def _order_connected(
    reasons: dict[str, list[str]],
    named_targets: Sequence[str],
    seed_paths: Sequence[str],
    graph: _Graph,
    terms: frozenset[str],
) -> list[str]:
    named = {_normalize(p) for p in named_targets}
    seeds = {_normalize(p) for p in seed_paths}

    def rank(path: str) -> tuple[int, int, int, int, int, str]:
        tier = 2
        if path in named:
            tier = 0
        elif path in seeds:
            tier = 1
        path_specific = 0 if _path_feature_tokens(path) & terms else 1
        feature_specific = 0 if _is_feature_relevant(graph, path, terms) else 1
        role_rank = _role_rank(graph.role.get(path, ""))
        return (
            tier,
            feature_specific,
            path_specific,
            role_rank,
            -len(reasons[path]),
            path.casefold(),
        )

    return sorted(reasons, key=rank)


def _goal_feature_tokens(goal: str) -> frozenset[str]:
    return frozenset(
        token
        for token in _TOKEN.findall(goal.casefold())
        if len(token) >= 4 and token not in _STOP_TOKENS
    )


def _path_feature_tokens(path: str) -> frozenset[str]:
    stem = Path(path).stem.casefold()
    return _feature_tokens_from_text(stem)


def _file_feature_tokens(index: RepoIndex, path: str) -> frozenset[str]:
    tokens: set[str] = set(_path_feature_tokens(path))
    for route in index.routes:
        if str(route.get("path")) == path:
            tokens.update(_feature_tokens_from_text(str(route.get("route", ""))))
            tokens.update(_feature_tokens_from_text(str(route.get("handler", ""))))
    for symbol in index.symbols:
        if str(symbol.get("path")) == path:
            tokens.update(_feature_tokens_from_text(str(symbol.get("name", ""))))
    for selector in index.css_selectors:
        if str(selector.get("path")) != path:
            continue
        tokens.update(_feature_tokens_from_text(str(selector.get("selector", ""))))
        for name in (
            *(selector.get("classes", []) or []),
            *(selector.get("ids", []) or []),
        ):
            tokens.update(_feature_tokens_from_text(str(name)))
    for record in index.js_assets:
        if str(record.get("path")) != path:
            continue
        for name in (
            *(record.get("lookup_classes", []) or []),
            *(record.get("managed_classes", []) or []),
            *(record.get("lookup_ids", []) or []),
        ):
            tokens.update(_feature_tokens_from_text(str(name)))
    for record in index.tests:
        if str(record.get("path")) != path:
            continue
        for name in (
            *(record.get("test_functions", []) or []),
            *(record.get("fixtures", []) or []),
            *(record.get("referenced_routes", []) or []),
            *(record.get("referenced_modules", []) or []),
        ):
            tokens.update(_feature_tokens_from_text(str(name)))
    return frozenset(tokens)


def _feature_tokens_from_text(text: str) -> frozenset[str]:
    return frozenset(
        token
        for token in _TOKEN.findall(text.casefold())
        if len(token) >= 4 and token not in _STOP_TOKENS
    )


def _matched_feature_terms(
    graph: _Graph, path: str, terms: frozenset[str]
) -> tuple[str, ...]:
    if not terms:
        return ()
    lowered = path.casefold()
    direct = [
        term
        for term in sorted(terms)
        if term in lowered or term in graph.feature_tokens.get(path, frozenset())
    ]
    return tuple(direct)


def _is_feature_relevant(graph: _Graph, path: str, terms: frozenset[str]) -> bool:
    if not terms:
        return True
    return bool(_matched_feature_terms(graph, path, terms))


def _should_expand_neighbor(
    graph: _Graph, seed: str, neighbor: str, terms: frozenset[str]
) -> bool:
    if not terms:
        return True
    seed_role = graph.role.get(seed, "")
    neighbor_role = graph.role.get(neighbor, "")
    if neighbor_role in {_STATIC_CSS, _STATIC_JS, _TEST}:
        return _is_feature_relevant(graph, neighbor, terms)
    if seed_role in {_STATIC_CSS, _STATIC_JS, _TEST}:
        return _is_feature_relevant(graph, neighbor, terms)
    return True


def _role_rank(role: str) -> int:
    if role == _WEB_ROUTE:
        return 0
    if role == _WEB_PAGE:
        return 1
    if role == _STATIC_CSS:
        return 2
    if role == _STATIC_JS:
        return 3
    if role == _TEST:
        return 4
    return 9


def _css_tokens(index: RepoIndex, css_path: str) -> set[str]:
    tokens: set[str] = set()
    for record in index.css_selectors:
        if str(record.get("path")) == css_path:
            tokens.update(record.get("classes", []) or [])
            tokens.update(record.get("ids", []) or [])
    return tokens


def _js_tokens(index: RepoIndex, js_path: str) -> set[str]:
    tokens: set[str] = set()
    for record in index.js_assets:
        if str(record.get("path")) == js_path:
            tokens.update(record.get("lookup_classes", []) or [])
            tokens.update(record.get("managed_classes", []) or [])
            tokens.update(record.get("lookup_ids", []) or [])
    return tokens


def _resolve_asset(asset: str, asset_index: dict[str, str]) -> str | None:
    lowered = asset.casefold().lstrip("/")
    for key, path in asset_index.items():
        if key.endswith(lowered) or lowered.endswith(key):
            return path
    return None


def _module_to_path(module: str) -> str:
    return module.replace(".", "/") + ".py"


def _source_for_test(test_path: str) -> tuple[str, ...]:
    name = Path(test_path).name
    if not name.startswith("test_"):
        return ()
    stem = name[len("test_") :].removesuffix(".py")
    return (f"app/{stem}.py", f"app/dev/{stem}.py", f"{stem}.py")


def _matches_terms(text: str, terms: frozenset[str]) -> bool:
    lowered = text.casefold()
    return any(term in lowered for term in terms)


def _normalize(path: str) -> str:
    return str(path).strip().replace("\\", "/")


def _short(path: str) -> str:
    return Path(path).name


def _bounded(values: Sequence[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= limit:
            break
    return out
