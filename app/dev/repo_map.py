"""Persistent, local, static repo-index storage and incremental indexing.

Builds an inspectable repo map under ``.lewis-dev/repo-index/`` from a single
*approved* root only: the host checkout (``ROOT_DIR``) or one known, approved
isolated worktree at ``.lewis-dev/worktrees/<job_id>``. Arbitrary roots are
refused. Index artifacts are plain JSON/JSONL the model never edits; building
the index reads source files but never mutates them.

Discovery reuses the existing freestyle safety policy: generated/cache trees,
the dev-state directory itself, secrets/``.env``, database files, binaries,
oversized files, and unsupported types are excluded, and approval-sensitive
files are recorded only as metadata (no symbols, selectors, chunks, or content)
so the map never makes them easier to edit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any

from app.code_proposal_path_safety import validate_relative_project_path
from app.config import ROOT_DIR
from app.dev.freestyle_artifacts import encode_json_artifact, write_artifact_bytes
from app.dev.freestyle_coding import (
    DEV_STATE_DIR_NAME,
    FreestyleJobRefusedError,
    _utc_now,
)
from app.dev.freestyle_policy import (
    APPROVAL_STORE_PATH_NAMES,
    APPROVAL_STORE_TOKEN,
    GENERATED_OR_CACHE_PARTS,
    contains_sensitive_content,
    defines_approval_store,
    is_job_id,
)
from app.dev.repo_map_extract import SCHEMA_VERSION, extract_file

INDEX_DIR_NAME = "repo-index"
HOST_SCOPE_ID = "host"
INDEXER_VERSION = 1

MAX_INDEX_FILES = 6_000
MAX_INDEX_FILE_BYTES = 512 * 1024

# Directory parts never scanned: caches/build output (shared policy set) plus
# version-control and the dev-state directory (worktrees, jobs, approval-store
# persistence, and the index itself all live there).
_EXCLUDED_PARTS = GENERATED_OR_CACHE_PARTS | {".git", DEV_STATE_DIR_NAME}

# Database/persistence file extensions never indexed.
_DATABASE_SUFFIXES = frozenset({"sqlite", "sqlite3", "db"})

# Source-like extensions worth indexing. Files outside this set are skipped as
# unsupported so the index stays source-shaped rather than absorbing binaries
# or opaque data blobs.
_INDEXABLE_SUFFIXES = frozenset(
    {
        "cfg",
        "css",
        "csv",
        "html",
        "htm",
        "ini",
        "js",
        "json",
        "jsx",
        "md",
        "mjs",
        "py",
        "rst",
        "sql",
        "toml",
        "ts",
        "tsx",
        "txt",
        "yaml",
        "yml",
    }
)

_RECORD_FILES = (
    "files.jsonl",
    "symbols.jsonl",
    "routes.jsonl",
    "css_selectors.jsonl",
    "js_assets.jsonl",
    "tests.jsonl",
    "chunks.jsonl",
    "relationships.jsonl",
)
_RECORD_ATTRS = (
    "files",
    "symbols",
    "routes",
    "css_selectors",
    "js_assets",
    "tests",
    "chunks",
    "relationships",
)


@dataclass(frozen=True)
class IndexScope:
    """The validated identity of one indexable root."""

    scope: str  # "host" or "worktree"
    scope_id: str  # directory key under repo-index/ (HOST_SCOPE_ID or job_id)
    root_path: str
    root_id: str
    job_id: str | None = None


@dataclass(frozen=True)
class RepoIndex:
    """An in-memory view of one persisted repo-map scope."""

    manifest: dict[str, Any]
    files: tuple[dict[str, Any], ...] = ()
    symbols: tuple[dict[str, Any], ...] = ()
    routes: tuple[dict[str, Any], ...] = ()
    css_selectors: tuple[dict[str, Any], ...] = ()
    js_assets: tuple[dict[str, Any], ...] = ()
    tests: tuple[dict[str, Any], ...] = ()
    chunks: tuple[dict[str, Any], ...] = ()
    relationships: tuple[dict[str, Any], ...] = ()

    @property
    def scope_id(self) -> str:
        return str(self.manifest.get("scope_id", ""))

    @property
    def indexed_at(self) -> str | None:
        value = self.manifest.get("indexed_at")
        return value if isinstance(value, str) else None

    @property
    def file_count(self) -> int:
        return len(self.files)

    def descriptor_for(self, path: str) -> dict[str, Any] | None:
        lowered = path.casefold()
        for record in self.files:
            if str(record.get("path", "")).casefold() == lowered:
                return record
        return None


def repo_index_root(state: Path) -> Path:
    return state / INDEX_DIR_NAME


def resolve_index_scope(root: Path, state: Path) -> IndexScope:
    """Validate that ``root`` is an approved indexable root and identify it.

    Only the host checkout (``ROOT_DIR``) and an existing approved worktree at
    ``<state>/worktrees/<job_id>`` are accepted; anything else is refused so the
    indexer can never be pointed at an arbitrary directory.
    """
    resolved = root.resolve()
    if resolved == ROOT_DIR.resolve():
        return IndexScope(
            scope="host",
            scope_id=HOST_SCOPE_ID,
            root_path=str(resolved),
            root_id=_root_identity(resolved),
        )
    worktrees = (state / "worktrees").resolve()
    try:
        relative = resolved.relative_to(worktrees)
    except ValueError as exc:
        raise FreestyleJobRefusedError(
            "Repo index root must be the host checkout or an approved worktree."
        ) from exc
    parts = relative.parts
    if len(parts) != 1 or not is_job_id(parts[0]):
        raise FreestyleJobRefusedError(
            "Repo index worktree root must be a single approved job worktree."
        )
    job_id = parts[0]
    if not resolved.is_dir() or resolved.is_symlink():
        raise FreestyleJobRefusedError("Repo index worktree root is missing or unsafe.")
    return IndexScope(
        scope="worktree",
        scope_id=job_id,
        root_path=str(resolved),
        root_id=_root_identity(resolved),
        job_id=job_id,
    )


def build_or_refresh_index(
    root: Path,
    *,
    state: Path,
    now: str | None = None,
) -> RepoIndex:
    """Build or incrementally refresh the repo index for one approved root."""
    state = state.resolve()
    scope = resolve_index_scope(root, state)
    resolved_root = Path(scope.root_path)
    previous = _load_scope(state, scope.scope_id)
    prior_by_path = _group_records_by_path(previous)
    prior_hashes = (
        previous.manifest.get("file_hashes", {}) if previous is not None else {}
    )
    if not isinstance(prior_hashes, dict):
        prior_hashes = {}

    file_hashes: dict[str, str] = {}
    records: dict[str, list[dict[str, Any]]] = {attr: [] for attr in _RECORD_ATTRS}
    counts = {
        "indexed": 0,
        "reused": 0,
        "changed": 0,
        "new": 0,
        "approval_sensitive": 0,
    }
    skipped: dict[str, int] = {}
    scanned = 0

    for relative, path, text, reason in _iter_candidate_files(resolved_root):
        if reason is not None:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        scanned += 1
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        file_hashes[relative] = digest
        approval_sensitive = _is_approval_sensitive(relative, path.name, text)
        if approval_sensitive:
            counts["approval_sensitive"] += 1
        prior_digest = prior_hashes.get(relative)
        if prior_digest == digest and relative in prior_by_path:
            for attr in _RECORD_ATTRS:
                records[attr].extend(prior_by_path[relative][attr])
            counts["reused"] += 1
        else:
            extracted = extract_file(
                relative, text, approval_sensitive=approval_sensitive
            )
            _append_extracted(records, extracted)
            counts["new" if prior_digest is None else "changed"] += 1
        counts["indexed"] += 1

    deleted = sorted(set(prior_hashes) - set(file_hashes))
    manifest = _build_manifest(
        scope=scope,
        now=now or _utc_now(),
        file_hashes=file_hashes,
        counts=counts,
        skipped=skipped,
        deleted=deleted,
        scanned=scanned,
    )
    index = RepoIndex(
        manifest=manifest,
        **{attr: tuple(records[attr]) for attr in _RECORD_ATTRS},
    )
    _persist_index(state, scope.scope_id, index)
    return index


def load_index(root: Path, *, state: Path) -> RepoIndex | None:
    """Load the persisted index for one approved root, fail-open to ``None``."""
    state = state.resolve()
    try:
        scope = resolve_index_scope(root, state)
    except FreestyleJobRefusedError:
        return None
    return _load_scope(state, scope.scope_id)


def load_host_index(state: Path) -> RepoIndex | None:
    return _load_scope(state.resolve(), HOST_SCOPE_ID)


def _build_manifest(
    *,
    scope: IndexScope,
    now: str,
    file_hashes: dict[str, str],
    counts: dict[str, int],
    skipped: dict[str, int],
    deleted: list[str],
    scanned: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "indexer_version": INDEXER_VERSION,
        "scope": scope.scope,
        "scope_id": scope.scope_id,
        "job_id": scope.job_id,
        "root_path": scope.root_path,
        "root_id": scope.root_id,
        "indexed_at": now,
        "file_count": counts["indexed"],
        "scanned_file_count": scanned,
        "changed_file_count": counts["changed"],
        "new_file_count": counts["new"],
        "reused_file_count": counts["reused"],
        "deleted_file_count": len(deleted),
        "approval_sensitive_file_count": counts["approval_sensitive"],
        "deleted_paths": deleted[:200],
        "skipped_counts": dict(sorted(skipped.items())),
        "skipped_total": sum(skipped.values()),
        "file_hashes": dict(sorted(file_hashes.items())),
    }


def _iter_candidate_files(root: Path):
    """Yield ``(relative, path, text, skip_reason)`` for every scanned entry."""
    scanned = 0
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        kept: list[str] = []
        for name in sorted(directory_names):
            child = directory_path / name
            if child.is_symlink() or name.casefold() in {
                part.casefold() for part in _EXCLUDED_PARTS
            }:
                continue
            kept.append(name)
        directory_names[:] = kept
        for name in sorted(file_names):
            if scanned >= MAX_INDEX_FILES:
                directory_names[:] = []
                yield "<scan-limit>", directory_path / name, "", "scan_limit"
                return
            scanned += 1
            path = directory_path / name
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                yield name, path, "", "outside_root"
                continue
            text, reason = _read_indexable_text(relative, path, root)
            yield relative, path, text, reason


def _read_indexable_text(
    relative: str, path: Path, root: Path
) -> tuple[str, str | None]:
    parts = tuple(part.casefold() for part in PurePosixPath(relative).parts)
    if path.is_symlink():
        return "", "symlink"
    if any(part in {p.casefold() for p in _EXCLUDED_PARTS} for part in parts):
        return "", "generated_or_cache"
    name = path.name.casefold()
    if name.startswith(".env"):
        return "", "secret_file"
    suffix = name.rsplit(".", 1)[-1] if "." in name else ""
    if suffix in _DATABASE_SUFFIXES:
        return "", "database_file"
    if suffix not in _INDEXABLE_SUFFIXES:
        return "", "unsupported_file_type"
    try:
        validate_relative_project_path(relative, root)
    except (ValueError, OSError):
        return "", "unsafe_path"
    try:
        if not path.is_file():
            return "", "unsupported_file_type"
        if path.stat().st_size > MAX_INDEX_FILE_BYTES:
            return "", "oversized"
        data = path.read_bytes()
    except OSError:
        return "", "unreadable"
    if b"\x00" in data[:4_096]:
        return "", "binary"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return "", "binary"
    if contains_sensitive_content(text):
        return "", "sensitive_content"
    return text, None


def _is_approval_sensitive(relative: str, name: str, text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", name.casefold())
    if (
        name.casefold() in APPROVAL_STORE_PATH_NAMES
        or APPROVAL_STORE_TOKEN in normalized
    ):
        return True
    return defines_approval_store(text)


def _append_extracted(records: dict[str, list[dict[str, Any]]], extracted) -> None:
    records["files"].append(extracted.descriptor.to_record())
    records["symbols"].extend(extracted.symbols)
    records["routes"].extend(extracted.routes)
    records["css_selectors"].extend(extracted.css_selectors)
    records["js_assets"].extend(extracted.js_assets)
    records["tests"].extend(extracted.tests)
    records["chunks"].extend(extracted.chunks)
    records["relationships"].extend(extracted.relationships)


def _group_records_by_path(
    index: RepoIndex | None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    if index is None:
        return grouped
    for attr in _RECORD_ATTRS:
        for record in getattr(index, attr):
            path = _record_path(record)
            if path is None:
                continue
            grouped.setdefault(path, {a: [] for a in _RECORD_ATTRS})[attr].append(
                record
            )
    return grouped


def _record_path(record: dict[str, Any]) -> str | None:
    path = record.get("path")
    if isinstance(path, str) and path:
        return path
    # Relationship edges store a "source" like "<path>:<symbol>".
    source = record.get("source")
    if isinstance(source, str) and source:
        return source.split(":", 1)[0]
    return None


def _persist_index(state: Path, scope_id: str, index: RepoIndex) -> None:
    write_artifact_bytes(
        state,
        INDEX_DIR_NAME,
        scope_id,
        "manifest.json",
        encode_json_artifact(index.manifest),
        label="repo index",
        allow_replace=True,
    )
    for filename, attr in zip(_RECORD_FILES, _RECORD_ATTRS):
        write_artifact_bytes(
            state,
            INDEX_DIR_NAME,
            scope_id,
            filename,
            _encode_jsonl(getattr(index, attr)),
            label="repo index",
            allow_replace=True,
        )


def _encode_jsonl(records: tuple[dict[str, Any], ...]) -> bytes:
    lines = [json.dumps(record, sort_keys=True) for record in records]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def _load_scope(state: Path, scope_id: str) -> RepoIndex | None:
    scope_dir = repo_index_root(state) / scope_id
    manifest_path = scope_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    payload: dict[str, tuple[dict[str, Any], ...]] = {}
    for filename, attr in zip(_RECORD_FILES, _RECORD_ATTRS):
        payload[attr] = _read_jsonl(scope_dir / filename)
    return RepoIndex(manifest=manifest, **payload)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file() or path.is_symlink():
        return ()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return ()
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return tuple(records)


def _root_identity(resolved_root: Path) -> str:
    return hashlib.sha256(str(resolved_root).encode("utf-8")).hexdigest()[:16]
