"""Deterministic, developer-only edit-bundle applier for freestyle jobs.

Applies one structured JSON edit bundle to the isolated worktree of one
approved, materialized freestyle job. This slice is the deterministic safety
layer that later model-generated edit bundles must pass through; it does not
generate edits and never contacts a model or network service. It never runs
checks, formatters, package managers, repository scripts, shell commands, or
git mutations, and it never touches the main checkout, remotes, approval
records, or runtime databases.

Edit bundle schema (version 1, one JSON object):

    {
      "schema_version": 1,
      "job_id": "<32-character lowercase hex freestyle job id>",
      "created_at": "<ISO-8601 timestamp>",
      "operation_count": <integer equal to len(operations)>,
      "proposal_id": "<optional bounded provenance text>",
      "goal": "<optional bounded provenance text>",
      "operations": [
        {"op": "create_file", "path": "relative/posix/path",
         "content": "<full new text>"},
        {"op": "replace_file", "path": "relative/posix/path",
         "content": "<full new text>",
         "expected_sha256": "<sha256 hex of the current file bytes>"},
        {"op": "delete_file", "path": "relative/posix/path",
         "expected_sha256": "<sha256 hex of the current file bytes>"},
        {"op": "replace_text", "path": "relative/posix/path",
         "old_text": "<existing text that must match exactly once>",
         "new_text": "<replacement text>"}
      ]
    }

Only relative paths to safe text files inside the job worktree are accepted.
A path may carry more than one operation only when every operation for it is
``replace_text`` and their exact matched regions do not overlap; any
``create_file``, ``replace_file``, or ``delete_file`` must be the sole
operation for its path. Same-file replacements are computed against the
original bytes and spliced in descending match-offset order so earlier
offsets stay valid, producing one journaled write per file. The whole bundle
is validated before the first write, every mutation is journaled, and any
failure rolls the worktree back so it is never left partially edited. Written
files are never marked executable. Whether a path appeared in the job's proposal
context is intentionally not enforced here: the proposal artifact records
omissions only up to a bounded count, so this module revalidates every path
and file independently instead, and a stricter proposal-context allowlist
remains a later policy hook.

Invocation (developer command only):

    python -m app.dev.freestyle_coding apply-edits <job-id> <bundle.json>
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any
import uuid

from app.code_proposal_path_safety import (
    SECRET_NAME_PATTERN,
    validate_relative_project_path,
)
from app.config import ROOT_DIR
from app.dev.freestyle_artifacts import encode_json_artifact, write_artifact_bytes
from app.dev.freestyle_coding import (
    FreestyleJob,
    FreestyleJobRefusedError,
    _bounded_error,
    _require_explicit_developer_approval,
    _utc_now,
    _validated_job_worktree,
    _WINDOWS_RESERVED_NAMES,
    _write_job,
    get_freestyle_job,
)
from app.dev.freestyle_policy import (
    APPROVAL_STORE_PATH_NAMES,
    APPROVAL_STORE_TOKEN,
    GENERATED_OR_CACHE_PARTS,
    approval_sensitive_region_labels,
    contains_sensitive_content,
)
from app.dev.freestyle_state import (
    require_materialized,
    require_no_active_edit_application,
    resolved_roots,
)

EDIT_BUNDLE_SCHEMA_VERSION = 1
MAX_BUNDLE_BYTES = 2_000_000
MAX_BUNDLE_OPERATIONS = 100
MAX_EDIT_FILE_BYTES = 1_000_000
MAX_BUNDLE_TOTAL_CONTENT_BYTES = 5_000_000
MAX_EDIT_PATH_LENGTH = 240
MAX_PROVENANCE_TEXT_CHARS = 500

_EXPECTED_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_EXECUTABLE_SUFFIXES = {
    ".bat",
    ".cmd",
    ".com",
    ".dll",
    ".dylib",
    ".exe",
    ".msi",
    ".ps1",
    ".psd1",
    ".psm1",
    ".scr",
    ".sh",
    ".so",
    ".vbs",
}
_BUNDLE_REQUIRED_FIELDS = frozenset(
    {"schema_version", "job_id", "created_at", "operation_count", "operations"}
)
_BUNDLE_OPTIONAL_FIELDS = frozenset({"proposal_id", "goal"})
_OPERATION_FIELDS = {
    "create_file": frozenset({"op", "path", "content"}),
    "replace_file": frozenset({"op", "path", "content", "expected_sha256"}),
    "delete_file": frozenset({"op", "path", "expected_sha256"}),
    "replace_text": frozenset({"op", "path", "old_text", "new_text"}),
}


class EditBundleExactMatchError(FreestyleJobRefusedError):
    """One replace_text old_text matched the current file exactly zero times.

    Raised with the same refusal message as before so every existing caller
    still fails closed unchanged; it additionally carries the failed path and
    the operation's anchor text so edit generation can attempt one bounded
    anchor-repair retry with fresh exact snippets. Never raised for schema,
    path, content, ambiguous-match, or approval-sensitive failures, which
    must never trigger any retry.
    """

    def __init__(self, message: str, *, path: str, old_text: str, new_text: str):
        super().__init__(message)
        self.path = path
        self.old_text = old_text
        self.new_text = new_text


@dataclass(frozen=True)
class _LocatedReplacement:
    """One validated replace_text span located in the original file text.

    ``start``/``end`` are character offsets into the original decoded text; the
    span is computed before any write so same-file replacements can be checked
    for overlap and spliced in descending offset order.
    """

    old_text: str
    new_text: str
    start: int
    end: int
    label: str


@dataclass(frozen=True)
class _OperationPlan:
    """One validated operation before same-path replace_text grouping."""

    kind: str
    path: str
    target: Path
    original_bytes: bytes | None
    new_bytes: bytes | None
    located: _LocatedReplacement | None


@dataclass(frozen=True)
class PlannedEdit:
    """One fully validated per-file edit unit, computed before any write.

    A ``replace_text`` unit may fold several non-overlapping same-file
    replacements into one journaled write; ``replacement_count``,
    ``replacement_ranges`` (original-text offsets), and
    ``descending_offset_application`` record that folding for artifacts.
    """

    op: str
    path: str
    target: Path
    new_bytes: bytes | None
    original_bytes: bytes | None
    before_sha256: str | None
    after_sha256: str | None
    replacement_count: int = 1
    replacement_ranges: tuple[tuple[int, int], ...] = ()
    descending_offset_application: bool = False


@dataclass(frozen=True)
class EditBundleValidation:
    """A parsed bundle and its fully validated, non-mutating edit plan."""

    bundle: dict[str, Any]
    planned: tuple[PlannedEdit, ...]


@dataclass
class _EditJournal:
    """Mutations performed so far, in order, for fail-closed rollback."""

    entries: list[tuple[str, Path, bytes | None]] = field(default_factory=list)
    created_directories: list[Path] = field(default_factory=list)


def apply_edit_bundle(
    job_id: str,
    bundle_path: str | Path,
    *,
    project_root: Path | None = None,
    state_root: Path | None = None,
) -> FreestyleJob:
    """Apply one validated edit bundle inside one materialized job worktree."""
    root, state = resolved_roots(project_root or ROOT_DIR, state_root)
    job = get_freestyle_job(job_id, project_root=root, state_root=state)
    _require_explicit_developer_approval(job, "apply edits")
    require_materialized(
        job, "must be successfully materialized before edits can be applied."
    )
    require_no_active_edit_application(job, "; refusing to apply another bundle.")

    started = replace(
        job,
        edit_apply_status="applying",
        edit_apply_started_at=_utc_now(),
        edit_applied_at=None,
        edit_errors=(),
        updated_at=_utc_now(),
    )
    _write_job(started, state)
    journal = _EditJournal()
    try:
        worktree = _validated_job_worktree(started, root, state)
        resolved_worktree = worktree.resolve(strict=True)
        bundle_bytes = _read_bundle_bytes(bundle_path)
        validation = validate_edit_bundle_bytes(started.job_id, bundle_bytes, worktree)
        bundle = validation.bundle
        planned = validation.planned
        _apply_planned_edits(planned, worktree, resolved_worktree, journal)
        completed_at = _utc_now()
        artifact = _build_edit_artifact(
            started, bundle, bundle_bytes, planned, str(bundle_path), completed_at
        )
        artifact_path = _write_edit_artifact(artifact, state, started.job_id)
    except Exception as exc:
        rollback_problem = _rollback_journal(journal)
        errors = [_bounded_error(str(exc), "Edit bundle application failed.")]
        status = "error"
        if rollback_problem is not None:
            status = "inconsistent"
            errors.append(_bounded_error(rollback_problem, "Rollback failed."))
        failed = replace(
            started,
            edit_apply_status=status,
            edit_errors=tuple(errors),
            updated_at=_utc_now(),
        )
        _write_job(failed, state)
        raise FreestyleJobRefusedError(
            f"Could not apply the edit bundle to freestyle job {job_id}; "
            "the error was recorded in local dev state."
        ) from exc

    completed = replace(
        started,
        edit_apply_status="applied",
        edit_applied_at=completed_at,
        edit_applied_file_count=len(planned),
        edit_artifact_path=artifact_path,
        edit_errors=(),
        updated_at=completed_at,
    )
    _write_job(completed, state)
    return completed


# ---------------------------------------------------------------------------
# Bundle reading and schema validation
# ---------------------------------------------------------------------------


def edit_bundle_schema_for_prompt() -> dict[str, Any]:
    """Return the exact accepted version-one field and operation constraints."""
    timestamp = "<ISO-8601 timestamp>"
    sha256 = "<64 lowercase hex SHA-256 of current file bytes>"
    return {
        "bundle": {
            "required_fields": [
                "schema_version",
                "job_id",
                "created_at",
                "operation_count",
                "operations",
            ],
            "optional_fields": ["proposal_id", "goal"],
            "additional_fields_allowed": False,
            "field_constraints": {
                "schema_version": EDIT_BUNDLE_SCHEMA_VERSION,
                "job_id": "<exact 32-character lowercase hex job id supplied>",
                "created_at": timestamp,
                "operation_count": "<integer exactly equal to len(operations)>",
                "operations": f"<array containing 1..{MAX_BUNDLE_OPERATIONS} operations>",
                "proposal_id": f"<optional non-empty text, max {MAX_PROVENANCE_TEXT_CHARS} chars>",
                "goal": f"<optional non-empty text, max {MAX_PROVENANCE_TEXT_CHARS} chars>",
            },
        },
        "operation_variants": [
            {
                "op": "create_file",
                "path": "<safe relative POSIX path for a missing text file>",
                "content": "<full new UTF-8 text>",
            },
            {
                "op": "replace_file",
                "path": "<safe relative POSIX path for an existing text file>",
                "content": "<full replacement UTF-8 text>",
                "expected_sha256": sha256,
            },
            {
                "op": "delete_file",
                "path": "<safe relative POSIX path for an existing text file>",
                "expected_sha256": sha256,
            },
            {
                "op": "replace_text",
                "path": "<safe relative POSIX path for an existing text file>",
                "old_text": "<non-empty UTF-8 text matching exactly once>",
                "new_text": "<replacement UTF-8 text different from old_text>",
            },
        ],
        "operation_constraints": {
            "additional_fields_allowed": False,
            "multiple_operations_per_path": (
                "allowed only when every operation for that path is replace_text "
                "and their exact old_text regions do not overlap; create_file, "
                "replace_file, and delete_file must each be the only operation "
                "for their path (paths compared case-insensitively)"
            ),
            "max_path_chars": MAX_EDIT_PATH_LENGTH,
            "max_file_content_bytes": MAX_EDIT_FILE_BYTES,
            "max_total_content_bytes": MAX_BUNDLE_TOTAL_CONTENT_BYTES,
            "unsafe_secret_db_cache_generated_approval_store_and_executable_paths": "forbidden",
            "approval_sensitive_code_regions": (
                "forbidden in old_text, new_text, and content, and in the "
                "current content of replace_file/delete_file targets; route "
                "wiring elsewhere in the same file is allowed"
            ),
            "binary_or_nul_content": "forbidden",
            "absolute_traversal_backslash_symlink_and_outside_worktree_paths": "forbidden",
        },
    }


def validate_edit_bundle_bytes(
    job_id: str, bundle_bytes: bytes, worktree: Path
) -> EditBundleValidation:
    """Validate one in-memory edit bundle without writing source or metadata."""
    if not isinstance(bundle_bytes, bytes):
        raise FreestyleJobRefusedError("Edit bundle content must be bytes.")
    if len(bundle_bytes) > MAX_BUNDLE_BYTES:
        raise FreestyleJobRefusedError("Edit bundle exceeds the size limit.")
    if worktree.is_symlink() or not worktree.is_dir():
        raise FreestyleJobRefusedError("Job worktree is unsafe for bundle validation.")
    try:
        resolved_worktree = worktree.resolve(strict=True)
    except OSError as exc:
        raise FreestyleJobRefusedError(
            "Job worktree is unsafe for bundle validation."
        ) from exc
    bundle = _parse_bundle(bundle_bytes, job_id)
    planned = _plan_operations(bundle["operations"], worktree, resolved_worktree)
    return EditBundleValidation(bundle=bundle, planned=planned)


def _read_bundle_bytes(bundle_path: str | Path) -> bytes:
    if isinstance(bundle_path, Path):
        path = bundle_path
    elif (
        isinstance(bundle_path, str)
        and bundle_path.strip()
        and "\x00" not in bundle_path
    ):
        path = Path(bundle_path)
    else:
        raise FreestyleJobRefusedError(
            "Edit bundle path must be a non-empty path string."
        )
    if path.suffix.casefold() != ".json":
        raise FreestyleJobRefusedError("Edit bundle must be a .json file.")
    name = path.name.casefold()
    if name.startswith(".") or SECRET_NAME_PATTERN.search(name):
        raise FreestyleJobRefusedError("Edit bundle file name is not safe to read.")
    if path.is_symlink() or not path.is_file():
        raise FreestyleJobRefusedError(
            "Edit bundle path must be an existing regular file."
        )
    try:
        if path.stat().st_size > MAX_BUNDLE_BYTES:
            raise FreestyleJobRefusedError("Edit bundle file exceeds the size limit.")
        return path.read_bytes()
    except OSError as exc:
        raise FreestyleJobRefusedError("Edit bundle file is unreadable.") from exc


def _parse_bundle(bundle_bytes: bytes, job_id: str) -> dict[str, Any]:
    try:
        payload = json.loads(bundle_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FreestyleJobRefusedError("Edit bundle is not valid UTF-8 JSON.") from exc
    if not isinstance(payload, dict):
        raise FreestyleJobRefusedError("Edit bundle must be one JSON object.")
    fields = set(payload)
    if not _BUNDLE_REQUIRED_FIELDS <= fields:
        raise FreestyleJobRefusedError("Edit bundle is missing required fields.")
    if fields - _BUNDLE_REQUIRED_FIELDS - _BUNDLE_OPTIONAL_FIELDS:
        raise FreestyleJobRefusedError("Edit bundle contains unexpected fields.")
    version = payload["schema_version"]
    if isinstance(version, bool) or version != EDIT_BUNDLE_SCHEMA_VERSION:
        raise FreestyleJobRefusedError("Edit bundle schema_version is unsupported.")
    bundle_job_id = payload["job_id"]
    if (
        not isinstance(bundle_job_id, str)
        or len(bundle_job_id) != 32
        or any(character not in "0123456789abcdef" for character in bundle_job_id)
    ):
        raise FreestyleJobRefusedError("Edit bundle job_id is invalid.")
    if bundle_job_id != job_id:
        raise FreestyleJobRefusedError(
            f"Edit bundle is bound to a different job id ({bundle_job_id})."
        )
    created_at = payload["created_at"]
    if not isinstance(created_at, str) or len(created_at) > 64:
        raise FreestyleJobRefusedError(
            "Edit bundle created_at must be a short ISO timestamp."
        )
    try:
        datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise FreestyleJobRefusedError(
            "Edit bundle created_at must be a short ISO timestamp."
        ) from exc
    operations = payload["operations"]
    if (
        not isinstance(operations, list)
        or not 1 <= len(operations) <= MAX_BUNDLE_OPERATIONS
    ):
        raise FreestyleJobRefusedError(
            f"Edit bundle operations must contain 1..{MAX_BUNDLE_OPERATIONS} items."
        )
    count = payload["operation_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count != len(operations):
        raise FreestyleJobRefusedError(
            "Edit bundle operation_count does not match its operations."
        )
    for field_name in ("proposal_id", "goal"):
        if field_name in payload:
            value = payload[field_name]
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > MAX_PROVENANCE_TEXT_CHARS
                or any(ord(character) < 32 for character in value)
            ):
                raise FreestyleJobRefusedError(
                    f"Edit bundle {field_name} is invalid."
                )
    return payload


# ---------------------------------------------------------------------------
# Operation planning (validation only; never mutates the worktree)
# ---------------------------------------------------------------------------


def _plan_operations(
    raw_operations: list[Any], worktree: Path, resolved_worktree: Path
) -> tuple[PlannedEdit, ...]:
    operations = _validate_operations(raw_operations, worktree, resolved_worktree)
    planned = _grouped_planned_edits(operations)
    _require_non_conflicting_edit_paths([item.path for item in planned])
    total_content_bytes = sum(
        len(item.original_bytes or b"") + len(item.new_bytes or b"")
        for item in planned
    )
    if total_content_bytes > MAX_BUNDLE_TOTAL_CONTENT_BYTES:
        raise FreestyleJobRefusedError(
            "Edit bundle exceeds the total content size limit."
        )
    return planned


def _validate_operations(
    raw_operations: list[Any], worktree: Path, resolved_worktree: Path
) -> list[_OperationPlan]:
    """Validate every operation independently before same-path grouping.

    replace_text operations are only *located* here (their unique match span is
    computed against the original file); the actual splice happens once per file
    in grouping so several same-file replacements compose correctly.
    """
    operations: list[_OperationPlan] = []
    for index, raw_operation in enumerate(raw_operations):
        label = f"operations[{index}]"
        if not isinstance(raw_operation, dict):
            raise FreestyleJobRefusedError(f"Edit bundle {label} must be an object.")
        kind = raw_operation.get("op")
        if not isinstance(kind, str) or kind not in _OPERATION_FIELDS:
            raise FreestyleJobRefusedError(
                f"Edit bundle {label} has an unsupported op kind."
            )
        if set(raw_operation) != _OPERATION_FIELDS[kind]:
            raise FreestyleJobRefusedError(
                f"Edit bundle {label} contains unexpected or missing fields."
            )
        path = _validated_edit_path(raw_operation.get("path"), worktree, label)
        located: _LocatedReplacement | None = None
        if kind == "create_file":
            target = _validated_edit_target(
                path, worktree, resolved_worktree, label, expect_file=False
            )
            new_bytes: bytes | None = _validated_content_bytes(
                raw_operation.get("content"), label, "content"
            )
            original_bytes: bytes | None = None
        else:
            target = _validated_edit_target(
                path, worktree, resolved_worktree, label, expect_file=True
            )
            original_bytes = _read_current_text_file(target, path, label)
            if kind == "replace_text":
                located = _located_replacement(
                    original_bytes.decode("utf-8"), raw_operation, path, label
                )
                new_bytes = None
            else:
                _require_fresh_expected_sha256(
                    raw_operation.get("expected_sha256"), original_bytes, path, label
                )
                new_bytes = (
                    _validated_content_bytes(
                        raw_operation.get("content"), label, "content"
                    )
                    if kind == "replace_file"
                    else None
                )
        _require_no_approval_sensitive_text(
            kind, raw_operation, original_bytes, path, label
        )
        operations.append(
            _OperationPlan(
                kind=kind,
                path=path,
                target=target,
                original_bytes=original_bytes,
                new_bytes=new_bytes,
                located=located,
            )
        )
    return operations


def _grouped_planned_edits(
    operations: list[_OperationPlan],
) -> tuple[PlannedEdit, ...]:
    """Fold same-path replace_text operations into one per-file edit unit.

    A path with more than one operation is accepted only when every operation
    for it is replace_text; any whole-file operation must be the sole operation
    for its path, so it fails closed otherwise.
    """
    groups: dict[str, list[_OperationPlan]] = {}
    order: list[str] = []
    for operation in operations:
        if operation.path not in groups:
            groups[operation.path] = []
            order.append(operation.path)
        groups[operation.path].append(operation)
    planned: list[PlannedEdit] = []
    for path in order:
        group = groups[path]
        if len(group) > 1 and any(item.kind != "replace_text" for item in group):
            raise FreestyleJobRefusedError(
                "Edit bundle contains more than one operation for the same path."
            )
        if group[0].kind == "replace_text":
            planned.append(_combined_text_edit(group, path))
        else:
            planned.append(_whole_file_planned_edit(group[0]))
    return tuple(planned)


def _whole_file_planned_edit(operation: _OperationPlan) -> PlannedEdit:
    return PlannedEdit(
        op=operation.kind,
        path=operation.path,
        target=operation.target,
        new_bytes=operation.new_bytes,
        original_bytes=operation.original_bytes,
        before_sha256=_sha256_or_none(operation.original_bytes),
        after_sha256=_sha256_or_none(operation.new_bytes),
    )


def _combined_text_edit(group: list[_OperationPlan], path: str) -> PlannedEdit:
    """Splice one or more non-overlapping same-file replacements into one write.

    Every span was already required to match the original file exactly once.
    Here the spans are additionally required not to overlap, then applied in
    descending start-offset order so each splice leaves the earlier (smaller)
    offsets of the original text valid.
    """
    original_bytes = group[0].original_bytes or b""
    text = original_bytes.decode("utf-8")
    replacements = [item.located for item in group if item.located is not None]
    ordered = sorted(replacements, key=lambda item: item.start)
    for previous, current in zip(ordered, ordered[1:]):
        if current.start < previous.end:
            raise FreestyleJobRefusedError(
                f"Edit bundle replace_text operations for {path!r} overlap; "
                "refusing an ambiguous multi-edit."
            )
    result = text
    for replacement in sorted(
        replacements, key=lambda item: item.start, reverse=True
    ):
        result = (
            result[: replacement.start]
            + replacement.new_text
            + result[replacement.end :]
        )
    new_bytes = result.encode("utf-8")
    if len(new_bytes) > MAX_EDIT_FILE_BYTES:
        raise FreestyleJobRefusedError(
            f"Edit bundle replace_text result for {path!r} exceeds the edit "
            "size limit."
        )
    return PlannedEdit(
        op="replace_text",
        path=path,
        target=group[0].target,
        new_bytes=new_bytes,
        original_bytes=original_bytes,
        before_sha256=_sha256_or_none(original_bytes),
        after_sha256=_sha256_or_none(new_bytes),
        replacement_count=len(replacements),
        replacement_ranges=tuple((item.start, item.end) for item in ordered),
        descending_offset_application=len(replacements) > 1,
    )


def _validated_edit_path(raw_path: Any, worktree: Path, label: str) -> str:
    if not isinstance(raw_path, str) or not raw_path:
        raise FreestyleJobRefusedError(
            f"Edit bundle {label} path must be non-empty text."
        )
    if len(raw_path) > MAX_EDIT_PATH_LENGTH:
        raise FreestyleJobRefusedError(f"Edit bundle {label} path is too long.")
    if (
        "\x00" in raw_path
        or "\\" in raw_path
        or any(ord(character) < 32 for character in raw_path)
        or raw_path.endswith("/")
    ):
        raise FreestyleJobRefusedError(
            f"Edit bundle {label} path contains an invalid character or form."
        )
    parts = raw_path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise FreestyleJobRefusedError(
            f"Edit bundle {label} path uses traversal or empty segments."
        )
    for part in parts:
        if (
            any(character in '<>:"|?*' for character in part)
            or part.endswith((" ", "."))
            or part.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_NAMES
        ):
            raise FreestyleJobRefusedError(
                f"Edit bundle {label} path contains an unsupported file system name."
            )
    posix_path = PurePosixPath(raw_path)
    windows_path = PureWindowsPath(raw_path)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise FreestyleJobRefusedError(f"Edit bundle {label} path must be relative.")
    try:
        safe_path = validate_relative_project_path(
            raw_path, worktree, f"Edit bundle {label} paths"
        )
    except ValueError as exc:
        raise FreestyleJobRefusedError(str(exc)) from exc
    lowered = tuple(part.casefold() for part in posix_path.parts)
    if any(part in GENERATED_OR_CACHE_PARTS for part in lowered):
        raise FreestyleJobRefusedError(
            f"Edit bundle {label} path targets a generated or cache directory."
        )
    name = lowered[-1]
    normalized_name = re.sub(r"[^a-z0-9]", "", name)
    if name in APPROVAL_STORE_PATH_NAMES or APPROVAL_STORE_TOKEN in normalized_name:
        raise FreestyleJobRefusedError(
            f"Edit bundle {label} path targets approval-record code."
        )
    if posix_path.suffix.casefold() in _EXECUTABLE_SUFFIXES:
        raise FreestyleJobRefusedError(
            f"Edit bundle {label} path would create or edit an executable file type."
        )
    return safe_path


def _validated_edit_target(
    safe_path: str,
    worktree: Path,
    resolved_worktree: Path,
    label: str,
    *,
    expect_file: bool,
) -> Path:
    parts = PurePosixPath(safe_path).parts
    current = worktree
    for part in parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise FreestyleJobRefusedError(
                f"Edit bundle {label} path crosses a symlink at {part!r}."
            )
        if current.exists():
            if not current.is_dir():
                raise FreestyleJobRefusedError(
                    f"Edit bundle {label} path uses existing file {part!r} "
                    "as a directory."
                )
            _require_inside_worktree(current, resolved_worktree, label)
    target = worktree.joinpath(*parts)
    if target.is_symlink():
        raise FreestyleJobRefusedError(
            f"Edit bundle {label} target {safe_path!r} is a symlink."
        )
    if expect_file:
        if not target.is_file():
            raise FreestyleJobRefusedError(
                f"Edit bundle {label} target {safe_path!r} is missing or not "
                "a regular file."
            )
        _require_inside_worktree(target, resolved_worktree, label)
    elif target.exists():
        raise FreestyleJobRefusedError(
            f"Edit bundle {label} create_file target {safe_path!r} already exists."
        )
    return target


def _require_inside_worktree(path: Path, resolved_worktree: Path, label: str) -> None:
    try:
        path.resolve(strict=True).relative_to(resolved_worktree)
    except (OSError, ValueError) as exc:
        raise FreestyleJobRefusedError(
            f"Edit bundle {label} path escapes the isolated job worktree."
        ) from exc


def _read_current_text_file(target: Path, path: str, label: str) -> bytes:
    try:
        if target.stat().st_size > MAX_EDIT_FILE_BYTES:
            raise FreestyleJobRefusedError(
                f"Edit bundle {label} target {path!r} is oversized for text edits."
            )
        data = target.read_bytes()
    except OSError as exc:
        raise FreestyleJobRefusedError(
            f"Edit bundle {label} target {path!r} is unreadable."
        ) from exc
    if b"\x00" in data:
        raise FreestyleJobRefusedError(
            f"Edit bundle {label} target {path!r} is a binary file."
        )
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FreestyleJobRefusedError(
            f"Edit bundle {label} target {path!r} is a binary file."
        ) from exc
    return data


def _validated_content_bytes(value: Any, label: str, field_name: str) -> bytes:
    if not isinstance(value, str):
        raise FreestyleJobRefusedError(
            f"Edit bundle {label} {field_name} must be text."
        )
    if "\x00" in value:
        raise FreestyleJobRefusedError(
            f"Edit bundle {label} {field_name} contains binary (NUL) content."
        )
    data = value.encode("utf-8")
    if len(data) > MAX_EDIT_FILE_BYTES:
        raise FreestyleJobRefusedError(
            f"Edit bundle {label} {field_name} exceeds the "
            f"{MAX_EDIT_FILE_BYTES}-byte edit limit."
        )
    return data


def _require_fresh_expected_sha256(
    expected: Any, original_bytes: bytes, path: str, label: str
) -> None:
    if (
        not isinstance(expected, str)
        or _EXPECTED_SHA256_PATTERN.fullmatch(expected) is None
    ):
        raise FreestyleJobRefusedError(
            f"Edit bundle {label} expected_sha256 must be 64 lowercase "
            "hex characters."
        )
    actual = hashlib.sha256(original_bytes).hexdigest()
    if actual != expected:
        raise FreestyleJobRefusedError(
            f"Edit bundle {label} expected_sha256 is stale for {path!r}: "
            f"the bundle expects {expected} but the current file is {actual}."
        )


def _located_replacement(
    text: str, raw_operation: dict[str, Any], path: str, label: str
) -> _LocatedReplacement:
    """Validate one replace_text op and locate its unique span in ``text``.

    Raises ``EditBundleExactMatchError`` (and only that) for zero matches so a
    bounded anchor-repair retry can be attempted upstream; ambiguous matches,
    identical text, and oversized or binary text stay plain refusals that never
    trigger any retry.
    """
    old_text = raw_operation.get("old_text")
    new_text = raw_operation.get("new_text")
    if not isinstance(old_text, str) or not old_text:
        raise FreestyleJobRefusedError(
            f"Edit bundle {label} old_text must be non-empty text."
        )
    for field_name, value in (("old_text", old_text), ("new_text", new_text)):
        if not isinstance(value, str) or "\x00" in value:
            raise FreestyleJobRefusedError(
                f"Edit bundle {label} {field_name} must be text without NUL bytes."
            )
        if len(value.encode("utf-8")) > MAX_EDIT_FILE_BYTES:
            raise FreestyleJobRefusedError(
                f"Edit bundle {label} {field_name} exceeds the edit size limit."
            )
    if old_text == new_text:
        raise FreestyleJobRefusedError(
            f"Edit bundle {label} old_text and new_text are identical."
        )
    occurrences = text.count(old_text)
    if occurrences == 0:
        raise EditBundleExactMatchError(
            f"Edit bundle {label} replace_text found no exact match in {path!r}; "
            "the expected old_text is stale or wrong.",
            path=path,
            old_text=old_text,
            new_text=new_text,
        )
    if occurrences > 1:
        raise FreestyleJobRefusedError(
            f"Edit bundle {label} replace_text matched {occurrences} times in "
            f"{path!r}; refusing an ambiguous replacement."
        )
    start = text.index(old_text)
    return _LocatedReplacement(
        old_text=old_text,
        new_text=new_text,
        start=start,
        end=start + len(old_text),
        label=label,
    )


def _require_no_approval_sensitive_text(
    kind: str,
    raw_operation: dict[str, Any],
    original_bytes: bytes | None,
    path: str,
    label: str,
) -> None:
    """Fail closed when one operation would touch approval-sensitive code.

    Path-level checks already block the approval-record implementation files
    outright. This region-level guard additionally protects files that merely
    reference approval code (routes, dashboards, tests): a replace_text whose
    old_text or new_text touches an approval-sensitive marker is refused, and
    whole-file create/replace/delete operations are refused whenever the
    written content or the targeted current content contains one. replace_text
    old_text/new_text is additionally refused when it carries sensitive secret
    content. Edits to the rest of the same file remain allowed.
    """
    if kind == "replace_text":
        scanned = [
            ("old_text", str(raw_operation.get("old_text", ""))),
            ("new_text", str(raw_operation.get("new_text", ""))),
        ]
    elif kind == "create_file":
        scanned = [("content", str(raw_operation.get("content", "")))]
    else:
        scanned = [
            ("current target content", (original_bytes or b"").decode("utf-8"))
        ]
        if kind == "replace_file":
            scanned.append(("content", str(raw_operation.get("content", ""))))
    for field_name, text in scanned:
        matched = approval_sensitive_region_labels(text)
        if matched:
            raise FreestyleJobRefusedError(
                f"Edit bundle {label} {field_name} touches an "
                f"approval-sensitive region in {path!r} "
                f"({', '.join(matched)}); approval-related code is only "
                "changed by a human."
            )
        if kind == "replace_text" and contains_sensitive_content(text):
            raise FreestyleJobRefusedError(
                f"Edit bundle {label} {field_name} touches sensitive secret "
                f"content in {path!r}; secrets are only changed by a human."
            )


def _require_non_conflicting_edit_paths(paths: list[str]) -> None:
    folded = sorted(path.casefold() for path in paths)
    if len(folded) != len(set(folded)):
        raise FreestyleJobRefusedError(
            "Edit bundle contains more than one operation for the same path."
        )
    for previous, current in zip(folded, folded[1:]):
        if current.startswith(previous + "/"):
            raise FreestyleJobRefusedError(
                "Edit bundle contains conflicting file and directory paths."
            )


# ---------------------------------------------------------------------------
# Journaled application and rollback
# ---------------------------------------------------------------------------


def _apply_planned_edits(
    planned: tuple[PlannedEdit, ...],
    worktree: Path,
    resolved_worktree: Path,
    journal: _EditJournal,
) -> None:
    for item in planned:
        if item.op == "create_file":
            _create_missing_parents(item.target, worktree, resolved_worktree, journal)
            if item.target.exists() or item.target.is_symlink():
                raise FreestyleJobRefusedError(
                    f"Edit target {item.path!r} changed during application."
                )
            journal.entries.append(("created", item.target, None))
            _write_new_file(item.target, item.new_bytes or b"")
        elif item.op == "delete_file":
            _require_unchanged_regular_file(item.target, item.path)
            journal.entries.append(("deleted", item.target, item.original_bytes))
            _remove_file(item.target)
        else:
            _require_unchanged_regular_file(item.target, item.path)
            journal.entries.append(("replaced", item.target, item.original_bytes))
            _overwrite_file(item.target, item.new_bytes or b"")


def _require_unchanged_regular_file(target: Path, path: str) -> None:
    if target.is_symlink() or not target.is_file():
        raise FreestyleJobRefusedError(
            f"Edit target {path!r} changed during application."
        )


def _create_missing_parents(
    target: Path, worktree: Path, resolved_worktree: Path, journal: _EditJournal
) -> None:
    relative_parts = target.relative_to(worktree).parts[:-1]
    current = worktree
    for part in relative_parts:
        current = current / part
        if current.is_symlink():
            raise FreestyleJobRefusedError(
                "Edit parent directory changed during application."
            )
        if not current.exists():
            current.mkdir()
            journal.created_directories.append(current)
        elif not current.is_dir():
            raise FreestyleJobRefusedError(
                "Edit parent directory changed during application."
            )
        else:
            _require_inside_worktree(current, resolved_worktree, "application")


def _write_new_file(target: Path, data: bytes) -> None:
    """Create one new non-executable file; fails if the target appears."""
    with target.open("xb") as handle:
        handle.write(data)
    target.chmod(0o644)


def _overwrite_file(target: Path, data: bytes) -> None:
    """Replace the full contents of one existing validated text file."""
    with target.open("wb") as handle:
        handle.write(data)
    target.chmod(0o644)


def _remove_file(target: Path) -> None:
    target.unlink()


def _rollback_journal(journal: _EditJournal) -> str | None:
    """Restore every journaled mutation in reverse order; report any failure."""
    problems: list[str] = []
    for action, target, original_bytes in reversed(journal.entries):
        try:
            if action == "created":
                target.unlink(missing_ok=True)
            elif original_bytes is not None:
                with target.open("wb") as handle:
                    handle.write(original_bytes)
                target.chmod(0o644)
        except OSError as exc:
            problems.append(f"{target.name}: {exc}")
    for directory in reversed(journal.created_directories):
        with suppress(OSError):
            directory.rmdir()
    if problems:
        return "Rollback could not restore: " + "; ".join(problems)
    return None


# ---------------------------------------------------------------------------
# Result artifact
# ---------------------------------------------------------------------------


def _build_edit_artifact(
    job: FreestyleJob,
    bundle: dict[str, Any],
    bundle_bytes: bytes,
    planned: tuple[PlannedEdit, ...],
    bundle_path_text: str,
    applied_at: str,
) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "branch": job.branch,
        "base_commit": job.base_commit,
        "schema_version": EDIT_BUNDLE_SCHEMA_VERSION,
        "bundle_path": bundle_path_text[:MAX_PROVENANCE_TEXT_CHARS],
        "bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "bundle_created_at": bundle["created_at"],
        "proposal_id": bundle.get("proposal_id"),
        "goal": bundle.get("goal"),
        "applied_at": applied_at,
        "operation_count": sum(item.replacement_count for item in planned),
        "touched_file_count": len(planned),
        "operations": [_operation_artifact_record(item) for item in planned],
        "policy_notes": [
            "Proposal-context allowlisting is a later policy hook; every path "
            "was revalidated independently before this application.",
            "Multiple replace_text operations on one path are spliced once in "
            "descending original-offset order; whole-file operations stay "
            "one-per-path.",
        ],
    }


def _operation_artifact_record(item: PlannedEdit) -> dict[str, Any]:
    """Summarize one applied per-file edit unit, including multi-op folding."""
    record: dict[str, Any] = {
        "op": item.op,
        "path": item.path,
        "before_sha256": item.before_sha256,
        "after_sha256": item.after_sha256,
        "replacement_count": item.replacement_count,
    }
    if item.op == "replace_text":
        record["replacement_ranges"] = [
            [start, end] for start, end in item.replacement_ranges
        ]
        record["descending_offset_application"] = item.descending_offset_application
    return record


def _write_edit_artifact(artifact: dict[str, Any], state: Path, job_id: str) -> str:
    return write_artifact_bytes(
        state,
        "edits",
        job_id,
        f"apply-{uuid.uuid4().hex[:12]}.json",
        encode_json_artifact(artifact),
        label="edit",
    )


def _sha256_or_none(data: bytes | None) -> str | None:
    if data is None:
        return None
    return hashlib.sha256(data).hexdigest()
