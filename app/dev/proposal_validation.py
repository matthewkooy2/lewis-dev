"""Stage 7B — Pure validation foundation for a future local branch applicator.

Validates approved code-proposal approval payloads and parses/validates
unified diff metadata. Does NOT apply patches, create branches, run git,
run tests, or modify files.

Explicitly deferred to Stage 7C (filesystem preflight / post-apply
verification), because a static parser cannot decide them:

* Deletions expressed as edits that leave a file empty (no ``/dev/null``
  and no ``deleted file mode`` metadata) cannot be detected here; Stage 7C
  must verify post-apply that no touched file was effectively deleted.
* Existing symlink targets and symlinked parent directories inside the
  project root require an on-disk preflight check before application.
* Stage 7C must revalidate the approval payload immediately before
  application; ParsedPatch/ValidatedProposal are immutable evidence of one
  validation pass, not a standing capability.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from app.approvals import ApprovalItem
from app.code_proposal_path_safety import validate_relative_project_path
from app.config import ROOT_DIR

# ---------------------------------------------------------------------------
# Conservative limits (documented per Rule 18 from the Stage 7A design)
# ---------------------------------------------------------------------------
MAX_PATCH_SIZE_BYTES = 64 * 1024  # 64 KiB
MAX_TOUCHED_FILES = 20
MAX_HUNKS_PER_FILE = 20
MAX_TOTAL_HUNKS = 40
MAX_PATH_LENGTH = 240
APPLICATOR_CONTRACT_VERSION = 1

_HUNK_HEADER_PATTERN = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?: .*)?$"
)
_NO_NEWLINE_MARKER = r"\ No newline at end of file"
_SUBMODULE_BODY_PATTERN = re.compile(r"^Subproject commit [0-9a-fA-F]{40}$")
# Conservative Git index metadata shape: "index <hex>..<hex>" with an
# optional six-digit octal mode. Anything else is refused.
_INDEX_LINE_PATTERN = re.compile(
    r"^index [0-9a-f]{4,64}\.\.[0-9a-f]{4,64}(?: [0-7]{6})?$"
)
# Characters Windows forbids in file names. ":" is checked separately.
_WINDOWS_ILLEGAL_PATH_CHARACTERS = set('<>"|?*')
_WINDOWS_RESERVED_NAMES = {
    "aux",
    "con",
    "conin$",
    "conout$",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}
# ---------------------------------------------------------------------------
# Error / result types
# ---------------------------------------------------------------------------


class ApplicatorValidationError(ValueError):
    """Raised when an approval payload fails applicator validation."""


@dataclass(frozen=True)
class ParsedPatch:
    """Parsed metadata extracted from a unified diff preview.

    All fields are read-only and immutable (tuples, not lists), so a
    ParsedPatch cannot be mutated between validation and any future use.
    Stage 7C must still revalidate immediately before application.
    Nothing is applied.
    """

    touched_paths: tuple[str, ...]
    file_count: int
    hunk_count: int
    total_added_lines: int
    total_removed_lines: int
    patch_size_bytes: int
    is_unified_diff: bool
    has_binary_marker: bool
    has_dev_null: bool
    has_rename: bool
    has_copy: bool
    has_mode_change: bool
    has_symlink: bool
    has_submodule: bool
    has_combined_diff: bool
    has_absolute_path: bool
    has_parent_traversal: bool
    has_backslash_path: bool
    has_empty_patch: bool
    has_malformed_header: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidatedProposal:
    """A successfully validated approved code-proposal ready for future application.

    Immutable (frozen dataclass with tuple fields). Stage 7C must still
    revalidate the approval payload immediately before application; this
    object is evidence of one validation pass, not a standing capability.
    """

    approval_id: int
    action_type: str
    requested_goal: str
    patch_preview: str
    patch_preview_kind: str
    patch_is_inert: bool
    patch_preview_implementation_ready: bool
    touched_paths: tuple[str, ...]
    selected_strategy: str
    parsed_patch: ParsedPatch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_approved_code_proposal(approval: ApprovalItem) -> ValidatedProposal:
    """Validate a raw approval item for future branch application.

    Returns a ValidatedProposal on success. Raises ApplicatorValidationError on
    any refusal.
    """
    _validate_approval_item(approval)
    payload = approval.payload
    _validate_payload_contract(payload)
    patch_preview = payload["patch_preview"]
    parsed = parse_unified_diff_metadata(patch_preview)
    _validate_patch_metadata(parsed)
    validate_patch_contract(payload, parsed)
    return ValidatedProposal(
        approval_id=approval.id,
        action_type=approval.action_type,
        requested_goal=payload.get("requested_goal", ""),
        patch_preview=patch_preview,
        patch_preview_kind=payload["patch_preview_kind"],
        patch_is_inert=payload["patch_is_inert"],
        patch_preview_implementation_ready=payload[
            "patch_preview_implementation_ready"
        ],
        touched_paths=parsed.touched_paths,
        selected_strategy=payload.get("selected_strategy", ""),
        parsed_patch=parsed,
    )


def parse_unified_diff_metadata(patch_preview: str) -> ParsedPatch:
    """Parse metadata from a unified diff string without applying it.

    Returns a ParsedPatch dataclass with all extracted metadata fields.
    Never modifies files, runs git, or executes subprocesses.
    """
    patch_size = len(patch_preview.encode("utf-8"))
    if patch_size == 0 or not patch_preview.strip():
        return _parsed_patch(
            patch_size=patch_size, empty=True, errors=["Patch preview is empty."]
        )

    errors: list[str] = []
    lines = _split_patch_lines(patch_preview, errors)
    flags = _empty_patch_flags()
    touched_paths: list[str] = []
    seen_paths: set[str] = set()
    hunk_count = 0
    total_added = 0
    total_removed = 0
    has_malformed_header = False

    if "\x00" in patch_preview:
        errors.append("Patch content contains a NUL byte.")

    section_starts = [
        index for index, line in enumerate(lines) if line.startswith("diff --git ")
    ]
    if not section_starts:
        has_malformed_header = True
        errors.append("No diff --git header found in patch preview.")
    elif section_starts[0] != 0:
        has_malformed_header = True
        errors.append("Patch content must begin with a diff --git header.")

    for section_index, start in enumerate(section_starts):
        end = (
            section_starts[section_index + 1]
            if section_index + 1 < len(section_starts)
            else len(lines)
        )
        result = _parse_diff_section(lines[start:end], section_index + 1, errors, flags)
        hunk_count += result["hunk_count"]
        total_added += result["total_added"]
        total_removed += result["total_removed"]
        path = result["path"]
        if path is None:
            has_malformed_header = True
        elif path in seen_paths:
            has_malformed_header = True
            errors.append(f"Duplicate diff section for path {path!r}.")
        else:
            seen_paths.add(path)
            touched_paths.append(path)

    for line in lines[: section_starts[0] if section_starts else len(lines)]:
        _record_unsupported_metadata(line, flags)
    if flags["combined"]:
        has_malformed_header = True

    _append_flag_errors(flags, errors)

    return ParsedPatch(
        touched_paths=tuple(touched_paths),
        file_count=len(touched_paths),
        hunk_count=hunk_count,
        total_added_lines=total_added,
        total_removed_lines=total_removed,
        patch_size_bytes=patch_size,
        is_unified_diff=bool(section_starts) and not has_malformed_header,
        has_binary_marker=flags["binary"],
        has_dev_null=flags["dev_null"],
        has_rename=flags["rename"],
        has_copy=flags["copy"],
        has_mode_change=flags["mode"],
        has_symlink=flags["symlink"],
        has_submodule=flags["submodule"],
        has_combined_diff=flags["combined"],
        has_absolute_path=flags["absolute"],
        has_parent_traversal=flags["traversal"],
        has_backslash_path=flags["backslash"],
        has_empty_patch=False,
        has_malformed_header=has_malformed_header,
        errors=tuple(errors),
    )


def validate_patch_contract(payload: dict[str, Any], parsed_patch: ParsedPatch) -> None:
    """Cross-validate parsed patch metadata against payload contract fields.

    Raises ApplicatorValidationError if any contract check fails.
    """
    declared_paths = sorted(payload.get("touched_paths", []))
    parsed_paths = sorted(parsed_patch.touched_paths)

    if (
        len(declared_paths) != len(set(declared_paths))
        or declared_paths != parsed_paths
    ):
        raise ApplicatorValidationError(
            f"Parsed patch paths {parsed_paths} do not match "
            f"payload touched_paths {declared_paths}."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_approval_item(approval: ApprovalItem) -> None:
    """Validate the ApprovalItem envelope (action type and status)."""
    if approval.action_type != "propose_code_change":
        raise ApplicatorValidationError(
            f"Approval {approval.id} is not a code change proposal "
            f"(action_type={approval.action_type!r})."
        )
    if approval.status != "approved":
        raise ApplicatorValidationError(
            f"Approval {approval.id} has status {approval.status!r}, "
            f"expected 'approved'."
        )
    if not isinstance(approval.payload, dict):
        raise ApplicatorValidationError(
            f"Approval {approval.id} payload is not a dict."
        )


def _validate_payload_contract(payload: dict[str, Any]) -> None:
    """Validate the full replacement-engine structured contract fields.

    Checks are ordered: presence checks first, then type checks, then value checks.
    This ensures missing fields produce clear "Missing required structured field"
    diagnostics before value-level errors.
    """
    # --- Presence checks (Stage 6B structured fields) ---
    for field_name in (
        "applicator_contract_version",
        "selected_strategy",
        "patch_preview_kind",
        "patch_is_inert",
        "commands_run",
        "files_modified",
    ):
        if field_name not in payload:
            raise ApplicatorValidationError(
                f"Missing required structured field: {field_name!r}. "
                f"Legacy/non-contract propose_code_change payloads are not "
                f"applicator-eligible."
            )

    # --- Type checks ---
    _require_field(payload, "applicator_contract_version", int)
    _require_field(payload, "selected_strategy", str)
    _require_field(payload, "patch_preview_kind", str)
    _require_field(payload, "patch_is_inert", bool)

    # --- Gating field value checks ---
    patch_preview = payload.get("patch_preview")
    if not isinstance(patch_preview, str) or not patch_preview.strip():
        raise ApplicatorValidationError("Approval payload has no patch_preview string.")

    if payload["applicator_contract_version"] != APPLICATOR_CONTRACT_VERSION:
        raise ApplicatorValidationError(
            f"applicator_contract_version must be {APPLICATOR_CONTRACT_VERSION}."
        )

    if not payload["selected_strategy"].strip():
        raise ApplicatorValidationError("selected_strategy must be a non-empty string.")

    patch_preview_kind = payload.get("patch_preview_kind")
    if patch_preview_kind != "unified diff":
        raise ApplicatorValidationError(
            f"patch_preview_kind must be 'unified diff' (space), "
            f"got {patch_preview_kind!r}."
        )

    if payload.get("patch_is_inert") is not True:
        raise ApplicatorValidationError("patch_is_inert must be True.")

    impl_ready = payload.get("patch_preview_implementation_ready")
    if impl_ready is not True:
        raise ApplicatorValidationError(
            "patch_preview_implementation_ready must be exactly True."
        )

    # --- Touched paths checks ---
    touched_paths = payload.get("touched_paths")
    if not isinstance(touched_paths, list) or not touched_paths:
        raise ApplicatorValidationError(
            "touched_paths must be a non-empty list of strings."
        )
    if not all(isinstance(p, str) and p for p in touched_paths):
        raise ApplicatorValidationError(
            "All touched_paths entries must be non-empty strings."
        )

    for raw_path in touched_paths:
        try:
            _validate_safe_path(raw_path)
        except (ValueError, OSError) as exc:
            raise ApplicatorValidationError(str(exc)) from exc

    # --- Audit / proof fields with value checks ---
    commands_run = payload.get("commands_run")
    if not isinstance(commands_run, list):
        raise ApplicatorValidationError("commands_run must be a list.")

    files_modified = payload.get("files_modified")
    if not isinstance(files_modified, list):
        raise ApplicatorValidationError("files_modified must be a list.")
    if files_modified != []:
        raise ApplicatorValidationError(
            f"files_modified must be empty (producer assertion), "
            f"got {files_modified!r}."
        )


def _validate_patch_metadata(parsed: ParsedPatch) -> None:
    """Validate parsed patch metadata against conservative limits and safety rules.

    These checks are independent of the payload assertions.
    """
    if parsed.errors:
        raise ApplicatorValidationError(
            "Patch metadata validation failed: " + "; ".join(parsed.errors)
        )

    if parsed.has_empty_patch:
        raise ApplicatorValidationError("Patch preview is empty.")

    if not parsed.is_unified_diff:
        raise ApplicatorValidationError("Patch preview is not a unified diff.")

    if parsed.has_malformed_header:
        raise ApplicatorValidationError(
            "Patch preview has malformed or missing diff headers."
        )

    if parsed.has_combined_diff:
        raise ApplicatorValidationError("Combined diffs are not supported.")

    if parsed.has_binary_marker:
        raise ApplicatorValidationError(
            "Binary patches are not supported (GIT binary patch marker detected)."
        )

    if parsed.has_dev_null:
        raise ApplicatorValidationError("File deletion via /dev/null is not supported.")

    if parsed.has_rename:
        raise ApplicatorValidationError("Rename diffs are not supported.")

    if parsed.has_copy:
        raise ApplicatorValidationError("Copy diffs are not supported.")

    if parsed.has_mode_change:
        raise ApplicatorValidationError("Mode changes are not supported.")

    if parsed.has_symlink:
        raise ApplicatorValidationError("Symlink changes are not supported.")

    if parsed.has_submodule:
        raise ApplicatorValidationError("Submodule changes are not supported.")

    if parsed.has_absolute_path:
        raise ApplicatorValidationError("Patch contains absolute paths.")

    if parsed.has_parent_traversal:
        raise ApplicatorValidationError(
            "Patch paths must not contain parent traversal (..)."
        )

    if parsed.has_backslash_path:
        raise ApplicatorValidationError(
            "Patch paths must use POSIX-style forward slashes."
        )

    # Conservative limits
    if parsed.patch_size_bytes > MAX_PATCH_SIZE_BYTES:
        raise ApplicatorValidationError(
            f"Patch size {parsed.patch_size_bytes} bytes exceeds "
            f"maximum {MAX_PATCH_SIZE_BYTES} bytes."
        )

    if parsed.file_count > MAX_TOUCHED_FILES:
        raise ApplicatorValidationError(
            f"Patch touches {parsed.file_count} files, exceeding "
            f"maximum {MAX_TOUCHED_FILES}."
        )

    if parsed.hunk_count > MAX_TOTAL_HUNKS:
        raise ApplicatorValidationError(
            f"Patch has {parsed.hunk_count} total hunks, exceeding "
            f"maximum {MAX_TOTAL_HUNKS}."
        )


def _require_field(
    payload: dict[str, Any], field_name: str, expected_type: type
) -> None:
    """Require a field exists and has the correct type."""
    if field_name not in payload:
        raise ApplicatorValidationError(f"Missing required field: {field_name!r}.")
    value = payload[field_name]
    if type(value) is not expected_type:
        raise ApplicatorValidationError(
            f"Field {field_name!r} must be {expected_type.__name__}, "
            f"got {type(value).__name__}."
        )


def _validate_safe_path(raw_path: str) -> str:
    """Validate a single path through project safety rules.

    The applicator contract deliberately accepts only conservative, unambiguous
    ASCII POSIX paths. Simple spaces are allowed; quoting is decoded before this
    function is called.
    """
    if len(raw_path) > MAX_PATH_LENGTH:
        raise ValueError(
            f"Applicator touched_paths cannot exceed {MAX_PATH_LENGTH} characters."
        )
    if not raw_path.isascii():
        raise ValueError(
            "Applicator touched_paths must use ASCII paths; Unicode aliases are refused."
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in raw_path):
        raise ValueError(
            "Applicator touched_paths cannot contain tabs or control characters."
        )
    if "\\" in raw_path:
        raise ValueError("Applicator touched_paths cannot contain backslash paths.")
    if any(part in {"", ".", ".."} for part in raw_path.split("/")):
        raise ValueError("Applicator touched_paths cannot use empty or dot segments.")
    for part in raw_path.split("/"):
        if part.endswith((" ", ".")):
            raise ValueError(
                "Applicator touched_paths cannot contain trailing dots or spaces."
            )
        if ":" in part:
            raise ValueError(
                "Applicator touched_paths cannot contain Windows drive or stream syntax."
            )
        if any(character in _WINDOWS_ILLEGAL_PATH_CHARACTERS for character in part):
            raise ValueError(
                "Applicator touched_paths cannot contain Windows-illegal "
                "characters (wildcards, quotes, pipes, or angle brackets)."
            )
        if part.split(".", 1)[0].lower() in _WINDOWS_RESERVED_NAMES:
            raise ValueError(
                "Applicator touched_paths cannot contain Windows reserved names."
            )
    try:
        return validate_relative_project_path(
            raw_path,
            ROOT_DIR,
            field_label="Applicator touched_paths",
        )
    except OSError as exc:
        # Defense in depth: Path.resolve() can raise OSError on Windows for
        # reserved device names or wildcard characters that slip past the
        # checks above. Attacker-controlled payloads must never crash
        # validation, so convert any OS-level failure into a refusal.
        raise ValueError(
            "Applicator touched_paths include an invalid or reserved file system name."
        ) from exc


def _parsed_patch(*, patch_size: int, empty: bool, errors: list[str]) -> ParsedPatch:
    return ParsedPatch(
        touched_paths=(),
        file_count=0,
        hunk_count=0,
        total_added_lines=0,
        total_removed_lines=0,
        patch_size_bytes=patch_size,
        is_unified_diff=False,
        has_binary_marker=False,
        has_dev_null=False,
        has_rename=False,
        has_copy=False,
        has_mode_change=False,
        has_symlink=False,
        has_submodule=False,
        has_combined_diff=False,
        has_absolute_path=False,
        has_parent_traversal=False,
        has_backslash_path=False,
        has_empty_patch=empty,
        has_malformed_header=False,
        errors=tuple(errors),
    )


def _split_patch_lines(patch_preview: str, errors: list[str]) -> list[str]:
    """Normalize accepted CRLF input while refusing ambiguous bare CR input."""
    normalized = patch_preview.replace("\r\n", "\n")
    if "\r" in normalized:
        errors.append("Patch content contains a bare carriage return.")
    lines = normalized.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _empty_patch_flags() -> dict[str, bool]:
    return {
        "binary": False,
        "dev_null": False,
        "rename": False,
        "copy": False,
        "mode": False,
        "symlink": False,
        "submodule": False,
        "combined": False,
        "absolute": False,
        "traversal": False,
        "backslash": False,
    }


def _append_flag_errors(flags: dict[str, bool], errors: list[str]) -> None:
    messages = (
        ("binary", "Binary patch marker detected; Binary patches are not supported."),
        ("dev_null", "File deletion (/dev/null) is not supported."),
        ("rename", "Rename diffs are not supported."),
        ("copy", "Copy diffs are not supported."),
        ("mode", "Mode changes are not supported."),
        ("symlink", "Symlink changes are not supported."),
        ("submodule", "Submodule changes are not supported."),
        ("combined", "Combined diffs are not supported."),
    )
    for flag, message in messages:
        if flags[flag] and message not in errors:
            errors.append(message)


def _record_unsupported_metadata(line: str, flags: dict[str, bool]) -> bool:
    """Record unsupported markers only while the parser is outside a hunk."""
    if line.startswith(("diff --cc ", "diff --combined ", "@@@")):
        flags["combined"] = True
        return True
    if line.startswith(("GIT binary patch", "Binary files ")):
        flags["binary"] = True
        return True
    if line.startswith(("rename from ", "rename to ")):
        flags["rename"] = True
        return True
    if line.startswith(("copy from ", "copy to ")):
        flags["copy"] = True
        return True
    if line.startswith(
        ("old mode ", "new mode ", "new file mode ", "deleted file mode ")
    ):
        flags["mode"] = True
        flags["symlink"] = flags["symlink"] or "120000" in line
        flags["submodule"] = flags["submodule"] or "160000" in line
        return True
    if _SUBMODULE_BODY_PATTERN.fullmatch(line):
        flags["submodule"] = True
        return True
    if line.startswith(("--- /dev/null", "+++ /dev/null")):
        flags["dev_null"] = True
        return True
    return False


def _update_path_flags(raw_path: str, flags: dict[str, bool]) -> None:
    normalized = raw_path.replace("\\", "/")
    flags["backslash"] = flags["backslash"] or "\\" in raw_path
    flags["absolute"] = (
        flags["absolute"]
        or normalized.startswith("/")
        or bool(re.match(r"^[A-Za-z]:/", normalized))
    )
    flags["traversal"] = flags["traversal"] or ".." in normalized.split("/")


def _parse_diff_section(
    lines: list[str],
    section_number: int,
    errors: list[str],
    flags: dict[str, bool],
) -> dict[str, Any]:
    path: str | None = None
    header_paths = _parse_diff_header(lines[0], section_number, errors, flags)
    if header_paths is not None:
        path_a, path_b = header_paths
        if path_a != path_b:
            errors.append(
                f"Diff section {section_number} changes path from {path_a!r} to {path_b!r}; "
                "renames are not supported."
            )
        else:
            path = _validate_parsed_path(path_a, section_number, errors)

    old_path: str | None = None
    new_path: str | None = None
    old_header_count = 0
    new_header_count = 0
    hunk_count = 0
    total_added = 0
    total_removed = 0
    saw_hunk = False
    prev_old_end = 0
    prev_new_end = 0
    index = 1

    while index < len(lines):
        line = lines[index]
        if line.startswith("@@"):
            if old_header_count != 1 or new_header_count != 1:
                errors.append(
                    f"Diff section {section_number} must contain exactly one --- "
                    "and exactly one +++ path line before its first hunk."
                )
            saw_hunk = True
            hunk = _parse_hunk(lines, index, section_number, errors, flags)
            hunk_count += int(hunk["valid_header"])
            total_added += hunk["total_added"]
            total_removed += hunk["total_removed"]
            if hunk["valid_header"]:
                # Hunks must be strictly ascending and non-overlapping on
                # both the old and new sides of the same file.
                if (
                    hunk["old_start"] <= prev_old_end
                    or hunk["new_start"] <= prev_new_end
                ):
                    errors.append(
                        f"Diff section {section_number} has descending or "
                        "overlapping hunk ranges."
                    )
                prev_old_end = max(
                    prev_old_end,
                    hunk["old_start"] + max(hunk["old_count"] - 1, 0),
                )
                prev_new_end = max(
                    prev_new_end,
                    hunk["new_start"] + max(hunk["new_count"] - 1, 0),
                )
            index = hunk["next_index"]
            continue

        if saw_hunk:
            if _record_unsupported_metadata(line, flags):
                errors.append(
                    f"Diff section {section_number} contains unsupported metadata "
                    "after a hunk."
                )
            elif line.startswith((" ", "+", "-", "\\")):
                errors.append(
                    f"Diff section {section_number} contains an extra body-like line "
                    "outside an active hunk."
                )
            else:
                errors.append(
                    f"Diff section {section_number} contains unsupported or malformed "
                    "content after a hunk."
                )
            index += 1
            continue

        if line.startswith("--- "):
            old_header_count += 1
            if new_header_count:
                errors.append(
                    f"Diff section {section_number} must place --- before +++."
                )
            parsed_path = _parse_body_path(line, "---", section_number, errors, flags)
            if old_header_count == 1:
                old_path = parsed_path
            index += 1
            continue
        if line.startswith("+++ "):
            new_header_count += 1
            if old_header_count != 1:
                errors.append(
                    f"Diff section {section_number} must place --- before +++."
                )
            parsed_path = _parse_body_path(line, "+++", section_number, errors, flags)
            if new_header_count == 1:
                new_path = parsed_path
            index += 1
            continue
        if line.startswith("index "):
            if old_header_count or new_header_count:
                errors.append(
                    f"Diff section {section_number} places index metadata after "
                    "a path header."
                )
            if _INDEX_LINE_PATTERN.fullmatch(line) is None:
                errors.append(
                    f"Diff section {section_number} has malformed index metadata."
                )
            # Index metadata is never trusted for safety decisions; the mode
            # suffix is only used to refuse symlink/submodule entries early.
            flags["symlink"] = flags["symlink"] or line.endswith(" 120000")
            flags["submodule"] = flags["submodule"] or line.endswith(" 160000")
            index += 1
            continue
        if _record_unsupported_metadata(line, flags):
            errors.append(
                f"Diff section {section_number} contains unsupported metadata."
            )
        elif line.startswith((" ", "+", "-", "\\")):
            errors.append(
                f"Diff section {section_number} contains a body-like line outside "
                "an active hunk."
            )
        else:
            errors.append(
                f"Diff section {section_number} contains unsupported or malformed "
                "metadata."
            )
        index += 1

    if old_header_count != 1:
        errors.append(
            f"Diff section {section_number} must contain exactly one --- path line."
        )
    if new_header_count != 1:
        errors.append(
            f"Diff section {section_number} must contain exactly one +++ path line."
        )
    if (
        header_paths is not None
        and old_path is not None
        and old_path != header_paths[0]
    ):
        errors.append(
            f"Diff section {section_number} --- path does not match diff --git path."
        )
    if (
        header_paths is not None
        and new_path is not None
        and new_path != header_paths[1]
    ):
        errors.append(
            f"Diff section {section_number} +++ path does not match diff --git path."
        )
    if old_path is not None and new_path is not None and old_path != new_path:
        errors.append(f"Diff section {section_number} --- and +++ paths do not match.")
    for body_path in (old_path, new_path):
        if body_path is not None and body_path != "/dev/null":
            _validate_parsed_path(body_path, section_number, errors)

    if not saw_hunk:
        errors.append(f"Diff section {section_number} has no hunk header.")
    if hunk_count > MAX_HUNKS_PER_FILE:
        errors.append(
            f"Diff section {section_number} has {hunk_count} hunks, exceeding "
            f"per-file maximum {MAX_HUNKS_PER_FILE}."
        )

    return {
        "path": path,
        "hunk_count": hunk_count,
        "total_added": total_added,
        "total_removed": total_removed,
    }


def _parse_hunk(
    lines: list[str],
    header_index: int,
    section_number: int,
    errors: list[str],
    flags: dict[str, bool],
) -> dict[str, Any]:
    header = lines[header_index]
    match = _HUNK_HEADER_PATTERN.fullmatch(header)
    if match is None:
        errors.append(f"Diff section {section_number} has a malformed hunk header.")
        return {
            "valid_header": False,
            "next_index": header_index + 1,
            "total_added": 0,
            "total_removed": 0,
            "old_start": 0,
            "old_count": 0,
            "new_start": 0,
            "new_count": 0,
        }

    old_start = int(match.group("old_start"))
    old_count = int(match.group("old_count") or "1")
    new_start = int(match.group("new_start"))
    new_count = int(match.group("new_count") or "1")
    # Zero starts only occur in new-file (-0,0) and whole-file-delete (+0,0)
    # hunks, which this applicator does not support. Refuse all of them
    # conservatively rather than special-casing the zero-count forms.
    if old_start == 0 or new_start == 0:
        errors.append(
            f"Diff section {section_number} has a hunk range starting at line "
            "zero; new-file and delete operations are not supported."
        )
    old_consumed = 0
    new_consumed = 0
    body_lines = 0
    total_added = 0
    total_removed = 0
    old_side_closed = False
    new_side_closed = False
    last_kind = ""
    index = header_index + 1

    while index < len(lines):
        line = lines[index]
        counts_satisfied = old_consumed == old_count and new_consumed == new_count
        if counts_satisfied:
            if line == _NO_NEWLINE_MARKER:
                if last_kind not in {" ", "+", "-"}:
                    errors.append(
                        f"Diff section {section_number} has a malformed no-newline "
                        "marker placement."
                    )
                index += 1
            break
        if line.startswith("@@"):
            break

        if line == _NO_NEWLINE_MARKER:
            # The marker annotates the immediately preceding hunk body line
            # and means that file side ends there: after "-" no further old
            # lines may appear, after "+" no further new lines, and after a
            # context line both sides are finished.
            if last_kind == "-":
                old_side_closed = True
            elif last_kind == "+":
                new_side_closed = True
            elif last_kind == " ":
                old_side_closed = True
                new_side_closed = True
            else:
                errors.append(
                    f"Diff section {section_number} has a malformed no-newline "
                    "marker placement."
                )
            last_kind = "\\"
            index += 1
            continue
        if line.startswith("\\"):
            errors.append(
                f"Diff section {section_number} has an unsupported backslash hunk line."
            )
            index += 1
            continue
        if not line or line[0] not in {" ", "+", "-"}:
            errors.append(
                f"Diff section {section_number} contains unsupported or malformed "
                "content inside a hunk."
            )
            break

        kind = line[0]
        if (kind in {" ", "-"} and old_side_closed) or (
            kind in {" ", "+"} and new_side_closed
        ):
            errors.append(
                f"Diff section {section_number} has hunk lines after a no-newline "
                "marker ended that file side."
            )
            break
        old_increment = int(kind in {" ", "-"})
        new_increment = int(kind in {" ", "+"})
        if (
            old_consumed + old_increment > old_count
            or new_consumed + new_increment > new_count
        ):
            errors.append(f"Diff section {section_number} has an overlong hunk body.")
            break

        body_lines += 1
        old_consumed += old_increment
        new_consumed += new_increment
        total_added += int(kind == "+")
        total_removed += int(kind == "-")
        if kind in {"+", "-"} and _SUBMODULE_BODY_PATTERN.fullmatch(line[1:]):
            flags["submodule"] = True
        last_kind = kind
        index += 1

    if body_lines == 0:
        errors.append(f"Diff section {section_number} has an empty hunk body.")
    elif total_added == 0 and total_removed == 0:
        errors.append(
            f"Diff section {section_number} has a context-only hunk with no "
            "added or removed lines."
        )
    if old_consumed != old_count or new_consumed != new_count:
        errors.append(
            f"Diff section {section_number} has a truncated hunk body: "
            f"expected old/new counts {old_count}/{new_count}, consumed "
            f"{old_consumed}/{new_consumed}."
        )
    return {
        "valid_header": True,
        "next_index": index,
        "total_added": total_added,
        "total_removed": total_removed,
        "old_start": old_start,
        "old_count": old_count,
        "new_start": new_start,
        "new_count": new_count,
    }


def _parse_diff_header(
    line: str,
    section_number: int,
    errors: list[str],
    flags: dict[str, bool],
) -> tuple[str, str] | None:
    tokens = _split_diff_tokens(line)
    if len(tokens) != 4 or tokens[:2] != ["diff", "--git"]:
        errors.append(
            f"Diff section {section_number} has a malformed diff --git header."
        )
        return None
    path_a = _parse_prefixed_path(
        tokens[2], "a/", section_number, "diff --git", errors, flags
    )
    path_b = _parse_prefixed_path(
        tokens[3], "b/", section_number, "diff --git", errors, flags
    )
    if path_a is None or path_b is None:
        return None
    return path_a, path_b


def _parse_body_path(
    line: str,
    marker: str,
    section_number: int,
    errors: list[str],
    flags: dict[str, bool],
) -> str | None:
    raw = line[4:]
    if raw == "/dev/null":
        flags["dev_null"] = True
        return raw
    prefix = "a/" if marker == "---" else "b/"
    return _parse_prefixed_path(raw, prefix, section_number, marker, errors, flags)


def _parse_prefixed_path(
    raw: str,
    prefix: str,
    section_number: int,
    label: str,
    errors: list[str],
    flags: dict[str, bool],
) -> str | None:
    decoded = _decode_simple_path(raw, section_number, label, errors)
    if decoded is None:
        _update_path_flags(raw.strip('"'), flags)
        return None
    if not decoded.startswith(prefix):
        errors.append(
            f"Diff section {section_number} {label} path must begin with {prefix}."
        )
        _update_path_flags(decoded, flags)
        return None
    path = decoded[len(prefix) :]
    _update_path_flags(path, flags)
    return path


def _decode_simple_path(
    raw: str, section_number: int, label: str, errors: list[str]
) -> str | None:
    if raw.startswith('"'):
        if not raw.endswith('"') or len(raw) < 2:
            errors.append(
                f"Diff section {section_number} {label} path has malformed quoting."
            )
            return None
        decoded = raw[1:-1]
        if '"' in decoded or "\\" in decoded:
            errors.append(
                f"Diff section {section_number} {label} path uses unsupported "
                "Git C-style escaping."
            )
            return None
        return decoded
    if '"' in raw or any(character.isspace() for character in raw):
        errors.append(
            f"Diff section {section_number} {label} path must use simple quotes "
            "when it contains spaces."
        )
        return None
    return raw


def _validate_parsed_path(
    raw_path: str, section_number: int, errors: list[str]
) -> str | None:
    try:
        return _validate_safe_path(raw_path)
    except (ValueError, OSError) as exc:
        errors.append(f"Diff section {section_number} path is unsafe: {exc}")
        return None


# Diff header tokenization is kept local so validation stays runtime-isolated.


def _split_diff_tokens(line: str) -> list[str]:
    """Split a diff header line into tokens."""
    tokens: list[str] = []
    current: list[str] = []
    in_quote = False
    for ch in line:
        if ch == '"':
            in_quote = not in_quote
            current.append(ch)
        elif ch in (" ", "\t") and not in_quote:
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(ch)
    if current:
        tokens.append("".join(current))
    if in_quote:
        return []
    return tokens
