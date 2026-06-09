"""Stage 7B — Pure validation tests for app/dev/proposal_validation.py.

Tests validate approval payloads and parse/validate unified diff metadata.
No branch creation, patch application, git calls, subprocess calls, or file
modifications occur.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.approvals import ApprovalItem
from app.dev.proposal_validation import (
    MAX_HUNKS_PER_FILE,
    MAX_PATCH_SIZE_BYTES,
    MAX_PATH_LENGTH,
    MAX_TOUCHED_FILES,
    MAX_TOTAL_HUNKS,
    ApplicatorValidationError,
    ParsedPatch,
    parse_unified_diff_metadata,
    validate_approved_code_proposal,
    validate_patch_contract,
)

# ---------------------------------------------------------------------------
# Helpers — construct ApprovalItem objects for testing
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _approval(
    approval_id: int = 1,
    action_type: str = "propose_code_change",
    status: str = "approved",
    payload: dict | None = None,
    title: str = "Test proposal",
) -> ApprovalItem:
    if payload is None:
        payload = _minimal_valid_payload()
    return ApprovalItem(
        id=approval_id,
        action_type=action_type,
        title=title,
        payload=payload,
        status=status,
        created_at=_now(),
        updated_at=_now(),
    )


def _minimal_valid_payload(
    *,
    patch_preview: str | None = None,
    touched_paths: list[str] | None = None,
    requested_goal: str = "Make the chat text blue.",
) -> dict:
    if patch_preview is None:
        patch_preview = _simple_diff(paths=touched_paths)
    if touched_paths is None:
        touched_paths = ["app/static/lewis.css"]
    return {
        "applicator_contract_version": 1,
        "requested_goal": requested_goal,
        "target_surface": "command center web UI",
        "relevant_files": ["app/static/lewis.css"],
        "proposed_approach": "Change the CSS color property.",
        "patch_preview": patch_preview,
        "patch_preview_path_classifications": [
            {"path": "app/static/lewis.css", "classification": "valid_project_file"}
        ],
        "patch_preview_implementation_ready": True,
        "patch_preview_kind": "unified diff",
        "patch_is_inert": True,
        "touched_paths": touched_paths,
        "selected_strategy": "chat_text_color",
        "commands_run": [],
        "files_modified": [],
        "safety_notes": ["Pending code proposal only."],
        "suggested_test_commands": [],
    }


def _simple_diff(
    paths: list[str] | None = None,
    path: str = "app/static/lewis.css",
    content_before: str = ".chat-log .bubble {\n    color: var(--ink);\n}\n",
    content_after: str = ".chat-log .bubble {\n    color: blue;\n}\n",
) -> str:
    """Generate a minimal valid unified diff for testing."""
    if paths is not None:
        # Build a multi-file diff
        parts: list[str] = []
        for p in paths:
            parts.append(
                f"diff --git a/{p} b/{p}\n"
                f"--- a/{p}\n"
                f"+++ b/{p}\n"
                f"@@ -1,1 +1,1 @@\n"
                f"-old content\n"
                f"+new content\n"
            )
        return "".join(parts)

    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        f"@@ -1,3 +1,3 @@\n"
        f" {content_before.strip().split(chr(10))[0]}\n"
        f"-{content_before.strip().split(chr(10))[1]}\n"
        f"+{content_after.strip().split(chr(10))[1]}\n"
        f" {content_before.strip().split(chr(10))[2]}\n"
    )


def _hunk_diff(
    hunk_header: str,
    body_lines: list[str],
    *,
    path: str = "app/safe.py",
    line_ending: str = "\n",
) -> str:
    lines = [
        f"diff --git a/{path} b/{path}",
        f"--- a/{path}",
        f"+++ b/{path}",
        hunk_header,
        *body_lines,
    ]
    return line_ending.join(lines) + line_ending


# ---------------------------------------------------------------------------
# Valid payload tests — three supported proposal types
# ---------------------------------------------------------------------------


def _chat_text_color_payload() -> dict:
    path = "app/static/lewis.css"
    patch = (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        f"@@ -1,4 +1,4 @@\n"
        f" .chat-log .bubble {{\n"
        f"-    color: var(--ink);\n"
        f"+    color: blue;\n"
        f"     padding: 0.75rem;\n"
        f" }}\n"
    )
    return {
        "applicator_contract_version": 1,
        "requested_goal": "Make the text blue instead of black for this chat.",
        "target_surface": "command center web UI",
        "relevant_files": [path, "app/web_pages.py"],
        "proposed_approach": "Change .chat-log .bubble color from var(--ink) to blue.",
        "patch_preview": patch,
        "patch_preview_path_classifications": [
            {"path": path, "classification": "valid_project_file"}
        ],
        "patch_preview_implementation_ready": True,
        "patch_preview_kind": "unified diff",
        "patch_is_inert": True,
        "touched_paths": [path],
        "selected_strategy": "chat_text_color",
        "commands_run": [],
        "files_modified": [],
        "safety_notes": ["Pending code proposal only."],
        "suggested_test_commands": [],
    }


def _thinking_animation_payload() -> dict:
    css_path = "app/static/lewis.css"
    js_path = "app/static/chat.js"
    patch = (
        f"diff --git a/{css_path} b/{css_path}\n"
        f"--- a/{css_path}\n"
        f"+++ b/{css_path}\n"
        f"@@ -45,3 +45,12 @@\n"
        f" .bubble {{\n"
        f"     margin: 0.25rem 0;\n"
        f" }}\n"
        f"+\n"
        f"+.bubble.thinking {{\n"
        f"+    animation: pulse 1.5s ease-in-out infinite;\n"
        f"+}}\n"
        f"+\n"
        f"+@keyframes pulse {{\n"
        f"+    0%, 100% {{ opacity: 1; }}\n"
        f"+    50% {{ opacity: 0.5; }}\n"
        f"+}}\n"
        f"diff --git a/{js_path} b/{js_path}\n"
        f"--- a/{js_path}\n"
        f"+++ b/{js_path}\n"
        f"@@ -1,3 +1,5 @@\n"
        f" function showThinkingBubble() {{\n"
        f"     return 'thinking';\n"
        f" }}\n"
        f"+\n"
        f"+showThinkingBubble();\n"
    )
    return {
        "applicator_contract_version": 1,
        "requested_goal": "Make a new animation for when Lewis is thinking.",
        "target_surface": "command center web UI",
        "relevant_files": [css_path, js_path],
        "proposed_approach": "Add CSS keyframe animation and JS trigger.",
        "patch_preview": patch,
        "patch_preview_path_classifications": [
            {"path": css_path, "classification": "valid_project_file"},
            {"path": js_path, "classification": "valid_project_file"},
        ],
        "patch_preview_implementation_ready": True,
        "patch_preview_kind": "unified diff",
        "patch_is_inert": True,
        "touched_paths": [css_path, js_path],
        "selected_strategy": "thinking_animation",
        "commands_run": ["search_project_files", "read_project_file"],
        "files_modified": [],
        "safety_notes": ["Pending code proposal only."],
        "suggested_test_commands": [],
    }


def _tool_use_limit_payload() -> dict:
    path = "app/tool_call_limits.py"
    patch = (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        f"@@ -1,2 +1,2 @@\n"
        f' LEWIS_MAX_TOOL_CALLS", 5\n'
        f'-LEWIS_MAX_TOOL_CALLS", 5\n'
        f'+LEWIS_MAX_TOOL_CALLS", 8\n'
    )
    return {
        "applicator_contract_version": 1,
        "requested_goal": "Increase the tool use limit.",
        "target_surface": "backend configuration",
        "relevant_files": [path, "app/config.py"],
        "proposed_approach": "Change default max_tool_calls from 5 to 8.",
        "patch_preview": patch,
        "patch_preview_path_classifications": [
            {"path": path, "classification": "valid_project_file"}
        ],
        "patch_preview_implementation_ready": True,
        "patch_preview_kind": "unified diff",
        "patch_is_inert": True,
        "touched_paths": [path],
        "selected_strategy": "setting_limit",
        "commands_run": [],
        "files_modified": [],
        "safety_notes": ["Pending code proposal only."],
        "suggested_test_commands": [],
    }


class TestValidPayloads:
    """Valid payloads for all three supported proposal types."""

    def test_chat_text_color_payload_validates(self):
        approval = _approval(payload=_chat_text_color_payload())
        result = validate_approved_code_proposal(approval)
        assert result.approval_id == 1
        assert result.action_type == "propose_code_change"
        assert result.selected_strategy == "chat_text_color"
        assert result.touched_paths == ("app/static/lewis.css",)
        assert result.parsed_patch.file_count == 1
        assert result.parsed_patch.is_unified_diff is True
        assert result.parsed_patch.errors == ()
        assert not result.parsed_patch.has_empty_patch

    def test_thinking_animation_payload_validates(self):
        approval = _approval(payload=_thinking_animation_payload())
        result = validate_approved_code_proposal(approval)
        assert result.selected_strategy == "thinking_animation"
        assert len(result.touched_paths) == 2
        assert "app/static/lewis.css" in result.touched_paths
        assert "app/static/chat.js" in result.touched_paths
        assert result.parsed_patch.file_count == 2
        assert result.parsed_patch.errors == ()

    def test_tool_use_limit_payload_validates(self):
        approval = _approval(payload=_tool_use_limit_payload())
        result = validate_approved_code_proposal(approval)
        assert result.selected_strategy == "setting_limit"
        assert result.touched_paths == ("app/tool_call_limits.py",)
        assert result.parsed_patch.file_count == 1
        assert result.parsed_patch.errors == ()


# ---------------------------------------------------------------------------
# Refusal: wrong action type / status / not a dict
# ---------------------------------------------------------------------------


class TestApprovalEnvelopeRefusal:
    def test_wrong_action_type_raises(self):
        approval = _approval(action_type="propose_memory")
        with pytest.raises(
            ApplicatorValidationError, match="not a code change proposal"
        ):
            validate_approved_code_proposal(approval)

    def test_status_not_approved_raises(self):
        approval = _approval(status="pending")
        with pytest.raises(ApplicatorValidationError, match="expected 'approved'"):
            validate_approved_code_proposal(approval)

    def test_status_rejected_raises(self):
        approval = _approval(status="rejected")
        with pytest.raises(ApplicatorValidationError, match="expected 'approved'"):
            validate_approved_code_proposal(approval)

    def test_status_completed_raises(self):
        approval = _approval(status="completed")
        with pytest.raises(ApplicatorValidationError, match="expected 'approved'"):
            validate_approved_code_proposal(approval)

    def test_payload_not_dict_raises(self):
        item = ApprovalItem(
            id=1,
            action_type="propose_code_change",
            title="Test",
            payload="not a dict",  # type: ignore
            status="approved",
            created_at=_now(),
            updated_at=_now(),
        )
        with pytest.raises(ApplicatorValidationError, match="payload is not a dict"):
            validate_approved_code_proposal(item)


# ---------------------------------------------------------------------------
# Refusal: missing or wrong gating fields
# ---------------------------------------------------------------------------


class TestMissingGatingFields:
    def test_missing_patch_preview_raises(self):
        payload = _minimal_valid_payload()
        del payload["patch_preview"]
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="no patch_preview"):
            validate_approved_code_proposal(approval)

    def test_empty_patch_preview_raises(self):
        payload = _minimal_valid_payload(patch_preview="   ")
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="no patch_preview"):
            validate_approved_code_proposal(approval)

    def test_patch_preview_kind_with_underscore_raises(self):
        payload = _minimal_valid_payload()
        payload["patch_preview_kind"] = "unified_diff"
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="patch_preview_kind"):
            validate_approved_code_proposal(approval)

    def test_patch_preview_kind_wrong_value_raises(self):
        payload = _minimal_valid_payload()
        payload["patch_preview_kind"] = "inline"
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="patch_preview_kind"):
            validate_approved_code_proposal(approval)

    def test_patch_is_inert_false_raises(self):
        payload = _minimal_valid_payload()
        payload["patch_is_inert"] = False
        approval = _approval(payload=payload)
        with pytest.raises(
            ApplicatorValidationError, match="patch_is_inert must be True"
        ):
            validate_approved_code_proposal(approval)

    def test_patch_is_inert_missing_raises(self):
        payload = _minimal_valid_payload()
        del payload["patch_is_inert"]
        approval = _approval(payload=payload)
        with pytest.raises(
            ApplicatorValidationError, match="Missing required structured field"
        ):
            validate_approved_code_proposal(approval)

    def test_patch_preview_implementation_ready_missing_raises(self):
        payload = _minimal_valid_payload()
        del payload["patch_preview_implementation_ready"]
        approval = _approval(payload=payload)
        with pytest.raises(
            ApplicatorValidationError, match="implementation_ready must be exactly True"
        ):
            validate_approved_code_proposal(approval)

    def test_patch_preview_implementation_ready_false_raises(self):
        payload = _minimal_valid_payload()
        payload["patch_preview_implementation_ready"] = False
        approval = _approval(payload=payload)
        with pytest.raises(
            ApplicatorValidationError, match="implementation_ready must be exactly True"
        ):
            validate_approved_code_proposal(approval)

    def test_patch_preview_implementation_ready_explicit_false_raises(self):
        payload = _minimal_valid_payload()
        payload["patch_preview_implementation_ready"] = False
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError):
            validate_approved_code_proposal(approval)

    def test_patch_preview_implementation_ready_truthy_string_raises(self):
        payload = _minimal_valid_payload()
        payload["patch_preview_implementation_ready"] = "yes"
        with pytest.raises(ApplicatorValidationError, match="exactly True"):
            validate_approved_code_proposal(_approval(payload=payload))

    def test_empty_selected_strategy_raises(self):
        payload = _minimal_valid_payload()
        payload["selected_strategy"] = "  "
        with pytest.raises(ApplicatorValidationError, match="non-empty"):
            validate_approved_code_proposal(_approval(payload=payload))


# ---------------------------------------------------------------------------
# Refusal: missing Stage 6B structured fields
# ---------------------------------------------------------------------------


class TestMissingStructuredFields:
    def test_missing_selected_strategy_raises(self):
        payload = _minimal_valid_payload()
        del payload["selected_strategy"]
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="selected_strategy"):
            validate_approved_code_proposal(approval)

    def test_missing_touched_paths_raises(self):
        payload = _minimal_valid_payload()
        del payload["touched_paths"]
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="touched_paths"):
            validate_approved_code_proposal(approval)

    def test_empty_touched_paths_raises(self):
        payload = _minimal_valid_payload(
            patch_preview=_simple_diff(path="app/static/lewis.css"),
            touched_paths=[],
        )
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="non-empty list"):
            validate_approved_code_proposal(approval)

    def test_missing_patch_preview_kind_raises(self):
        payload = _minimal_valid_payload()
        del payload["patch_preview_kind"]
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="patch_preview_kind"):
            validate_approved_code_proposal(approval)

    def test_missing_patch_is_inert_raises(self):
        payload = _minimal_valid_payload()
        del payload["patch_is_inert"]
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="patch_is_inert"):
            validate_approved_code_proposal(approval)

    def test_missing_commands_run_raises(self):
        payload = _minimal_valid_payload()
        del payload["commands_run"]
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="commands_run"):
            validate_approved_code_proposal(approval)

    def test_missing_files_modified_raises(self):
        payload = _minimal_valid_payload()
        del payload["files_modified"]
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="files_modified"):
            validate_approved_code_proposal(approval)


# ---------------------------------------------------------------------------
# Refusal: commands_run non-empty, files_modified non-empty
# ---------------------------------------------------------------------------


class TestCommandsRunAndFilesModified:
    def test_commands_run_non_empty_audit_metadata_accepts(self):
        payload = _minimal_valid_payload()
        payload["commands_run"] = ["search_project_files"]
        approval = _approval(payload=payload)
        result = validate_approved_code_proposal(approval)
        assert result.touched_paths == ("app/static/lewis.css",)

    def test_files_modified_non_empty_raises(self):
        payload = _minimal_valid_payload()
        payload["files_modified"] = ["app/static/lewis.css"]
        approval = _approval(payload=payload)
        with pytest.raises(
            ApplicatorValidationError, match="files_modified must be empty"
        ):
            validate_approved_code_proposal(approval)

    def test_commands_run_not_list_raises(self):
        payload = _minimal_valid_payload()
        payload["commands_run"] = "search_project_files"  # not a list
        approval = _approval(payload=payload)
        with pytest.raises(
            ApplicatorValidationError, match="commands_run must be a list"
        ):
            validate_approved_code_proposal(approval)

    def test_files_modified_not_list_raises(self):
        payload = _minimal_valid_payload()
        payload["files_modified"] = "file.txt"  # not a list
        approval = _approval(payload=payload)
        with pytest.raises(
            ApplicatorValidationError, match="files_modified must be a list"
        ):
            validate_approved_code_proposal(approval)


# ---------------------------------------------------------------------------
# Refusal: touched_paths mismatch with parsed patch
# ---------------------------------------------------------------------------


class TestTouchedPathsMismatch:
    def test_extra_declared_path_raises(self):
        # Diff touches lewis.css only, but touched_paths declares an extra file
        patch = _simple_diff(path="app/static/lewis.css")
        payload = _minimal_valid_payload(
            patch_preview=patch,
            touched_paths=["app/static/lewis.css", "app/extra.css"],
        )
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="do not match"):
            validate_approved_code_proposal(approval)

    def test_missing_declared_path_raises(self):
        # Payload declares only lewis.css, but diff touches lewis.css
        # Actually payload matches diff in this case. Let's change the diff instead.
        patch = _simple_diff(path="app/static/other.css")
        payload = _minimal_valid_payload(
            patch_preview=patch,
            touched_paths=["app/static/lewis.css"],
        )
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="do not match"):
            validate_approved_code_proposal(approval)

    def test_completely_different_paths_raises(self):
        patch = _simple_diff(path="app/different.css")
        payload = _minimal_valid_payload(
            patch_preview=patch,
            touched_paths=["app/static/lewis.css"],
        )
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="do not match"):
            validate_approved_code_proposal(approval)

    def test_valid_paths_must_have_exact_equality(self):
        """validate_patch_contract requires exact sorted equality."""
        payload = _minimal_valid_payload(
            touched_paths=["app/static/lewis.css"],
        )
        parsed = ParsedPatch(
            touched_paths=("app/static/lewis.css",),
            file_count=1,
            hunk_count=1,
            total_added_lines=1,
            total_removed_lines=1,
            patch_size_bytes=200,
            is_unified_diff=True,
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
            has_empty_patch=False,
            has_malformed_header=False,
        )
        # Should not raise
        validate_patch_contract(payload, parsed)


# ---------------------------------------------------------------------------
# Refusal: unsafe paths
# ---------------------------------------------------------------------------


class TestUnsafePaths:
    def test_absolute_path_in_diff_raises(self):
        patch = """diff --git a//etc/passwd b//etc/passwd
--- a//etc/passwd
+++ b//etc/passwd
@@ -1,1 +1,1 @@
-old
+new
"""
        payload = _minimal_valid_payload(
            patch_preview=patch,
            touched_paths=["/etc/passwd"],
        )
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError):
            validate_approved_code_proposal(approval)

    def test_parent_traversal_path_raises(self):
        patch = """diff --git a/../outside/file.txt b/../outside/file.txt
--- a/../outside/file.txt
+++ b/../outside/file.txt
@@ -1,1 +1,1 @@
-old
+new
"""
        payload = _minimal_valid_payload(
            patch_preview=patch,
            touched_paths=["../outside/file.txt"],
        )
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError):
            validate_approved_code_proposal(approval)

    def test_dot_env_path_raises(self):
        payload = _minimal_valid_payload(touched_paths=[".env"])
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="blocked file name"):
            validate_approved_code_proposal(approval)

    def test_sqlite_path_raises(self):
        payload = _minimal_valid_payload(touched_paths=["data/lewis.sqlite3"])
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="blocked"):
            validate_approved_code_proposal(approval)

    def test_data_runtime_path_raises(self):
        payload = _minimal_valid_payload(touched_paths=["data/something.json"])
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="blocked runtime path"):
            validate_approved_code_proposal(approval)

    def test_dot_prefixed_path_raises(self):
        payload = _minimal_valid_payload(touched_paths=[".hidden/file.py"])
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="blocked file name"):
            validate_approved_code_proposal(approval)

    def test_secret_path_raises(self):
        payload = _minimal_valid_payload(touched_paths=["config/secret_key.py"])
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="sensitive path name"):
            validate_approved_code_proposal(approval)


# ---------------------------------------------------------------------------
# Refusal: diff format issues
# ---------------------------------------------------------------------------


class TestDiffFormatIssues:
    @pytest.mark.parametrize(
        "patch",
        [
            """diff --git a/app/safe.py b/app/safe.py
--- a/app/safe.py
+++ b/data/lewis.sqlite3
@@ -1 +1 @@
-old
+new
""",
            """diff --git a/app/safe.py b/app/safe.py
--- a/.env
+++ b/app/safe.py
@@ -1 +1 @@
-old
+new
""",
            """diff --git a/app/safe.py b/app/safe.py
--- a/app/other.py
+++ b/app/safe.py
@@ -1 +1 @@
-old
+new
""",
            """diff --git a/app/safe.py b/app/safe.py
--- a/app/safe.py
+++ b/app/other.py
@@ -1 +1 @@
-old
+new
""",
        ],
    )
    def test_body_paths_must_be_safe_and_match_header(self, patch):
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/safe.py"]
        )
        with pytest.raises(ApplicatorValidationError):
            validate_approved_code_proposal(_approval(payload=payload))

    def test_header_only_patch_raises(self):
        patch = "diff --git a/app/safe.py b/app/safe.py\n--- a/app/safe.py\n+++ b/app/safe.py\n"
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/safe.py"]
        )
        with pytest.raises(ApplicatorValidationError, match="no hunk"):
            validate_approved_code_proposal(_approval(payload=payload))

    def test_diff_header_without_body_headers_raises(self):
        patch = "diff --git a/app/safe.py b/app/safe.py\n@@ -1 +1 @@\n-old\n+new\n"
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/safe.py"]
        )
        with pytest.raises(ApplicatorValidationError, match="exactly one ---"):
            validate_approved_code_proposal(_approval(payload=payload))

    def test_malformed_additional_diff_section_raises(self):
        patch = _simple_diff(path="app/safe.py") + "diff --git malformed\n"
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/safe.py"]
        )
        with pytest.raises(ApplicatorValidationError, match="malformed"):
            validate_approved_code_proposal(_approval(payload=payload))

    def test_mixed_valid_diff_plus_diff_cc_raises(self):
        patch = _simple_diff(path="app/safe.py") + "diff --cc app/merged.py\n"
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/safe.py"]
        )
        with pytest.raises(ApplicatorValidationError, match="Combined diffs"):
            validate_approved_code_proposal(_approval(payload=payload))

    def test_unmarked_rename_raises(self):
        patch = """diff --git a/app/old.py b/app/new.py
--- a/app/old.py
+++ b/app/new.py
@@ -1 +1 @@
-old
+new
"""
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/old.py", "app/new.py"]
        )
        with pytest.raises(
            ApplicatorValidationError, match="renames are not supported"
        ):
            validate_approved_code_proposal(_approval(payload=payload))

    def test_nul_byte_raises(self):
        patch = _simple_diff(path="app/safe.py") + "\x00"
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/safe.py"]
        )
        with pytest.raises(ApplicatorValidationError, match="NUL byte"):
            validate_approved_code_proposal(_approval(payload=payload))

    def test_unsupported_metadata_after_hunk_raises(self):
        patch = _simple_diff(path="app/safe.py") + "old mode 100644\n"
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/safe.py"]
        )
        with pytest.raises(ApplicatorValidationError, match="Mode changes"):
            validate_approved_code_proposal(_approval(payload=payload))

    def test_unknown_metadata_after_hunk_raises(self):
        patch = _simple_diff(path="app/safe.py") + "similarity index 100%\n"
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/safe.py"]
        )
        with pytest.raises(ApplicatorValidationError, match="content after a hunk"):
            validate_approved_code_proposal(_approval(payload=payload))

    def test_file_delete_dev_null_raises(self):
        patch = """diff --git a/app/removed.py b/app/removed.py
deleted file mode 100644
--- a/app/removed.py
+++ /dev/null
@@ -1,3 +0,0 @@
-old line 1
-old line 2
-old line 3
"""
        payload = _minimal_valid_payload(
            patch_preview=patch,
            touched_paths=["app/removed.py"],
        )
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="/dev/null"):
            validate_approved_code_proposal(approval)

    def test_rename_diff_raises(self):
        patch = """diff --git a/app/old_name.py b/app/new_name.py
rename from app/old_name.py
rename to app/new_name.py
"""
        payload = _minimal_valid_payload(
            patch_preview=patch,
            touched_paths=["app/old_name.py", "app/new_name.py"],
        )
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="Rename diffs"):
            validate_approved_code_proposal(approval)

    def test_copy_diff_raises(self):
        patch = """diff --git a/app/original.py b/app/copy.py
copy from app/original.py
copy to app/copy.py
--- a/app/original.py
+++ b/app/copy.py
@@ -1,1 +1,1 @@
-same
+same
"""
        payload = _minimal_valid_payload(
            patch_preview=patch,
            touched_paths=["app/original.py", "app/copy.py"],
        )
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="Copy diffs"):
            validate_approved_code_proposal(approval)

    def test_mode_change_raises(self):
        patch = """diff --git a/app/script.py b/app/script.py
old mode 100644
new mode 100755
"""
        payload = _minimal_valid_payload(
            patch_preview=patch,
            touched_paths=["app/script.py"],
        )
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="Mode changes"):
            validate_approved_code_proposal(approval)

    def test_binary_patch_marker_raises(self):
        patch = """diff --git a/app/image.png b/app/image.png
GIT binary patch
literal 1234
"""
        payload = _minimal_valid_payload(
            patch_preview=patch,
            touched_paths=["app/image.png"],
        )
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="Binary patches"):
            validate_approved_code_proposal(approval)

    def test_combined_diff_raises(self):
        patch = """diff --combined app/merged.py
--- a/app/merged.py
+++ b/app/merged.py
@@@ -1,1 -1,1 +1,1 @@@
-old
++new
"""
        payload = _minimal_valid_payload(
            patch_preview=patch,
            touched_paths=["app/merged.py"],
        )
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="Combined diffs"):
            validate_approved_code_proposal(approval)

    def test_empty_patch_raises(self):
        payload = _minimal_valid_payload(
            patch_preview="",
            touched_paths=["app/empty.py"],
        )
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError):
            validate_approved_code_proposal(approval)

    def test_malformed_diff_no_header_raises(self):
        patch = """--- a/app/file.py
+++ b/app/file.py
@@ -1,1 +1,1 @@
-old
+new
"""
        payload = _minimal_valid_payload(
            patch_preview=patch,
            touched_paths=["app/file.py"],
        )
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="diff --git header"):
            validate_approved_code_proposal(approval)

    def test_backslash_path_raises(self):
        patch = """diff --git "a/app\\static\\file.css" "b/app\\static\\file.css"
--- "a/app\\static\\file.css"
+++ "b/app\\static\\file.css"
@@ -1,1 +1,1 @@
-old
+new
"""
        payload = _minimal_valid_payload(
            patch_preview=patch,
            touched_paths=["app\\static\\file.css"],
        )
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="backslash"):
            validate_approved_code_proposal(approval)


# ---------------------------------------------------------------------------
# Strict hunk state machine and metadata-state tests
# ---------------------------------------------------------------------------


class TestStrictHunkStateMachine:
    def _assert_refused(self, patch: str, match: str | None = None) -> None:
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/safe.py"]
        )
        with pytest.raises(ApplicatorValidationError, match=match):
            validate_approved_code_proposal(_approval(payload=payload))

    def test_empty_zero_count_hunk_refuses(self):
        self._assert_refused(_hunk_diff("@@ -1,0 +1,0 @@", []), "empty hunk")

    def test_hunk_header_with_no_body_refuses(self):
        self._assert_refused(_hunk_diff("@@ -1,1 +1,1 @@", []), "empty hunk")

    def test_truncated_hunk_with_fewer_old_lines_refuses(self):
        patch = _hunk_diff("@@ -1,2 +1,1 @@", ["-old", "+new"])
        self._assert_refused(patch, "truncated hunk")

    def test_truncated_hunk_with_fewer_new_lines_refuses(self):
        patch = _hunk_diff("@@ -1,1 +1,2 @@", ["-old", "+new"])
        self._assert_refused(patch, "truncated hunk")

    def test_overlong_hunk_with_too_many_old_lines_refuses(self):
        patch = _hunk_diff("@@ -1,1 +1,2 @@", ["-old", "+new", "-extra"])
        self._assert_refused(patch, "overlong hunk")

    def test_overlong_hunk_with_too_many_new_lines_refuses(self):
        patch = _hunk_diff("@@ -1,2 +1,1 @@", ["-old", "+new", "+extra"])
        self._assert_refused(patch, "overlong hunk")

    def test_extra_body_line_after_satisfied_counts_refuses(self):
        patch = _hunk_diff("@@ -1,1 +1,1 @@", ["-old", "+new", " extra"])
        self._assert_refused(patch, "outside an active hunk")

    @pytest.mark.parametrize("outside_line", ["+outside", "-outside", " outside"])
    def test_body_line_outside_active_hunk_refuses(self, outside_line):
        patch = (
            "diff --git a/app/safe.py b/app/safe.py\n"
            "--- a/app/safe.py\n"
            "+++ b/app/safe.py\n"
            f"{outside_line}\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n"
            "+new\n"
        )
        self._assert_refused(patch, "outside an active hunk")

    def test_no_newline_marker_before_changed_line_refuses(self):
        patch = _hunk_diff(
            "@@ -1,1 +1,1 @@",
            [r"\ No newline at end of file", "-old", "+new"],
        )
        self._assert_refused(patch, "no-newline")

    def test_no_newline_marker_after_final_context_validates(self):
        # Git accepts a no-newline marker annotating a final context line.
        # The hunk must contain a real change because context-only hunks
        # are refused.
        patch = _hunk_diff(
            "@@ -1,2 +1,2 @@",
            ["-old", "+new", " same", r"\ No newline at end of file"],
        )
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/safe.py"]
        )
        result = validate_approved_code_proposal(_approval(payload=payload))
        assert result.parsed_patch.errors == ()

    def test_no_newline_marker_after_changed_lines_validates(self):
        patch = _hunk_diff(
            "@@ -1,1 +1,1 @@",
            [
                "-old",
                r"\ No newline at end of file",
                "+new",
                r"\ No newline at end of file",
            ],
        )
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/safe.py"]
        )
        result = validate_approved_code_proposal(_approval(payload=payload))
        assert result.parsed_patch.errors == ()

    def test_metadata_looking_rename_text_inside_hunk_validates(self):
        patch = _hunk_diff(
            "@@ -1,1 +1,1 @@",
            ["-rename from old text", "+rename from new text"],
        )
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/safe.py"]
        )
        result = validate_approved_code_proposal(_approval(payload=payload))
        assert result.parsed_patch.has_rename is False

    def test_metadata_looking_old_mode_text_inside_hunk_validates(self):
        patch = _hunk_diff(
            "@@ -1,1 +1,1 @@",
            ["-old mode 100644", "+old mode 100755"],
        )
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/safe.py"]
        )
        result = validate_approved_code_proposal(_approval(payload=payload))
        assert result.parsed_patch.has_mode_change is False

    @pytest.mark.parametrize(
        ("old_text", "new_text"),
        [
            ("/dev/null is documentation text", "/dev/null remains documentation text"),
            (
                "Subproject commit is ordinary prose",
                "Subproject commit remains ordinary prose",
            ),
        ],
    )
    def test_other_metadata_looking_hunk_content_validates(self, old_text, new_text):
        patch = _hunk_diff("@@ -1,1 +1,1 @@", [f"-{old_text}", f"+{new_text}"])
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/safe.py"]
        )
        result = validate_approved_code_proposal(_approval(payload=payload))
        assert result.parsed_patch.errors == ()

    @pytest.mark.parametrize(
        ("metadata", "match"),
        [
            ("rename from app/old.py\nrename to app/safe.py\n", "Rename diffs"),
            ("copy from app/old.py\ncopy to app/safe.py\n", "Copy diffs"),
            ("old mode 100644\nnew mode 100755\n", "Mode changes"),
        ],
    )
    def test_actual_unsupported_metadata_outside_hunk_refuses(self, metadata, match):
        patch = (
            "diff --git a/app/safe.py b/app/safe.py\n"
            f"{metadata}"
            "--- a/app/safe.py\n"
            "+++ b/app/safe.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n"
            "+new\n"
        )
        self._assert_refused(patch, match)

    def test_actual_subproject_commit_body_refuses(self):
        old_hash = "1" * 40
        new_hash = "2" * 40
        patch = _hunk_diff(
            "@@ -1,1 +1,1 @@",
            [f"-Subproject commit {old_hash}", f"+Subproject commit {new_hash}"],
        )
        self._assert_refused(patch, "Submodule changes")

    def test_mixed_valid_section_plus_malformed_hunk_section_refuses(self):
        patch = _simple_diff(path="app/safe.py") + _hunk_diff(
            "@@ -1,2 +1,2 @@", ["-old", "+new"], path="app/other.py"
        )
        payload = _minimal_valid_payload(
            patch_preview=patch,
            touched_paths=["app/safe.py", "app/other.py"],
        )
        with pytest.raises(ApplicatorValidationError, match="truncated hunk"):
            validate_approved_code_proposal(_approval(payload=payload))

    def test_crlf_hunk_input_validates(self):
        patch = _hunk_diff("@@ -1 +1 @@", ["-old", "+new"], line_ending="\r\n")
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/safe.py"]
        )
        result = validate_approved_code_proposal(_approval(payload=payload))
        assert result.parsed_patch.errors == ()

    def test_bare_carriage_return_refuses(self):
        patch = _hunk_diff("@@ -1 +1 @@", ["-old", "+new"]).replace("-old\n", "-old\r")
        self._assert_refused(patch, "bare carriage return")


class TestConservativePatchPathPolicy:
    @staticmethod
    def _patch_with_quoted_path(path: str) -> str:
        return (
            f'diff --git "a/{path}" "b/{path}"\n'
            f'--- "a/{path}"\n'
            f'+++ "b/{path}"\n'
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )

    def _assert_parsed_path_refused(self, path: str, match: str) -> None:
        parsed = parse_unified_diff_metadata(self._patch_with_quoted_path(path))
        assert any(match in error for error in parsed.errors), parsed.errors

    @pytest.mark.parametrize("path", ["app/tab\tname.py", "app/control\x01name.py"])
    def test_tab_or_control_character_path_refuses(self, path):
        self._assert_parsed_path_refused(path, "control characters")

    def test_windows_drive_path_refuses(self):
        self._assert_parsed_path_refused("C:/temp/file.py", "Windows drive")

    def test_extremely_long_path_refuses(self):
        path = f"app/{'x' * MAX_PATH_LENGTH}.py"
        self._assert_parsed_path_refused(path, "cannot exceed")

    @pytest.mark.parametrize("path", ["app/trailing.", "app/trailing /file.py"])
    def test_trailing_dot_or_space_path_refuses(self, path):
        self._assert_parsed_path_refused(path, "trailing dots or spaces")

    def test_windows_reserved_name_refuses(self):
        self._assert_parsed_path_refused("app/CON.py", "Windows reserved")

    def test_unicode_path_refuses_to_avoid_normalization_aliases(self):
        self._assert_parsed_path_refused("app/caf\u00e9.py", "Unicode aliases")

    def test_simple_quoted_path_with_spaces_validates(self):
        patch = (
            'diff --git "a/app/my file.py" "b/app/my file.py"\n'
            '--- "a/app/my file.py"\n'
            '+++ "b/app/my file.py"\n'
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/my file.py"]
        )
        result = validate_approved_code_proposal(_approval(payload=payload))
        assert result.touched_paths == ("app/my file.py",)

    def test_git_c_style_escaped_quoted_path_refuses(self):
        patch = (
            r'diff --git "a/app/my\040file.py" "b/app/my\040file.py"'
            "\n"
            r'--- "a/app/my\040file.py"'
            "\n"
            r'+++ "b/app/my\040file.py"'
            "\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/my file.py"]
        )
        with pytest.raises(ApplicatorValidationError, match="C-style escaping"):
            validate_approved_code_proposal(_approval(payload=payload))


# ---------------------------------------------------------------------------
# Stage 7B.3 — applicator contract version refusal
# ---------------------------------------------------------------------------


class TestApplicatorContractVersion:
    def test_wrong_contract_version_refuses(self):
        payload = _minimal_valid_payload()
        payload["applicator_contract_version"] = 2
        with pytest.raises(
            ApplicatorValidationError, match="applicator_contract_version must be 1"
        ):
            validate_approved_code_proposal(_approval(payload=payload))

    def test_missing_contract_version_refuses(self):
        payload = _minimal_valid_payload()
        del payload["applicator_contract_version"]
        with pytest.raises(
            ApplicatorValidationError, match="Missing required structured field"
        ):
            validate_approved_code_proposal(_approval(payload=payload))

    def test_string_contract_version_refuses(self):
        payload = _minimal_valid_payload()
        payload["applicator_contract_version"] = "1"
        with pytest.raises(ApplicatorValidationError, match="must be int"):
            validate_approved_code_proposal(_approval(payload=payload))

    def test_boolean_contract_version_refuses(self):
        payload = _minimal_valid_payload()
        payload["applicator_contract_version"] = True
        with pytest.raises(ApplicatorValidationError, match="must be int"):
            validate_approved_code_proposal(_approval(payload=payload))


# ---------------------------------------------------------------------------
# Stage 7B.3 — Windows-illegal / OSError-raising paths refuse cleanly
# ---------------------------------------------------------------------------


class TestWindowsIllegalPathRobustness:
    """Paths that make Path.resolve() raise OSError on Windows must become
    clean validation refusals, never crashes."""

    _ILLEGAL_PATHS = [
        "app/CONIN$",
        "app/CONOUT$",
        "app/file*.py",
        "app/file?.py",
        "app/file<name>.py",
        "app/file|name.py",
    ]

    @pytest.mark.parametrize("path", _ILLEGAL_PATHS)
    def test_payload_touched_path_refuses_cleanly(self, path):
        payload = _minimal_valid_payload(touched_paths=[path])
        with pytest.raises(ApplicatorValidationError):
            validate_approved_code_proposal(_approval(payload=payload))

    @pytest.mark.parametrize("path", _ILLEGAL_PATHS)
    def test_parser_path_refuses_cleanly_without_crash(self, path):
        patch = _simple_diff(paths=[path])
        parsed = parse_unified_diff_metadata(patch)
        assert any("unsafe" in error for error in parsed.errors), parsed.errors


class TestPayloadLevelConservativePathPolicy:
    """touched_paths entries are refused at payload level before parsing."""

    @pytest.mark.parametrize(
        ("path", "match"),
        [
            ("app/tab\tname.py", "control characters"),
            ("app/control\x01name.py", "control characters"),
            ("C:/temp/file.py", "Windows drive"),
            (f"app/{'x' * MAX_PATH_LENGTH}.py", "cannot exceed"),
            ("app/café.py", "Unicode aliases"),
        ],
    )
    def test_unsafe_payload_path_refuses(self, path, match):
        payload = _minimal_valid_payload(touched_paths=[path])
        with pytest.raises(ApplicatorValidationError, match=match):
            validate_approved_code_proposal(_approval(payload=payload))


# ---------------------------------------------------------------------------
# Stage 7B.3 — hunk semantic exactness
# ---------------------------------------------------------------------------


class TestHunkSemanticExactness:
    def _assert_refused(self, patch: str, match: str | None = None) -> None:
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/safe.py"]
        )
        with pytest.raises(ApplicatorValidationError, match=match):
            validate_approved_code_proposal(_approval(payload=payload))

    def test_context_only_hunk_refuses(self):
        patch = _hunk_diff("@@ -1,1 +1,1 @@", [" same"])
        self._assert_refused(patch, "context-only hunk")

    def test_zero_start_positive_count_hunk_refuses(self):
        patch = _hunk_diff("@@ -0,1 +0,1 @@", ["-old", "+new"])
        self._assert_refused(patch, "starting at line zero")

    def test_new_file_style_zero_start_hunk_refuses(self):
        patch = _hunk_diff("@@ -0,0 +1,2 @@", ["+a", "+b"])
        self._assert_refused(patch, "zero")

    def test_delete_style_zero_start_hunk_refuses(self):
        patch = _hunk_diff("@@ -1,2 +0,0 @@", ["-a", "-b"])
        self._assert_refused(patch, "zero")

    def test_descending_hunk_ranges_refuse(self):
        patch = _hunk_diff(
            "@@ -10,2 +10,2 @@",
            ["-old", "+new", " ctx", "@@ -1,2 +1,2 @@", "-old2", "+new2", " ctx2"],
        )
        self._assert_refused(patch, "descending or overlapping")

    def test_overlapping_hunk_ranges_refuse(self):
        patch = _hunk_diff(
            "@@ -1,3 +1,3 @@",
            [" a", "-b", "+B", " c", "@@ -3,2 +3,2 @@", "-c", "+C", " d"],
        )
        self._assert_refused(patch, "descending or overlapping")

    def test_ascending_non_overlapping_hunks_validate(self):
        patch = _hunk_diff(
            "@@ -1,2 +1,2 @@",
            ["-old", "+new", " ctx", "@@ -10,2 +10,2 @@", " ctx2", "-old2", "+new2"],
        )
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/safe.py"]
        )
        result = validate_approved_code_proposal(_approval(payload=payload))
        assert result.parsed_patch.hunk_count == 2
        assert result.parsed_patch.errors == ()


# ---------------------------------------------------------------------------
# Stage 7B.3 — no-newline marker semantics
# ---------------------------------------------------------------------------


class TestNoNewlineMarkerSemantics:
    def _assert_refused(self, patch: str, match: str | None = None) -> None:
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/safe.py"]
        )
        with pytest.raises(ApplicatorValidationError, match=match):
            validate_approved_code_proposal(_approval(payload=payload))

    def test_marker_after_final_context_with_real_change_validates(self):
        patch = _hunk_diff(
            "@@ -1,2 +1,2 @@",
            ["-old", "+new", " same", r"\ No newline at end of file"],
        )
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/safe.py"]
        )
        result = validate_approved_code_proposal(_approval(payload=payload))
        assert result.parsed_patch.errors == ()

    def test_marker_after_mid_hunk_context_before_changes_refuses(self):
        patch = _hunk_diff(
            "@@ -1,2 +1,2 @@",
            [" ctx", r"\ No newline at end of file", "-old", "+new"],
        )
        self._assert_refused(patch, "no-newline")

    def test_marker_after_removal_then_context_refuses(self):
        patch = _hunk_diff(
            "@@ -1,2 +1,2 @@",
            ["-old", r"\ No newline at end of file", " ctx", "+new"],
        )
        self._assert_refused(patch, "no-newline")

    def test_marker_before_any_body_line_refuses(self):
        patch = _hunk_diff(
            "@@ -1,1 +1,1 @@",
            [r"\ No newline at end of file", "-old", "+new"],
        )
        self._assert_refused(patch, "no-newline")

    def test_repeated_marker_refuses(self):
        patch = _hunk_diff(
            "@@ -1,1 +1,1 @@",
            [
                "-old",
                r"\ No newline at end of file",
                r"\ No newline at end of file",
                "+new",
            ],
        )
        self._assert_refused(patch, "no-newline")


# ---------------------------------------------------------------------------
# Stage 7B.3 — index metadata validation
# ---------------------------------------------------------------------------


def _index_diff(index_line: str) -> str:
    return (
        "diff --git a/app/safe.py b/app/safe.py\n"
        f"{index_line}\n"
        "--- a/app/safe.py\n"
        "+++ b/app/safe.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
    )


class TestIndexMetadata:
    def _validate(self, patch: str):
        payload = _minimal_valid_payload(
            patch_preview=patch, touched_paths=["app/safe.py"]
        )
        return validate_approved_code_proposal(_approval(payload=payload))

    def test_valid_index_line_with_mode_validates(self):
        result = self._validate(_index_diff("index 0123abc..4567def 100644"))
        assert result.parsed_patch.errors == ()

    def test_valid_index_line_without_mode_validates(self):
        result = self._validate(_index_diff("index 0123abc..4567def"))
        assert result.parsed_patch.errors == ()

    @pytest.mark.parametrize(
        "index_line",
        [
            "index definitely-not-valid",
            "index 0123abc..4567def 100644 extra",
            "index 0123ABC..4567DEF 100644",
            "index 0123abc...4567def 100644",
            "index ..",
            "index 0123abc..4567def 999999",
        ],
    )
    def test_malformed_index_line_refuses(self, index_line):
        with pytest.raises(ApplicatorValidationError, match="malformed index"):
            self._validate(_index_diff(index_line))

    def test_symlink_mode_index_line_refuses(self):
        with pytest.raises(ApplicatorValidationError, match="Symlink"):
            self._validate(_index_diff("index 0123abc..4567def 120000"))

    def test_submodule_mode_index_line_refuses(self):
        with pytest.raises(ApplicatorValidationError, match="Submodule"):
            self._validate(_index_diff("index 0123abc..4567def 160000"))


# ---------------------------------------------------------------------------
# parse_unified_diff_metadata unit tests
# ---------------------------------------------------------------------------


class TestParseUnifiedDiffMetadata:
    def test_parses_simple_diff(self):
        patch = _simple_diff()
        parsed = parse_unified_diff_metadata(patch)
        assert parsed.touched_paths == ("app/static/lewis.css",)
        assert parsed.file_count == 1
        assert parsed.hunk_count == 1
        assert parsed.is_unified_diff is True
        assert parsed.has_binary_marker is False
        assert parsed.has_dev_null is False
        assert parsed.has_empty_patch is False
        assert parsed.errors == ()

    def test_detects_binary_marker(self):
        patch = """diff --git a/app/image.png b/app/image.png
GIT binary patch
literal 1234
"""
        parsed = parse_unified_diff_metadata(patch)
        assert parsed.has_binary_marker is True
        assert "Binary patch marker" in " ".join(parsed.errors)

    def test_detects_dev_null(self):
        patch = """diff --git a/app/file.py b/app/file.py
deleted file mode 100644
--- a/app/file.py
+++ /dev/null
@@ -1,1 +0,0 @@
-old
"""
        parsed = parse_unified_diff_metadata(patch)
        assert parsed.has_dev_null is True
        assert any("/dev/null" in e for e in parsed.errors)

    def test_detects_rename(self):
        patch = """diff --git a/app/old.py b/app/new.py
rename from app/old.py
rename to app/new.py
"""
        parsed = parse_unified_diff_metadata(patch)
        assert parsed.has_rename is True
        assert any("Rename diffs" in e for e in parsed.errors)

    def test_detects_copy(self):
        patch = """diff --git a/app/original.py b/app/copy.py
copy from app/original.py
copy to app/copy.py
--- a/app/original.py
+++ b/app/copy.py
@@ -1,1 +1,1 @@
-old
+new
"""
        parsed = parse_unified_diff_metadata(patch)
        assert parsed.has_copy is True

    def test_detects_empty_patch(self):
        parsed = parse_unified_diff_metadata("")
        assert parsed.has_empty_patch is True
        assert parsed.is_unified_diff is False
        assert parsed.errors == ("Patch preview is empty.",)

    def test_detects_absolute_path(self):
        patch = """diff --git a//etc/hosts b//etc/hosts
--- a//etc/hosts
+++ b//etc/hosts
@@ -1,1 +1,1 @@
-old
+new
"""
        parsed = parse_unified_diff_metadata(patch)
        assert parsed.has_absolute_path is True

    def test_detects_parent_traversal(self):
        patch = """diff --git a/../outside/file.py b/../outside/file.py
--- a/../outside/file.py
+++ b/../outside/file.py
@@ -1,1 +1,1 @@
-old
+new
"""
        parsed = parse_unified_diff_metadata(patch)
        assert parsed.has_parent_traversal is True

    def test_detects_backslash_path(self):
        patch = """diff --git "a/app\\file.py" "b/app\\file.py"
--- "a/app\\file.py"
+++ "b/app\\file.py"
@@ -1,1 +1,1 @@
-old
+new
"""
        parsed = parse_unified_diff_metadata(patch)
        assert parsed.has_backslash_path is True

    def test_counts_hunks_correctly(self):
        patch = """diff --git a/app/file.py b/app/file.py
--- a/app/file.py
+++ b/app/file.py
@@ -1,3 +1,3 @@
 a
-b
+B
 c
@@ -10,3 +10,3 @@
 x
-y
+Y
 z
"""
        parsed = parse_unified_diff_metadata(patch)
        assert parsed.hunk_count == 2
        assert parsed.errors == ()


# ---------------------------------------------------------------------------
# Limit enforcement tests
# ---------------------------------------------------------------------------


class TestLimits:
    def test_patch_size_limit_exceeded_raises(self):
        # Create a patch larger than MAX_PATCH_SIZE_BYTES
        filler = "x" * (MAX_PATCH_SIZE_BYTES + 100)
        patch = (
            f"diff --git a/app/file.py b/app/file.py\n"
            f"--- a/app/file.py\n"
            f"+++ b/app/file.py\n"
            f"@@ -1,1 +1,1 @@\n"
            f"-old\n"
            f"+{filler}\n"
        )
        payload = _minimal_valid_payload(
            patch_preview=patch,
            touched_paths=["app/file.py"],
        )
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="Patch size"):
            validate_approved_code_proposal(approval)

    def test_file_count_limit_exceeded_raises(self):
        paths = [f"app/file_{i}.py" for i in range(MAX_TOUCHED_FILES + 1)]
        parts: list[str] = []
        for p in paths:
            parts.append(
                f"diff --git a/{p} b/{p}\n"
                f"--- a/{p}\n"
                f"+++ b/{p}\n"
                f"@@ -1,1 +1,1 @@\n"
                f"-old\n"
                f"+new\n"
            )
        patch = "".join(parts)
        payload = _minimal_valid_payload(
            patch_preview=patch,
            touched_paths=paths,
        )
        approval = _approval(payload=payload)
        with pytest.raises(
            ApplicatorValidationError, match="touches.*files.*exceeding"
        ):
            validate_approved_code_proposal(approval)

    def test_hunk_limit_exceeded_raises(self):
        parts = [
            "diff --git a/app/file.py b/app/file.py\n--- a/app/file.py\n+++ b/app/file.py\n"
        ]
        for i in range(MAX_TOTAL_HUNKS + 1):
            parts.append(f"@@ -{i + 1},1 +{i + 1},1 @@\n-old\n+new\n")
        patch = "".join(parts)
        payload = _minimal_valid_payload(
            patch_preview=patch,
            touched_paths=["app/file.py"],
        )
        approval = _approval(payload=payload)
        with pytest.raises(ApplicatorValidationError, match="exceeding"):
            validate_approved_code_proposal(approval)

    def test_per_file_hunk_limit_exceeded_raises(self):
        parts = [
            "diff --git a/app/file.py b/app/file.py\n"
            "--- a/app/file.py\n"
            "+++ b/app/file.py\n"
        ]
        for i in range(MAX_HUNKS_PER_FILE + 1):
            parts.append(f"@@ -{i + 1},1 +{i + 1},1 @@\n-old\n+new\n")
        payload = _minimal_valid_payload(
            patch_preview="".join(parts), touched_paths=["app/file.py"]
        )
        with pytest.raises(ApplicatorValidationError, match="per-file maximum"):
            validate_approved_code_proposal(_approval(payload=payload))


# ---------------------------------------------------------------------------
# Legacy payload refusal
# ---------------------------------------------------------------------------


class TestLegacyPayloadRefusal:
    """Ensure legacy/non-contract propose_code_change payloads are refused."""

    def test_legacy_payload_no_structured_fields_raises(self):
        """A payload without selected_strategy, commands_run, etc. is refused."""
        legacy = {
            "requested_goal": "Some goal",
            "target_surface": "some target",
            "relevant_files": [],
            "proposed_approach": "some approach",
            "patch_preview": _simple_diff(),
            "patch_preview_implementation_ready": True,
            "patch_preview_kind": "unified diff",
            "patch_is_inert": True,
            "touched_paths": ["app/static/lewis.css"],
            # missing: selected_strategy, commands_run, files_modified
        }
        approval = _approval(payload=legacy)
        with pytest.raises(
            ApplicatorValidationError, match="Missing required structured field"
        ):
            validate_approved_code_proposal(approval)


# ---------------------------------------------------------------------------
# ParsedPatch dataclass tests
# ---------------------------------------------------------------------------


class TestParsedPatchDataclass:
    def test_default_errors_empty(self):
        parsed = ParsedPatch(
            touched_paths=(),
            file_count=0,
            hunk_count=0,
            total_added_lines=0,
            total_removed_lines=0,
            patch_size_bytes=0,
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
            has_empty_patch=False,
            has_malformed_header=True,
        )
        assert parsed.errors == ()
        assert parsed.file_count == 0


# ---------------------------------------------------------------------------
# isort verification (no-op but ensures import order is checked)
# ---------------------------------------------------------------------------


def test_no_chat_runtime_imports_in_validation_module():
    """Verify the validation module does not import chat/session/runtime modules."""
    import app.dev.proposal_validation as pv

    source = pv.__file__
    if source is None:
        pytest.skip("Cannot find proposal_validation module source.")
    with open(source, encoding="utf-8") as f:
        content = f.read()
    # Must not import from chat/session/runtime modules
    forbidden_imports = [
        "from app.llm",
        "from app.tools",
        "from app.main",
        "from app.web_routes",
        "from app.web_pages",
        "from app.web_components",
        "from app.dashboard",
        "from app.executors",
    ]
    for forbidden in forbidden_imports:
        assert forbidden not in content, f"Validation module imports {forbidden}"
