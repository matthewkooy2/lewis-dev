"""Direct coverage for the shared freestyle policy/state/artifact helpers."""

from __future__ import annotations

import json

import pytest

import app.dev.freestyle_review as review
from app.dev.freestyle_artifacts import (
    encode_json_artifact,
    safe_artifact_reference,
    write_artifact_bytes,
)
from app.dev.freestyle_coding import FreestyleJob, FreestyleJobRefusedError
from app.dev.freestyle_policy import (
    APPROVAL_STORE_CLASS_NAME,
    GENERATED_OR_CACHE_PARTS,
    approval_fixture_allowed_for_path,
    approval_sensitive_region_labels,
    approval_test_fixture_refusal,
    defines_approval_store,
    is_job_id,
    is_test_file_path,
    recovery_message,
)
from app.dev.freestyle_state import (
    ACTIVE_STATES,
    INCONSISTENT_STATES,
    is_materialized,
    require_materialized,
    require_no_active_edit_application,
    resolved_roots,
)

_JOB_ID = "a" * 32


def _job(**overrides) -> FreestyleJob:
    fields = {
        "job_id": _JOB_ID,
        "title": "Helper test job",
        "goal": "Exercise shared guards.",
        "branch": f"lewis/freestyle-{_JOB_ID[:12]}",
        "worktree_path": "/tmp/unused",
        "base_commit": "b" * 40,
        "status": "materialized",
        "created_at": "2026-06-11T00:00:00+00:00",
        "updated_at": "2026-06-11T00:00:00+00:00",
        "developer_approved": True,
        "materialization_status": "materialized",
    }
    fields.update(overrides)
    return FreestyleJob(**fields)


def test_resolved_roots_defaults_state_inside_project(tmp_path):
    root, state = resolved_roots(tmp_path, None)
    assert root == tmp_path.resolve()
    assert state == (tmp_path / ".lewis-dev").resolve()


def test_resolved_roots_refuses_state_outside_project(tmp_path):
    outside = tmp_path.parent / "elsewhere"
    with pytest.raises(FreestyleJobRefusedError, match="inside the project root"):
        resolved_roots(tmp_path, outside)


def test_resolved_roots_refuses_state_equal_to_project_root(tmp_path):
    with pytest.raises(FreestyleJobRefusedError, match="cannot be the project root"):
        resolved_roots(tmp_path, tmp_path)


def test_require_materialized_fails_closed_for_partial_states():
    require_materialized(_job(), "must be successfully materialized first.")
    for overrides in (
        {"status": "ready", "materialization_status": "not_materialized"},
        {"status": "materialized", "materialization_status": "error"},
        {"status": "materialization_error", "materialization_status": "materialized"},
    ):
        partial = _job(**overrides)
        assert not is_materialized(partial)
        with pytest.raises(FreestyleJobRefusedError, match="successfully materialized"):
            require_materialized(partial, "must be successfully materialized first.")


@pytest.mark.parametrize("blocked", ["applying", "inconsistent"])
def test_require_no_active_edit_application_fails_closed(blocked):
    with pytest.raises(
        FreestyleJobRefusedError, match="in-progress or inconsistent"
    ) as raised:
        require_no_active_edit_application(
            _job(edit_apply_status=blocked), "; refusing for the test."
        )
    assert str(raised.value).endswith("; refusing for the test.")
    require_no_active_edit_application(
        _job(edit_apply_status="applied"), "; refusing for the test."
    )


def test_active_states_cover_every_stage_lock_without_inconsistent():
    assert set(ACTIVE_STATES) == {
        "status",
        "materialization_status",
        "proposal_status",
        "edit_generation_status",
        "edit_apply_status",
        "review_status",
        "checks_status",
        "latest_run_status",
        "latest_refinement_status",
    }
    assert "inconsistent" not in ACTIVE_STATES["edit_apply_status"]
    assert INCONSISTENT_STATES == {"edit_apply_status": {"inconsistent"}}


def test_policy_markers_and_review_cache_superset():
    assert APPROVAL_STORE_CLASS_NAME == "Approval" + "Store"
    assert "cache" not in GENERATED_OR_CACHE_PARTS
    assert review._GENERATED_OR_CACHE_PARTS == GENERATED_OR_CACHE_PARTS | {"cache"}


def test_defines_approval_store_matches_definitions_not_references():
    class_name = APPROVAL_STORE_CLASS_NAME
    assert defines_approval_store(f"class {class_name}:\n    pass\n")
    assert defines_approval_store(f"class  {class_name}(Base):\n    pass\n")
    for reference_only in (
        f"from app.approvals import {class_name}\n",
        f"store = {class_name}(settings.database_path)\n",
        f"def _store(request) -> {class_name}:\n    ...\n",
        f"class {class_name}Stub:\n    pass\n",
        "ROUTES = ['/shenanigans']\n",
    ):
        assert not defines_approval_store(reference_only)


def test_approval_sensitive_region_labels_match_markers_without_page_helpers():
    class_name = APPROVAL_STORE_CLASS_NAME
    assert approval_sensitive_region_labels(
        f"from app.approvals import {class_name}"
    ) == (
        "approval-store class reference",
        "approval module reference",
    )
    assert approval_sensitive_region_labels("store = _approval_store(request)") == (
        "approval-store helper reference",
    )
    assert approval_sensitive_region_labels(
        "item = approve_approval(store, approval_id)"
    ) == ("approval mutation call",)
    assert approval_sensitive_region_labels("reject_approval(store, 3)") == (
        "approval mutation call",
    )
    assert approval_sensitive_region_labels("result = execute_approved(store, 3)") == (
        "approved-execution call",
    )
    assert approval_sensitive_region_labels("from app.approval_service import x") == (
        "approval module reference",
    )
    for safe_text in (
        "from app.web_pages import approvals_page, shenanigans_page\n",
        'return _page("Shenanigans", request, shenanigans_page())\n',
        "approvals_page(items, message)\n",
        "the approval store is documented elsewhere\n",
        "app.approvals_page\n",
    ):
        assert approval_sensitive_region_labels(safe_text) == ()


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tests/test_calendar_page.py", True),
        ("conftest.py", True),
        ("pkg/tests/test_thing.py", True),
        ("app/dev/freestyle_proposal.py", False),
        ("app/web_routes.py", False),
        ("src/contests/runner.py", False),
    ],
)
def test_is_test_file_path_only_matches_test_locations(path, expected):
    assert is_test_file_path(path) is expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # Non-approval feature test files qualify for the fixture allowance.
        ("tests/test_calendar_page.py", True),
        ("tests/calendar/test_view.py", True),
        ("conftest.py", True),
        # Approval-focused test files are held to the full region rules.
        ("tests/test_approvals.py", False),
        ("tests/test_approval_flow.py", False),
        ("tests/test_approval_service.py", False),
        # Non-test files never qualify, approval-named or not.
        ("app/approvals.py", False),
        ("app/web_routes.py", False),
    ],
)
def test_approval_fixture_allowance_is_limited_to_non_approval_tests(path, expected):
    assert approval_fixture_allowed_for_path(path) is expected


def test_approval_test_fixture_refusal_allows_only_proposal_seeding():
    class_name = APPROVAL_STORE_CLASS_NAME
    # Fixture seeding for an unrelated feature: import + construct the store and
    # add a local proposal record. None of this changes approval behavior.
    safe_fixture = (
        f"from app.approvals import {class_name}\n"
        "\n"
        "def test_calendar_page_lists_proposals(tmp_path):\n"
        f"    store = {class_name}(tmp_path / 'db.sqlite3')\n"
        "    store.add_proposal(kind='calendar', payload={'day': 1})\n"
        "    assert store.list_proposals()\n"
    )
    assert approval_test_fixture_refusal(safe_fixture) is None

    # Each forbidden signal is reported, never silently allowed.
    assert "approval route literal" in approval_test_fixture_refusal(
        '    response = client.get("/approvals")\n'
    )
    assert "approval helper definition" in approval_test_fixture_refusal(
        "def _approval_store(request):\n    return None\n"
    )
    assert "approval-store class definition" in approval_test_fixture_refusal(
        f"class {class_name}:\n    pass\n"
    )
    for behavior in (
        "    approve_approval(store, 1)\n",
        "    reject_approval(store, 1)\n",
        "    execute_approved(store, 1)\n",
        "    store = _approval_store(request)\n",
        "from app.approval_service import approve_approval\n",
    ):
        reason = approval_test_fixture_refusal(behavior)
        assert reason is not None
        assert "approval behavior reference" in reason


def test_recovery_message_matches_slice_texts():
    assert recovery_message("retrying review") == (
        "Recover or verify the isolated worktree and job metadata before "
        "retrying review."
    )


@pytest.mark.parametrize(
    "value",
    [None, "", "x" * 501, "a\x00b", "a\\b", "a\tb", "/absolute/path", "a/../b"],
)
def test_safe_artifact_reference_rejects_unsafe_values(value):
    assert safe_artifact_reference(value) is None


def test_safe_artifact_reference_returns_relative_posix_path():
    assert safe_artifact_reference("reviews/abc/review-1.json") == (
        "reviews/abc/review-1.json"
    )


def test_is_job_id_accepts_only_lowercase_hex_of_length_32():
    assert is_job_id(_JOB_ID)
    for bad in ("", _JOB_ID[:-1], _JOB_ID + "a", _JOB_ID.upper(), "g" * 32, None, 5):
        assert not is_job_id(bad)


def test_encode_json_artifact_is_sorted_indented_and_newline_terminated():
    encoded = encode_json_artifact({"b": 1, "a": [2]})
    assert encoded == b'{\n  "a": [\n    2\n  ],\n  "b": 1\n}\n'
    assert json.loads(encoded.decode("utf-8")) == {"a": [2], "b": 1}


def test_write_artifact_bytes_writes_atomically_under_state(tmp_path):
    state = tmp_path / ".lewis-dev"
    state.mkdir()
    relative = write_artifact_bytes(
        state, "reviews", _JOB_ID, "review-1.json", b"{}\n", label="review"
    )
    assert relative == f"reviews/{_JOB_ID}/review-1.json"
    target = state / "reviews" / _JOB_ID / "review-1.json"
    assert target.read_bytes() == b"{}\n"
    assert not target.with_name("review-1.json.tmp").exists()


def test_write_artifact_bytes_refuses_existing_artifact_without_replace(tmp_path):
    state = tmp_path / ".lewis-dev"
    state.mkdir()
    write_artifact_bytes(
        state, "reviews", _JOB_ID, "review-1.json", b"{}\n", label="review"
    )
    with pytest.raises(FreestyleJobRefusedError, match="artifact path is unsafe"):
        write_artifact_bytes(
            state, "reviews", _JOB_ID, "review-1.json", b"{}\n", label="review"
        )


def test_write_artifact_bytes_allow_replace_overwrites_regular_file_only(tmp_path):
    state = tmp_path / ".lewis-dev"
    state.mkdir()
    for payload in (b"first\n", b"second\n"):
        write_artifact_bytes(
            state,
            "proposals",
            _JOB_ID,
            "proposal.json",
            payload,
            label="proposal",
            allow_replace=True,
        )
    target = state / "proposals" / _JOB_ID / "proposal.json"
    assert target.read_bytes() == b"second\n"
    target.unlink()
    target.mkdir()
    with pytest.raises(FreestyleJobRefusedError, match="artifact path is unsafe"):
        write_artifact_bytes(
            state,
            "proposals",
            _JOB_ID,
            "proposal.json",
            b"third\n",
            label="proposal",
            allow_replace=True,
        )


def test_write_artifact_bytes_refuses_symlinked_directories_and_temp_files(tmp_path):
    state = tmp_path / ".lewis-dev"
    state.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (state / "checks").symlink_to(outside)
    with pytest.raises(FreestyleJobRefusedError, match="artifact path is unsafe"):
        write_artifact_bytes(
            state, "checks", _JOB_ID, "checks-1.json", b"{}\n", label="check"
        )
    job_dir = state / "runs" / _JOB_ID
    job_dir.mkdir(parents=True)
    (job_dir / "run-1.json.tmp").write_bytes(b"stale")
    with pytest.raises(
        FreestyleJobRefusedError, match="temporary artifact path is unsafe"
    ):
        write_artifact_bytes(
            state, "runs", _JOB_ID, "run-1.json", b"{}\n", label="orchestration"
        )
