"""Tests for the deterministic freestyle edit-bundle applier."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

import app.dev.freestyle_coding as freestyle
import app.dev.freestyle_edit_bundle as edit_bundle
from app.dev.freestyle_coding import (
    FreestyleJobRefusedError,
    apply_freestyle_edit_bundle,
    create_freestyle_job,
    get_freestyle_job,
    materialize_freestyle_job,
)
from app.dev.freestyle_policy import APPROVAL_STORE_CLASS_NAME

TRACKED_SHA = hashlib.sha256(b"committed\n").hexdigest()
NOTES_SHA = hashlib.sha256(b"note\n").hexdigest()
SAMPLE_SHA = hashlib.sha256(b"VALUE = 1\n").hexdigest()

# A worktree file with several distinct, non-overlapping anchors for
# multi-replace_text tests.
_MULTI_ANCHOR_TEXT = "ALPHA = 1\nBETA = 2\nGAMMA = 3\n"

# A routes file that references approval code without defining it, mirroring
# the real app/web_routes.py. Built with the concatenated class name so this
# test file itself never contains the literal class text.
_APPROVAL_REFERENCING_ROUTES = f"""from app.approvals import {APPROVAL_STORE_CLASS_NAME}
from app.approval_service import approve_approval
from app.web_pages import approvals_page, calendar_page, chat_page


def register_web_routes(app):
    @app.get("/chat")
    def chat(request):
        return _page("Chat", request, chat_page())

    @app.get("/approvals")
    def approvals(request, message=""):
        items = list_pending_approvals(_approval_store(request))
        return _page("Approvals", request, approvals_page(items, message))

    @app.get("/calendar")
    def calendar(request, message=""):
        return _page("Calendar", request, calendar_page(list_calendar_proposals(_approval_store(request)), message))

    @app.post("/approvals/{{approval_id}}/approve")
    def approve_item(request, approval_id):
        return approve_approval(_approval_store(request), approval_id)


def _approval_store(request):
    return {APPROVAL_STORE_CLASS_NAME}(request.app.state.settings.database_path)


def _page(title, request, content):
    return content
"""

_SAFE_ROUTE_ANCHOR = '        return _page("Chat", request, chat_page())'
_WIRED_ROUTE_TEXT = (
    '        return _page("Chat", request, chat_page())\n'
    "\n"
    '    @app.get("/shenanigans")\n'
    "    def shenanigans(request):\n"
    '        return _page("Shenanigans", request, shenanigans_page())'
)

# The unedited /calendar route body: a non-approval feature route that merely
# *calls* the route-level _approval_store(request) helper to obtain the store.
_CALENDAR_ANCHOR = (
    '        return _page("Calendar", request, '
    "calendar_page(list_calendar_proposals(_approval_store(request)), message))"
)
# A safe refinement that parses month/day query params and passes extra page
# parameters while preserving list_calendar_proposals(_approval_store(request)).
_CALENDAR_WITH_QUERY = (
    '        month = request.query_params.get("month")\n'
    '        day = request.query_params.get("day")\n'
    "        proposals = list_calendar_proposals(_approval_store(request))\n"
    '        return _page("Calendar", request, '
    "calendar_page(proposals, message, month=month, day=day))"
)
# The unedited /approvals GET route body, whose only approval marker is the
# preserved _approval_store(request) call; editing it is still an approval
# route change because the route path itself appears in the edited region.
_APPROVALS_ROUTE_ANCHOR = (
    '    @app.get("/approvals")\n'
    '    def approvals(request, message=""):\n'
    "        items = list_pending_approvals(_approval_store(request))\n"
    '        return _page("Approvals", request, approvals_page(items, message))'
)
# The route-level _approval_store helper *definition* region.
_APPROVAL_HELPER_DEFINITION = (
    "def _approval_store(request):\n"
    f"    return {APPROVAL_STORE_CLASS_NAME}"
    "(request.app.state.settings.database_path)"
)


def _routes_job(root: Path):
    """Materialize one job whose worktree contains the routes file above."""
    job = _materialized_job(root, goal="Wire the shenanigans page route.")
    routes = Path(job.worktree_path) / "app" / "web_routes.py"
    routes.parent.mkdir(parents=True, exist_ok=True)
    routes.write_text(_APPROVAL_REFERENCING_ROUTES, encoding="utf-8", newline="\n")
    return job, routes


def _multi_anchor_job(root: Path):
    """Materialize one job whose worktree holds a multi-anchor text file."""
    job = _materialized_job(root)
    target = Path(job.worktree_path) / "src" / "multi.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_MULTI_ANCHOR_TEXT, encoding="utf-8", newline="\n")
    return job, target


def _replace_text_op(path: str, old_text: str, new_text: str) -> dict:
    return {
        "op": "replace_text",
        "path": path,
        "old_text": old_text,
        "new_text": new_text,
    }


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        shell=False,
    )
    return completed.stdout.decode().strip()


def _init_repo(tmp_path: Path) -> Path:
    # Files are written as bytes and autocrlf is pinned off so that committed,
    # archived, and materialized bytes are deterministic for hash assertions.
    root = tmp_path / "project"
    root.mkdir()
    (root / ".gitignore").write_text(".lewis-dev/\n", encoding="utf-8")
    (root / "tracked.txt").write_bytes(b"committed\n")
    (root / "notes.txt").write_bytes(b"note\n")
    (root / "src").mkdir()
    (root / "src" / "sample.py").write_bytes(b"VALUE = 1\n")
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "init")
    return root


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _materialized_job(root: Path, goal: str = "Apply structured edits safely."):
    job = create_freestyle_job(goal, developer_approved=True, project_root=root)
    return materialize_freestyle_job(job.job_id, project_root=root)


def _bundle_payload(job_id: str, operations: list[dict], **overrides) -> dict:
    payload = {
        "schema_version": 1,
        "job_id": job_id,
        "created_at": "2026-06-10T00:00:00+00:00",
        "operation_count": len(operations),
        "operations": operations,
    }
    payload.update(overrides)
    return payload


def _write_bundle(directory: Path, payload: dict, name: str = "bundle.json") -> Path:
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _create_op(path: str = "created.txt", content: str = "created\n") -> dict:
    return {"op": "create_file", "path": path, "content": content}


def _worktree_snapshot(worktree: Path) -> dict[str, bytes]:
    return {
        path.relative_to(worktree).as_posix(): path.read_bytes()
        for path in worktree.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def _checkout_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and ".git" not in path.relative_to(root).parts
        and freestyle.DEV_STATE_DIR_NAME not in path.relative_to(root).parts
    }


def _refused_apply(
    root: Path, job_id: str, bundle_path: Path
) -> "freestyle.FreestyleJob":
    """Apply expecting the wrapped refusal; return the persisted failed job."""
    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        apply_freestyle_edit_bundle(job_id, str(bundle_path), project_root=root)
    return get_freestyle_job(job_id, project_root=root)


def test_applies_create_replace_delete_and_replace_text_inside_worktree_only(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree = Path(job.worktree_path)
    checkout_before = _checkout_snapshot(root)
    status_before = _git(root, "status", "--porcelain")
    operations = [
        {
            "op": "create_file",
            "path": "src/new_module.py",
            "content": "CREATED = True\n",
        },
        {
            "op": "replace_file",
            "path": "tracked.txt",
            "content": "replaced\n",
            "expected_sha256": TRACKED_SHA,
        },
        {"op": "delete_file", "path": "notes.txt", "expected_sha256": NOTES_SHA},
        {
            "op": "replace_text",
            "path": "src/sample.py",
            "old_text": "VALUE = 1",
            "new_text": "VALUE = 2",
        },
    ]
    bundle_path = _write_bundle(
        tmp_path,
        _bundle_payload(job.job_id, operations, proposal_id="proposal.json"),
    )

    updated = apply_freestyle_edit_bundle(
        job.job_id, str(bundle_path), project_root=root
    )

    assert (worktree / "src" / "new_module.py").read_text(encoding="utf-8") == (
        "CREATED = True\n"
    )
    assert (worktree / "tracked.txt").read_text(encoding="utf-8") == "replaced\n"
    assert not (worktree / "notes.txt").exists()
    assert (worktree / "src" / "sample.py").read_text(encoding="utf-8") == "VALUE = 2\n"

    assert updated.status == "materialized"
    assert updated.edit_apply_status == "applied"
    assert updated.edit_apply_started_at
    assert updated.edit_applied_at
    assert updated.edit_applied_file_count == 4
    assert updated.edit_errors == ()
    assert updated.edit_artifact_path
    assert updated.edit_artifact_path.startswith(f"edits/{job.job_id}/")

    artifact = json.loads(
        (root / freestyle.DEV_STATE_DIR_NAME / updated.edit_artifact_path).read_text(
            encoding="utf-8"
        )
    )
    assert artifact["job_id"] == job.job_id
    assert artifact["proposal_id"] == "proposal.json"
    assert artifact["bundle_sha256"] == _sha(bundle_path.read_bytes())
    assert artifact["operation_count"] == 4
    assert artifact["touched_file_count"] == 4
    by_path = {entry["path"]: entry for entry in artifact["operations"]}
    assert by_path["src/new_module.py"]["before_sha256"] is None
    assert by_path["src/new_module.py"]["after_sha256"] == _sha(b"CREATED = True\n")
    assert by_path["tracked.txt"]["before_sha256"] == TRACKED_SHA
    assert by_path["tracked.txt"]["after_sha256"] == _sha(b"replaced\n")
    assert by_path["notes.txt"]["before_sha256"] == NOTES_SHA
    assert by_path["notes.txt"]["after_sha256"] is None
    assert by_path["src/sample.py"]["before_sha256"] == _sha(b"VALUE = 1\n")
    assert by_path["src/sample.py"]["after_sha256"] == _sha(b"VALUE = 2\n")

    # The main checkout and its branch state remain untouched.
    assert _checkout_snapshot(root) == checkout_before
    assert _git(root, "status", "--porcelain") == status_before
    assert _git(root, "rev-parse", "--abbrev-ref", "HEAD") == "main"


def test_rejects_jobs_that_are_not_successfully_materialized(tmp_path):
    root = _init_repo(tmp_path)
    job = create_freestyle_job(
        "Edit after materialization only.",
        developer_approved=True,
        project_root=root,
    )
    bundle_path = _write_bundle(tmp_path, _bundle_payload(job.job_id, [_create_op()]))

    with pytest.raises(FreestyleJobRefusedError, match="materialized before"):
        apply_freestyle_edit_bundle(job.job_id, str(bundle_path), project_root=root)

    refreshed = get_freestyle_job(job.job_id, project_root=root)
    assert refreshed.edit_apply_status == "not_applied"
    assert [entry.name for entry in Path(job.worktree_path).iterdir()] == [".git"]


def test_rejects_bundle_bound_to_a_different_job_id(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree_before = _worktree_snapshot(Path(job.worktree_path))
    bundle_path = _write_bundle(tmp_path, _bundle_payload("f" * 32, [_create_op()]))

    failed = _refused_apply(root, job.job_id, bundle_path)

    assert failed.edit_apply_status == "error"
    assert "different job id" in failed.edit_errors[0]
    assert _worktree_snapshot(Path(job.worktree_path)) == worktree_before


@pytest.mark.parametrize(
    "bad_path",
    [
        "C:/escape.txt",
        "C:\\escape.txt",
        "/escape.txt",
        "../escape.txt",
        "nested/../escape.txt",
        "nested\\escape.txt",
    ],
)
def test_rejects_absolute_and_traversal_paths(tmp_path, bad_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree_before = _worktree_snapshot(Path(job.worktree_path))
    bundle_path = _write_bundle(
        tmp_path, _bundle_payload(job.job_id, [_create_op(path=bad_path)])
    )

    failed = _refused_apply(root, job.job_id, bundle_path)

    assert failed.edit_apply_status == "error"
    assert _worktree_snapshot(Path(job.worktree_path)) == worktree_before
    assert not (tmp_path / "escape.txt").exists()
    assert not (root / "escape.txt").exists()


def test_rejects_symlink_escape_targets(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree = Path(job.worktree_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "real.txt").write_text("outside\n", encoding="utf-8")
    try:
        (worktree / "linked").symlink_to(outside, target_is_directory=True)
        (worktree / "alias.txt").symlink_to(outside / "real.txt")
    except OSError:
        pytest.skip("symlink creation is not supported in this environment")

    through_directory = _write_bundle(
        tmp_path,
        _bundle_payload(job.job_id, [_create_op(path="linked/escape.txt")]),
        name="through-directory.json",
    )
    failed = _refused_apply(root, job.job_id, through_directory)
    error = failed.edit_errors[0]
    assert "symlink" in error or "escapes" in error or "stay inside" in error
    assert not (outside / "escape.txt").exists()

    through_file = _write_bundle(
        tmp_path,
        _bundle_payload(
            job.job_id,
            [
                {
                    "op": "replace_file",
                    "path": "alias.txt",
                    "content": "evil\n",
                    "expected_sha256": _sha(b"outside\n"),
                }
            ],
        ),
        name="through-file.json",
    )
    failed = _refused_apply(root, job.job_id, through_file)
    error = failed.edit_errors[0]
    assert "symlink" in error or "escapes" in error or "stay inside" in error
    assert (outside / "real.txt").read_text(encoding="utf-8") == "outside\n"


@pytest.mark.skipif(os.name != "nt", reason="junctions are Windows-only")
def test_rejects_windows_junction_escape_targets(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree = Path(job.worktree_path)
    outside = tmp_path / "outside-junction"
    outside.mkdir()
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(worktree / "junct"), str(outside)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip("junction creation is not supported in this environment")
    bundle_path = _write_bundle(
        tmp_path, _bundle_payload(job.job_id, [_create_op(path="junct/escape.txt")])
    )

    failed = _refused_apply(root, job.job_id, bundle_path)

    # The junction is refused either by resolve-based containment ("must stay
    # inside") or by the symlink parent-chain check, depending on how the
    # Python version classifies junctions.
    error = failed.edit_errors[0]
    assert "symlink" in error or "escapes" in error or "stay inside" in error
    assert not (outside / "escape.txt").exists()


@pytest.mark.parametrize(
    "blocked_path",
    [
        ".git/config",
        ".env",
        ".lewis-dev/jobs/tampered.json",
        "config/secret_token.txt",
        "data/lewis.sqlite3",
        "store.db",
        "approvals.py",
        "app/approval_service.py",
        "approval_store_helper.py",
        "__pycache__/cached.py",
        "node_modules/pkg/index.js",
        "build/output.txt",
        "tools/run.exe",
        "scripts/setup.ps1",
    ],
)
def test_rejects_protected_secret_db_cache_and_executable_paths(tmp_path, blocked_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree_before = _worktree_snapshot(Path(job.worktree_path))
    bundle_path = _write_bundle(
        tmp_path, _bundle_payload(job.job_id, [_create_op(path=blocked_path)])
    )

    failed = _refused_apply(root, job.job_id, bundle_path)

    assert failed.edit_apply_status == "error"
    assert _worktree_snapshot(Path(job.worktree_path)) == worktree_before


def test_route_wiring_replace_text_is_allowed_in_approval_referencing_file(
    tmp_path,
):
    """Regression for the Tetris refinement false positive: wiring an
    ordinary page route into a file that references approval code must be
    applicable when the edited region itself touches no approval marker."""
    root = _init_repo(tmp_path)
    job, routes = _routes_job(root)
    bundle_path = _write_bundle(
        tmp_path,
        _bundle_payload(
            job.job_id,
            [
                {
                    "op": "replace_text",
                    "path": "app/web_routes.py",
                    "old_text": _SAFE_ROUTE_ANCHOR,
                    "new_text": _WIRED_ROUTE_TEXT,
                }
            ],
        ),
    )

    updated = apply_freestyle_edit_bundle(
        job.job_id, str(bundle_path), project_root=root
    )

    assert updated.edit_apply_status == "applied"
    wired = routes.read_text(encoding="utf-8")
    assert '"/shenanigans"' in wired
    assert "def _approval_store(request):" in wired
    assert "approve_approval(_approval_store(request), approval_id)" in wired


@pytest.mark.parametrize(
    ("old_text", "new_text", "expected_fragment"),
    [
        pytest.param(
            "        return approve_approval(_approval_store(request), approval_id)",
            "        return None",
            "approval-sensitive region touched (approval mutation call)",
            id="old_text_targets_approval_mutation",
        ),
        pytest.param(
            _SAFE_ROUTE_ANCHOR,
            '        return _page("Chat", request, chat_page(_approval_store(request)))',
            "approval-store helper reference changed",
            id="new_text_introduces_approval_reference",
        ),
    ],
)
def test_replace_text_touching_approval_sensitive_region_is_refused(
    tmp_path, old_text, new_text, expected_fragment
):
    root = _init_repo(tmp_path)
    job, routes = _routes_job(root)
    before = routes.read_text(encoding="utf-8")
    bundle_path = _write_bundle(
        tmp_path,
        _bundle_payload(
            job.job_id,
            [
                {
                    "op": "replace_text",
                    "path": "app/web_routes.py",
                    "old_text": old_text,
                    "new_text": new_text,
                }
            ],
        ),
    )

    failed = _refused_apply(root, job.job_id, bundle_path)

    assert expected_fragment in failed.edit_errors[0]
    assert routes.read_text(encoding="utf-8") == before


def _routes_replace_bundle(
    tmp_path: Path, job_id: str, old_text: str, new_text: str, name: str
) -> Path:
    return _write_bundle(
        tmp_path,
        _bundle_payload(
            job_id,
            [
                {
                    "op": "replace_text",
                    "path": "app/web_routes.py",
                    "old_text": old_text,
                    "new_text": new_text,
                }
            ],
        ),
        name=name,
    )


def test_calendar_route_edit_preserving_approval_store_helper_is_allowed(tmp_path):
    """Regression for the Calendar refinement false positive: a non-approval
    feature route may parse query params and pass extra page parameters while
    preserving list_calendar_proposals(_approval_store(request)) unchanged."""
    root = _init_repo(tmp_path)
    job, routes = _routes_job(root)
    bundle_path = _routes_replace_bundle(
        tmp_path, job.job_id, _CALENDAR_ANCHOR, _CALENDAR_WITH_QUERY, "calendar-ok.json"
    )

    updated = apply_freestyle_edit_bundle(
        job.job_id, str(bundle_path), project_root=root
    )

    assert updated.edit_apply_status == "applied"
    wired = routes.read_text(encoding="utf-8")
    assert 'request.query_params.get("month")' in wired
    assert "list_calendar_proposals(_approval_store(request))" in wired
    # The /approvals routes and the helper definition are untouched.
    assert "def _approval_store(request):" in wired
    assert "approve_approval(_approval_store(request), approval_id)" in wired


def test_calendar_route_edit_changing_approval_store_helper_is_refused(tmp_path):
    root = _init_repo(tmp_path)
    job, routes = _routes_job(root)
    before = routes.read_text(encoding="utf-8")
    altered = _CALENDAR_ANCHOR.replace(
        "_approval_store(request)", "_approval_store(req)"
    )
    bundle_path = _routes_replace_bundle(
        tmp_path, job.job_id, _CALENDAR_ANCHOR, altered, "calendar-changed.json"
    )

    failed = _refused_apply(root, job.job_id, bundle_path)

    assert "approval-store helper reference changed" in failed.edit_errors[0]
    assert routes.read_text(encoding="utf-8") == before


def test_calendar_route_edit_removing_approval_store_helper_is_refused(tmp_path):
    root = _init_repo(tmp_path)
    job, routes = _routes_job(root)
    before = routes.read_text(encoding="utf-8")
    removed = _CALENDAR_ANCHOR.replace(
        "list_calendar_proposals(_approval_store(request))",
        "list_calendar_proposals(None)",
    )
    bundle_path = _routes_replace_bundle(
        tmp_path, job.job_id, _CALENDAR_ANCHOR, removed, "calendar-removed.json"
    )

    failed = _refused_apply(root, job.job_id, bundle_path)

    assert "approval-store helper reference changed" in failed.edit_errors[0]
    assert routes.read_text(encoding="utf-8") == before


def test_calendar_route_edit_adding_approval_store_helper_is_refused(tmp_path):
    root = _init_repo(tmp_path)
    job, routes = _routes_job(root)
    before = routes.read_text(encoding="utf-8")
    old_text = '    def calendar(request, message=""):'
    new_text = (
        '    def calendar(request, message=""):\n'
        "        store = _approval_store(request)"
    )
    bundle_path = _routes_replace_bundle(
        tmp_path, job.job_id, old_text, new_text, "calendar-added.json"
    )

    failed = _refused_apply(root, job.job_id, bundle_path)

    assert "approval-store helper reference changed" in failed.edit_errors[0]
    assert routes.read_text(encoding="utf-8") == before


def test_approvals_route_edit_is_refused_even_when_helper_preserved(tmp_path):
    """An /approvals route edit is blocked by the route path itself, even when
    every _approval_store(request) call is preserved byte-for-byte."""
    root = _init_repo(tmp_path)
    job, routes = _routes_job(root)
    before = routes.read_text(encoding="utf-8")
    new_text = _APPROVALS_ROUTE_ANCHOR.replace(
        'def approvals(request, message=""):',
        'def approvals(request, message="", page=1):',
    ).replace(
        "approvals_page(items, message)",
        "approvals_page(items, message, page)",
    )
    bundle_path = _routes_replace_bundle(
        tmp_path, job.job_id, _APPROVALS_ROUTE_ANCHOR, new_text, "approvals-edit.json"
    )

    failed = _refused_apply(root, job.job_id, bundle_path)

    assert "approval route edit blocked" in failed.edit_errors[0]
    assert routes.read_text(encoding="utf-8") == before


def test_approval_store_helper_definition_edit_is_refused(tmp_path):
    root = _init_repo(tmp_path)
    job, routes = _routes_job(root)
    before = routes.read_text(encoding="utf-8")
    new_text = _APPROVAL_HELPER_DEFINITION.replace("database_path", "db_path")
    bundle_path = _routes_replace_bundle(
        tmp_path, job.job_id, _APPROVAL_HELPER_DEFINITION, new_text, "helper-def.json"
    )

    failed = _refused_apply(root, job.job_id, bundle_path)

    assert "approval helper definition edit blocked" in failed.edit_errors[0]
    assert routes.read_text(encoding="utf-8") == before


@pytest.mark.parametrize(
    "blocked_path", ["app/approvals.py", "app/approval_service.py"]
)
def test_replace_text_on_approval_files_is_refused_by_path(tmp_path, blocked_path):
    """Approval-record implementation files stay blocked at the path layer for
    every operation kind, including replace_text."""
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree_before = _worktree_snapshot(Path(job.worktree_path))
    bundle_path = _write_bundle(
        tmp_path,
        _bundle_payload(
            job.job_id,
            [_replace_text_op(blocked_path, "old", "new")],
        ),
        name=f"approval-file-{blocked_path.replace('/', '-')}.json",
    )

    failed = _refused_apply(root, job.job_id, bundle_path)

    assert "approval-record code" in failed.edit_errors[0]
    assert _worktree_snapshot(Path(job.worktree_path)) == worktree_before


def test_whole_file_operations_never_touch_approval_referencing_files(tmp_path):
    root = _init_repo(tmp_path)
    job, routes = _routes_job(root)
    before = routes.read_text(encoding="utf-8")
    routes_sha = hashlib.sha256(before.encode("utf-8")).hexdigest()
    operations = [
        {
            "op": "replace_file",
            "path": "app/web_routes.py",
            "content": "ROUTES = {}\n",
            "expected_sha256": routes_sha,
        },
        {
            "op": "delete_file",
            "path": "app/web_routes.py",
            "expected_sha256": routes_sha,
        },
        {
            "op": "create_file",
            "path": "app/approval_glue.py",
            "content": f"store = {APPROVAL_STORE_CLASS_NAME}('db')\n",
        },
        {
            "op": "replace_file",
            "path": "tracked.txt",
            "content": "from app.approval_service import approve_approval\n",
            "expected_sha256": TRACKED_SHA,
        },
    ]
    for operation in operations:
        bundle_path = _write_bundle(
            tmp_path,
            _bundle_payload(job.job_id, [operation]),
            name=f"bundle-{operation['op']}-{operation['path'].replace('/', '-')}.json",
        )
        failed = _refused_apply(root, job.job_id, bundle_path)
        assert "approval-sensitive region" in failed.edit_errors[0], operation
    assert routes.read_text(encoding="utf-8") == before
    assert not (Path(job.worktree_path) / "app" / "approval_glue.py").exists()


# ---------------------------------------------------------------------------
# Approval-policy precision: a non-approval test file may seed local proposal
# fixtures (importing and constructing the approval store) without that being
# read as an approval *behavior* change, while every approval behavior signal
# and every approval implementation path stays refused. Built with the
# concatenated class name so this test file never holds the literal class text.
# ---------------------------------------------------------------------------

_CALENDAR_FIXTURE_TEST = f"""from app.approvals import {APPROVAL_STORE_CLASS_NAME}
from app.web_pages import calendar_page


def test_calendar_page_lists_seeded_proposals(tmp_path):
    store = {APPROVAL_STORE_CLASS_NAME}(tmp_path / "db.sqlite3")
    store.add_proposal(kind="calendar", payload={{"day": 1}})
    assert "calendar" in calendar_page(store.list_proposals())
"""


def _create_file_bundle(
    tmp_path: Path, job_id: str, path: str, content: str, name: str
) -> Path:
    return _write_bundle(
        tmp_path,
        _bundle_payload(
            job_id, [{"op": "create_file", "path": path, "content": content}]
        ),
        name=name,
    )


def test_new_non_approval_test_may_seed_local_proposal_fixture(tmp_path):
    """A new Calendar test importing and constructing the approval store to
    seed local proposal records is allowed: it changes no approval behavior."""
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree = Path(job.worktree_path)
    bundle_path = _create_file_bundle(
        tmp_path,
        job.job_id,
        "tests/test_calendar_page.py",
        _CALENDAR_FIXTURE_TEST,
        "calendar-fixture-ok.json",
    )

    updated = apply_freestyle_edit_bundle(
        job.job_id, str(bundle_path), project_root=root
    )

    assert updated.edit_apply_status == "applied"
    assert (worktree / "tests" / "test_calendar_page.py").read_text(
        encoding="utf-8"
    ) == _CALENDAR_FIXTURE_TEST


def test_modifying_existing_non_approval_test_fixture_is_allowed(tmp_path):
    """The allowance also covers a *modified* non-approval test file: a
    replace_file whose old and new content are both fixture-only is applied."""
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    target = Path(job.worktree_path) / "tests" / "test_calendar_page.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_CALENDAR_FIXTURE_TEST, encoding="utf-8", newline="\n")
    updated_content = _CALENDAR_FIXTURE_TEST + (
        "\n\ndef test_calendar_page_handles_two_days(tmp_path):\n"
        f'    store = {APPROVAL_STORE_CLASS_NAME}(tmp_path / "db.sqlite3")\n'
        '    store.add_proposal(kind="calendar", payload={"day": 2})\n'
        "    assert store.list_proposals()\n"
    )
    bundle_path = _write_bundle(
        tmp_path,
        _bundle_payload(
            job.job_id,
            [
                {
                    "op": "replace_file",
                    "path": "tests/test_calendar_page.py",
                    "content": updated_content,
                    "expected_sha256": _sha(_CALENDAR_FIXTURE_TEST.encode("utf-8")),
                }
            ],
        ),
        name="calendar-fixture-modify.json",
    )

    updated = apply_freestyle_edit_bundle(
        job.job_id, str(bundle_path), project_root=root
    )

    assert updated.edit_apply_status == "applied"
    assert target.read_text(encoding="utf-8") == updated_content


def test_seed_fixture_test_with_approval_route_literal_is_refused(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree_before = _worktree_snapshot(Path(job.worktree_path))
    content = _CALENDAR_FIXTURE_TEST + (
        "\n\ndef test_calendar_redirects_from_approvals(client):\n"
        '    assert client.get("/approvals").status_code == 200\n'
    )
    bundle_path = _create_file_bundle(
        tmp_path,
        job.job_id,
        "tests/test_calendar_page.py",
        content,
        "calendar-route-literal.json",
    )

    failed = _refused_apply(root, job.job_id, bundle_path)

    assert "approval route literal" in failed.edit_errors[0]
    assert _worktree_snapshot(Path(job.worktree_path)) == worktree_before
    assert not (Path(job.worktree_path) / "tests" / "test_calendar_page.py").exists()


@pytest.mark.parametrize(
    ("approval_call", "marker"),
    [
        ("approve_approval(store, 1)", "approval mutation call"),
        ("reject_approval(store, 1)", "approval mutation call"),
        ("execute_approved(store, 1)", "approved-execution call"),
        ("store = _approval_store(request)", "approval-store helper reference"),
        (
            "from app.approval_service import approve_approval",
            "approval service reference",
        ),
    ],
)
def test_seed_fixture_test_invoking_approval_behavior_is_refused(
    tmp_path, approval_call, marker
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree_before = _worktree_snapshot(Path(job.worktree_path))
    content = _CALENDAR_FIXTURE_TEST + (
        "\n\ndef test_calendar_touches_approval_behavior(request):\n"
        f"    {approval_call}\n"
    )
    bundle_path = _create_file_bundle(
        tmp_path,
        job.job_id,
        "tests/test_calendar_page.py",
        content,
        f"calendar-behavior-{marker.replace(' ', '-')}.json",
    )

    failed = _refused_apply(root, job.job_id, bundle_path)

    error = failed.edit_errors[0]
    assert "approval behavior reference" in error
    assert marker in error
    assert _worktree_snapshot(Path(job.worktree_path)) == worktree_before


def test_approval_focused_test_file_is_not_exempt_from_region_rules(tmp_path):
    """The fixture allowance is limited to non-approval test files: an
    approval-named test is held to the full approval-sensitive-region scan, so
    it cannot import the approval store via a whole-file create."""
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree_before = _worktree_snapshot(Path(job.worktree_path))
    bundle_path = _create_file_bundle(
        tmp_path,
        job.job_id,
        "tests/test_approval_widget.py",
        _CALENDAR_FIXTURE_TEST,
        "approval-named-test.json",
    )

    failed = _refused_apply(root, job.job_id, bundle_path)

    assert "approval-sensitive region" in failed.edit_errors[0]
    assert _worktree_snapshot(Path(job.worktree_path)) == worktree_before


def test_create_file_on_approval_implementation_path_is_refused(tmp_path):
    """Approval-record implementation files stay blocked at the path layer even
    with innocuous content and even though tests may now seed fixtures."""
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree_before = _worktree_snapshot(Path(job.worktree_path))
    bundle_path = _create_file_bundle(
        tmp_path,
        job.job_id,
        "app/approvals.py",
        "VALUE = 1\n",
        "approval-impl-path.json",
    )

    failed = _refused_apply(root, job.job_id, bundle_path)

    assert "approval-record code" in failed.edit_errors[0]
    assert _worktree_snapshot(Path(job.worktree_path)) == worktree_before


def test_replace_text_introducing_approval_route_handler_is_refused(tmp_path):
    """The allowance never reaches approval route handlers: adding a new
    /approvals route via replace_text is blocked by the route path itself."""
    root = _init_repo(tmp_path)
    job, routes = _routes_job(root)
    before = routes.read_text(encoding="utf-8")
    new_text = (
        '        return _page("Chat", request, chat_page())\n'
        "\n"
        '    @app.post("/approvals/{approval_id}/reject")\n'
        "    def reject_item(request, approval_id):\n"
        "        return reject_approval(_approval_store(request), approval_id)"
    )
    bundle_path = _routes_replace_bundle(
        tmp_path, job.job_id, _SAFE_ROUTE_ANCHOR, new_text, "new-approval-route.json"
    )

    failed = _refused_apply(root, job.job_id, bundle_path)

    assert "approval route edit blocked" in failed.edit_errors[0]
    assert routes.read_text(encoding="utf-8") == before


def test_rejects_binary_and_oversized_content_and_files(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree = Path(job.worktree_path)

    nul_content = _write_bundle(
        tmp_path,
        _bundle_payload(job.job_id, [_create_op(content="\x00binary")]),
        name="nul-content.json",
    )
    failed = _refused_apply(root, job.job_id, nul_content)
    assert "binary" in failed.edit_errors[0]

    oversized_content = _write_bundle(
        tmp_path,
        _bundle_payload(
            job.job_id,
            [_create_op(content="x" * (edit_bundle.MAX_EDIT_FILE_BYTES + 1))],
        ),
        name="oversized-content.json",
    )
    failed = _refused_apply(root, job.job_id, oversized_content)
    assert "exceeds" in failed.edit_errors[0]

    binary_current = b"\x00binary-on-disk"
    (worktree / "blob.txt").write_bytes(binary_current)
    binary_target = _write_bundle(
        tmp_path,
        _bundle_payload(
            job.job_id,
            [
                {
                    "op": "replace_file",
                    "path": "blob.txt",
                    "content": "text\n",
                    "expected_sha256": _sha(binary_current),
                }
            ],
        ),
        name="binary-target.json",
    )
    failed = _refused_apply(root, job.job_id, binary_target)
    assert "binary" in failed.edit_errors[0]
    assert (worktree / "blob.txt").read_bytes() == binary_current

    oversized_current = b"y" * (edit_bundle.MAX_EDIT_FILE_BYTES + 1)
    (worktree / "big.txt").write_bytes(oversized_current)
    oversized_target = _write_bundle(
        tmp_path,
        _bundle_payload(
            job.job_id,
            [
                {
                    "op": "replace_file",
                    "path": "big.txt",
                    "content": "small\n",
                    "expected_sha256": _sha(oversized_current),
                }
            ],
        ),
        name="oversized-target.json",
    )
    failed = _refused_apply(root, job.job_id, oversized_target)
    assert "oversized" in failed.edit_errors[0]
    assert (worktree / "big.txt").read_bytes() == oversized_current


def test_rejects_stale_expected_hashes_and_stale_or_ambiguous_text(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree = Path(job.worktree_path)
    stale_sha = _sha(b"some other content\n")

    stale_replace = _write_bundle(
        tmp_path,
        _bundle_payload(
            job.job_id,
            [
                {
                    "op": "replace_file",
                    "path": "tracked.txt",
                    "content": "new\n",
                    "expected_sha256": stale_sha,
                }
            ],
        ),
        name="stale-replace.json",
    )
    failed = _refused_apply(root, job.job_id, stale_replace)
    assert "stale" in failed.edit_errors[0]
    assert TRACKED_SHA in failed.edit_errors[0]

    stale_delete = _write_bundle(
        tmp_path,
        _bundle_payload(
            job.job_id,
            [{"op": "delete_file", "path": "notes.txt", "expected_sha256": stale_sha}],
        ),
        name="stale-delete.json",
    )
    failed = _refused_apply(root, job.job_id, stale_delete)
    assert "stale" in failed.edit_errors[0]
    assert (worktree / "notes.txt").exists()

    missing_text = _write_bundle(
        tmp_path,
        _bundle_payload(
            job.job_id,
            [
                {
                    "op": "replace_text",
                    "path": "src/sample.py",
                    "old_text": "NOT_PRESENT = 9",
                    "new_text": "VALUE = 9",
                }
            ],
        ),
        name="missing-text.json",
    )
    failed = _refused_apply(root, job.job_id, missing_text)
    assert "no exact match" in failed.edit_errors[0]

    (worktree / "dup.txt").write_bytes(b"same\nsame\n")
    ambiguous_text = _write_bundle(
        tmp_path,
        _bundle_payload(
            job.job_id,
            [
                {
                    "op": "replace_text",
                    "path": "dup.txt",
                    "old_text": "same\n",
                    "new_text": "different\n",
                }
            ],
        ),
        name="ambiguous-text.json",
    )
    failed = _refused_apply(root, job.job_id, ambiguous_text)
    assert "ambiguous" in failed.edit_errors[0]
    assert (worktree / "dup.txt").read_text(encoding="utf-8") == "same\nsame\n"


def test_rejects_malformed_bundles_and_schema_violations(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree_before = _worktree_snapshot(Path(job.worktree_path))

    cases: list[tuple[str, dict | str]] = [
        (
            "wrong-version.json",
            _bundle_payload(job.job_id, [_create_op()], schema_version=2),
        ),
        (
            "count-mismatch.json",
            _bundle_payload(job.job_id, [_create_op()], operation_count=5),
        ),
        (
            "extra-field.json",
            _bundle_payload(job.job_id, [_create_op()], surprise=True),
        ),
        ("no-operations.json", _bundle_payload(job.job_id, [])),
        (
            "bad-timestamp.json",
            _bundle_payload(job.job_id, [_create_op()], created_at="not a date"),
        ),
        ("not-json.json", "this is not json"),
        (
            "extra-op-field.json",
            _bundle_payload(
                job.job_id,
                [{"op": "create_file", "path": "x.txt", "content": "x", "mode": "755"}],
            ),
        ),
        (
            "unknown-op.json",
            _bundle_payload(
                job.job_id, [{"op": "run_command", "path": "x.txt", "content": "x"}]
            ),
        ),
        (
            "bad-sha.json",
            _bundle_payload(
                job.job_id,
                [{"op": "delete_file", "path": "notes.txt", "expected_sha256": "xyz"}],
            ),
        ),
        (
            "duplicate-paths.json",
            _bundle_payload(
                job.job_id, [_create_op(path="a.txt"), _create_op(path="A.TXT")]
            ),
        ),
        (
            "dir-file-conflict.json",
            _bundle_payload(
                job.job_id, [_create_op(path="a.txt"), _create_op(path="a.txt/b.txt")]
            ),
        ),
    ]
    for name, payload in cases:
        path = tmp_path / name
        if isinstance(payload, str):
            path.write_text(payload, encoding="utf-8")
        else:
            _write_bundle(tmp_path, payload, name=name)
        failed = _refused_apply(root, job.job_id, path)
        assert failed.edit_apply_status == "error", name
        assert _worktree_snapshot(Path(job.worktree_path)) == worktree_before, name

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        apply_freestyle_edit_bundle(
            job.job_id, str(tmp_path / "does-not-exist.json"), project_root=root
        )
    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        wrong_suffix = tmp_path / "bundle.txt"
        wrong_suffix.write_text("{}", encoding="utf-8")
        apply_freestyle_edit_bundle(job.job_id, str(wrong_suffix), project_root=root)
    assert _worktree_snapshot(Path(job.worktree_path)) == worktree_before


def test_validates_entire_bundle_before_applying_any_operation(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree = Path(job.worktree_path)
    worktree_before = _worktree_snapshot(worktree)
    operations = [
        _create_op(path="ok.txt"),
        {
            "op": "replace_file",
            "path": "tracked.txt",
            "content": "new\n",
            "expected_sha256": _sha(b"stale expectation\n"),
        },
    ]
    bundle_path = _write_bundle(tmp_path, _bundle_payload(job.job_id, operations))

    failed = _refused_apply(root, job.job_id, bundle_path)

    assert failed.edit_apply_status == "error"
    assert not (worktree / "ok.txt").exists()
    assert _worktree_snapshot(worktree) == worktree_before


def test_rolls_back_to_an_unchanged_worktree_on_midway_failure(tmp_path, monkeypatch):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree = Path(job.worktree_path)
    worktree_before = _worktree_snapshot(worktree)
    checkout_before = _checkout_snapshot(root)
    real_overwrite = edit_bundle._overwrite_file
    real_write_new = edit_bundle._write_new_file

    def failing_overwrite(target, data):
        raise OSError("simulated disk failure")

    # Scenario A: the create succeeds, then the replace fails -> create rolled back.
    monkeypatch.setattr(edit_bundle, "_overwrite_file", failing_overwrite)
    replace_after_create = _write_bundle(
        tmp_path,
        _bundle_payload(
            job.job_id,
            [
                _create_op(path="nested/dir/alpha.txt"),
                {
                    "op": "replace_file",
                    "path": "tracked.txt",
                    "content": "replaced\n",
                    "expected_sha256": TRACKED_SHA,
                },
            ],
        ),
        name="scenario-a.json",
    )
    failed = _refused_apply(root, job.job_id, replace_after_create)
    assert failed.edit_apply_status == "error"
    assert "simulated disk failure" in failed.edit_errors[0]
    assert _worktree_snapshot(worktree) == worktree_before
    assert not (worktree / "nested").exists()

    # Scenario B: the delete succeeds, then the create fails -> delete restored.
    monkeypatch.setattr(edit_bundle, "_overwrite_file", real_overwrite)

    def failing_write_new(target, data):
        raise OSError("simulated create failure")

    monkeypatch.setattr(edit_bundle, "_write_new_file", failing_write_new)
    create_after_delete = _write_bundle(
        tmp_path,
        _bundle_payload(
            job.job_id,
            [
                {
                    "op": "delete_file",
                    "path": "notes.txt",
                    "expected_sha256": NOTES_SHA,
                },
                _create_op(path="beta.txt"),
            ],
        ),
        name="scenario-b.json",
    )
    failed = _refused_apply(root, job.job_id, create_after_delete)
    assert failed.edit_apply_status == "error"
    assert _worktree_snapshot(worktree) == worktree_before
    assert (worktree / "notes.txt").read_text(encoding="utf-8") == "note\n"

    # After rollback the job is retryable and the same bundle applies cleanly.
    monkeypatch.setattr(edit_bundle, "_write_new_file", real_write_new)
    updated = apply_freestyle_edit_bundle(
        job.job_id, str(create_after_delete), project_root=root
    )
    assert updated.edit_apply_status == "applied"
    assert not (worktree / "notes.txt").exists()
    assert (worktree / "beta.txt").read_text(encoding="utf-8") == "created\n"
    assert _checkout_snapshot(root) == checkout_before


def test_refuses_in_progress_or_inconsistent_edit_state(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    bundle_path = _write_bundle(tmp_path, _bundle_payload(job.job_id, [_create_op()]))
    metadata_path = root / freestyle.DEV_STATE_DIR_NAME / "jobs" / f"{job.job_id}.json"

    for blocked_status in ("applying", "inconsistent"):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["edit_apply_status"] = blocked_status
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(
            FreestyleJobRefusedError, match="in-progress or inconsistent"
        ):
            apply_freestyle_edit_bundle(job.job_id, str(bundle_path), project_root=root)
        refreshed = get_freestyle_job(job.job_id, project_root=root)
        assert refreshed.edit_apply_status == blocked_status


def test_apply_uses_only_fixed_readonly_git_and_no_shell(tmp_path, monkeypatch):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    bundle_path = _write_bundle(tmp_path, _bundle_payload(job.job_id, [_create_op()]))
    calls: list[tuple[list[str], dict]] = []
    real_run = subprocess.run

    def recording_run(args, **kwargs):
        calls.append((list(args), kwargs))
        return real_run(args, **kwargs)

    monkeypatch.setattr(freestyle.subprocess, "run", recording_run)
    updated = apply_freestyle_edit_bundle(
        job.job_id, str(bundle_path), project_root=root
    )

    assert updated.edit_apply_status == "applied"
    assert calls
    assert all(call[0][0] == "git" for call in calls)
    assert all(call[1]["shell"] is False for call in calls)
    assert all(
        call[0][:3]
        in (
            ["git", "rev-parse", "--show-toplevel"],
            ["git", "rev-parse", "HEAD"],
            ["git", "rev-parse", "--abbrev-ref"],
        )
        for call in calls
    )
    forbidden = {
        "apply",
        "checkout",
        "commit",
        "fetch",
        "merge",
        "pull",
        "push",
        "remote",
        "reset",
    }
    assert all(forbidden.isdisjoint(call[0]) for call in calls)


def test_multiple_non_overlapping_replace_text_apply_in_descending_offset_order(
    tmp_path,
):
    """Several replace_text ops on one file validate, fold into one journaled
    write, and apply correctly. The first edit grows the text, so a correct
    result is only possible if same-file splices use the original offsets in
    descending order."""
    root = _init_repo(tmp_path)
    job, target = _multi_anchor_job(root)
    operations = [
        _replace_text_op("src/multi.py", "ALPHA = 1", "ALPHA = 1111"),
        _replace_text_op("src/multi.py", "GAMMA = 3", "GAMMA = 33"),
    ]
    bundle_path = _write_bundle(tmp_path, _bundle_payload(job.job_id, operations))

    updated = apply_freestyle_edit_bundle(
        job.job_id, str(bundle_path), project_root=root
    )

    assert updated.edit_apply_status == "applied"
    assert updated.edit_applied_file_count == 1
    assert target.read_text(encoding="utf-8") == (
        "ALPHA = 1111\nBETA = 2\nGAMMA = 33\n"
    )

    artifact = json.loads(
        (root / freestyle.DEV_STATE_DIR_NAME / updated.edit_artifact_path).read_text(
            encoding="utf-8"
        )
    )
    assert artifact["operation_count"] == 2
    assert artifact["touched_file_count"] == 1
    entry = artifact["operations"][0]
    assert entry["op"] == "replace_text"
    assert entry["path"] == "src/multi.py"
    assert entry["replacement_count"] == 2
    assert entry["descending_offset_application"] is True
    alpha_start = _MULTI_ANCHOR_TEXT.index("ALPHA = 1")
    gamma_start = _MULTI_ANCHOR_TEXT.index("GAMMA = 3")
    assert entry["replacement_ranges"] == [
        [alpha_start, alpha_start + len("ALPHA = 1")],
        [gamma_start, gamma_start + len("GAMMA = 3")],
    ]
    assert entry["before_sha256"] == _sha(_MULTI_ANCHOR_TEXT.encode("utf-8"))
    assert entry["after_sha256"] == _sha(b"ALPHA = 1111\nBETA = 2\nGAMMA = 33\n")


def test_overlapping_same_file_replace_text_fails_closed(tmp_path):
    root = _init_repo(tmp_path)
    job, target = _multi_anchor_job(root)
    before = target.read_text(encoding="utf-8")
    operations = [
        _replace_text_op("src/multi.py", "ALPHA = 1\nBETA = 2", "X = 0"),
        _replace_text_op("src/multi.py", "BETA = 2\nGAMMA = 3", "Y = 0"),
    ]
    bundle_path = _write_bundle(tmp_path, _bundle_payload(job.job_id, operations))

    failed = _refused_apply(root, job.job_id, bundle_path)

    assert "overlap" in failed.edit_errors[0]
    assert target.read_text(encoding="utf-8") == before


def test_multi_replace_text_missing_or_ambiguous_match_fails_closed(tmp_path):
    root = _init_repo(tmp_path)
    job, target = _multi_anchor_job(root)
    before = target.read_text(encoding="utf-8")

    missing = _write_bundle(
        tmp_path,
        _bundle_payload(
            job.job_id,
            [
                _replace_text_op("src/multi.py", "ALPHA = 1", "ALPHA = 9"),
                _replace_text_op("src/multi.py", "NOPE = 0", "NOPE = 1"),
            ],
        ),
        name="missing.json",
    )
    failed = _refused_apply(root, job.job_id, missing)
    assert "no exact match" in failed.edit_errors[0]
    assert target.read_text(encoding="utf-8") == before

    target.write_text(
        "DUP = 1\nDUP = 1\nKEEP = 2\n",
        encoding="utf-8",
        newline="\n",
    )
    before_dup = target.read_text(encoding="utf-8")
    ambiguous = _write_bundle(
        tmp_path,
        _bundle_payload(
            job.job_id,
            [
                _replace_text_op("src/multi.py", "KEEP = 2", "KEEP = 3"),
                _replace_text_op("src/multi.py", "DUP = 1", "DUP = 2"),
            ],
        ),
        name="ambiguous.json",
    )
    failed = _refused_apply(root, job.job_id, ambiguous)
    assert "ambiguous" in failed.edit_errors[0]
    assert target.read_text(encoding="utf-8") == before_dup


def test_same_path_whole_file_op_with_any_other_op_fails_closed(tmp_path):
    """create_file, replace_file, and delete_file must each be the only
    operation for their path; mixing them with anything else fails closed."""
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree = Path(job.worktree_path)
    worktree_before = _worktree_snapshot(worktree)

    cases: list[tuple[str, list[dict]]] = [
        (
            "replace-file-plus-text.json",
            [
                {
                    "op": "replace_file",
                    "path": "src/sample.py",
                    "content": "VALUE = 5\n",
                    "expected_sha256": SAMPLE_SHA,
                },
                _replace_text_op("src/sample.py", "VALUE = 1", "VALUE = 2"),
            ],
        ),
        (
            "delete-file-plus-text.json",
            [
                {
                    "op": "delete_file",
                    "path": "src/sample.py",
                    "expected_sha256": SAMPLE_SHA,
                },
                _replace_text_op("src/sample.py", "VALUE = 1", "VALUE = 2"),
            ],
        ),
        (
            "create-file-twice.json",
            [
                _create_op(path="src/created.py", content="A = 1\n"),
                _create_op(path="src/created.py", content="A = 2\n"),
            ],
        ),
    ]
    for name, operations in cases:
        bundle_path = _write_bundle(
            tmp_path, _bundle_payload(job.job_id, operations), name=name
        )
        failed = _refused_apply(root, job.job_id, bundle_path)
        assert "more than one operation for the same path" in failed.edit_errors[0], (
            name
        )
        assert _worktree_snapshot(worktree) == worktree_before, name
    assert not (worktree / "src" / "created.py").exists()


def test_same_file_multi_replace_text_touching_approval_marker_fails_closed(tmp_path):
    root = _init_repo(tmp_path)
    job, routes = _routes_job(root)
    before = routes.read_text(encoding="utf-8")
    operations = [
        _replace_text_op("app/web_routes.py", _SAFE_ROUTE_ANCHOR, _WIRED_ROUTE_TEXT),
        _replace_text_op(
            "app/web_routes.py",
            "        return approve_approval(_approval_store(request), approval_id)",
            "        return None",
        ),
    ]
    bundle_path = _write_bundle(tmp_path, _bundle_payload(job.job_id, operations))

    failed = _refused_apply(root, job.job_id, bundle_path)

    assert "approval-sensitive region" in failed.edit_errors[0]
    assert routes.read_text(encoding="utf-8") == before


def test_multi_replace_text_file_rolls_back_when_a_later_op_fails(
    tmp_path, monkeypatch
):
    """The folded multi-replace_text write is journaled like any other, so a
    later failing operation rolls the whole worktree back unchanged."""
    root = _init_repo(tmp_path)
    job, target = _multi_anchor_job(root)
    worktree = Path(job.worktree_path)
    worktree_before = _worktree_snapshot(worktree)
    real_overwrite = edit_bundle._overwrite_file
    state = {"calls": 0}

    def flaky_overwrite(path, data):
        state["calls"] += 1
        if state["calls"] == 2:
            raise OSError("simulated disk failure")
        return real_overwrite(path, data)

    monkeypatch.setattr(edit_bundle, "_overwrite_file", flaky_overwrite)
    operations = [
        _replace_text_op("src/multi.py", "ALPHA = 1", "ALPHA = 9"),
        _replace_text_op("src/multi.py", "GAMMA = 3", "GAMMA = 9"),
        {
            "op": "replace_file",
            "path": "tracked.txt",
            "content": "new\n",
            "expected_sha256": TRACKED_SHA,
        },
    ]
    bundle_path = _write_bundle(tmp_path, _bundle_payload(job.job_id, operations))

    failed = _refused_apply(root, job.job_id, bundle_path)

    assert failed.edit_apply_status == "error"
    assert "simulated disk failure" in failed.edit_errors[0]
    assert _worktree_snapshot(worktree) == worktree_before


def test_module_has_no_model_shell_network_or_runtime_dependencies():
    source = Path(edit_bundle.__file__).read_text(encoding="utf-8")
    forbidden = [
        "deepseek",
        "DeepSeek",
        "openai",
        "OpenAI",
        "subprocess",
        "os.system",
        "shutil",
        "ApprovalStore",
        "sqlite",
        "load_dotenv",
        "get_settings",
        "DEEPSEEK_API_KEY",
        "from app.llm",
        "from app.tools",
        "from app.main",
        "from app.web",
        "from app.approvals",
        "requests",
        "urllib",
        "httpx",
    ]
    for text in forbidden:
        assert text not in source, text


def test_apply_edits_cli_dispatches_job_id_and_bundle_path(
    tmp_path, monkeypatch, capsys
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    calls: list[tuple[str, str]] = []

    def fake_apply(job_id, bundle_path):
        calls.append((job_id, bundle_path))
        return job

    monkeypatch.setattr(freestyle, "apply_freestyle_edit_bundle", fake_apply)
    exit_code = freestyle.main(["apply-edits", job.job_id, "bundle.json"])

    assert exit_code == 0
    assert calls == [(job.job_id, "bundle.json")]
    assert json.loads(capsys.readouterr().out)["job_id"] == job.job_id


def test_recoverable_operation_errors_carry_categories_and_paths(tmp_path):
    """Deterministic model-mistake refusals expose a stable category while
    keeping the exact refusal message; safety refusals never gain one."""
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree = Path(job.worktree_path)
    (worktree / "dup.py").write_bytes(b"X = 1\nX = 1\n")

    scenarios = [
        (
            {"op": "create_file", "path": "src/sample.py", "content": "Y = 1\n"},
            "create_existing_file",
            "src/sample.py",
        ),
        (
            {
                "op": "replace_file",
                "path": "missing.py",
                "content": "Y = 1\n",
                "expected_sha256": "0" * 64,
            },
            "missing_target_file",
            "missing.py",
        ),
        (
            {
                "op": "replace_file",
                "path": "src/sample.py",
                "content": "Y = 1\n",
                "expected_sha256": "0" * 64,
            },
            "stale_expected_sha256",
            "src/sample.py",
        ),
        (
            {
                "op": "replace_text",
                "path": "src/sample.py",
                "old_text": "VALUE = 1",
                "new_text": "VALUE = 1",
            },
            "identical_replacement",
            "src/sample.py",
        ),
        (
            {
                "op": "replace_text",
                "path": "dup.py",
                "old_text": "X = 1",
                "new_text": "X = 2",
            },
            "ambiguous_replacement",
            "dup.py",
        ),
    ]
    for operation, category, path in scenarios:
        bundle = json.dumps(
            {
                "schema_version": 1,
                "job_id": job.job_id,
                "created_at": "2026-06-10T12:00:00+00:00",
                "operation_count": 1,
                "operations": [operation],
            }
        ).encode("utf-8")
        with pytest.raises(edit_bundle.EditBundleRecoverableOperationError) as info:
            edit_bundle.validate_edit_bundle_bytes(job.job_id, bundle, worktree)
        assert info.value.category == category
        assert info.value.path == path

    unsafe = json.dumps(
        {
            "schema_version": 1,
            "job_id": job.job_id,
            "created_at": "2026-06-10T12:00:00+00:00",
            "operation_count": 1,
            "operations": [
                {"op": "create_file", "path": ".env", "content": "TOKEN=x\n"}
            ],
        }
    ).encode("utf-8")
    with pytest.raises(FreestyleJobRefusedError) as unsafe_info:
        edit_bundle.validate_edit_bundle_bytes(job.job_id, unsafe, worktree)
    assert not isinstance(
        unsafe_info.value, edit_bundle.EditBundleRecoverableOperationError
    )
