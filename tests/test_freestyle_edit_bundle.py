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

TRACKED_SHA = hashlib.sha256(b"committed\n").hexdigest()
NOTES_SHA = hashlib.sha256(b"note\n").hexdigest()


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


def _refused_apply(root: Path, job_id: str, bundle_path: Path) -> "freestyle.FreestyleJob":
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
        {"op": "create_file", "path": "src/new_module.py", "content": "CREATED = True\n"},
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
    assert "symlink" in failed.edit_errors[0]
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
    assert "symlink" in failed.edit_errors[0]
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
        ("wrong-version.json", _bundle_payload(job.job_id, [_create_op()], schema_version=2)),
        ("count-mismatch.json", _bundle_payload(job.job_id, [_create_op()], operation_count=5)),
        ("extra-field.json", _bundle_payload(job.job_id, [_create_op()], surprise=True)),
        ("no-operations.json", _bundle_payload(job.job_id, [])),
        ("bad-timestamp.json", _bundle_payload(job.job_id, [_create_op()], created_at="not a date")),
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
        apply_freestyle_edit_bundle(
            job.job_id, str(wrong_suffix), project_root=root
        )
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
                {"op": "delete_file", "path": "notes.txt", "expected_sha256": NOTES_SHA},
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
        with pytest.raises(FreestyleJobRefusedError, match="in-progress or inconsistent"):
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
