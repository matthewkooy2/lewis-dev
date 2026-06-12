"""Tests for the developer-only freestyle job scaffold."""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.dev.freestyle_coding as freestyle
from app.dev.freestyle_coding import (
    FreestyleJobRefusedError,
    create_freestyle_job,
    get_freestyle_job,
    list_freestyle_jobs,
    materialize_freestyle_job,
    propose_freestyle_job,
    validate_goal,
)
import app.dev.freestyle_proposal as proposal
from app.dev.freestyle_proposal import DeepSeekProposalConfig


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
    root = tmp_path / "project"
    root.mkdir()
    (root / "tracked.txt").write_text("committed\n", encoding="utf-8")
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-m", "init")
    return root


def _tar_bytes(*entries: tuple[str, bytes | None, str]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, content, kind in entries:
            member = tarfile.TarInfo(name)
            if kind == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = str(content or b"outside").removeprefix("b'")
                archive.addfile(member)
            else:
                payload = content or b""
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
    return output.getvalue()


def _deepseek_config(api_key: str = "test-deepseek-key") -> DeepSeekProposalConfig:
    return DeepSeekProposalConfig(
        api_key=api_key,
        model="deepseek-test-code",
        base_url=proposal.DEEPSEEK_API_BASE_URL,
    )


def _proposal_content() -> str:
    return json.dumps(
        {
            "proposed_implementation_steps": [
                "Update the isolated dev implementation.",
                "Add focused regression coverage.",
            ],
            "likely_files_to_edit": [
                "app/dev/freestyle_coding.py",
                "tests/test_freestyle_coding.py",
            ],
            "risks_unknowns": ["Existing metadata may require backward compatibility."],
            "suggested_allowlisted_checks": ["pytest", "ruff"],
        }
    )


class _FakeCompletions:
    def __init__(self, content: str | None = None, error: Exception | None = None):
        self.content = content or _proposal_content()
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            model="deepseek-test-code",
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=self.content))
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        )


class _FakeDeepSeekClient:
    def __init__(self, content: str | None = None, error: Exception | None = None):
        self.completions = _FakeCompletions(content, error)
        self.chat = SimpleNamespace(completions=self.completions)


class _SequenceCompletions:
    """Returns each content in turn, repeating the last one when exhausted."""

    def __init__(self, contents: list[str]):
        self.contents = list(contents)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.contents[min(len(self.calls) - 1, len(self.contents) - 1)]
        return SimpleNamespace(
            model="deepseek-test-code",
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=content))
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        )


class _SequenceDeepSeekClient:
    def __init__(self, contents: list[str]):
        self.completions = _SequenceCompletions(contents)
        self.chat = SimpleNamespace(completions=self.completions)


def _materialized_job(root: Path, goal: str = "Improve the dev proposal flow."):
    job = create_freestyle_job(
        goal,
        developer_approved=True,
        project_root=root,
    )
    return materialize_freestyle_job(job.job_id, project_root=root)


def _worktree_snapshot(worktree: Path) -> dict[str, bytes]:
    return {
        path.relative_to(worktree).as_posix(): path.read_bytes()
        for path in worktree.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def _repository_file_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def _job_metadata_path(root: Path, job_id: str) -> Path:
    return root / freestyle.DEV_STATE_DIR_NAME / "jobs" / f"{job_id}.json"


def _proposal_failure_artifact(root: Path, job) -> dict:
    assert job.proposal_failure_artifact_path
    path = root / freestyle.DEV_STATE_DIR_NAME / job.proposal_failure_artifact_path
    return json.loads(path.read_text(encoding="utf-8"))


def _set_approval_marker(root: Path, job_id: str, value=...):
    path = _job_metadata_path(root, job_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if value is ...:
        payload.pop("developer_approved", None)
    else:
        payload["developer_approved"] = value
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_goal_validation_rejects_empty_and_huge_goals():
    with pytest.raises(FreestyleJobRefusedError, match="empty"):
        validate_goal(" \n ")
    with pytest.raises(FreestyleJobRefusedError, match="maximum"):
        validate_goal("x" * (freestyle.MAX_GOAL_LENGTH + 1))


def test_creation_requires_explicit_developer_approval(tmp_path):
    root = _init_repo(tmp_path)

    with pytest.raises(FreestyleJobRefusedError, match="developer approval"):
        create_freestyle_job("Inspect the code.", project_root=root)

    assert not (root / freestyle.DEV_STATE_DIR_NAME).exists()
    assert _git(root, "branch", "--list", "lewis/freestyle-*") == ""


def test_creates_unique_branch_and_no_checkout_worktree_from_committed_head(tmp_path):
    root = _init_repo(tmp_path)
    head = _git(root, "rev-parse", "HEAD")
    (root / "tracked.txt").write_text("dirty main checkout\n", encoding="utf-8")
    (root / "untracked.txt").write_text("main only\n", encoding="utf-8")

    job = create_freestyle_job(
        "Inspect the committed project safely.",
        title="Safe inspection",
        developer_approved=True,
        project_root=root,
    )

    worktree = Path(job.worktree_path)
    assert job.status == "ready"
    assert job.base_commit == head
    assert job.branch.startswith(freestyle.BRANCH_PREFIX)
    assert worktree.is_dir()
    assert worktree.is_relative_to(root / freestyle.DEV_STATE_DIR_NAME / "worktrees")
    assert not (worktree / "tracked.txt").exists()
    assert not (worktree / "untracked.txt").exists()
    assert (root / "tracked.txt").read_text(encoding="utf-8") == "dirty main checkout\n"
    assert _git(root, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert _git(worktree, "rev-parse", "--abbrev-ref", "HEAD") == job.branch


def test_persists_minimal_metadata_and_supports_list_and_read(tmp_path):
    root = _init_repo(tmp_path)
    job = create_freestyle_job(
        "Prepare an isolated inspection.",
        developer_approved=True,
        project_root=root,
    )

    loaded = get_freestyle_job(job.job_id, project_root=root)
    listed = list_freestyle_jobs(project_root=root)

    assert loaded == job
    assert listed == [job]
    assert loaded.goal == "Prepare an isolated inspection."
    assert loaded.created_at
    assert loaded.updated_at
    assert loaded.errors == ()
    assert loaded.materialization_status == "not_materialized"
    assert loaded.materialized_at is None
    assert loaded.materialized_file_count == 0
    assert loaded.developer_approved is True
    assert loaded.approval_state == "approved"
    persisted = json.loads(
        _job_metadata_path(root, job.job_id).read_text(encoding="utf-8")
    )
    assert persisted["developer_approved"] is True


@pytest.mark.parametrize("approval_value", [..., False, 1, "true", {}, []])
def test_operations_fail_closed_without_explicit_persisted_approval(
    tmp_path, monkeypatch, approval_value
):
    root = _init_repo(tmp_path)
    job = create_freestyle_job(
        "Refuse execution without persisted approval.",
        developer_approved=True,
        project_root=root,
    )
    _set_approval_marker(root, job.job_id, approval_value)
    source_before = (root / "tracked.txt").read_bytes()

    def forbidden_subprocess(*args, **kwargs):
        raise AssertionError("approval refusal must happen before subprocess execution")

    monkeypatch.setattr(freestyle.subprocess, "run", forbidden_subprocess)
    operations = (
        lambda: materialize_freestyle_job(job.job_id, project_root=root),
        lambda: propose_freestyle_job(
            job.job_id, project_root=root, deepseek_client=object()
        ),
        lambda: freestyle.generate_freestyle_edit_bundle(
            job.job_id, project_root=root, deepseek_client=object()
        ),
        lambda: freestyle.apply_freestyle_edit_bundle(
            job.job_id, "missing-bundle.json", project_root=root
        ),
    )
    for operation in operations:
        with pytest.raises(
            FreestyleJobRefusedError, match="explicit persisted developer approval"
        ):
            operation()

    assert (root / "tracked.txt").read_bytes() == source_before
    loaded = get_freestyle_job(job.job_id, project_root=root)
    expected_state = (
        "missing"
        if approval_value is ...
        else "not_approved"
        if approval_value is False
        else "invalid"
    )
    assert loaded.approval_state == expected_state


def test_list_and_show_surface_missing_approval_without_approving_job(
    tmp_path, monkeypatch, capsys
):
    root = _init_repo(tmp_path)
    job = create_freestyle_job(
        "Display a legacy job safely.",
        developer_approved=True,
        project_root=root,
    )
    _set_approval_marker(root, job.job_id)
    monkeypatch.setattr(freestyle, "ROOT_DIR", root)

    listed = list_freestyle_jobs(project_root=root)
    shown = get_freestyle_job(job.job_id, project_root=root)
    assert listed[0].developer_approved is None
    assert listed[0].approval_state == "missing"
    assert shown.developer_approved is None
    assert shown.approval_state == "missing"

    assert freestyle.main(["list"]) == 0
    list_output = capsys.readouterr().out
    assert '"approval_state": "missing"' in list_output
    assert '"developer_approved": null' in list_output
    assert freestyle.main(["show", job.job_id]) == 0
    show_output = capsys.readouterr().out
    assert '"approval_state": "missing"' in show_output
    assert '"developer_approved": null' in show_output

    _set_approval_marker(root, job.job_id, "true")
    assert freestyle.main(["list"]) == 0
    invalid_list_output = capsys.readouterr().out
    assert '"approval_state": "invalid"' in invalid_list_output
    assert freestyle.main(["show", job.job_id]) == 0
    invalid_show_output = capsys.readouterr().out
    assert '"approval_state": "invalid"' in invalid_show_output


def test_reapprove_requires_explicit_flag_and_only_updates_job_metadata(
    tmp_path, monkeypatch, capsys
):
    root = _init_repo(tmp_path)
    job = create_freestyle_job(
        "Explicitly reapprove legacy metadata.",
        developer_approved=True,
        project_root=root,
    )
    metadata_path = _job_metadata_path(root, job.job_id)
    _set_approval_marker(root, job.job_id)
    source_before = (root / "tracked.txt").read_bytes()
    metadata_before = metadata_path.read_bytes()
    repository_before = _repository_file_snapshot(root)
    monkeypatch.setattr(freestyle, "ROOT_DIR", root)

    assert freestyle.main(["reapprove", job.job_id]) == 1
    error = json.loads(capsys.readouterr().err)
    assert error["error_type"] == "freestyle_job_refused"
    assert "Explicit developer approval" in error["message"]
    assert metadata_path.read_bytes() == metadata_before

    def forbidden_subprocess(*args, **kwargs):
        raise AssertionError("reapproval must not execute subprocesses")

    monkeypatch.setattr(freestyle.subprocess, "run", forbidden_subprocess)
    assert freestyle.main(["reapprove", job.job_id, "--approve"]) == 0
    assert '"approval_state": "approved"' in capsys.readouterr().out
    approved = get_freestyle_job(job.job_id, project_root=root)

    assert approved.developer_approved is True
    assert approved.approval_state == "approved"
    assert (root / "tracked.txt").read_bytes() == source_before
    assert metadata_path.read_bytes() != metadata_before
    assert json.loads(metadata_path.read_text(encoding="utf-8"))[
        "developer_approved"
    ] is True
    repository_after = _repository_file_snapshot(root)
    changed_paths = {
        path
        for path in repository_before.keys() | repository_after.keys()
        if repository_before.get(path) != repository_after.get(path)
    }
    assert changed_paths == {
        f"{freestyle.DEV_STATE_DIR_NAME}/jobs/{job.job_id}.json"
    }


def test_materializes_only_valid_existing_job_worktree(tmp_path, monkeypatch):
    root = _init_repo(tmp_path)
    (root / "safe.py").write_text("print('committed')\n", encoding="utf-8")
    (root / ".env").write_text("SECRET=do-not-materialize\n", encoding="utf-8")
    (root / "private.key").write_text("do-not-materialize\n", encoding="utf-8")
    _git(root, "add", "safe.py", ".env", "private.key")
    _git(root, "commit", "-m", "add snapshot files")
    (root / "safe.py").write_text("dirty main checkout\n", encoding="utf-8")
    blob_calls: list[tuple[str, ...]] = []
    real_run_blob_batch = freestyle._run_git_blob_batch

    def recording_blob_batch(blobs, run_root, *, check_only=False):
        blob_calls.append(tuple(blob.path for blob in blobs))
        return real_run_blob_batch(blobs, run_root, check_only=check_only)

    monkeypatch.setattr(freestyle, "_run_git_blob_batch", recording_blob_batch)

    job = create_freestyle_job(
        "Materialize the committed snapshot.",
        developer_approved=True,
        project_root=root,
    )
    main_status_before = _git(root, "status", "--short")
    materialized = materialize_freestyle_job(job.job_id, project_root=root)
    worktree = Path(materialized.worktree_path)

    assert materialized.status == "materialized"
    assert materialized.materialization_status == "materialized"
    assert materialized.materialization_started_at
    assert materialized.materialized_at
    assert materialized.materialized_file_count == 2
    assert (worktree / "tracked.txt").read_text(encoding="utf-8") == "committed\n"
    assert (worktree / "safe.py").read_text(encoding="utf-8") == "print('committed')\n"
    assert not (worktree / ".env").exists()
    assert not (worktree / "private.key").exists()
    assert (root / "safe.py").read_text(encoding="utf-8") == "dirty main checkout\n"
    assert _git(root, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert _git(root, "status", "--short") == main_status_before
    assert _git(worktree, "status", "--short") == ""
    assert _git(worktree, "rev-parse", "HEAD") == job.base_commit
    assert _git(worktree, "rev-parse", "--abbrev-ref", "HEAD") == job.branch
    index_entries = set(_git(worktree, "ls-files", "-v").splitlines())
    assert "H safe.py" in index_entries
    assert "H tracked.txt" in index_entries
    assert "S .env" in index_entries
    assert "S private.key" in index_entries
    assert blob_calls
    assert all(".env" not in path for call in blob_calls for path in call)
    assert all("private.key" not in path for call in blob_calls for path in call)

    with pytest.raises(FreestyleJobRefusedError, match="already materialized"):
        materialize_freestyle_job(job.job_id, project_root=root)


def test_materialization_rejects_tampered_or_dirty_job_worktree_paths(tmp_path):
    root = _init_repo(tmp_path)
    job = create_freestyle_job(
        "Materialize safely.",
        developer_approved=True,
        project_root=root,
    )
    metadata_path = (
        root / freestyle.DEV_STATE_DIR_NAME / "jobs" / f"{job.job_id}.json"
    )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["worktree_path"] = str(root)
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        materialize_freestyle_job(job.job_id, project_root=root)
    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.status == "materialization_error"
    assert failed.materialization_status == "error"
    assert "does not match local dev state" in failed.errors[0]

    second = create_freestyle_job(
        "Reject dirty worktree.",
        developer_approved=True,
        project_root=root,
    )
    (Path(second.worktree_path) / "unexpected.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        materialize_freestyle_job(second.job_id, project_root=root)
    dirty_failed = get_freestyle_job(second.job_id, project_root=root)
    assert "already dirty" in dirty_failed.errors[0]


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (("../escape.txt", b"escape", "file"), "path traversal"),
        ((".git/config", b"tamper", "file"), ".git tampering"),
        (("link", b"../../outside", "symlink"), "symlink"),
    ],
)
def test_materialization_blocks_malicious_archive_members(
    tmp_path, monkeypatch, entry, message
):
    root = _init_repo(tmp_path)
    job = create_freestyle_job(
        "Reject malicious archive.",
        developer_approved=True,
        project_root=root,
    )
    malicious = _tar_bytes(entry)
    monkeypatch.setattr(freestyle, "_archive_snapshot", lambda blobs, root: (malicious,))
    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        materialize_freestyle_job(job.job_id, project_root=root)

    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.status == "materialization_error"
    assert failed.materialization_status == "error"
    assert message in failed.errors[0]
    assert not (root.parent / "escape.txt").exists()
    assert not (Path(job.worktree_path) / "link").exists()


def test_materialization_failure_error_is_bounded(tmp_path, monkeypatch):
    root = _init_repo(tmp_path)
    job = create_freestyle_job(
        "Record bounded failure.",
        developer_approved=True,
        project_root=root,
    )
    monkeypatch.setattr(
        freestyle,
        "_run_git_blob_batch",
        lambda blobs, root, check_only=False: freestyle.GitBytesCommandResult(
            ("git", "cat-file", "--batch-check" if check_only else "--batch"),
            1,
            b"",
            "x" * 2_000,
        ),
    )
    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        materialize_freestyle_job(job.job_id, project_root=root)

    failed = get_freestyle_job(job.job_id, project_root=root)
    assert len(failed.errors[0]) == 500
    assert failed.materialization_started_at
    assert failed.materialized_at is None
    assert failed.materialized_file_count == 0


def test_materialization_records_index_consistency_verification_failure(
    tmp_path, monkeypatch
):
    root = _init_repo(tmp_path)
    job = create_freestyle_job(
        "Fail closed if the isolated index cannot be verified.",
        developer_approved=True,
        project_root=root,
    )
    real_run = freestyle._run_materialization_git_bytes

    def failing_verification(args, worktree, *, omitted_paths=()):
        if "diff-index" in args:
            return freestyle.GitBytesCommandResult(tuple(args), 1, b"", "")
        return real_run(args, worktree, omitted_paths=omitted_paths)

    monkeypatch.setattr(
        freestyle, "_run_materialization_git_bytes", failing_verification
    )
    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        materialize_freestyle_job(job.job_id, project_root=root)

    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.status == "materialization_error"
    assert failed.materialization_status == "error"
    assert failed.materialized_at is None
    assert failed.materialized_file_count == 0
    assert failed.errors == ("Git index consistency verification failed.",)


def test_extraction_blocks_preexisting_symlink_escape(tmp_path):
    worktree = tmp_path / "worktree"
    outside = tmp_path / "outside"
    worktree.mkdir()
    outside.mkdir()
    try:
        (worktree / "linked").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is not supported in this environment")
    archive = _tar_bytes(("linked/escape.txt", b"escape", "file"))
    members = freestyle._preflight_archive(archive)

    with pytest.raises(FreestyleJobRefusedError, match="symlink or file escape"):
        freestyle._extract_preflighted_archive(archive, members, worktree)

    assert not (outside / "escape.txt").exists()


def test_worktree_failure_is_persisted_as_error_metadata(tmp_path, monkeypatch):
    root = _init_repo(tmp_path)
    real_run_git = freestyle._run_git

    def failing_worktree(args, run_root):
        if args[:3] == ["git", "worktree", "add"]:
            return freestyle.GitCommandResult(
                args=tuple(args),
                returncode=1,
                stdout="",
                stderr="simulated local worktree failure",
            )
        return real_run_git(args, run_root)

    monkeypatch.setattr(freestyle, "_run_git", failing_worktree)
    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        create_freestyle_job(
            "Prepare an isolated inspection.",
            developer_approved=True,
            project_root=root,
        )

    jobs = list_freestyle_jobs(project_root=root)
    assert len(jobs) == 1
    assert jobs[0].status == "error"
    assert jobs[0].errors == ("simulated local worktree failure",)


def test_git_wrapper_accepts_only_fixed_local_setup_operations(tmp_path):
    root = _init_repo(tmp_path)

    forbidden = [
        ["git", "fetch"],
        ["git", "pull"],
        ["git", "push"],
        ["git", "merge", "main"],
        ["git", "checkout", "main"],
        ["git", "reset", "--hard"],
        ["git", "remote", "-v"],
        ["git", "worktree", "remove", "anything"],
        [
            "git",
            "archive",
            "--format=tar",
            _git(root, "rev-parse", "HEAD"),
            "--",
            "tracked.txt",
        ],
        ["git", "archive", "--format=zip", _git(root, "rev-parse", "HEAD")],
    ]
    for command in forbidden:
        with pytest.raises(FreestyleJobRefusedError, match="forbidden git"):
            freestyle._run_git(command, root)

    with pytest.raises(FreestyleJobRefusedError, match="forbidden materialization git"):
        freestyle._run_materialization_git_bytes(
            [
                *freestyle._MATERIALIZATION_GIT_PREFIX,
                "checkout",
                "--",
                "tracked.txt",
            ],
            root,
        )


def test_worktree_git_command_is_host_generated_and_shell_false(tmp_path, monkeypatch):
    root = _init_repo(tmp_path)
    calls: list[tuple[list[str], dict]] = []
    real_run = subprocess.run

    def recording_run(args, **kwargs):
        calls.append((list(args), kwargs))
        return real_run(args, **kwargs)

    monkeypatch.setattr(freestyle.subprocess, "run", recording_run)
    job = create_freestyle_job(
        "Goal text; git push origin main",
        title="Title with shell | syntax",
        developer_approved=True,
        project_root=root,
    )

    assert calls
    assert all(call[0][0] == "git" for call in calls)
    assert all(call[1]["shell"] is False for call in calls)
    worktree_call = calls[-1][0]
    assert worktree_call[:5] == ["git", "worktree", "add", "--no-checkout", "-b"]
    assert worktree_call[5] == job.branch
    assert worktree_call[7] == job.base_commit
    assert "push" not in worktree_call
    assert "shell" not in worktree_call


def test_no_runtime_module_imports_freestyle_scaffold():
    app_dir = Path(freestyle.__file__).resolve().parents[1]
    for module_path in app_dir.rglob("*.py"):
        relative = module_path.relative_to(app_dir)
        if relative.parts[0] == "dev":
            continue
        content = module_path.read_text(encoding="utf-8")
        assert "freestyle_coding" not in content, f"{relative} references freestyle"


def test_scaffold_has_no_approval_sqlite_env_or_runtime_execution_dependencies():
    source = Path(freestyle.__file__).read_text(encoding="utf-8")
    forbidden = [
        "ApprovalStore",
        "sqlite",
        "load_dotenv",
        "get_settings",
        "DEEPSEEK_API_KEY",
        "from app.llm",
        "from app.tools",
        "from app.main",
        "from app.web",
    ]
    for text in forbidden:
        assert text not in source


def test_materialization_uses_only_fixed_git_and_no_runtime_dependencies(
    tmp_path, monkeypatch
):
    root = _init_repo(tmp_path)
    calls: list[tuple[list[str], dict]] = []
    real_run = subprocess.run

    def recording_run(args, **kwargs):
        calls.append((list(args), kwargs))
        return real_run(args, **kwargs)

    monkeypatch.setattr(freestyle.subprocess, "run", recording_run)
    job = create_freestyle_job(
        "Materialize without arbitrary runtime.",
        developer_approved=True,
        project_root=root,
    )
    materialize_freestyle_job(job.job_id, project_root=root)

    assert calls
    assert all(call[0][0] == "git" for call in calls)
    assert all(call[1]["shell"] is False for call in calls)
    assert any(call[0] == ["git", "cat-file", "--batch-check"] for call in calls)
    assert any(call[0] == ["git", "cat-file", "--batch"] for call in calls)
    consistency_calls = [
        call[0]
        for call in calls
        if any(
            operation in call[0]
            for operation in ("read-tree", "update-index", "diff-index", "ls-files")
        )
    ]
    assert consistency_calls
    assert all(
        call[: len(freestyle._MATERIALIZATION_GIT_PREFIX)]
        == list(freestyle._MATERIALIZATION_GIT_PREFIX)
        for call in consistency_calls
    )
    assert any("read-tree" in call for call in consistency_calls)
    assert any("diff-index" in call for call in consistency_calls)
    assert any("ls-files" in call for call in consistency_calls)
    assert all("checkout" not in call for call in calls)
    assert all("reset" not in call for call in calls)
    assert all("archive" not in call for call in calls)
    assert all("fetch" not in call[0] for call in calls)
    assert all("push" not in call[0] for call in calls)


def test_materialization_index_synchronization_does_not_run_git_hooks(tmp_path):
    root = _init_repo(tmp_path)
    job = create_freestyle_job(
        "Materialize without running repository hooks.",
        developer_approved=True,
        project_root=root,
    )
    sentinel = tmp_path / "post-index-change-ran"
    hook = root / ".git" / "hooks" / "post-index-change"
    hook.write_text(
        f"#!/bin/sh\nprintf ran > '{sentinel.as_posix()}'\n",
        encoding="utf-8",
    )
    hook.chmod(0o755)

    materialized = materialize_freestyle_job(job.job_id, project_root=root)

    assert materialized.status == "materialized"
    assert not sentinel.exists()


def test_materialization_raw_blob_extraction_does_not_run_configured_filters(tmp_path):
    root = _init_repo(tmp_path)
    (root / ".gitattributes").write_text(
        "tracked.txt filter=must-not-run\n", encoding="utf-8"
    )
    _git(root, "add", ".gitattributes")
    _git(root, "commit", "-m", "add filter attributes")
    job = create_freestyle_job(
        "Materialize without running repository filters.",
        developer_approved=True,
        project_root=root,
    )
    _git(root, "config", "filter.must-not-run.clean", "false")
    _git(root, "config", "filter.must-not-run.smudge", "false")
    _git(root, "config", "filter.must-not-run.required", "true")

    materialized = materialize_freestyle_job(job.job_id, project_root=root)

    assert materialized.status == "materialized"
    assert (Path(job.worktree_path) / "tracked.txt").read_text(encoding="utf-8") == (
        "committed\n"
    )
    assert not (Path(job.worktree_path) / ".gitattributes").exists()


def test_proposal_requires_successfully_materialized_job(tmp_path):
    root = _init_repo(tmp_path)
    job = create_freestyle_job(
        "Propose only after materialization.",
        developer_approved=True,
        project_root=root,
    )
    client = _FakeDeepSeekClient()

    with pytest.raises(FreestyleJobRefusedError, match="materialized first"):
        propose_freestyle_job(
            job.job_id,
            project_root=root,
            deepseek_config=_deepseek_config(),
            deepseek_client=client,
        )

    assert client.completions.calls == []
    assert get_freestyle_job(job.job_id, project_root=root).proposal_status == (
        "not_proposed"
    )


def test_propose_cli_dispatches_only_the_requested_job_id(
    tmp_path, monkeypatch, capsys
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    calls: list[str] = []

    def fake_propose(job_id):
        calls.append(job_id)
        return job

    monkeypatch.setattr(freestyle, "propose_freestyle_job", fake_propose)
    exit_code = freestyle.main(["propose", job.job_id])

    assert exit_code == 0
    assert calls == [job.job_id]
    assert json.loads(capsys.readouterr().out)["job_id"] == job.job_id


def test_proposal_context_reads_only_the_isolated_worktree(tmp_path):
    root = _init_repo(tmp_path)
    (root / "safe.py").write_text("COMMITTED_WORKTREE_ONLY = True\n", encoding="utf-8")
    _git(root, "add", "safe.py")
    _git(root, "commit", "-m", "add safe source")
    job = _materialized_job(root, "Update safe.py.")
    (root / "safe.py").write_text("DIRTY_MAIN_CHECKOUT = True\n", encoding="utf-8")
    (root / "main-only.txt").write_text("MAIN_ONLY_CONTENT\n", encoding="utf-8")
    client = _FakeDeepSeekClient()

    propose_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=_deepseek_config(),
        deepseek_client=client,
    )

    prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "COMMITTED_WORKTREE_ONLY" in prompt
    assert "DIRTY_MAIN_CHECKOUT" not in prompt
    assert "MAIN_ONLY_CONTENT" not in prompt


def test_proposal_excludes_secret_git_dev_state_db_cache_and_approval_store_paths(
    tmp_path,
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root, "Inspect safe.py.")
    worktree = Path(job.worktree_path)
    (worktree / "safe.py").write_text("SAFE_CONTEXT_MARKER = True\n", encoding="utf-8")
    (worktree / ".env").write_text("ENV_SECRET_MARKER=bad\n", encoding="utf-8")
    (worktree / "secret-token.txt").write_text("TOKEN_MARKER\n", encoding="utf-8")
    (worktree / ".lewis-dev").mkdir()
    (worktree / ".lewis-dev" / "state.json").write_text(
        "DEV_STATE_MARKER\n", encoding="utf-8"
    )
    (worktree / "data").mkdir()
    (worktree / "data" / "runtime.sqlite3").write_text(
        "DB_MARKER\n", encoding="utf-8"
    )
    (worktree / "__pycache__").mkdir()
    (worktree / "__pycache__" / "cached.py").write_text(
        "CACHE_MARKER\n", encoding="utf-8"
    )
    (worktree / "approvalstore.txt").write_text(
        "APPROVAL_STORE_MARKER\n", encoding="utf-8"
    )
    (worktree / "notes.txt").write_text(
        ("safe prefix\n" * 2_000) + "-----BEGIN PRIVATE KEY-----\nLATE_SECRET\n",
        encoding="utf-8",
    )
    client = _FakeDeepSeekClient()

    completed = propose_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=_deepseek_config(),
        deepseek_client=client,
    )

    prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "SAFE_CONTEXT_MARKER" in prompt
    for forbidden in (
        "ENV_SECRET_MARKER",
        "TOKEN_MARKER",
        "DEV_STATE_MARKER",
        "DB_MARKER",
        "CACHE_MARKER",
        "APPROVAL_STORE_MARKER",
        "LATE_SECRET",
        ".git",
        ".lewis-dev",
    ):
        assert forbidden not in prompt
    artifact = json.loads(
        (
            root / freestyle.DEV_STATE_DIR_NAME / completed.proposal_artifact_path
        ).read_text(encoding="utf-8")
    )
    omission_counts = artifact["context_summary"]["omission_counts"]
    assert omission_counts["safety"] >= 1
    assert omission_counts["generated_or_cache"] >= 1
    assert omission_counts["approval_store"] >= 1
    assert omission_counts["sensitive_content"] >= 1


def test_proposal_skips_binary_and_oversized_files(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root, "Inspect safe.py.")
    worktree = Path(job.worktree_path)
    (worktree / "safe.py").write_text("SAFE_SMALL_FILE\n", encoding="utf-8")
    (worktree / "binary.bin").write_bytes(b"\x00binary marker")
    (worktree / "oversized.txt").write_text(
        "x" * (proposal.MAX_CONTEXT_FILE_BYTES + 1),
        encoding="utf-8",
    )
    client = _FakeDeepSeekClient()

    completed = propose_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=_deepseek_config(),
        deepseek_client=client,
    )

    prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "SAFE_SMALL_FILE" in prompt
    assert "binary marker" not in prompt
    artifact_path = (
        root / freestyle.DEV_STATE_DIR_NAME / completed.proposal_artifact_path
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["context_summary"]["omission_counts"]["binary"] == 1
    assert artifact["context_summary"]["omission_counts"]["oversized"] == 1


def test_proposal_uses_deepseek_only_path_and_persists_artifact_metadata(
    tmp_path, monkeypatch
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    client = _FakeDeepSeekClient()
    factory_calls: list[dict[str, str]] = []

    def fake_factory(**kwargs):
        factory_calls.append(kwargs)
        return client

    monkeypatch.setattr(proposal, "create_deepseek_client", fake_factory)
    completed = propose_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=_deepseek_config(),
    )

    assert factory_calls == [
        {
            "api_key": "test-deepseek-key",
            "base_url": proposal.DEEPSEEK_API_BASE_URL,
        }
    ]
    assert completed.proposal_status == "proposed"
    assert completed.proposal_started_at
    assert completed.proposal_completed_at
    assert completed.proposal_artifact_path
    assert completed.proposal_failure_artifact_path is None
    assert completed.proposal_inspected_file_count > 0
    assert completed.proposal_model == "deepseek-test-code"
    assert completed.proposal_errors == ()
    call = client.completions.calls[0]
    assert call["model"] == "deepseek-test-code"
    assert "tools" not in call
    artifact_path = (
        root / freestyle.DEV_STATE_DIR_NAME / completed.proposal_artifact_path
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["job_id"] == job.job_id
    assert artifact["goal"] == job.goal
    assert artifact["relevant_files_inspected"]
    assert artifact["proposed_implementation_steps"]
    assert artifact["likely_files_to_edit"]
    assert artifact["risks_unknowns"]
    assert artifact["suggested_allowlisted_checks"] == ["pytest", "ruff"]
    assert artifact["model_metadata"]["provider"] == "deepseek"
    assert artifact["model_metadata"]["usage"]["total_tokens"] == 150


def test_proposal_failure_persists_bounded_redacted_error_metadata(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    api_key = "test-deepseek-secret-key"
    client = _FakeDeepSeekClient(
        error=RuntimeError(f"api_key={api_key} " + ("x" * 2_000))
    )

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        propose_freestyle_job(
            job.job_id,
            project_root=root,
            deepseek_config=_deepseek_config(api_key),
            deepseek_client=client,
        )

    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.status == "materialized"
    assert failed.proposal_status == "error"
    assert failed.proposal_started_at
    assert failed.proposal_completed_at
    assert failed.proposal_artifact_path is None
    assert failed.proposal_failure_artifact_path
    assert failed.proposal_inspected_file_count > 0
    assert len(failed.proposal_errors[0]) <= 500
    assert api_key not in failed.proposal_errors[0]
    artifact = _proposal_failure_artifact(root, failed)
    assert artifact["source_mutation_performed"] is False
    assert artifact["proposal_attempts"][0]["status"] == "provider_error"
    assert api_key not in json.dumps(artifact)


@pytest.mark.parametrize(
    ("contents", "statuses"),
    [
        pytest.param(["", ""], ["empty_response", "empty_response"], id="empty"),
        pytest.param(
            ["not json", "still not json"],
            ["invalid_json", "invalid_json"],
            id="invalid-json",
        ),
    ],
)
def test_proposal_shape_failure_persists_attempt_diagnostics(
    tmp_path, contents, statuses
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        propose_freestyle_job(
            job.job_id,
            project_root=root,
            deepseek_config=_deepseek_config(),
            deepseek_client=_SequenceDeepSeekClient(contents),
        )

    failed = get_freestyle_job(job.job_id, project_root=root)
    artifact = _proposal_failure_artifact(root, failed)
    assert failed.proposal_artifact_path is None
    assert failed.proposal_failure_artifact_path
    assert [item["status"] for item in artifact["proposal_attempts"]] == statuses
    assert artifact["proposal_attempts"][1]["repair_retry"] is True
    assert artifact["context_plan"]
    assert artifact["context_summary"]["inspected_file_count"] > 0


def test_unsafe_likely_path_failure_records_redacted_reason_without_retry(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    unsafe = json.dumps(
        {
            "proposed_implementation_steps": ["Inspect the rejected path."],
            "likely_files_to_edit": ["/Users/private/secret-token.txt"],
            "risks_unknowns": [],
            "suggested_allowlisted_checks": ["pytest"],
        }
    )
    client = _FakeDeepSeekClient(content=unsafe)

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        propose_freestyle_job(
            job.job_id,
            project_root=root,
            deepseek_config=_deepseek_config(),
            deepseek_client=client,
        )

    failed = get_freestyle_job(job.job_id, project_root=root)
    attempt = _proposal_failure_artifact(root, failed)["proposal_attempts"][0]
    assert len(client.completions.calls) == 1
    assert attempt["status"] == "unsafe_likely_path"
    assert attempt["validator_category"] == "absolute"
    assert attempt["rejected_path_preview"] == "[REDACTED_ABSOLUTE]"
    assert attempt["retry_skipped_reason"] == "safety_related"
    assert "/Users/private" not in json.dumps(attempt)


@pytest.mark.parametrize(
    ("raw_path", "category", "expected_preview"),
    [
        ("/Users/private/file.py", "absolute", "[REDACTED_ABSOLUTE]"),
        ("secret-token.txt", "secrets", "[REDACTED_SECRETS]"),
        ("src/control\nname.py", "control_character", '"src/control\\nname.py"'),
    ],
)
def test_rejected_path_diagnostics_are_redacted_escaped_and_capped(
    raw_path, category, expected_preview
):
    detail = proposal._rejected_path_detail(raw_path, category)

    assert detail["validator_category"] == category
    assert detail["rejected_path_preview"] == expected_preview
    assert len(detail["rejected_path_preview"]) <= proposal.MAX_REJECTED_PATH_PREVIEW_CHARS


def test_model_output_preview_redacts_sensitive_relative_paths_and_controls():
    preview = proposal._redacted_preview(
        'likely_files_to_edit=["config/secret-token.txt"]\nnext', ""
    )

    assert "secret-token.txt" not in preview
    assert "[REDACTED_SENSITIVE_PATH]" in preview
    assert "\\n" in preview


def test_proposal_invalid_config_still_persists_bounded_failure_metadata(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        propose_freestyle_job(
            job.job_id,
            project_root=root,
            deepseek_config=object(),
            deepseek_client=_FakeDeepSeekClient(),
        )

    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.proposal_status == "error"
    assert failed.proposal_completed_at
    assert failed.proposal_errors == ("DeepSeek proposal configuration is invalid.",)


def test_proposal_rejects_shell_package_manager_and_non_allowlisted_check_text(
    tmp_path,
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    unsafe_content = json.dumps(
        {
            "proposed_implementation_steps": ["Run npm install before editing."],
            "likely_files_to_edit": ["tracked.txt"],
            "risks_unknowns": [],
            "suggested_allowlisted_checks": ["pytest", "curl"],
        }
    )

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        propose_freestyle_job(
            job.job_id,
            project_root=root,
            deepseek_config=_deepseek_config(),
            deepseek_client=_FakeDeepSeekClient(content=unsafe_content),
        )

    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.proposal_status == "error"
    assert failed.proposal_artifact_path is None


def test_proposal_repairs_malformed_json_with_exactly_one_retry(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    client = _SequenceDeepSeekClient(["this is not json", _proposal_content()])

    completed = propose_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=_deepseek_config(),
        deepseek_client=client,
    )

    assert completed.proposal_status == "proposed"
    assert len(client.completions.calls) == 2
    repair_messages = client.completions.calls[1]["messages"]
    assert [message["role"] for message in repair_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    repair_text = repair_messages[-1]["content"]
    assert "previous_response_error" in repair_text
    assert "required_response_schema" in repair_text
    assert "JSON object only" in repair_text
    artifact = json.loads(
        (
            root / freestyle.DEV_STATE_DIR_NAME / completed.proposal_artifact_path
        ).read_text(encoding="utf-8")
    )
    assert [record["status"] for record in artifact["attempts"]] == [
        "invalid_json",
        "ok",
    ]
    assert artifact["attempts"][1]["repair_retry"] is True
    preview = artifact["attempts"][0]["model_output_preview"]
    assert "this is not json" in preview
    assert len(preview) <= 2_000


def test_proposal_repair_retry_is_bounded_to_exactly_one_and_fails_closed(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    client = _FakeDeepSeekClient(content="never valid json")

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        propose_freestyle_job(
            job.job_id,
            project_root=root,
            deepseek_config=_deepseek_config(),
            deepseek_client=client,
        )

    assert len(client.completions.calls) == 2
    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.proposal_status == "error"
    assert failed.proposal_artifact_path is None
    assert any("not valid JSON" in error for error in failed.proposal_errors)
    assert all(len(error) <= 500 for error in failed.proposal_errors)


@pytest.mark.parametrize(
    "unsafe_payload",
    [
        pytest.param(
            {
                "proposed_implementation_steps": ["Run npm install before editing."],
                "likely_files_to_edit": ["tracked.txt"],
                "risks_unknowns": [],
                "suggested_allowlisted_checks": ["pytest"],
            },
            id="forbidden_command",
        ),
        pytest.param(
            {
                "proposed_implementation_steps": ["Edit the escape file."],
                "likely_files_to_edit": ["../escape.txt"],
                "risks_unknowns": [],
                "suggested_allowlisted_checks": ["pytest"],
            },
            id="unsafe_path",
        ),
        pytest.param(
            {
                "proposed_implementation_steps": ["Update the tracked file."],
                "likely_files_to_edit": ["tracked.txt"],
                "risks_unknowns": [],
                "suggested_allowlisted_checks": ["curl"],
            },
            id="non_allowlisted_check",
        ),
    ],
)
def test_proposal_safety_failures_never_trigger_a_repair_retry(
    tmp_path, unsafe_payload
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    client = _FakeDeepSeekClient(content=json.dumps(unsafe_payload))

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        propose_freestyle_job(
            job.job_id,
            project_root=root,
            deepseek_config=_deepseek_config(),
            deepseek_client=client,
        )

    assert len(client.completions.calls) == 1
    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.proposal_status == "error"
    assert failed.proposal_artifact_path is None


def test_proposal_does_not_modify_worktree_source_files(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree = Path(job.worktree_path)
    before = _worktree_snapshot(worktree)

    propose_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=_deepseek_config(),
        deepseek_client=_FakeDeepSeekClient(),
    )

    assert _worktree_snapshot(worktree) == before


def test_proposal_keeps_forbidden_shell_git_and_runtime_operations_unavailable(
    tmp_path, monkeypatch
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    calls: list[tuple[list[str], dict]] = []
    real_run = subprocess.run

    def recording_run(args, **kwargs):
        calls.append((list(args), kwargs))
        return real_run(args, **kwargs)

    monkeypatch.setattr(freestyle.subprocess, "run", recording_run)
    propose_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=_deepseek_config(),
        deepseek_client=_FakeDeepSeekClient(),
    )

    assert calls
    assert all(call[0][0] == "git" for call in calls)
    assert all(call[1]["shell"] is False for call in calls)
    assert all(
        call[0][:3] in (
            ["git", "rev-parse", "--show-toplevel"],
            ["git", "rev-parse", "HEAD"],
            ["git", "rev-parse", "--abbrev-ref"],
        )
        for call in calls
    )
    forbidden = {"fetch", "pull", "push", "merge", "checkout", "remote", "apply"}
    assert all(forbidden.isdisjoint(call[0]) for call in calls)


def test_proposal_has_no_fallback_provider_or_runtime_registration():
    proposal_source = Path(proposal.__file__).read_text(encoding="utf-8")
    forbidden = [
        "Anthropic",
        "ollama",
        "from app.approvals",
        "from app.llm",
        "from app.tools",
        "from app.main",
        "from app.web",
    ]
    for text in forbidden:
        assert text not in proposal_source
