"""Tests for allowlisted, isolated freestyle check execution."""

from __future__ import annotations

import importlib.util
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import app.dev.freestyle_checks as checks
import app.dev.freestyle_coding as freestyle
from app.dev.freestyle_coding import (
    FreestyleJobRefusedError,
    create_freestyle_job,
    get_freestyle_job,
    materialize_freestyle_job,
    run_freestyle_checks,
)

_RUFF_AVAILABLE = importlib.util.find_spec("ruff") is not None

_PASSING_TEST = "def test_freestyle_sample_passes():\n    assert True\n"
_FAILING_TEST = "def test_freestyle_sample_fails():\n    assert False\n"
_MUTATING_TEST = """\
from pathlib import Path


def test_freestyle_sample_mutates_worktree():
    Path("created_by_check.txt").write_text("made by check\\n", encoding="utf-8")
    Path("untracked_seed.txt").write_text("changed by check\\n", encoding="utf-8")
    Path(".env").write_text("SECRET_ENV_MARKER=rewritten-longer\\n", encoding="utf-8")
"""


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
    (root / "tests").mkdir(parents=True)
    (root / ".gitignore").write_text(
        ".env\n.lewis-dev/\n__pycache__/\n*.db\n", encoding="utf-8"
    )
    (root / "tracked.txt").write_text("committed\n", encoding="utf-8")
    (root / "tests" / "test_freestyle_sample.py").write_text(
        _PASSING_TEST, encoding="utf-8"
    )
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "init")
    return root


def _materialized_job(root: Path):
    created = create_freestyle_job(
        "Run allowlisted checks for isolated freestyle edits.",
        developer_approved=True,
        project_root=root,
    )
    return materialize_freestyle_job(created.job_id, project_root=root)


def _metadata_path(root: Path, job_id: str) -> Path:
    return root / freestyle.DEV_STATE_DIR_NAME / "jobs" / f"{job_id}.json"


def _set_metadata(root: Path, job_id: str, **values) -> None:
    path = _metadata_path(root, job_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(values)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _artifact(root: Path, job) -> dict:
    assert job.checks_artifact_path
    path = root / freestyle.DEV_STATE_DIR_NAME / job.checks_artifact_path
    return json.loads(path.read_text(encoding="utf-8"))


def _source_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and ".git" not in path.relative_to(root).parts
        and freestyle.DEV_STATE_DIR_NAME not in path.relative_to(root).parts
    }


def _forbid_subprocess(monkeypatch) -> None:
    def forbidden_run(*args, **kwargs):
        raise AssertionError("subprocess.run must not execute for this case")

    monkeypatch.setattr(checks.subprocess, "run", forbidden_run)


def _record_subprocess(monkeypatch) -> list[tuple[list[str], dict]]:
    calls: list[tuple[list[str], dict]] = []
    real_run = subprocess.run

    def recording_run(args, **kwargs):
        calls.append((list(args), kwargs))
        return real_run(args, **kwargs)

    monkeypatch.setattr(checks.subprocess, "run", recording_run)
    return calls


@pytest.mark.parametrize("approval", [None, False, "yes"])
def test_run_checks_requires_explicit_approval_and_persists_bounded_error(
    tmp_path, monkeypatch, approval
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    _set_metadata(root, job.job_id, developer_approved=approval)
    _forbid_subprocess(monkeypatch)

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        run_freestyle_checks(job.job_id, project_root=root)

    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.checks_status == "error"
    assert failed.checks_started_at
    assert failed.checks_completed_at
    assert failed.checks_artifact_path is None
    assert len(failed.checks_errors[0]) <= 500
    assert "approval" in failed.checks_errors[0]


def test_run_checks_requires_successfully_materialized_job(tmp_path, monkeypatch):
    root = _init_repo(tmp_path)
    job = create_freestyle_job(
        "Run checks only after materialization.",
        developer_approved=True,
        project_root=root,
    )
    _forbid_subprocess(monkeypatch)

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        run_freestyle_checks(job.job_id, project_root=root)

    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.checks_status == "error"
    assert "successfully materialized" in failed.checks_errors[0]


@pytest.mark.parametrize("edit_status", ["applying", "inconsistent"])
def test_in_progress_or_inconsistent_edit_state_fails_closed(
    tmp_path, monkeypatch, edit_status
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    _set_metadata(root, job.job_id, edit_apply_status=edit_status)
    _forbid_subprocess(monkeypatch)

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        run_freestyle_checks(job.job_id, project_root=root)

    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.checks_status == "error"
    assert "Recover or verify" in failed.checks_errors[0]


def test_in_progress_check_run_fails_closed_without_clearing_the_lock(
    tmp_path, monkeypatch
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    _set_metadata(root, job.job_id, checks_status="running")
    _forbid_subprocess(monkeypatch)

    with pytest.raises(FreestyleJobRefusedError, match="in progress"):
        run_freestyle_checks(job.job_id, project_root=root)

    persisted = get_freestyle_job(job.job_id, project_root=root)
    assert persisted.checks_status == "running"
    assert persisted.checks_errors == ()


@pytest.mark.parametrize(
    "check_ids, match",
    [
        (["pytest-freestyle; rm -rf /"], "Unknown check id"),
        (["git push --force"], "Unknown check id"),
        ([""], "Unknown check id"),
        (["pytest"], "Unknown check id"),
        (["pytest-freestyle", "pytest-freestyle"], "more than once"),
        ([], "at least one"),
        ("pytest-freestyle", "list of allowlisted"),
        ([{"argv": ["evil"]}], "check ids only"),
        ([b"pytest-freestyle"], "check ids only"),
    ],
)
def test_unknown_or_malformed_check_ids_are_rejected_before_execution(
    tmp_path, monkeypatch, check_ids, match
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    _forbid_subprocess(monkeypatch)

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        run_freestyle_checks(job.job_id, check_ids=check_ids, project_root=root)

    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.checks_status == "error"
    assert failed.checks_artifact_path is None
    assert re.search(match, failed.checks_errors[0])


def test_refusal_when_no_allowlisted_check_is_available(tmp_path, monkeypatch):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    monkeypatch.setattr(checks, "_module_available", lambda name: False)
    _forbid_subprocess(monkeypatch)

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        run_freestyle_checks(job.job_id, project_root=root)

    failed = get_freestyle_job(job.job_id, project_root=root)
    assert "No allowlisted checks are available" in failed.checks_errors[0]


def test_user_cannot_supply_command_args_env_or_cwd():
    for function in (freestyle.run_freestyle_checks, checks.run_freestyle_checks):
        parameters = set(inspect.signature(function).parameters)
        assert parameters == {"job_id", "check_ids", "project_root", "state_root"}


def test_only_identical_registry_definitions_can_execute(tmp_path):
    forged = checks.CheckDefinition(
        check_id="pytest-freestyle",
        description="forged duplicate of an allowlisted id",
        module="pytest",
        argv_suffix=("-q",),
        timeout_seconds=5,
        max_output_bytes=1_000,
        expected_failure="",
        requires_module="pytest",
    )
    with pytest.raises(FreestyleJobRefusedError, match="forbidden check execution"):
        checks._execute_check(forged, tmp_path)

    rogue = checks.CheckDefinition(
        check_id="rogue-check",
        description="not registered at all",
        module="pytest",
        argv_suffix=("-q",),
        timeout_seconds=5,
        max_output_bytes=1_000,
        expected_failure="",
        requires_module="pytest",
    )
    with pytest.raises(FreestyleJobRefusedError, match="forbidden check execution"):
        checks._execute_check(rogue, tmp_path)


def test_static_allowlist_is_conservative():
    assert {definition.check_id for definition in checks.CHECK_REGISTRY} == {
        "pytest-freestyle",
        "ruff-check",
    }
    for definition in checks.CHECK_REGISTRY:
        assert definition.module in {"pytest", "ruff"}
        assert definition.requires_module in {"pytest", "ruff"}
        tokens = {definition.module, *definition.argv_suffix}
        assert tokens.isdisjoint(checks._FORBIDDEN_CHECK_TOKENS)
        for token in definition.argv_suffix:
            assert token.isprintable()
            assert not any(
                character in checks._SHELL_METACHARACTERS for character in token
            )
        assert "install" not in " ".join(definition.argv_suffix)
        assert 0 < definition.timeout_seconds <= checks.MAX_CHECK_TIMEOUT_SECONDS
        assert 0 < definition.max_output_bytes <= checks.MAX_CHECK_OUTPUT_BYTES_LIMIT
        assert definition.description
        assert definition.expected_failure
    checks._require_safe_registry()


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"module": "bash"}, "forbidden module"),
        ({"requires_module": "requests"}, "forbidden requirement"),
        ({"argv_suffix": ("install",)}, "forbidden argv token"),
        ({"argv_suffix": ("a;b",)}, "forbidden argv token"),
        ({"argv_suffix": ("rm -rf /",)}, "forbidden argv token"),
        ({"timeout_seconds": 0}, "timeout is out of bounds"),
        ({"max_output_bytes": 10_000_000}, "output cap is out of bounds"),
        ({"check_id": "Evil Check"}, "unsafe id"),
    ],
)
def test_registry_validation_blocks_unsafe_drift(monkeypatch, overrides, match):
    fields = {
        "check_id": "evil-check",
        "description": "tampered",
        "module": "pytest",
        "argv_suffix": ("-q",),
        "timeout_seconds": 5,
        "max_output_bytes": 1_000,
        "expected_failure": "",
        "requires_module": "pytest",
        **overrides,
    }
    evil = checks.CheckDefinition(**fields)
    monkeypatch.setattr(checks, "CHECK_REGISTRY", (evil,))
    monkeypatch.setattr(checks, "CHECKS_BY_ID", {evil.check_id: evil})

    with pytest.raises(RuntimeError, match=match):
        checks._require_safe_registry()


def test_module_availability_probe_accepts_allowlisted_modules_only():
    with pytest.raises(FreestyleJobRefusedError, match="forbidden check requirement"):
        checks._module_available("bash")
    assert checks._module_available("pytest") is True


def test_check_subprocess_discipline_env_sanitization_and_artifact(
    tmp_path, monkeypatch
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek-value")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-aws-value")
    monkeypatch.setenv("GITHUB_TOKEN", "secret-github-value")
    (Path(job.worktree_path) / ".env").write_text(
        "SECRET_ENV_MARKER=topsecret\n", encoding="utf-8"
    )
    calls = _record_subprocess(monkeypatch)

    completed = run_freestyle_checks(
        job.job_id, check_ids=["pytest-freestyle"], project_root=root
    )

    assert completed.checks_status == "passed"
    check_calls = [call for call in calls if call[0][0] == sys.executable]
    assert len(check_calls) == 1
    argv, kwargs = check_calls[0]
    definition = checks.CHECKS_BY_ID["pytest-freestyle"]
    assert argv == [sys.executable, "-m", "pytest", *definition.argv_suffix]
    assert kwargs["shell"] is False
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert "input" not in kwargs
    assert Path(kwargs["cwd"]) == Path(job.worktree_path)
    assert kwargs["timeout"] == definition.timeout_seconds
    assert kwargs["env"] == checks._CHECK_ENVIRONMENT
    for secret in ("DEEPSEEK_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN"):
        assert secret not in kwargs["env"]
    assert all(call[0][0] in {"git", sys.executable} for call in calls)
    forbidden_git = {
        "push",
        "pull",
        "fetch",
        "merge",
        "commit",
        "rebase",
        "remote",
        "apply",
        "checkout",
        "reset",
        "worktree",
        "clone",
        "submodule",
    }
    git_calls = [call for call in calls if call[0][0] == "git"]
    assert git_calls
    assert all(forbidden_git.isdisjoint(call[0]) for call in git_calls)

    artifact = _artifact(root, completed)
    result = artifact["checks"][0]
    assert result["status"] == "passed"
    assert result["exit_code"] == 0
    assert result["timed_out"] is False
    assert result["duration_seconds"] >= 0
    assert "passed" in result["stdout"]
    assert artifact["environment"] == {
        "policy": "constructed_allowlist_only",
        "variables": sorted(checks._CHECK_ENVIRONMENT),
        "parent_environment_forwarded": False,
    }
    env_entries = [
        entry
        for entry in artifact["worktree_status"]["before"]
        if entry["path"] == ".env"
    ]
    assert len(env_entries) == 1
    assert env_entries[0]["content_digest"].startswith("stat:")
    encoded = json.dumps(artifact) + _metadata_path(root, job.job_id).read_text(
        encoding="utf-8"
    )
    for secret_value in (
        "secret-deepseek-value",
        "secret-aws-value",
        "secret-github-value",
        "SECRET_ENV_MARKER",
        "topsecret",
    ):
        assert secret_value not in encoded


def test_passing_checks_persist_artifact_and_metadata(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)

    completed = run_freestyle_checks(
        job.job_id, check_ids=["pytest-freestyle"], project_root=root
    )

    assert completed.checks_status == "passed"
    assert completed.checks_selected_ids == ("pytest-freestyle",)
    assert completed.checks_passed_count == 1
    assert completed.checks_failed_count == 0
    assert completed.checks_timeout_count == 0
    assert completed.checks_infra_error_count == 0
    assert completed.checks_modified_worktree is False
    assert completed.checks_started_at
    assert completed.checks_completed_at
    assert completed.checks_errors == ()
    assert completed.checks_artifact_path.startswith(f"checks/{job.job_id}/checks-")
    artifact_file = (
        root / freestyle.DEV_STATE_DIR_NAME / completed.checks_artifact_path
    )
    assert artifact_file.is_file()
    assert artifact_file.stat().st_size <= checks.MAX_CHECKS_ARTIFACT_BYTES

    artifact = _artifact(root, completed)
    assert artifact["schema_version"] == checks.CHECKS_SCHEMA_VERSION
    assert artifact["job"]["job_id"] == job.job_id
    assert artifact["job"]["goal"] == job.goal
    assert artifact["job"]["branch"] == job.branch
    assert artifact["job"]["worktree_path"] == job.worktree_path
    assert artifact["base_commit"] == job.base_commit
    assert artifact["current_head"] == {
        "branch": job.branch,
        "commit": job.base_commit,
    }
    assert artifact["selection"] == {
        "requested_check_ids": ["pytest-freestyle"],
        "selected_check_ids": ["pytest-freestyle"],
        "selection_mode": "explicit",
    }
    assert artifact["summary"] == {
        "total": 1,
        "passed": 1,
        "failed": 0,
        "timeout": 0,
        "infra_error": 0,
        "overall_status": "passed",
    }
    assert artifact["worktree_status"]["modified_by_checks"] is False
    assert artifact["worktree_status"]["modified_paths"] == []
    statement = artifact["execution_statement"]
    for forbidden_action in ("commit", "merge", "push", "PR creation", "auto-merge"):
        assert forbidden_action in statement

    raw = json.loads(_metadata_path(root, job.job_id).read_text(encoding="utf-8"))
    assert raw["checks_status"] == "passed"
    assert raw["checks_selected_ids"] == ["pytest-freestyle"]
    assert raw["checks_modified_worktree"] is False


def test_failing_check_persists_failed_artifact_and_metadata(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    (Path(job.worktree_path) / "tests" / "test_freestyle_sample.py").write_text(
        _FAILING_TEST, encoding="utf-8"
    )

    completed = run_freestyle_checks(
        job.job_id, check_ids=["pytest-freestyle"], project_root=root
    )

    assert completed.checks_status == "failed"
    assert completed.checks_failed_count == 1
    assert completed.checks_passed_count == 0
    artifact = _artifact(root, completed)
    result = artifact["checks"][0]
    assert result["status"] == "failed"
    assert result["exit_code"] == 1
    assert "1 failed" in result["stdout"]
    assert artifact["summary"]["overall_status"] == "failed"


@pytest.mark.skipif(not _RUFF_AVAILABLE, reason="ruff is not installed")
def test_default_selection_runs_all_available_checks_with_mixed_results(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    (Path(job.worktree_path) / "lint_error.py").write_text(
        "import os\n", encoding="utf-8"
    )

    completed = run_freestyle_checks(job.job_id, project_root=root)

    assert completed.checks_status == "failed"
    assert completed.checks_selected_ids == ("pytest-freestyle", "ruff-check")
    assert completed.checks_passed_count == 1
    assert completed.checks_failed_count == 1
    artifact = _artifact(root, completed)
    assert artifact["selection"]["selection_mode"] == "all_available"
    assert artifact["selection"]["requested_check_ids"] is None
    by_id = {result["check_id"]: result for result in artifact["checks"]}
    assert by_id["pytest-freestyle"]["status"] == "passed"
    assert by_id["ruff-check"]["status"] == "failed"
    assert "F401" in by_id["ruff-check"]["stdout"]


def test_timeout_produces_timeout_status_and_counts(tmp_path, monkeypatch):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    real_run = subprocess.run

    def fake_run(args, **kwargs):
        if args[0] == sys.executable:
            raise subprocess.TimeoutExpired(
                cmd=args,
                timeout=kwargs["timeout"],
                output=b"partial-stdout",
                stderr=b"partial-stderr",
            )
        return real_run(args, **kwargs)

    monkeypatch.setattr(checks.subprocess, "run", fake_run)

    completed = run_freestyle_checks(
        job.job_id, check_ids=["pytest-freestyle"], project_root=root
    )

    assert completed.checks_status == "failed"
    assert completed.checks_timeout_count == 1
    assert completed.checks_failed_count == 0
    result = _artifact(root, completed)["checks"][0]
    assert result["status"] == "timeout"
    assert result["timed_out"] is True
    assert result["exit_code"] is None
    assert result["stdout"] == "partial-stdout"
    assert "timeout" in result["note"]


def test_output_caps_set_truncation_flags_and_bound_the_artifact(
    tmp_path, monkeypatch
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    cap = checks.CHECKS_BY_ID["pytest-freestyle"].max_output_bytes
    real_run = subprocess.run

    def fake_run(args, **kwargs):
        if args[0] == sys.executable:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout=b"x" * (cap * 3),
                stderr=b"y" * (cap * 3),
            )
        return real_run(args, **kwargs)

    monkeypatch.setattr(checks.subprocess, "run", fake_run)

    completed = run_freestyle_checks(
        job.job_id, check_ids=["pytest-freestyle"], project_root=root
    )

    result = _artifact(root, completed)["checks"][0]
    marker_bytes = len(checks._OUTPUT_TRUNCATION_MARKER.encode("utf-8"))
    assert result["status"] == "passed"
    assert result["stdout_truncated"] is True
    assert result["stderr_truncated"] is True
    assert len(result["stdout"].encode("utf-8")) <= cap + marker_bytes
    assert len(result["stderr"].encode("utf-8")) <= cap + marker_bytes


def test_unstartable_check_records_infra_error(tmp_path, monkeypatch):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    real_run = subprocess.run

    def fake_run(args, **kwargs):
        if args[0] == sys.executable:
            raise OSError("spawn failed")
        return real_run(args, **kwargs)

    monkeypatch.setattr(checks.subprocess, "run", fake_run)

    completed = run_freestyle_checks(
        job.job_id, check_ids=["pytest-freestyle"], project_root=root
    )

    assert completed.checks_status == "failed"
    assert completed.checks_infra_error_count == 1
    result = _artifact(root, completed)["checks"][0]
    assert result["status"] == "infra_error"
    assert result["exit_code"] is None
    assert result["note"] == "Check process could not start."


def test_unavailable_selected_check_records_infra_error_without_executing(
    tmp_path, monkeypatch
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    monkeypatch.setattr(checks, "_module_available", lambda name: name == "pytest")
    real_run = subprocess.run

    def fake_run(args, **kwargs):
        assert args[0] != sys.executable, "unavailable check must not execute"
        return real_run(args, **kwargs)

    monkeypatch.setattr(checks.subprocess, "run", fake_run)

    completed = run_freestyle_checks(
        job.job_id, check_ids=["ruff-check"], project_root=root
    )

    assert completed.checks_status == "failed"
    assert completed.checks_infra_error_count == 1
    result = _artifact(root, completed)["checks"][0]
    assert result["status"] == "infra_error"
    assert "not importable" in result["note"]
    assert "nothing was installed" in result["note"]


def test_detects_worktree_modifications_caused_by_checks(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree = Path(job.worktree_path)
    (worktree / "tests" / "test_freestyle_sample.py").write_text(
        _MUTATING_TEST, encoding="utf-8"
    )
    (worktree / "untracked_seed.txt").write_text("seed\n", encoding="utf-8")
    (worktree / ".env").write_text("SECRET_ENV_MARKER=seed\n", encoding="utf-8")

    completed = run_freestyle_checks(
        job.job_id, check_ids=["pytest-freestyle"], project_root=root
    )

    assert completed.checks_status == "passed"
    assert completed.checks_modified_worktree is True
    artifact = _artifact(root, completed)
    assert artifact["worktree_status"]["modified_by_checks"] is True
    changes = {
        item["path"]: item["change"]
        for item in artifact["worktree_status"]["modified_paths"]
    }
    assert changes == {
        ".env": "content_changed",
        "created_by_check.txt": "appeared",
        "untracked_seed.txt": "content_changed",
    }
    encoded = json.dumps(artifact)
    assert "SECRET_ENV_MARKER" not in encoded
    assert "rewritten-longer" not in encoded


def test_clean_checks_leave_worktree_and_main_checkout_untouched(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    (root / "tracked.txt").write_text("dirty main marker\n", encoding="utf-8")
    main_before = _source_snapshot(root)
    worktree_before = _source_snapshot(Path(job.worktree_path))

    completed = run_freestyle_checks(
        job.job_id, check_ids=["pytest-freestyle"], project_root=root
    )

    assert completed.checks_status == "passed"
    assert completed.checks_modified_worktree is False
    assert _source_snapshot(root) == main_before
    assert _source_snapshot(Path(job.worktree_path)) == worktree_before
    assert _git(root, "rev-parse", "--abbrev-ref", "HEAD") == "main"


def test_unsafe_recorded_worktree_path_fails_and_persists_error(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    _set_metadata(root, job.job_id, worktree_path=str(root))

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        run_freestyle_checks(
            job.job_id, check_ids=["pytest-freestyle"], project_root=root
        )

    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.checks_status == "error"
    assert failed.checks_selected_ids == ("pytest-freestyle",)
    assert "worktree path does not match" in failed.checks_errors[0]


def test_checks_module_has_no_secret_env_model_or_runtime_dependencies():
    source = Path(checks.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "deepseek",
        "DeepSeek",
        "DEEPSEEK",
        "ApprovalStore",
        "sqlite",
        "load_dotenv",
        "dotenv",
        "os.environ",
        "getenv",
        "environb",
        "from app.llm",
        "from app.tools",
        "from app.main",
        "from app.web",
        "requests",
        "httpx",
        "urllib",
        "socket",
    ):
        assert forbidden not in source


def test_run_checks_cli_dispatches_job_id_and_selected_ids(
    tmp_path, monkeypatch, capsys
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    calls: list[tuple[str, list[str] | None]] = []

    def fake_run_checks(job_id, *, check_ids=None):
        calls.append((job_id, check_ids))
        return job

    monkeypatch.setattr(freestyle, "run_freestyle_checks", fake_run_checks)

    exit_code = freestyle.main(
        [
            "run-checks",
            job.job_id,
            "--check",
            "pytest-freestyle",
            "--check",
            "ruff-check",
        ]
    )
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["job_id"] == job.job_id

    exit_code = freestyle.main(["run-checks", job.job_id])
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["job_id"] == job.job_id
    assert calls == [
        (job.job_id, ["pytest-freestyle", "ruff-check"]),
        (job.job_id, None),
    ]


def test_run_checks_cli_refuses_unknown_ids_without_executing(
    tmp_path, monkeypatch, capsys
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    monkeypatch.setattr(freestyle, "ROOT_DIR", root)
    monkeypatch.setattr(checks, "ROOT_DIR", root)
    _forbid_subprocess(monkeypatch)

    exit_code = freestyle.main(
        ["run-checks", job.job_id, "--check", "evil; rm -rf /"]
    )

    assert exit_code == 1
    error = json.loads(capsys.readouterr().err)
    assert error["error_type"] == "freestyle_job_refused"
    assert error["command"] == "run-checks"
    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.checks_status == "error"
