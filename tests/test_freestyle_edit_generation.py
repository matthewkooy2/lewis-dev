"""Tests for DeepSeek-only freestyle edit-bundle generation."""

from __future__ import annotations

from dataclasses import replace
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.dev.freestyle_coding as freestyle
import app.dev.freestyle_edit_bundle as edit_bundle
import app.dev.freestyle_edit_generation as generation
import app.dev.freestyle_proposal as proposal
from app.dev.freestyle_coding import (
    FreestyleJobRefusedError,
    create_freestyle_job,
    generate_freestyle_edit_bundle,
    get_freestyle_job,
    materialize_freestyle_job,
    propose_freestyle_job,
)
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
    (root / "app" / "dev").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "app" / "dev" / "sample.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    (root / "tests" / "test_sample.py").write_text(
        "def test_value():\n    assert True\n", encoding="utf-8"
    )
    (root / "README.md").write_text("Sample project\n", encoding="utf-8")
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "init")
    return root


def _config(api_key: str = "test-deepseek-key") -> DeepSeekProposalConfig:
    return DeepSeekProposalConfig(
        api_key=api_key,
        model="deepseek-test-code",
        base_url=proposal.DEEPSEEK_API_BASE_URL,
    )


def _proposal_content() -> str:
    return json.dumps(
        {
            "proposed_implementation_steps": ["Update the sample value safely."],
            "likely_files_to_edit": [
                "app/dev/sample.py",
                "tests/test_sample.py",
            ],
            "risks_unknowns": ["Keep the change isolated."],
            "suggested_allowlisted_checks": ["pytest"],
        }
    )


def _bundle_content(job_id: str) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "job_id": job_id,
            "created_at": "2026-06-10T12:00:00+00:00",
            "operation_count": 1,
            "operations": [
                {
                    "op": "replace_text",
                    "path": "app/dev/sample.py",
                    "old_text": "VALUE = 1",
                    "new_text": "VALUE = 2",
                }
            ],
        }
    )


class _FakeCompletions:
    def __init__(self, content: str, error: Exception | None = None):
        self.content = content
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            model="deepseek-test-code",
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        )


class _FakeDeepSeekClient:
    def __init__(self, content: str, error: Exception | None = None):
        self.completions = _FakeCompletions(content, error)
        self.chat = SimpleNamespace(completions=self.completions)


def _materialized_job(root: Path):
    job = create_freestyle_job(
        "Update the sample value.",
        developer_approved=True,
        project_root=root,
    )
    return materialize_freestyle_job(job.job_id, project_root=root)


def _proposed_job(root: Path):
    job = _materialized_job(root)
    return propose_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=_FakeDeepSeekClient(_proposal_content()),
    )


def _source_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and ".git" not in path.relative_to(root).parts
        and freestyle.DEV_STATE_DIR_NAME not in path.relative_to(root).parts
    }


def _set_job_field(root: Path, job_id: str, field: str, value) -> None:
    path = root / freestyle.DEV_STATE_DIR_NAME / "jobs" / f"{job_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_generation_requires_approved_materialized_job_and_proposal(tmp_path):
    root = _init_repo(tmp_path)
    ready = create_freestyle_job(
        "Update the sample value.",
        developer_approved=True,
        project_root=root,
    )
    client = _FakeDeepSeekClient(_bundle_content(ready.job_id))

    with pytest.raises(FreestyleJobRefusedError, match="materialized"):
        generate_freestyle_edit_bundle(
            ready.job_id,
            project_root=root,
            deepseek_config=_config(),
            deepseek_client=client,
        )

    materialized = materialize_freestyle_job(ready.job_id, project_root=root)
    with pytest.raises(FreestyleJobRefusedError, match="proposal artifact"):
        generate_freestyle_edit_bundle(
            materialized.job_id,
            project_root=root,
            deepseek_config=_config(),
            deepseek_client=client,
        )

    proposed = propose_freestyle_job(
        materialized.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=_FakeDeepSeekClient(_proposal_content()),
    )
    _set_job_field(root, proposed.job_id, "developer_approved", False)
    with pytest.raises(
        FreestyleJobRefusedError, match="explicit persisted developer approval"
    ):
        generate_freestyle_edit_bundle(
            proposed.job_id,
            project_root=root,
            deepseek_config=_config(),
            deepseek_client=client,
        )

    assert client.completions.calls == []


@pytest.mark.parametrize("blocked_status", ["applying", "inconsistent"])
def test_generation_refuses_in_progress_or_inconsistent_edit_apply_state(
    tmp_path, blocked_status
):
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    _set_job_field(root, job.job_id, "edit_apply_status", blocked_status)
    client = _FakeDeepSeekClient(_bundle_content(job.job_id))

    with pytest.raises(FreestyleJobRefusedError, match="in-progress or inconsistent"):
        generate_freestyle_edit_bundle(
            job.job_id,
            project_root=root,
            deepseek_config=_config(),
            deepseek_client=client,
        )

    assert client.completions.calls == []
    assert get_freestyle_job(job.job_id, project_root=root).edit_apply_status == (
        blocked_status
    )


def test_generation_reads_only_isolated_worktree_and_excludes_unsafe_context(tmp_path):
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    worktree = Path(job.worktree_path)
    (root / "app" / "dev" / "sample.py").write_text(
        "MAIN_CHECKOUT_ONLY = True\n", encoding="utf-8"
    )
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
    (worktree / "generated").mkdir()
    (worktree / "generated" / "output.py").write_text(
        "GENERATED_MARKER\n", encoding="utf-8"
    )
    (worktree / "approvalstore.txt").write_text(
        "APPROVAL_STORE_MARKER\n", encoding="utf-8"
    )
    (worktree / "binary.bin").write_bytes(b"\x00BINARY_MARKER")
    (worktree / "oversized.txt").write_text(
        "x" * (proposal.MAX_CONTEXT_FILE_BYTES + 1), encoding="utf-8"
    )
    client = _FakeDeepSeekClient(_bundle_content(job.job_id))

    completed = generate_freestyle_edit_bundle(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "VALUE = 1" in prompt
    assert "MAIN_CHECKOUT_ONLY" not in prompt
    for marker in (
        "ENV_SECRET_MARKER",
        "TOKEN_MARKER",
        "DEV_STATE_MARKER",
        "DB_MARKER",
        "CACHE_MARKER",
        "GENERATED_MARKER",
        "APPROVAL_STORE_MARKER",
        "BINARY_MARKER",
    ):
        assert marker not in prompt
    artifact = json.loads(
        (
            root
            / freestyle.DEV_STATE_DIR_NAME
            / completed.edit_generation_artifact_path
        ).read_text(encoding="utf-8")
    )
    omissions = artifact["context_summary"]["omission_counts"]
    assert omissions["safety"] >= 1
    assert omissions["generated_or_cache"] >= 1
    assert omissions["approval_store"] >= 1
    assert omissions["binary"] >= 1
    assert omissions["oversized"] >= 1


def test_valid_deepseek_json_is_validated_persisted_and_never_applied(
    tmp_path, monkeypatch
):
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    worktree = Path(job.worktree_path)
    worktree_before = _source_snapshot(worktree)
    checkout_before = _source_snapshot(root)
    client = _FakeDeepSeekClient(_bundle_content(job.job_id))

    def forbidden_apply(*args, **kwargs):
        raise AssertionError("generation must not call an edit applier")

    monkeypatch.setattr(edit_bundle, "_apply_planned_edits", forbidden_apply)
    completed = generate_freestyle_edit_bundle(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.edit_generation_status == "generated"
    assert completed.edit_bundle_validation_status == "valid"
    assert completed.edit_bundle_candidate_file_count == 1
    assert completed.edit_bundle_path
    assert completed.edit_generation_artifact_path
    bundle_path = root / freestyle.DEV_STATE_DIR_NAME / completed.edit_bundle_path
    assert json.loads(bundle_path.read_text(encoding="utf-8"))["job_id"] == job.job_id
    artifact_path = (
        root / freestyle.DEV_STATE_DIR_NAME / completed.edit_generation_artifact_path
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["validation"]["status"] == "valid"
    assert artifact["validation"]["candidate_file_count"] == 1
    assert artifact["source_mutation_performed"] is False
    assert artifact["bundle_applied"] is False
    assert _source_snapshot(worktree) == worktree_before
    assert _source_snapshot(root) == checkout_before
    assert (worktree / "app" / "dev" / "sample.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 1\n"


@pytest.mark.parametrize(
    ("content_factory", "error_text"),
    [
        (lambda job_id: "not json", "valid UTF-8 JSON"),
        (
            lambda job_id: _bundle_content("f" * 32),
            "different job id",
        ),
        (
            lambda job_id: json.dumps(
                {
                    "schema_version": 1,
                    "job_id": job_id,
                    "created_at": "2026-06-10T12:00:00+00:00",
                    "operation_count": 1,
                    "operations": [
                        {"op": "run_command", "path": "x.txt", "content": "bad"}
                    ],
                }
            ),
            "unsupported op",
        ),
        (
            lambda job_id: json.dumps(
                {
                    "schema_version": 1,
                    "job_id": job_id,
                    "created_at": "2026-06-10T12:00:00+00:00",
                    "operation_count": 1,
                    "operations": [
                        {
                            "op": "replace_file",
                            "path": "app/dev/sample.py",
                            "content": "VALUE = 2\n",
                            "expected_sha256": "0" * 64,
                        }
                    ],
                }
            ),
            "stale",
        ),
    ],
)
def test_invalid_deepseek_output_fails_closed_and_records_bounded_errors(
    tmp_path, content_factory, error_text
):
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    worktree_before = _source_snapshot(Path(job.worktree_path))
    checkout_before = _source_snapshot(root)

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        generate_freestyle_edit_bundle(
            job.job_id,
            project_root=root,
            deepseek_config=_config(),
            deepseek_client=_FakeDeepSeekClient(content_factory(job.job_id)),
        )

    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.edit_generation_status == "error"
    assert failed.edit_bundle_validation_status == "invalid"
    assert failed.edit_bundle_path is None
    assert failed.edit_generation_artifact_path
    assert error_text in failed.edit_generation_errors[0]
    assert len(failed.edit_generation_errors[0]) <= 500
    artifact = json.loads(
        (
            root / freestyle.DEV_STATE_DIR_NAME / failed.edit_generation_artifact_path
        ).read_text(encoding="utf-8")
    )
    assert artifact["validation"]["status"] == "invalid"
    assert artifact["validation"]["errors"]
    assert _source_snapshot(Path(job.worktree_path)) == worktree_before
    assert _source_snapshot(root) == checkout_before


def test_generation_uses_existing_deepseek_path_exact_schema_and_no_execution(
    tmp_path, monkeypatch
):
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    client = _FakeDeepSeekClient(_bundle_content(job.job_id))
    factory_calls: list[dict[str, str]] = []
    subprocess_calls: list[tuple[list[str], dict]] = []
    real_run = subprocess.run

    def fake_factory(**kwargs):
        factory_calls.append(kwargs)
        return client

    def recording_run(args, **kwargs):
        subprocess_calls.append((list(args), kwargs))
        return real_run(args, **kwargs)

    monkeypatch.setattr(generation, "create_deepseek_client", fake_factory)
    monkeypatch.setattr(freestyle.subprocess, "run", recording_run)
    completed = generate_freestyle_edit_bundle(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
    )

    assert completed.edit_generation_status == "generated"
    assert factory_calls == [
        {
            "api_key": "test-deepseek-key",
            "base_url": proposal.DEEPSEEK_API_BASE_URL,
        }
    ]
    call = client.completions.calls[0]
    assert call["model"] == "deepseek-test-code"
    assert call["response_format"] == {"type": "json_object"}
    assert "tools" not in call
    system_prompt = call["messages"][0]["content"]
    user_prompt = call["messages"][1]["content"]
    assert "Output JSON only" in system_prompt
    assert "exact_edit_bundle_schema" in user_prompt
    assert '"additional_fields_allowed":false' in user_prompt
    assert '"replace_text"' in user_prompt
    assert len(system_prompt) + len(user_prompt) <= generation.MAX_EDIT_PROMPT_CHARS
    prompt_payload = json.loads(user_prompt)
    included_paths = [
        item["path"] for item in prompt_payload["safe_isolated_worktree_files"]
    ]
    assert set(included_paths[:2]) == {"app/dev/sample.py", "tests/test_sample.py"}
    assert subprocess_calls
    assert all(call_args[0][0] == "git" for call_args in subprocess_calls)
    assert all(call_args[1]["shell"] is False for call_args in subprocess_calls)
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
    assert all(forbidden.isdisjoint(call_args[0]) for call_args in subprocess_calls)


def test_generation_has_no_fallback_provider_shell_or_runtime_registration():
    source = Path(generation.__file__).read_text(encoding="utf-8")
    forbidden = [
        "Anthropic",
        "ollama",
        "subprocess",
        "os.system",
        "ApprovalStore",
        "sqlite",
        "load_dotenv",
        "get_settings",
        "from app.llm",
        "from app.tools",
        "from app.main",
        "from app.web",
        "from app.approvals",
        "requests",
        "httpx",
    ]
    for text in forbidden:
        assert text not in source, text


def test_generate_edits_cli_dispatches_only_the_requested_job_id(
    tmp_path, monkeypatch, capsys
):
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    calls: list[str] = []

    def fake_generate(job_id):
        calls.append(job_id)
        return replace(job, edit_generation_status="generated")

    monkeypatch.setattr(freestyle, "generate_freestyle_edit_bundle", fake_generate)
    exit_code = freestyle.main(["generate-edits", job.job_id])

    assert exit_code == 0
    assert calls == [job.job_id]
    assert json.loads(capsys.readouterr().out)["job_id"] == job.job_id
