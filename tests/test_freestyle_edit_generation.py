"""Tests for DeepSeek-only freestyle edit-bundle generation."""

from __future__ import annotations

from dataclasses import replace
import hashlib
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
    (root / ".gitignore").write_text(".lewis-dev/\n", encoding="utf-8")
    (root / "app" / "dev").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "app" / "dev" / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
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


def _lean_bundle_content(operations: list[dict] | None = None) -> str:
    """A lean model output: only the two root fields the model now controls."""
    return json.dumps(
        {
            "schema_version": 1,
            "operations": operations
            or [
                {
                    "op": "replace_text",
                    "path": "app/dev/sample.py",
                    "old_text": "VALUE = 1",
                    "new_text": "VALUE = 2",
                }
            ],
        }
    )


_ALL_INJECTED_FIELDS = [
    "created_at",
    "goal",
    "job_id",
    "operation_count",
    "proposal_id",
]


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
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
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
    (worktree / "data" / "runtime.sqlite3").write_text("DB_MARKER\n", encoding="utf-8")
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


def _generation_artifact(root: Path, job) -> dict:
    return json.loads(
        (
            root / freestyle.DEV_STATE_DIR_NAME / job.edit_generation_artifact_path
        ).read_text(encoding="utf-8")
    )


def test_full_bundle_injects_all_provenance_and_single_valid_attempt(tmp_path):
    """An old full-shape bundle (model echoing provenance) is still handled:
    every trusted provenance field is injected/overwritten from local state and
    the bundle validates without a retry."""
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    client = _FakeDeepSeekClient(_bundle_content(job.job_id))

    completed = generate_freestyle_edit_bundle(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.edit_generation_status == "generated"
    assert len(client.completions.calls) == 1
    artifact = _generation_artifact(root, completed)
    assert artifact["normalization"] == {
        "unwrapped_bundle_key": False,
        "injected_metadata_fields": _ALL_INJECTED_FIELDS,
    }
    assert artifact["attempts"] == [{"attempt": 1, "status": "valid"}]


def test_wrapped_bundle_key_is_normalized_validated_and_recorded(tmp_path):
    """Regression for the Tetris/Shenanigans job: DeepSeek returned the bundle
    wrapped in one top-level "bundle" key and validation failed with
    "Edit bundle is missing required fields"."""
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    worktree_before = _source_snapshot(Path(job.worktree_path))
    wrapped = json.dumps({"bundle": json.loads(_bundle_content(job.job_id))})
    client = _FakeDeepSeekClient(wrapped)

    completed = generate_freestyle_edit_bundle(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.edit_generation_status == "generated"
    assert completed.edit_bundle_validation_status == "valid"
    assert len(client.completions.calls) == 1
    artifact = _generation_artifact(root, completed)
    assert artifact["normalization"] == {
        "unwrapped_bundle_key": True,
        "injected_metadata_fields": _ALL_INJECTED_FIELDS,
    }
    assert artifact["attempts"] == [{"attempt": 1, "status": "valid"}]
    persisted = json.loads(
        (root / freestyle.DEV_STATE_DIR_NAME / completed.edit_bundle_path).read_text(
            encoding="utf-8"
        )
    )
    assert "bundle" not in persisted
    assert persisted["schema_version"] == 1
    assert persisted["job_id"] == job.job_id
    assert persisted["operation_count"] == 1
    assert _source_snapshot(Path(job.worktree_path)) == worktree_before


@pytest.mark.parametrize(
    "wrapper_factory",
    [
        pytest.param(
            lambda bundle: {
                "bundle": {k: v for k, v in bundle.items() if k != "operations"}
            },
            id="nested_without_operations",
        ),
        pytest.param(lambda bundle: {"data": bundle}, id="arbitrary_wrapper_key"),
        pytest.param(lambda bundle: {"result": bundle}, id="result_wrapper_key"),
        pytest.param(
            lambda bundle: {"bundle": bundle, "metadata": {}},
            id="multiple_top_level_keys",
        ),
        pytest.param(lambda bundle: {"bundle": [bundle]}, id="non_object_bundle_value"),
    ],
)
def test_unacceptable_wrapper_shapes_still_fail_closed(tmp_path, wrapper_factory):
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    worktree_before = _source_snapshot(Path(job.worktree_path))
    wrapped = json.dumps(wrapper_factory(json.loads(_bundle_content(job.job_id))))
    client = _FakeDeepSeekClient(wrapped)

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        generate_freestyle_edit_bundle(
            job.job_id,
            project_root=root,
            deepseek_config=_config(),
            deepseek_client=client,
        )

    assert len(client.completions.calls) == 2
    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.edit_generation_status == "error"
    assert failed.edit_bundle_validation_status == "invalid"
    assert failed.edit_bundle_path is None
    assert "missing required fields" in failed.edit_generation_errors[0]
    artifact = _generation_artifact(root, failed)
    assert [record["status"] for record in artifact["attempts"]] == [
        "format_error",
        "error",
    ]
    assert _source_snapshot(Path(job.worktree_path)) == worktree_before


def test_drifted_top_level_metadata_is_normalized_from_trusted_state(tmp_path):
    """The live "Edit bundle goal is invalid" failure: the model echoed a long
    multi-line goal (and other drifted provenance). All four trusted top-level
    fields are pinned from local job/proposal state so the otherwise valid
    bundle is accepted without a retry."""
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    drifted = json.dumps(
        {
            "schema_version": 1,
            "job_id": "f" * 32,
            "created_at": "not a real timestamp",
            "operation_count": 1,
            "proposal_id": "model-invented-proposal-id",
            "goal": ("x" * 900) + "\nsecond line\nthird line",
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
    client = _FakeDeepSeekClient(drifted)

    completed = generate_freestyle_edit_bundle(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.edit_generation_status == "generated"
    assert completed.edit_bundle_validation_status == "valid"
    assert len(client.completions.calls) == 1
    persisted = json.loads(
        (root / freestyle.DEV_STATE_DIR_NAME / completed.edit_bundle_path).read_text(
            encoding="utf-8"
        )
    )
    assert persisted["job_id"] == job.job_id
    # created_at is now a parseable ISO timestamp from trusted generation state.
    from datetime import datetime

    datetime.fromisoformat(persisted["created_at"])
    assert "\n" not in persisted["goal"]
    assert len(persisted["goal"]) <= 500
    assert persisted["goal"] == "Update the sample value."
    artifact = _generation_artifact(root, completed)
    assert persisted["proposal_id"] == artifact["proposal_id"]
    assert persisted["operation_count"] == 1
    assert artifact["normalization"]["injected_metadata_fields"] == _ALL_INJECTED_FIELDS


def test_metadata_injection_does_not_let_unsafe_operations_through(tmp_path):
    """Injecting trusted top-level metadata never repairs an unsafe operation:
    the bundle still fails closed on the path and is never retried."""
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    worktree_before = _source_snapshot(Path(job.worktree_path))
    bundle = json.dumps(
        {
            "schema_version": 1,
            "job_id": "f" * 32,
            "created_at": "bad timestamp",
            "operation_count": 1,
            "goal": "drifted goal\nwith a newline",
            "operations": [
                {"op": "create_file", "path": "../escape.txt", "content": "x\n"}
            ],
        }
    )
    client = _FakeDeepSeekClient(bundle)

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        generate_freestyle_edit_bundle(
            job.job_id,
            project_root=root,
            deepseek_config=_config(),
            deepseek_client=client,
        )

    assert len(client.completions.calls) == 1
    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.edit_generation_status == "error"
    assert failed.edit_bundle_path is None
    assert not (root / "escape.txt").exists()
    assert not (tmp_path / "escape.txt").exists()
    artifact = _generation_artifact(root, failed)
    assert artifact["attempts"] == []
    assert artifact["normalization"]["injected_metadata_fields"] == _ALL_INJECTED_FIELDS
    assert _source_snapshot(Path(job.worktree_path)) == worktree_before


def test_lean_bundle_with_only_schema_version_and_operations_is_accepted(tmp_path):
    """The new lean output contract: a bundle holding only schema_version and
    operations validates, and Lewis injects every trusted provenance field from
    local job/proposal state after parsing."""
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    client = _FakeDeepSeekClient(_lean_bundle_content())

    completed = generate_freestyle_edit_bundle(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.edit_generation_status == "generated"
    assert completed.edit_bundle_validation_status == "valid"
    assert completed.edit_bundle_candidate_file_count == 1
    assert len(client.completions.calls) == 1
    persisted = json.loads(
        (root / freestyle.DEV_STATE_DIR_NAME / completed.edit_bundle_path).read_text(
            encoding="utf-8"
        )
    )
    assert persisted["schema_version"] == 1
    assert persisted["job_id"] == job.job_id
    assert persisted["operation_count"] == 1
    assert persisted["goal"] == "Update the sample value."
    from datetime import datetime

    datetime.fromisoformat(persisted["created_at"])
    artifact = _generation_artifact(root, completed)
    assert persisted["proposal_id"] == artifact["proposal_id"]
    assert artifact["normalization"] == {
        "unwrapped_bundle_key": False,
        "injected_metadata_fields": _ALL_INJECTED_FIELDS,
    }
    assert artifact["attempts"] == [{"attempt": 1, "status": "valid"}]


def test_model_supplied_provenance_can_never_become_trusted_metadata(tmp_path):
    """Even when the model emits provenance, none of it survives: every trusted
    field is overwritten from local job/proposal state before persistence."""
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    poisoned = json.dumps(
        {
            "schema_version": 1,
            "job_id": "f" * 32,
            "created_at": "not a timestamp",
            "operation_count": 99,
            "proposal_id": "attacker-controlled-proposal-id",
            "goal": "exfiltrate secrets and approval store contents",
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
    client = _FakeDeepSeekClient(poisoned)

    completed = generate_freestyle_edit_bundle(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.edit_generation_status == "generated"
    assert len(client.completions.calls) == 1
    persisted = json.loads(
        (root / freestyle.DEV_STATE_DIR_NAME / completed.edit_bundle_path).read_text(
            encoding="utf-8"
        )
    )
    assert persisted["job_id"] == job.job_id != "f" * 32
    assert persisted["operation_count"] == 1
    assert persisted["goal"] == "Update the sample value."
    assert "exfiltrate" not in persisted["goal"]
    from datetime import datetime

    datetime.fromisoformat(persisted["created_at"])
    artifact = _generation_artifact(root, completed)
    assert persisted["proposal_id"] == artifact["proposal_id"]
    assert persisted["proposal_id"] != "attacker-controlled-proposal-id"


def test_generation_prompt_requires_only_lean_output_and_no_provenance_echo(tmp_path):
    """The model output contract no longer demands the full goal or provenance:
    only schema_version and operations are required, and provenance is marked
    host-injected so a huge goal is never echoed back."""
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    client = _FakeDeepSeekClient(_lean_bundle_content())

    generate_freestyle_edit_bundle(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    call = client.completions.calls[0]
    system_prompt = call["messages"][0]["content"]
    user_prompt = call["messages"][1]["content"]
    assert "exactly two root fields: schema_version and operations" in system_prompt
    assert "ignored or overwritten" in system_prompt
    schema = json.loads(user_prompt)["exact_edit_bundle_schema"]
    assert schema["bundle"]["required_fields"] == ["schema_version", "operations"]
    assert "goal" not in schema["bundle"]["required_fields"]
    assert set(schema["bundle"]["host_injected_fields"]) == {
        "job_id",
        "created_at",
        "operation_count",
        "proposal_id",
        "goal",
    }


def test_lean_bundle_unsafe_path_still_fails_closed(tmp_path):
    """Metadata injection never relaxes safety: a lean bundle with an unsafe
    path still fails closed on the path and is never retried."""
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    worktree_before = _source_snapshot(Path(job.worktree_path))
    lean_unsafe = _lean_bundle_content(
        [{"op": "create_file", "path": "../escape.txt", "content": "x\n"}]
    )
    client = _FakeDeepSeekClient(lean_unsafe)

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        generate_freestyle_edit_bundle(
            job.job_id,
            project_root=root,
            deepseek_config=_config(),
            deepseek_client=client,
        )

    assert len(client.completions.calls) == 1
    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.edit_generation_status == "error"
    assert failed.edit_bundle_path is None
    assert not (root / "escape.txt").exists()
    assert not (tmp_path / "escape.txt").exists()
    artifact = _generation_artifact(root, failed)
    assert artifact["attempts"] == []
    assert artifact["normalization"]["injected_metadata_fields"] == _ALL_INJECTED_FIELDS
    assert _source_snapshot(Path(job.worktree_path)) == worktree_before


def test_multiple_replace_text_ops_on_one_file_generate_and_persist(tmp_path):
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    worktree = Path(job.worktree_path)
    (worktree / "app" / "dev" / "sample.py").write_text(
        "ALPHA = 1\nBETA = 2\n", encoding="utf-8"
    )
    bundle = json.dumps(
        {
            "schema_version": 1,
            "job_id": job.job_id,
            "created_at": "2026-06-10T12:00:00+00:00",
            "operation_count": 2,
            "operations": [
                {
                    "op": "replace_text",
                    "path": "app/dev/sample.py",
                    "old_text": "ALPHA = 1",
                    "new_text": "ALPHA = 11",
                },
                {
                    "op": "replace_text",
                    "path": "app/dev/sample.py",
                    "old_text": "BETA = 2",
                    "new_text": "BETA = 22",
                },
            ],
        }
    )
    client = _FakeDeepSeekClient(bundle)

    completed = generate_freestyle_edit_bundle(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.edit_generation_status == "generated"
    assert completed.edit_bundle_validation_status == "valid"
    assert completed.edit_bundle_candidate_file_count == 1
    assert len(client.completions.calls) == 1
    assert (worktree / "app" / "dev" / "sample.py").read_text(
        encoding="utf-8"
    ) == "ALPHA = 1\nBETA = 2\n"
    artifact = _generation_artifact(root, completed)
    assert artifact["validation"]["operation_count"] == 2
    operation = artifact["validation"]["operations"][0]
    assert operation["op"] == "replace_text"
    assert operation["replacement_count"] == 2
    assert operation["descending_offset_application"] is True


def test_malformed_json_is_repaired_with_exactly_one_retry(tmp_path):
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    client = _SequenceDeepSeekClient(
        ["this is not json at all", _bundle_content(job.job_id)]
    )

    completed = generate_freestyle_edit_bundle(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.edit_generation_status == "generated"
    assert completed.edit_bundle_validation_status == "valid"
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
    assert "exact_edit_bundle_schema" in repair_text
    assert "Output JSON only" in repair_text
    artifact = _generation_artifact(root, completed)
    assert [record["status"] for record in artifact["attempts"]] == [
        "format_error",
        "valid",
    ]
    assert artifact["attempts"][1]["repair_retry"] is True
    preview = artifact["attempts"][0]["model_output_preview"]
    assert "this is not json at all" in preview
    assert len(preview) <= 2_000


@pytest.mark.parametrize(
    "content",
    [
        "never valid json",
        '```json\n{"schema_version": 1}\n```',
    ],
)
def test_repair_retry_is_bounded_to_exactly_one_and_fails_closed(tmp_path, content):
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    worktree_before = _source_snapshot(Path(job.worktree_path))
    client = _FakeDeepSeekClient(content)

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        generate_freestyle_edit_bundle(
            job.job_id,
            project_root=root,
            deepseek_config=_config(),
            deepseek_client=client,
        )

    assert len(client.completions.calls) == 2
    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.edit_generation_status == "error"
    assert failed.edit_bundle_path is None
    assert all(len(error) <= 500 for error in failed.edit_generation_errors)
    assert _source_snapshot(Path(job.worktree_path)) == worktree_before


@pytest.mark.parametrize(
    "unsafe_path",
    ["../escape.txt", ".env", "data/runtime.sqlite3", "run.sh"],
)
def test_unsafe_bundle_operations_never_trigger_a_repair_retry(tmp_path, unsafe_path):
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    bundle = json.dumps(
        {
            "schema_version": 1,
            "job_id": job.job_id,
            "created_at": "2026-06-10T12:00:00+00:00",
            "operation_count": 1,
            "operations": [
                {"op": "create_file", "path": unsafe_path, "content": "x\n"}
            ],
        }
    )
    client = _FakeDeepSeekClient(bundle)

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        generate_freestyle_edit_bundle(
            job.job_id,
            project_root=root,
            deepseek_config=_config(),
            deepseek_client=client,
        )

    assert len(client.completions.calls) == 1
    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.edit_generation_status == "error"
    assert failed.edit_bundle_path is None
    artifact = _generation_artifact(root, failed)
    assert [record["status"] for record in artifact["attempts"]] == []


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


def _stale_anchor_bundle(job_id: str) -> str:
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
                    "old_text": "VALUE = 999",
                    "new_text": "VALUE = 2",
                }
            ],
        }
    )


def test_prompt_includes_exact_anchor_snippets_and_anchor_instructions(tmp_path):
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    client = _FakeDeepSeekClient(_bundle_content(job.job_id))

    completed = generate_freestyle_edit_bundle(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    call = client.completions.calls[0]
    system_prompt = call["messages"][0]["content"]
    assert "copied exactly" in system_prompt
    assert "never invent" in system_prompt
    assert "avoid whole-file replace_file rewrites" in system_prompt
    payload = json.loads(call["messages"][1]["content"])
    snippets = payload["exact_anchor_snippets"]
    snippet_paths = {item["path"] for item in snippets}
    assert {"app/dev/sample.py", "tests/test_sample.py"} <= snippet_paths
    worktree = Path(completed.worktree_path)
    for item in snippets:
        content = (worktree / Path(item["path"])).read_text(encoding="utf-8")
        assert item["exact_text"] in content
    labels = {item["label"] for item in snippets}
    assert "test:test_value" in labels
    artifact = _generation_artifact(root, completed)
    anchor = artifact["anchor_snippets"]
    assert {record["path"] for record in anchor["files"]} >= {
        "app/dev/sample.py",
        "tests/test_sample.py",
    }
    assert anchor["total_char_count"] > 0
    assert anchor["retry"] == {
        "used": False,
        "failed_path": None,
        "failed_old_text_preview": None,
        "fresh_snippet_labels": [],
        "current_file_sha256": None,
        "full_current_content_included": False,
        "stale_anchor_analysis": None,
        "error": None,
        "skipped_reason": None,
    }


def test_no_exact_match_triggers_exactly_one_anchor_repair_retry(tmp_path):
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    client = _SequenceDeepSeekClient(
        [_stale_anchor_bundle(job.job_id), _bundle_content(job.job_id)]
    )

    completed = generate_freestyle_edit_bundle(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.edit_generation_status == "generated"
    assert completed.edit_bundle_validation_status == "valid"
    assert len(client.completions.calls) == 2
    repair_messages = client.completions.calls[1]["messages"]
    assert [message["role"] for message in repair_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    repair_payload = json.loads(repair_messages[-1]["content"])
    assert repair_payload["failed_path"] == "app/dev/sample.py"
    assert "VALUE = 999" in repair_payload["failed_old_text_preview"]
    assert "smaller exact anchor" in repair_payload["instruction"]
    assert any(
        snippet["exact_text"] == "VALUE = 1"
        for snippet in repair_payload["fresh_exact_anchor_snippets"]
    )
    artifact = _generation_artifact(root, completed)
    assert [record["status"] for record in artifact["attempts"]] == [
        "exact_match_error",
        "valid",
    ]
    assert artifact["attempts"][0]["failed_path"] == "app/dev/sample.py"
    assert artifact["attempts"][1]["repair_retry"] is True
    assert artifact["attempts"][1]["anchor_repair"] is True
    retry = artifact["anchor_snippets"]["retry"]
    assert retry["used"] is True
    assert retry["failed_path"] == "app/dev/sample.py"
    assert "VALUE = 999" in retry["failed_old_text_preview"]
    assert retry["fresh_snippet_labels"] == ["file-tail"]
    assert retry["error"] is None


def test_second_exact_match_failure_fails_closed_with_recorded_retry(tmp_path):
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    worktree_before = _source_snapshot(Path(job.worktree_path))
    client = _FakeDeepSeekClient(_stale_anchor_bundle(job.job_id))

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        generate_freestyle_edit_bundle(
            job.job_id,
            project_root=root,
            deepseek_config=_config(),
            deepseek_client=client,
        )

    assert len(client.completions.calls) == 2
    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.edit_generation_status == "error"
    assert failed.edit_bundle_path is None
    assert "no exact match" in failed.edit_generation_errors[0]
    artifact = _generation_artifact(root, failed)
    assert [record["status"] for record in artifact["attempts"]] == [
        "exact_match_error",
        "error",
    ]
    retry = artifact["anchor_snippets"]["retry"]
    assert retry["used"] is True
    assert retry["failed_path"] == "app/dev/sample.py"
    assert "no exact match" in retry["error"]
    assert _source_snapshot(Path(job.worktree_path)) == worktree_before


def test_approval_sensitive_anchor_mismatch_never_retries(tmp_path):
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    bundle = json.dumps(
        {
            "schema_version": 1,
            "job_id": job.job_id,
            "created_at": "2026-06-10T12:00:00+00:00",
            "operation_count": 1,
            "operations": [
                {
                    "op": "replace_text",
                    "path": "app/dev/sample.py",
                    "old_text": "store = _approval_store(request)",
                    "new_text": "store = None",
                }
            ],
        }
    )
    client = _FakeDeepSeekClient(bundle)

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        generate_freestyle_edit_bundle(
            job.job_id,
            project_root=root,
            deepseek_config=_config(),
            deepseek_client=client,
        )

    assert len(client.completions.calls) == 1
    failed = get_freestyle_job(job.job_id, project_root=root)
    artifact = _generation_artifact(root, failed)
    assert [record["status"] for record in artifact["attempts"]] == [
        "exact_match_error"
    ]
    assert (
        artifact["attempts"][0]["retry_skipped_reason"] == "approval_sensitive_anchor"
    )
    retry = artifact["anchor_snippets"]["retry"]
    assert retry["used"] is False
    assert retry["skipped_reason"] == "approval_sensitive_anchor"


def test_ambiguous_match_and_marker_touching_ops_never_retry(tmp_path):
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    worktree = Path(job.worktree_path)
    (worktree / "app" / "dup.py").write_text("X = 1\nX = 1\n", encoding="utf-8")
    ambiguous = json.dumps(
        {
            "schema_version": 1,
            "job_id": job.job_id,
            "created_at": "2026-06-10T12:00:00+00:00",
            "operation_count": 1,
            "operations": [
                {
                    "op": "replace_text",
                    "path": "app/dup.py",
                    "old_text": "X = 1",
                    "new_text": "X = 2",
                }
            ],
        }
    )
    marker_touching = json.dumps(
        {
            "schema_version": 1,
            "job_id": job.job_id,
            "created_at": "2026-06-10T12:00:00+00:00",
            "operation_count": 1,
            "operations": [
                {
                    "op": "replace_text",
                    "path": "app/dev/sample.py",
                    "old_text": "VALUE = 1",
                    "new_text": "VALUE = execute_approved(1)",
                }
            ],
        }
    )
    for content, expected_error in (
        (ambiguous, "matched 2 times"),
        (marker_touching, "approval-sensitive region"),
    ):
        client = _FakeDeepSeekClient(content)
        with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
            generate_freestyle_edit_bundle(
                job.job_id,
                project_root=root,
                deepseek_config=_config(),
                deepseek_client=client,
            )
        assert len(client.completions.calls) == 1
        failed = get_freestyle_job(job.job_id, project_root=root)
        assert expected_error in failed.edit_generation_errors[0]
        artifact = _generation_artifact(root, failed)
        assert [record["status"] for record in artifact["attempts"]] == []
        assert artifact["anchor_snippets"]["retry"]["used"] is False


def test_truncated_high_priority_file_still_gets_exact_anchor_snippets(tmp_path):
    root = _init_repo(tmp_path)
    header = (
        "from app.web_pages import (\n"
        "    chat_page,\n"
        "    usage_page,\n"
        ")\n"
        "\n"
        "\n"
        "def register_web_routes(app):\n"
    )
    filler = "".join(
        f"    register_value_{index:04d} = {index}\n" for index in range(400)
    )
    late_routes = (
        "\n"
        '    @app.get("/usage")\n'
        "    def usage(request):\n"
        '        return _page("Usage", request, usage_page())\n'
        "\n"
        '    @app.get("/chat")\n'
        "    def chat(request):\n"
        '        return _page("Chat", request, chat_page())\n'
    )
    routes_text = header + filler + late_routes
    assert len(routes_text) > proposal.MAX_CONTEXT_CHARS_PER_FILE
    (root / "app" / "web_routes.py").write_text(routes_text, encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, shell=False)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-m", "routes"],
        check=True,
        shell=False,
        stdout=subprocess.PIPE,
    )
    job = create_freestyle_job(
        "Wire the /usage route on the web routes page.",
        developer_approved=True,
        project_root=root,
    )
    materialize_freestyle_job(job.job_id, project_root=root)
    propose_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=_FakeDeepSeekClient(
            json.dumps(
                {
                    "proposed_implementation_steps": ["Wire the usage route."],
                    "likely_files_to_edit": ["app/web_routes.py"],
                    "risks_unknowns": [],
                    "suggested_allowlisted_checks": ["pytest"],
                }
            )
        ),
    )
    chat_block = (
        '    @app.get("/chat")\n'
        "    def chat(request):\n"
        '        return _page("Chat", request, chat_page())'
    )
    bundle = json.dumps(
        {
            "schema_version": 1,
            "job_id": job.job_id,
            "created_at": "2026-06-10T12:00:00+00:00",
            "operation_count": 1,
            "operations": [
                {
                    "op": "replace_text",
                    "path": "app/web_routes.py",
                    "old_text": chat_block,
                    "new_text": chat_block + "\n\n    # wired\n",
                }
            ],
        }
    )
    client = _FakeDeepSeekClient(bundle)

    completed = generate_freestyle_edit_bundle(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.edit_generation_status == "generated"
    payload = json.loads(client.completions.calls[0]["messages"][1]["content"])
    included = {item["path"]: item for item in payload["safe_isolated_worktree_files"]}
    assert included["app/web_routes.py"]["truncated"] is True
    truncated_content = included["app/web_routes.py"]["content"]
    snippets = [
        item
        for item in payload["exact_anchor_snippets"]
        if item["path"] == "app/web_routes.py"
    ]
    assert {item["label"] for item in snippets} >= {
        "route:/usage",
        "route:/chat",
        "import-block:app.web_pages",
    }
    for item in snippets:
        assert item["exact_text"] in routes_text
    assert any(item["exact_text"] not in truncated_content for item in snippets)


def test_anchor_mismatch_failure_surfaces_concise_handoff_warning(tmp_path):
    from app.dev.freestyle_handoff import (
        get_freestyle_handoff,
        render_freestyle_handoff_markdown,
    )

    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    client = _FakeDeepSeekClient(_stale_anchor_bundle(job.job_id))
    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        generate_freestyle_edit_bundle(
            job.job_id,
            project_root=root,
            deepseek_config=_config(),
            deepseek_client=client,
        )

    handoff = get_freestyle_handoff(job.job_id, project_root=root)
    anchor_risks = [
        risk
        for risk in handoff["risks"]
        if "exact replace_text anchor mismatch" in risk
    ]
    assert len(anchor_risks) == 1
    assert "app/dev/sample.py" in anchor_risks[0]
    assert "one bounded repair retry" in anchor_risks[0]
    assert len(anchor_risks[0]) <= 500
    markdown = render_freestyle_handoff_markdown(handoff)
    assert "exact replace_text anchor mismatch" in markdown


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


def _syntax_error_bundle() -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "operations": [
                {
                    "op": "create_file",
                    "path": "app/dev/broken.py",
                    "content": "def broken(:\n    return 1\n",
                }
            ],
        }
    )


def test_python_syntax_preflight_triggers_one_bounded_repair(tmp_path):
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    client = _SequenceDeepSeekClient(
        [_syntax_error_bundle(), _bundle_content(job.job_id)]
    )

    completed = generate_freestyle_edit_bundle(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.edit_generation_status == "generated"
    assert len(client.completions.calls) == 2
    repair_payload = json.loads(client.completions.calls[1]["messages"][-1]["content"])
    assert repair_payload["failed_path"] == "app/dev/broken.py"
    assert repair_payload["syntax_error_line"] == 1
    assert "not" in repair_payload["instruction"]
    artifact = _generation_artifact(root, completed)
    assert [record["status"] for record in artifact["attempts"]] == [
        "preflight_error",
        "valid",
    ]
    assert artifact["attempts"][0]["failed_path"] == "app/dev/broken.py"
    assert artifact["attempts"][1]["syntax_repair"] is True


def test_syntax_preflight_never_persists_or_applies_a_broken_bundle(tmp_path):
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    worktree_before = _source_snapshot(Path(job.worktree_path))
    client = _FakeDeepSeekClient(_syntax_error_bundle())

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded") as info:
        generate_freestyle_edit_bundle(
            job.job_id,
            project_root=root,
            deepseek_config=_config(),
            deepseek_client=client,
        )

    assert info.value.failure_category == "python_syntax_error"
    assert info.value.failed_path == "app/dev/broken.py"
    assert len(client.completions.calls) == 2
    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.edit_bundle_path is None
    assert "not valid Python" in failed.edit_generation_errors[0]
    artifact = _generation_artifact(root, failed)
    assert artifact["validation"]["failure_category"] == "python_syntax_error"
    assert artifact["validation"]["failed_path"] == "app/dev/broken.py"
    assert _source_snapshot(Path(job.worktree_path)) == worktree_before


def test_anchor_repair_includes_current_content_and_replace_file_fallback(tmp_path):
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    client = _SequenceDeepSeekClient(
        [_stale_anchor_bundle(job.job_id), _bundle_content(job.job_id)]
    )

    completed = generate_freestyle_edit_bundle(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    repair_payload = json.loads(client.completions.calls[1]["messages"][-1]["content"])
    assert repair_payload["current_file_content"] == "VALUE = 1\n"
    assert len(repair_payload["current_file_sha256"]) == 64
    assert "replace_file" in repair_payload["instruction"]
    assert "current_file_sha256" in repair_payload["instruction"]
    artifact = _generation_artifact(root, completed)
    retry = artifact["anchor_snippets"]["retry"]
    assert retry["full_current_content_included"] is True
    assert len(retry["current_file_sha256"]) == 64
    analysis = retry["stale_anchor_analysis"]
    assert analysis["anchor_in_prompt_context"] is False
    assert analysis["anchor_source"] == "invented"


def test_recoverable_operation_failures_classify_without_generation_retry(tmp_path):
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    create_existing = json.dumps(
        {
            "schema_version": 1,
            "operations": [
                {
                    "op": "create_file",
                    "path": "app/dev/sample.py",
                    "content": "VALUE = 2\n",
                }
            ],
        }
    )
    identical = json.dumps(
        {
            "schema_version": 1,
            "operations": [
                {
                    "op": "replace_text",
                    "path": "app/dev/sample.py",
                    "old_text": "VALUE = 1",
                    "new_text": "VALUE = 1",
                }
            ],
        }
    )
    for content, category in (
        (create_existing, "create_existing_file"),
        (identical, "identical_replacement"),
    ):
        client = _FakeDeepSeekClient(content)
        with pytest.raises(
            FreestyleJobRefusedError, match="error was recorded"
        ) as info:
            generate_freestyle_edit_bundle(
                job.job_id,
                project_root=root,
                deepseek_config=_config(),
                deepseek_client=client,
            )
        # Operation-shape mistakes stay single-shot at this layer; the loop owns
        # the bounded regeneration for them.
        assert len(client.completions.calls) == 1
        assert info.value.failure_category == category
        assert info.value.failed_path == "app/dev/sample.py"
        failed = get_freestyle_job(job.job_id, project_root=root)
        artifact = _generation_artifact(root, failed)
        assert artifact["validation"]["failure_category"] == category


def test_duplicate_definition_preflight_is_refused_and_classified(tmp_path):
    root = _init_repo(tmp_path)
    job = _proposed_job(root)
    duplicated = (
        "def normalize(value):\n"
        "    import re\n"
        "\n"
        "def normalize(value):\n"
        "    return value\n"
    )
    bundle = json.dumps(
        {
            "schema_version": 1,
            "operations": [
                {
                    "op": "replace_file",
                    "path": "app/dev/sample.py",
                    "content": duplicated,
                    "expected_sha256": hashlib.sha256(b"VALUE = 1\n").hexdigest(),
                }
            ],
        }
    )
    client = _FakeDeepSeekClient(bundle)

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded") as info:
        generate_freestyle_edit_bundle(
            job.job_id,
            project_root=root,
            deepseek_config=_config(),
            deepseek_client=client,
        )

    assert info.value.failure_category == "python_syntax_error"
    assert len(client.completions.calls) == 2
    failed = get_freestyle_job(job.job_id, project_root=root)
    assert "defines 'normalize' 2 times" in failed.edit_generation_errors[0]
