"""Tests for provider-neutral freestyle model configuration.

Covers local OpenAI-compatible endpoint support (LM Studio/Qwen), preserved
DeepSeek validation, no-remote-fallback behavior, reduced local context
limits, provider metadata in artifacts, and the constrained stop-before-apply
workflow under the local provider.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.dev.freestyle_coding as freestyle
from app.dev.freestyle_coding import (
    FreestyleJobRefusedError,
    create_freestyle_job,
    generate_freestyle_edit_bundle,
    get_freestyle_job,
    materialize_freestyle_job,
    propose_freestyle_job,
)
from app.dev.freestyle_provider import (
    DEEPSEEK_API_BASE_URL,
    DEFAULT_LOCAL_BASE_URL,
    DEFAULT_LOCAL_MODEL,
    LOCAL_OPENAI_PROVIDER,
    LOCAL_PLACEHOLDER_API_KEY,
    ProviderProposalConfig,
    context_limits_for,
    describe_provider_config,
    effective_api_key,
    provider_config_from_settings,
    provider_metadata,
    validate_provider_config,
)
from app.dev.freestyle_run import run_freestyle_job


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


def _local_config(
    api_key: str = "",
    model: str = "qwen2.5-coder-7b-instruct",
    base_url: str = "http://127.0.0.1:1234/v1",
) -> ProviderProposalConfig:
    # Mirrors from_process_environment(): local endpoints do not get JSON
    # response format unless LEWIS_LOCAL_JSON_MODE is explicitly enabled.
    return ProviderProposalConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider=LOCAL_OPENAI_PROVIDER,
        local_only=True,
        supports_json_response_format=False,
    )


def _proposal_content() -> str:
    return json.dumps(
        {
            "proposed_implementation_steps": ["Update the sample value safely."],
            "likely_files_to_edit": ["app/dev/sample.py", "tests/test_sample.py"],
            "risks_unknowns": ["Keep the change isolated."],
            "suggested_allowlisted_checks": ["pytest"],
        }
    )


def _lean_bundle_content() -> str:
    return json.dumps(
        {
            "schema_version": 1,
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
            model="qwen2.5-coder-7b-instruct",
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        )


class _FakeLocalClient:
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


def _read_artifact(root: Path, reference: str) -> dict:
    path = root / freestyle.DEV_STATE_DIR_NAME
    for part in reference.split("/"):
        path = path / part
    return json.loads(path.read_text(encoding="utf-8"))


# --- configuration resolution -------------------------------------------------


def test_environment_defaults_to_deepseek_provider(monkeypatch):
    for name in (
        "LEWIS_FREESTYLE_PROVIDER",
        "LEWIS_FREESTYLE_LOCAL_ONLY",
        "LEWIS_LOCAL_BASE_URL",
        "LEWIS_LOCAL_CODE_MODEL",
        "LEWIS_LOCAL_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "remote-key")
    monkeypatch.setenv("DEEPSEEK_CODE_MODEL", "deepseek-test-code")

    config = ProviderProposalConfig.from_process_environment()

    assert config.provider == "deepseek"
    assert config.api_key == "remote-key"
    assert config.model == "deepseek-test-code"
    assert config.base_url == DEEPSEEK_API_BASE_URL
    assert config.local_only is False
    validate_provider_config(config)


@pytest.mark.parametrize("alias", ["local-openai", "local", "lmstudio", "LM-Studio"])
def test_environment_selects_local_provider_with_defaults(monkeypatch, alias):
    monkeypatch.setenv("LEWIS_FREESTYLE_PROVIDER", alias)
    for name in ("LEWIS_LOCAL_BASE_URL", "LEWIS_LOCAL_CODE_MODEL", "LEWIS_LOCAL_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    config = ProviderProposalConfig.from_process_environment()

    assert config.provider == LOCAL_OPENAI_PROVIDER
    assert config.base_url == DEFAULT_LOCAL_BASE_URL
    assert config.model == DEFAULT_LOCAL_MODEL
    assert config.api_key == ""
    assert config.local_only is True
    validate_provider_config(config)


def test_environment_unknown_provider_fails_closed(monkeypatch):
    monkeypatch.setenv("LEWIS_FREESTYLE_PROVIDER", "shadow-cloud")

    config = ProviderProposalConfig.from_process_environment()

    with pytest.raises(FreestyleJobRefusedError, match="Unknown freestyle model provider"):
        validate_provider_config(config)


def test_settings_based_local_provider_resolution():
    settings = SimpleNamespace(
        freestyle_provider="lmstudio",
        freestyle_local_only=False,
        local_llm_base_url="http://localhost:1234/v1",
        local_llm_model="qwen2.5-coder-7b-instruct",
        local_llm_api_key=None,
        local_llm_json_mode=False,
        deepseek_api_key="unused",
        deepseek_code_model="deepseek-v4-pro",
        deepseek_base_url=DEEPSEEK_API_BASE_URL,
    )

    config = provider_config_from_settings(settings)

    assert config.provider == LOCAL_OPENAI_PROVIDER
    assert config.base_url == "http://localhost:1234/v1"
    assert config.local_only is True
    validate_provider_config(config)


# --- local endpoint and credential validation ----------------------------------


def test_local_provider_accepts_empty_api_key_and_loopback_endpoints():
    for base_url in (
        "http://127.0.0.1:1234/v1",
        "http://localhost:1234/v1",
        "http://127.0.0.1:8080",
        "https://localhost:1234/v1",
    ):
        validate_provider_config(_local_config(base_url=base_url))
    assert effective_api_key(_local_config()) == LOCAL_PLACEHOLDER_API_KEY
    assert effective_api_key(_local_config(api_key="real-key")) == "real-key"


@pytest.mark.parametrize(
    "base_url",
    [
        "http://192.168.1.20:1234/v1",
        "http://api.deepseek.com/v1",
        "https://api.openai.com/v1",
        "http://127.0.0.1:1234/v1?fallback=remote",
        "http://user:pass@127.0.0.1:1234/v1",
        "ftp://127.0.0.1:1234/v1",
        "",
    ],
)
def test_local_provider_refuses_non_loopback_or_malformed_endpoints(base_url):
    with pytest.raises(FreestyleJobRefusedError, match="loopback"):
        validate_provider_config(_local_config(base_url=base_url))


def test_local_provider_does_not_require_deepseek_model_names():
    validate_provider_config(_local_config(model="qwen2.5-coder-7b-instruct"))
    validate_provider_config(_local_config(model="Qwen/Qwen2.5-Coder-7B-Instruct"))
    with pytest.raises(FreestyleJobRefusedError, match="LEWIS_LOCAL_CODE_MODEL"):
        validate_provider_config(_local_config(model=""))
    with pytest.raises(FreestyleJobRefusedError, match="LEWIS_LOCAL_CODE_MODEL"):
        validate_provider_config(_local_config(model="bad model name"))


def test_deepseek_provider_validation_is_preserved():
    valid = ProviderProposalConfig(
        api_key="remote-key",
        model="deepseek-test-code",
        base_url=DEEPSEEK_API_BASE_URL,
    )
    validate_provider_config(valid)
    with pytest.raises(FreestyleJobRefusedError, match="DEEPSEEK_API_KEY must be set"):
        validate_provider_config(
            ProviderProposalConfig(api_key="", model="deepseek-test-code")
        )
    with pytest.raises(FreestyleJobRefusedError, match="must name a DeepSeek model"):
        validate_provider_config(
            ProviderProposalConfig(
                api_key="remote-key", model="qwen2.5-coder-7b-instruct"
            )
        )
    with pytest.raises(FreestyleJobRefusedError, match="official DeepSeek API endpoint"):
        validate_provider_config(
            ProviderProposalConfig(
                api_key="remote-key",
                model="deepseek-test-code",
                base_url="http://127.0.0.1:1234/v1",
            )
        )


def test_local_only_mode_refuses_remote_deepseek_provider():
    config = ProviderProposalConfig(
        api_key="remote-key",
        model="deepseek-test-code",
        base_url=DEEPSEEK_API_BASE_URL,
        local_only=True,
    )
    with pytest.raises(FreestyleJobRefusedError, match="LEWIS_FREESTYLE_LOCAL_ONLY"):
        validate_provider_config(config)


# --- provider-derived limits and display ----------------------------------------


def test_local_provider_context_limits_are_reduced():
    local = context_limits_for(_local_config())
    remote = context_limits_for(
        ProviderProposalConfig(api_key="k", model="deepseek-test-code")
    )
    assert local.max_context_files < remote.max_context_files
    assert local.max_context_total_chars < remote.max_context_total_chars
    assert local.max_context_chars_per_file < remote.max_context_chars_per_file
    assert local.max_proposal_output_tokens < remote.max_proposal_output_tokens
    assert local.max_edit_output_tokens < remote.max_edit_output_tokens
    assert local.max_edit_prompt_chars < remote.max_edit_prompt_chars


def test_provider_display_is_secret_free():
    described = describe_provider_config(_local_config(api_key="topsecret"))
    assert "provider=local-openai" in described
    assert "endpoint=http://127.0.0.1:1234/v1" in described
    assert "model=qwen2.5-coder-7b-instruct" in described
    assert "local_only=true" in described
    assert "topsecret" not in described
    metadata = provider_metadata(_local_config(api_key="topsecret"))
    assert metadata["provider"] == LOCAL_OPENAI_PROVIDER
    assert metadata["endpoint"] == "http://127.0.0.1:1234/v1"
    assert metadata["local_only"] is True
    assert "topsecret" not in json.dumps(metadata)


# --- generation through the local provider --------------------------------------


def test_local_proposal_records_provider_metadata_and_reduced_output_budget(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    client = _FakeLocalClient(_proposal_content())

    proposed = propose_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=_local_config(),
        deepseek_client=client,
    )

    assert proposed.proposal_status == "proposed"
    artifact = _read_artifact(root, proposed.proposal_artifact_path)
    assert artifact["model_metadata"]["provider"] == LOCAL_OPENAI_PROVIDER
    assert artifact["model_metadata"]["endpoint"] == "http://127.0.0.1:1234/v1"
    assert artifact["model_metadata"]["local_only"] is True
    assert (
        artifact["model_metadata"]["requested_model"] == "qwen2.5-coder-7b-instruct"
    )
    call = client.completions.calls[0]
    assert call["model"] == "qwen2.5-coder-7b-instruct"
    assert call["max_tokens"] == context_limits_for(_local_config()).max_proposal_output_tokens


def test_local_edit_generation_omits_json_mode_unless_enabled(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    propose_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=_local_config(),
        deepseek_client=_FakeLocalClient(_proposal_content()),
    )
    client = _FakeLocalClient(_lean_bundle_content())

    generated = generate_freestyle_edit_bundle(
        job.job_id,
        project_root=root,
        deepseek_config=_local_config(),
        deepseek_client=client,
    )

    assert generated.edit_generation_status == "generated"
    assert generated.edit_bundle_validation_status == "valid"
    call = client.completions.calls[0]
    assert "response_format" not in call
    assert call["max_tokens"] == context_limits_for(_local_config()).max_edit_output_tokens
    artifact = _read_artifact(root, generated.edit_generation_artifact_path)
    assert artifact["model_metadata"]["provider"] == LOCAL_OPENAI_PROVIDER
    # The trusted-pipeline provenance gate accepted a local-provider proposal.
    assert generated.edit_apply_status == "not_applied"


def test_local_edit_generation_sends_json_mode_when_capability_enabled(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    json_mode_config = ProviderProposalConfig(
        api_key="",
        model="qwen2.5-coder-7b-instruct",
        base_url="http://127.0.0.1:1234/v1",
        provider=LOCAL_OPENAI_PROVIDER,
        local_only=True,
        supports_json_response_format=True,
    )
    propose_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=json_mode_config,
        deepseek_client=_FakeLocalClient(_proposal_content()),
    )
    client = _FakeLocalClient(_lean_bundle_content())

    generate_freestyle_edit_bundle(
        job.job_id,
        project_root=root,
        deepseek_config=json_mode_config,
        deepseek_client=client,
    )

    assert client.completions.calls[0]["response_format"] == {"type": "json_object"}


def test_unreachable_local_endpoint_fails_closed_without_deepseek_wording(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    client = _FakeLocalClient(
        _proposal_content(), error=ConnectionError("connection refused: 127.0.0.1:1234")
    )

    with pytest.raises(FreestyleJobRefusedError, match="Could not generate proposal"):
        propose_freestyle_job(
            job.job_id,
            project_root=root,
            deepseek_config=_local_config(),
            deepseek_client=client,
        )

    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.proposal_status == "error"
    assert failed.proposal_errors
    assert all("deepseek" not in error.casefold() for error in failed.proposal_errors)
    failure = _read_artifact(root, failed.proposal_failure_artifact_path)
    assert failure["model_metadata"]["provider"] == LOCAL_OPENAI_PROVIDER
    assert "deepseek" not in json.dumps(failure["validation_errors"]).casefold()
    # Exactly one provider call was attempted; nothing retried remotely.
    assert len(client.completions.calls) == 1


def test_local_malformed_json_gets_one_repair_retry_then_fails(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    client_contents = ["Sure! Here is my plan without JSON.", "still not json {"]
    client = _FakeLocalClient(client_contents[0])

    def _sequenced_create(**kwargs):
        client.completions.calls.append(kwargs)
        index = min(len(client.completions.calls) - 1, len(client_contents) - 1)
        return SimpleNamespace(
            model="qwen2.5-coder-7b-instruct",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=client_contents[index])
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100, completion_tokens=20, total_tokens=120
            ),
        )

    client.chat.completions.create = _sequenced_create

    with pytest.raises(FreestyleJobRefusedError):
        propose_freestyle_job(
            job.job_id,
            project_root=root,
            deepseek_config=_local_config(),
            deepseek_client=client,
        )

    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.proposal_status == "error"
    assert any("not valid JSON" in error for error in failed.proposal_errors)
    assert all("deepseek" not in error.casefold() for error in failed.proposal_errors)
    assert len(client.completions.calls) == 2
    failure = _read_artifact(root, failed.proposal_failure_artifact_path)
    statuses = [attempt["status"] for attempt in failure["proposal_attempts"]]
    assert statuses.count("invalid_json") == 2


def test_default_local_run_stops_before_apply(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree = Path(job.worktree_path)
    contents = [_proposal_content(), _lean_bundle_content()]
    client = _FakeLocalClient(contents[0])

    def _staged_create(**kwargs):
        client.completions.calls.append(kwargs)
        index = min(len(client.completions.calls) - 1, len(contents) - 1)
        return SimpleNamespace(
            model="qwen2.5-coder-7b-instruct",
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=contents[index]))
            ],
            usage=SimpleNamespace(
                prompt_tokens=100, completion_tokens=50, total_tokens=150
            ),
        )

    client.chat.completions.create = _staged_create

    finished = run_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=_local_config(),
        deepseek_client=client,
    )

    assert finished.latest_run_status == "completed"
    assert finished.edit_generation_status == "generated"
    assert finished.edit_bundle_validation_status == "valid"
    assert finished.edit_apply_status == "not_applied"
    assert (worktree / "app" / "dev" / "sample.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 1\n"
    assert (root / "app" / "dev" / "sample.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 1\n"
