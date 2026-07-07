"""Provider-neutral model configuration for freestyle generation slices.

Freestyle supports exactly two explicit providers and fails closed on
anything else:

- ``deepseek``: the original remote DeepSeek endpoint, with the original
  strict credential/model/endpoint validation preserved unchanged.
- ``local-openai``: one explicitly configured local OpenAI-compatible
  endpoint (LM Studio, llama.cpp, vLLM) bound to a loopback host. No real
  API key is required, remote hosts are refused, and there is no fallback
  path to any remote provider.

Provider selection is explicit configuration only. If the selected provider
is unavailable or misconfigured the job fails with a recorded error; nothing
here ever silently substitutes another provider.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any
from urllib.parse import urlparse

from app.dev.freestyle_coding import FreestyleJobRefusedError

DEEPSEEK_PROVIDER = "deepseek"
LOCAL_OPENAI_PROVIDER = "local-openai"
KNOWN_PROVIDERS = frozenset({DEEPSEEK_PROVIDER, LOCAL_OPENAI_PROVIDER})
_LOCAL_PROVIDER_ALIASES = frozenset(
    {"local", "local-openai", "lmstudio", "lm-studio", "openai-local"}
)

DEEPSEEK_API_BASE_URL = "https://api.deepseek.com"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_LOCAL_MODEL = "qwen2.5-coder-7b-instruct"
# The OpenAI SDK requires a non-empty api_key even when the endpoint ignores
# it; local endpoints get this placeholder when no key is configured.
LOCAL_PLACEHOLDER_API_KEY = "local-endpoint-no-key"
_LOOPBACK_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "::1"})
_LOCAL_MODEL_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@-]{0,119}")

UNKNOWN_PROVIDER_MESSAGE = (
    "Unknown freestyle model provider; set LEWIS_FREESTYLE_PROVIDER to "
    "'deepseek' or 'local-openai'."
)
LOCAL_ONLY_REMOTE_REFUSAL = (
    "LEWIS_FREESTYLE_LOCAL_ONLY is enabled; remote model providers are "
    "refused and no remote fallback is available."
)
LOCAL_BASE_URL_MESSAGE = (
    "LEWIS_LOCAL_BASE_URL must point at a loopback OpenAI-compatible "
    "endpoint such as http://127.0.0.1:1234/v1; non-loopback hosts are "
    "refused for the local provider."
)
LOCAL_MODEL_MESSAGE = (
    "LEWIS_LOCAL_CODE_MODEL must name the locally served model, for "
    "example qwen2.5-coder-7b-instruct."
)


@dataclass(frozen=True)
class ProviderProposalConfig:
    """One explicit provider selection for proposal/edit generation.

    Field order keeps positional compatibility with the original
    DeepSeek-only config (api_key, model, base_url).
    """

    api_key: str
    model: str
    base_url: str = DEEPSEEK_API_BASE_URL
    provider: str = DEEPSEEK_PROVIDER
    local_only: bool = False
    supports_json_response_format: bool = True

    @classmethod
    def from_process_environment(cls) -> ProviderProposalConfig:
        raw_provider = os.getenv("LEWIS_FREESTYLE_PROVIDER", DEEPSEEK_PROVIDER)
        provider = raw_provider.strip().casefold() or DEEPSEEK_PROVIDER
        local_only = _env_flag("LEWIS_FREESTYLE_LOCAL_ONLY")
        if provider in _LOCAL_PROVIDER_ALIASES:
            return cls(
                api_key=os.getenv("LEWIS_LOCAL_API_KEY", ""),
                model=os.getenv("LEWIS_LOCAL_CODE_MODEL", DEFAULT_LOCAL_MODEL),
                base_url=os.getenv("LEWIS_LOCAL_BASE_URL", DEFAULT_LOCAL_BASE_URL),
                provider=LOCAL_OPENAI_PROVIDER,
                # The local provider is always local-only: it can never fall
                # back to a remote endpoint regardless of the env flag.
                local_only=True,
                supports_json_response_format=_env_flag("LEWIS_LOCAL_JSON_MODE"),
            )
        if provider == DEEPSEEK_PROVIDER:
            return cls(
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                model=os.getenv("DEEPSEEK_CODE_MODEL", ""),
                base_url=os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_API_BASE_URL),
                provider=DEEPSEEK_PROVIDER,
                local_only=local_only,
            )
        # Unknown providers become a config that validation refuses, so the
        # failure is recorded through the normal fail-closed artifact path.
        return cls(
            api_key="",
            model="",
            base_url="",
            provider=provider,
            local_only=local_only,
        )


@dataclass(frozen=True)
class ProviderContextLimits:
    """Bounded prompt/response budgets, reduced for small local models."""

    max_context_files: int
    max_context_total_chars: int
    max_context_chars_per_file: int
    max_proposal_output_tokens: int
    max_edit_output_tokens: int
    max_edit_prompt_chars: int


_DEEPSEEK_LIMITS = ProviderContextLimits(
    max_context_files=12,
    max_context_total_chars=48_000,
    max_context_chars_per_file=8_000,
    max_proposal_output_tokens=10_000,
    max_edit_output_tokens=16_000,
    max_edit_prompt_chars=80_000,
)
_LOCAL_LIMITS = ProviderContextLimits(
    max_context_files=8,
    max_context_total_chars=24_000,
    max_context_chars_per_file=6_000,
    max_proposal_output_tokens=4_000,
    max_edit_output_tokens=8_000,
    max_edit_prompt_chars=40_000,
)


def context_limits_for(config: Any) -> ProviderContextLimits:
    if is_local_provider(config):
        return _LOCAL_LIMITS
    return _DEEPSEEK_LIMITS


def is_local_provider(config: Any) -> bool:
    return getattr(config, "provider", None) == LOCAL_OPENAI_PROVIDER


def effective_api_key(config: Any) -> str:
    """The key passed to the SDK client; local endpoints may omit a real key."""
    api_key = getattr(config, "api_key", "")
    if not isinstance(api_key, str):
        api_key = ""
    if is_local_provider(config) and not api_key.strip():
        return LOCAL_PLACEHOLDER_API_KEY
    return api_key


def provider_metadata(config: Any) -> dict[str, Any]:
    """Bounded, secret-free provider identity for artifacts and display."""
    provider = getattr(config, "provider", None)
    metadata: dict[str, Any] = {
        "provider": provider if isinstance(provider, str) and provider else "unknown"
    }
    base_url = getattr(config, "base_url", None)
    if isinstance(base_url, str) and base_url:
        metadata["endpoint"] = base_url[:200]
    if getattr(config, "local_only", False):
        metadata["local_only"] = True
    return metadata


def describe_provider_config(config: Any) -> str:
    """One secret-free line naming the effective provider, endpoint, model."""
    provider = getattr(config, "provider", None) or "unknown"
    base_url = getattr(config, "base_url", None) or "<unset>"
    model = getattr(config, "model", None) or "<unset>"
    parts = [f"provider={provider}", f"endpoint={base_url}", f"model={model}"]
    if getattr(config, "local_only", False):
        parts.append("local_only=true")
    return " ".join(str(part)[:220] for part in parts)


def validate_provider_config(config: Any) -> None:
    """Refuse any config that is not one fully valid explicit provider."""
    if not isinstance(config, ProviderProposalConfig):
        raise FreestyleJobRefusedError("DeepSeek proposal configuration is invalid.")
    if config.provider == DEEPSEEK_PROVIDER:
        if config.local_only:
            raise FreestyleJobRefusedError(LOCAL_ONLY_REMOTE_REFUSAL)
        _validate_deepseek_fields(config)
        return
    if config.provider == LOCAL_OPENAI_PROVIDER:
        _validate_local_fields(config)
        return
    raise FreestyleJobRefusedError(UNKNOWN_PROVIDER_MESSAGE)


def _validate_deepseek_fields(config: ProviderProposalConfig) -> None:
    if (
        not isinstance(config.api_key, str)
        or not config.api_key.strip()
        or config.api_key != config.api_key.strip()
        or len(config.api_key) > 512
        or any(ord(character) < 32 for character in config.api_key)
    ):
        raise FreestyleJobRefusedError(
            "DEEPSEEK_API_KEY must be set in the process environment."
        )
    if not isinstance(config.model, str):
        raise FreestyleJobRefusedError(
            "DEEPSEEK_CODE_MODEL must be set in the process environment."
        )
    model = config.model.strip()
    if not model:
        raise FreestyleJobRefusedError(
            "DEEPSEEK_CODE_MODEL must be set in the process environment."
        )
    if (
        len(model) > 120
        or model != config.model
        or re.fullmatch(r"deepseek-[A-Za-z0-9._-]+", model, re.IGNORECASE) is None
    ):
        raise FreestyleJobRefusedError("DEEPSEEK_CODE_MODEL must name a DeepSeek model.")
    if not isinstance(config.base_url, str):
        raise FreestyleJobRefusedError(
            "DEEPSEEK_BASE_URL must use the official DeepSeek API endpoint."
        )
    parsed = urlparse(config.base_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "api.deepseek.com"
        or parsed.port not in {None, 443}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") not in {"", "/v1"}
    ):
        raise FreestyleJobRefusedError(
            "DEEPSEEK_BASE_URL must use the official DeepSeek API endpoint."
        )


def _validate_local_fields(config: ProviderProposalConfig) -> None:
    if (
        not isinstance(config.api_key, str)
        or config.api_key != config.api_key.strip()
        or len(config.api_key) > 512
        or any(ord(character) < 32 for character in config.api_key)
    ):
        raise FreestyleJobRefusedError(
            "LEWIS_LOCAL_API_KEY contains invalid characters; leave it unset "
            "for a local endpoint that needs no key."
        )
    if (
        not isinstance(config.model, str)
        or config.model != config.model.strip()
        or _LOCAL_MODEL_NAME_PATTERN.fullmatch(config.model) is None
    ):
        raise FreestyleJobRefusedError(LOCAL_MODEL_MESSAGE)
    if not isinstance(config.base_url, str):
        raise FreestyleJobRefusedError(LOCAL_BASE_URL_MESSAGE)
    parsed = urlparse(config.base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in _LOOPBACK_HOSTNAMES
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") not in {"", "/v1"}
    ):
        raise FreestyleJobRefusedError(LOCAL_BASE_URL_MESSAGE)


def provider_config_from_settings(settings: Any) -> ProviderProposalConfig:
    """Build the effective provider config from loaded Lewis settings."""
    provider = (
        str(getattr(settings, "freestyle_provider", "") or DEEPSEEK_PROVIDER)
        .strip()
        .casefold()
    )
    local_only = bool(getattr(settings, "freestyle_local_only", False))
    if provider in _LOCAL_PROVIDER_ALIASES:
        return ProviderProposalConfig(
            api_key=str(getattr(settings, "local_llm_api_key", "") or ""),
            model=str(
                getattr(settings, "local_llm_model", "") or DEFAULT_LOCAL_MODEL
            ),
            base_url=str(
                getattr(settings, "local_llm_base_url", "") or DEFAULT_LOCAL_BASE_URL
            ),
            provider=LOCAL_OPENAI_PROVIDER,
            local_only=True,
            supports_json_response_format=bool(
                getattr(settings, "local_llm_json_mode", False)
            ),
        )
    if provider == DEEPSEEK_PROVIDER:
        return ProviderProposalConfig(
            api_key=str(getattr(settings, "deepseek_api_key", "") or ""),
            model=str(getattr(settings, "deepseek_code_model", "") or ""),
            base_url=str(
                getattr(settings, "deepseek_base_url", "") or DEEPSEEK_API_BASE_URL
            ),
            provider=DEEPSEEK_PROVIDER,
            local_only=local_only,
        )
    return ProviderProposalConfig(
        api_key="",
        model="",
        base_url="",
        provider=provider,
        local_only=local_only,
    )


def _env_flag(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().casefold() in {"1", "true", "yes", "on"}
