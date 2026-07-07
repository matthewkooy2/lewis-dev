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
STRUCTURED_OUTPUT_TEXT = "text"
STRUCTURED_OUTPUT_JSON_OBJECT = "json_object"
STRUCTURED_OUTPUT_JSON_SCHEMA = "json_schema"
STRUCTURED_OUTPUT_MODES = frozenset(
    {
        STRUCTURED_OUTPUT_TEXT,
        STRUCTURED_OUTPUT_JSON_OBJECT,
        STRUCTURED_OUTPUT_JSON_SCHEMA,
    }
)
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
    structured_output_mode: str | None = None
    # Optional deterministic sampling seed passed through to OpenAI-compatible
    # endpoints that support it (LM Studio, vLLM). None omits the parameter.
    sampling_seed: int | None = None

    def __post_init__(self) -> None:
        mode = self.structured_output_mode
        if mode is None:
            if self.supports_json_response_format:
                mode = (
                    STRUCTURED_OUTPUT_JSON_SCHEMA
                    if self.provider == LOCAL_OPENAI_PROVIDER
                    else STRUCTURED_OUTPUT_JSON_OBJECT
                )
            else:
                mode = STRUCTURED_OUTPUT_TEXT
        mode = str(mode).strip().casefold()
        object.__setattr__(self, "structured_output_mode", mode)
        object.__setattr__(
            self,
            "supports_json_response_format",
            mode != STRUCTURED_OUTPUT_TEXT,
        )

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
                structured_output_mode=(
                    STRUCTURED_OUTPUT_JSON_SCHEMA
                    if _env_flag("LEWIS_LOCAL_JSON_MODE")
                    else STRUCTURED_OUTPUT_TEXT
                ),
                sampling_seed=_env_seed("LEWIS_LOCAL_SEED"),
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
    metadata["structured_output_mode"] = structured_output_mode_for(config)
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
        _validate_structured_output_mode(config)
        if config.local_only:
            raise FreestyleJobRefusedError(LOCAL_ONLY_REMOTE_REFUSAL)
        _validate_deepseek_fields(config)
        return
    if config.provider == LOCAL_OPENAI_PROVIDER:
        _validate_structured_output_mode(config)
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
    if not isinstance(config.base_url, str):
        raise FreestyleJobRefusedError(
            "DEEPSEEK_BASE_URL must use the official DeepSeek API endpoint."
        )
    parsed = urlparse(config.base_url)
    loopback_endpoint = _is_explicit_loopback_endpoint(parsed)
    if (
        len(model) > 120
        or model != config.model
        or (
            not loopback_endpoint
            and re.fullmatch(r"deepseek-[A-Za-z0-9._-]+", model, re.IGNORECASE) is None
        )
    ):
        raise FreestyleJobRefusedError("DEEPSEEK_CODE_MODEL must name a DeepSeek model.")
    if not loopback_endpoint and (
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
    if not _is_explicit_loopback_endpoint(parsed):
        raise FreestyleJobRefusedError(LOCAL_BASE_URL_MESSAGE)


def _is_explicit_loopback_endpoint(parsed) -> bool:
    return bool(
        parsed.scheme in {"http", "https"}
        and parsed.hostname in _LOOPBACK_HOSTNAMES
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment
        and parsed.path.rstrip("/") in {"", "/v1"}
    )


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
            structured_output_mode=(
                STRUCTURED_OUTPUT_JSON_SCHEMA
                if bool(getattr(settings, "local_llm_json_mode", False))
                else STRUCTURED_OUTPUT_TEXT
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


def _env_seed(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None


def sampling_parameters_for(
    config: Any, *, temperature: float | None = None
) -> dict[str, Any]:
    """Sampling parameters for one generation request.

    Temperature 0 by default for deterministic first attempts; a bounded
    override (clamped to [0, 1]) lets recovery attempts escape a temperature-0
    attractor that would otherwise re-emit the identical failed output. A
    configured seed is passed only for the local provider, where
    OpenAI-compatible servers (LM Studio, vLLM) accept it.
    """
    value: float | int = 0
    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
        value = min(1.0, max(0.0, float(temperature)))
    parameters: dict[str, Any] = {"temperature": value}
    seed = getattr(config, "sampling_seed", None)
    if (
        is_local_provider(config)
        and isinstance(seed, int)
        and not isinstance(seed, bool)
    ):
        parameters["seed"] = seed
    return parameters


def structured_output_mode_for(config: Any) -> str:
    """Return the explicit structured-output mode for one provider config."""
    mode = getattr(config, "structured_output_mode", None)
    if isinstance(mode, str) and mode.strip().casefold() in STRUCTURED_OUTPUT_MODES:
        resolved = mode.strip().casefold()
    elif getattr(config, "supports_json_response_format", False):
        if is_local_provider(config):
            resolved = STRUCTURED_OUTPUT_JSON_SCHEMA
        else:
            resolved = STRUCTURED_OUTPUT_JSON_OBJECT
    else:
        resolved = STRUCTURED_OUTPUT_TEXT
    if (
        resolved == STRUCTURED_OUTPUT_JSON_OBJECT
        and _uses_loopback_json_schema_response_format(config)
    ):
        return STRUCTURED_OUTPUT_JSON_SCHEMA
    return resolved


def response_format_for_schema(
    config: Any,
    *,
    name: str,
    schema: dict[str, Any],
    strict: bool = True,
) -> dict[str, Any] | None:
    """Build the provider-specific response_format payload for a JSON stage.

    ``None`` means explicit text mode. Local LM Studio-compatible endpoints use
    the JSON Schema form when strict structured output is enabled; DeepSeek keeps
    the original json_object shape unless configured otherwise.
    """
    mode = structured_output_mode_for(config)
    if mode == STRUCTURED_OUTPUT_TEXT:
        return None
    if mode == STRUCTURED_OUTPUT_JSON_OBJECT:
        return {"type": STRUCTURED_OUTPUT_JSON_OBJECT}
    if mode == STRUCTURED_OUTPUT_JSON_SCHEMA:
        return {
            "type": STRUCTURED_OUTPUT_JSON_SCHEMA,
            "json_schema": {
                "name": _safe_schema_name(name),
                "strict": bool(strict),
                "schema": schema,
            },
        }
    raise FreestyleJobRefusedError(
        f"Unsupported structured output mode {mode!r} for freestyle provider."
    )


def _safe_schema_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name).strip())[:64]
    return cleaned or "freestyle_stage"


def _uses_loopback_json_schema_response_format(config: Any) -> bool:
    """LM Studio-compatible loopback endpoints require json_schema, not json_object."""
    base_url = getattr(config, "base_url", None)
    if not isinstance(base_url, str) or not base_url:
        return False
    parsed = urlparse(base_url)
    return _is_explicit_loopback_endpoint(parsed)


def _validate_structured_output_mode(config: ProviderProposalConfig) -> None:
    mode = structured_output_mode_for(config)
    if mode not in STRUCTURED_OUTPUT_MODES:
        raise FreestyleJobRefusedError(
            "Freestyle structured output mode must be text, json_object, or "
            "json_schema."
        )
    if config.provider == LOCAL_OPENAI_PROVIDER and mode == STRUCTURED_OUTPUT_JSON_OBJECT:
        raise FreestyleJobRefusedError(
            "The local-openai provider does not use json_object strict output; "
            "set LEWIS_LOCAL_JSON_MODE for json_schema or leave it disabled for text."
        )
