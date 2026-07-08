from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
DB_PATH = DATA_DIR / "lewis.sqlite3"


def load_dotenv(path: Path | None = None) -> None:
    """Small .env loader to keep dependencies minimal."""
    env_path = path or ROOT_DIR / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str | None
    deepseek_model: str
    deepseek_base_url: str
    debug: bool
    user_name: str
    user_name_configured: bool
    database_path: Path
    documents_dir: Path
    max_tool_calls_per_request: int
    max_estimated_daily_cost: float
    price_per_1k_input_tokens: float
    price_per_1k_output_tokens: float
    deepseek_code_model: str | None = None
    developer_code_workspace: bool = False
    developer_workspace_root: Path | None = None
    # Freestyle model provider selection: "deepseek" (default) or a local
    # OpenAI-compatible endpoint ("local-openai"/"lmstudio"/"local").
    freestyle_provider: str = "deepseek"
    freestyle_local_only: bool = False
    local_llm_base_url: str | None = None
    local_llm_model: str | None = None
    local_llm_api_key: str | None = None
    local_llm_json_mode: bool = False
    # Dedicated Freestyle web UI flag. Enables the /dev/freestyle routes and
    # its "Developer" navigation link without turning on unrelated broad
    # debug functionality (LEWIS_DEBUG also still enables it, for existing
    # developer workflows). Every job-creation, apply, checks, refine,
    # cleanup, and recover action still requires its own explicit form
    # confirmation regardless of this flag.
    freestyle_web_ui: bool = False
    # Dedicated Coding Agents web UI flag. Enables the /dev/coding-agents
    # routes and navigation link for bounded read-only Codex/Claude CLI jobs.
    coding_agents_web_ui: bool = False
    coding_agent_repositories: tuple[Path, ...] = ()
    codex_cli_path: Path | None = None
    claude_cli_path: Path | None = None
    coding_agent_default_timeout_seconds: int = 600
    coding_agent_max_timeout_seconds: int = 3600

    @property
    def has_api_key(self) -> bool:
        return bool(self.deepseek_api_key)

    @property
    def freestyle_ui_enabled(self) -> bool:
        """Whether the Freestyle web UI (routes and nav link) is enabled."""
        return self.debug or self.freestyle_web_ui

    @property
    def coding_agents_ui_enabled(self) -> bool:
        """Whether the Coding Agents web UI (routes and nav link) is enabled."""
        return self.debug or self.coding_agents_web_ui


class DeveloperWorkspaceRootError(ValueError):
    """Raised when LEWIS_DEVELOPER_WORKSPACE_ROOT is set but unusable."""


MISSING_API_KEY_MESSAGE = (
    "DEEPSEEK_API_KEY is not set yet.\n\n"
    "To start Lewis, copy .env.example to .env and add your DeepSeek API key:\n"
    "  copy .env.example .env\n\n"
    "Then edit .env and set DEEPSEEK_API_KEY."
)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_text(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    text = value.strip()
    return text or None


def _env_optional_path(name: str) -> Path | None:
    text = _env_optional_text(name)
    return Path(text).expanduser() if text else None


def _env_path_tuple(name: str) -> tuple[Path, ...]:
    value = os.getenv(name)
    if not value:
        return ()
    paths: list[Path] = []
    for item in value.split(os.pathsep):
        text = item.strip().strip('"').strip("'")
        if text:
            paths.append(Path(text).expanduser())
    return tuple(paths)


def get_settings() -> Settings:
    load_dotenv()
    raw_user_name = os.getenv("LEWIS_USER_NAME")
    user_name = raw_user_name.strip() if raw_user_name else ""
    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        debug=_env_bool("LEWIS_DEBUG"),
        user_name=user_name or "the user",
        user_name_configured=bool(user_name),
        database_path=Path(os.getenv("LEWIS_DB_PATH", DB_PATH)),
        documents_dir=DOCUMENTS_DIR,
        max_tool_calls_per_request=_env_int("LEWIS_MAX_TOOL_CALLS", 8),
        max_estimated_daily_cost=_env_float("LEWIS_MAX_DAILY_COST_USD", 1.00),
        price_per_1k_input_tokens=_env_float("DEEPSEEK_PRICE_PER_1K_INPUT", 0.0),
        price_per_1k_output_tokens=_env_float("DEEPSEEK_PRICE_PER_1K_OUTPUT", 0.0),
        deepseek_code_model=_env_optional_text(
            "DEEPSEEK_CODE_MODEL",
            "deepseek-v4-pro",
        ),
        developer_code_workspace=_env_bool("LEWIS_DEVELOPER_CODE_WORKSPACE"),
        developer_workspace_root=_env_optional_path("LEWIS_DEVELOPER_WORKSPACE_ROOT"),
        freestyle_provider=os.getenv("LEWIS_FREESTYLE_PROVIDER", "deepseek"),
        freestyle_local_only=_env_bool("LEWIS_FREESTYLE_LOCAL_ONLY"),
        local_llm_base_url=_env_optional_text("LEWIS_LOCAL_BASE_URL"),
        local_llm_model=_env_optional_text("LEWIS_LOCAL_CODE_MODEL"),
        local_llm_api_key=_env_optional_text("LEWIS_LOCAL_API_KEY"),
        local_llm_json_mode=_env_bool("LEWIS_LOCAL_JSON_MODE"),
        freestyle_web_ui=_env_bool("LEWIS_FREESTYLE_UI"),
        coding_agents_web_ui=_env_bool("LEWIS_CODING_AGENTS_UI"),
        coding_agent_repositories=_env_path_tuple("LEWIS_CODING_AGENT_REPOSITORIES"),
        codex_cli_path=_env_optional_path("LEWIS_CODEX_CLI_PATH"),
        claude_cli_path=_env_optional_path("LEWIS_CLAUDE_CLI_PATH"),
        coding_agent_default_timeout_seconds=_env_int(
            "LEWIS_CODING_AGENT_DEFAULT_TIMEOUT_SECONDS", 600
        ),
        coding_agent_max_timeout_seconds=_env_int(
            "LEWIS_CODING_AGENT_MAX_TIMEOUT_SECONDS", 3600
        ),
    )


def resolve_developer_workspace_root(
    settings: Settings, default: Path | None = None
) -> Path:
    """Resolve the root developer code tools should operate on.

    When LEWIS_DEVELOPER_WORKSPACE_ROOT is unset, developer code tools keep
    inspecting Lewis itself (``default``, normally ``ROOT_DIR``). When it is
    set, the path is validated and used as-is; an external workspace that is
    missing or not a directory raises instead of silently falling back to
    Lewis, so a misconfigured path never traces or exposes the wrong repo.
    """
    if settings.developer_workspace_root is None:
        return default if default is not None else ROOT_DIR
    root = settings.developer_workspace_root
    if not root.exists() or not root.is_dir():
        raise DeveloperWorkspaceRootError(
            "LEWIS_DEVELOPER_WORKSPACE_ROOT is set to "
            f"{root} but that path does not exist or is not a directory."
        )
    return root.resolve(strict=True)
