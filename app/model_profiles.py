from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import Settings


CODE_RELATED_TOOL_NAMES = frozenset({"propose_code_change", "search_project_files"})
CODE_PROFILE_NAME = "code"
DEFAULT_CHAT_PROFILE_NAME = "default_chat"

_CODE_INTENT_PATTERNS = (
    r"\bcode (?:change|proposal|review|plan|planning|help|work)\b",
    r"\b(?:implement|refactor|debug|fix|patch|diff|test) (?:lewis|the app|this app|code|a bug|bug|feature)\b",
    r"\b(?:add|build|change|update) .{0,80}\b(?:feature|toggle|setting|config|configuration|model profile|command center|web ui|ui)\b",
    r"\b(?:tests|pytest|ruff|frontend|backend|module|function|class|migration)\b",
    r"\b(?:propose|plan) .{0,80}\bcode\b",
)


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model: str


def default_chat_profile(settings: Settings) -> ModelProfile:
    return ModelProfile(name=DEFAULT_CHAT_PROFILE_NAME, model=settings.deepseek_model)


def code_profile(settings: Settings) -> ModelProfile:
    model = (settings.deepseek_code_model or "").strip()
    if not model:
        return default_chat_profile(settings)
    return ModelProfile(name=CODE_PROFILE_NAME, model=model)


def profile_for_user_input(settings: Settings, user_input: str) -> ModelProfile:
    if is_code_related_user_input(user_input):
        return code_profile(settings)
    return default_chat_profile(settings)


def profile_after_tool_call(
    settings: Settings,
    current_profile: ModelProfile,
    tool_name: str,
) -> ModelProfile:
    if tool_name in CODE_RELATED_TOOL_NAMES:
        return code_profile(settings)
    return current_profile


def is_code_related_user_input(user_input: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\s./_-]", " ", user_input.lower())
    normalized = " ".join(normalized.split())
    return any(re.search(pattern, normalized) for pattern in _CODE_INTENT_PATTERNS)
