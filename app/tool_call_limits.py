from __future__ import annotations

from app.config import Settings
from app.model_profiles import ModelProfile


def effective_tool_call_limit(settings: Settings, profile: ModelProfile) -> int:
    del profile
    return max(0, settings.max_tool_calls_per_request)
