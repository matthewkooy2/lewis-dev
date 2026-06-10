from __future__ import annotations

from openai import OpenAI


def create_deepseek_client(*, api_key: str, base_url: str) -> OpenAI:
    """Create the OpenAI-compatible SDK client used only for DeepSeek."""
    return OpenAI(api_key=api_key, base_url=base_url)
