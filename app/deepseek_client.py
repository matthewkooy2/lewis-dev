from __future__ import annotations

from openai import OpenAI


def create_deepseek_client(*, api_key: str, base_url: str) -> OpenAI:
    """Create the OpenAI-compatible SDK client.

    Despite the historical name, this serves any OpenAI-compatible endpoint:
    DeepSeek or an explicitly configured local server (LM Studio, llama.cpp,
    vLLM). The name is kept because callers and tests patch it by name.
    """
    return OpenAI(api_key=api_key, base_url=base_url)


# Provider-neutral alias for new call sites.
create_openai_compatible_client = create_deepseek_client
