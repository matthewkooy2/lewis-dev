from pathlib import Path

import pytest

from app.config import DOCUMENTS_DIR, MISSING_API_KEY_MESSAGE, Settings, get_settings
from app.llm import LewisAgent, build_system_prompt
from app.model_profiles import code_profile


def test_missing_api_key_message_is_friendly():
    assert "DEEPSEEK_API_KEY" in MISSING_API_KEY_MESSAGE
    assert ".env.example" in MISSING_API_KEY_MESSAGE
    assert ".env" in MISSING_API_KEY_MESSAGE


def test_agent_refuses_to_run_without_api_key(tmp_path):
    settings = Settings(
        deepseek_api_key=None,
        deepseek_model="test-model",
        deepseek_base_url="https://api.deepseek.com",
        debug=False,
        user_name="the user",
        user_name_configured=False,
        database_path=tmp_path / "test.sqlite3",
        documents_dir=Path("documents"),
        max_tool_calls_per_request=5,
        max_estimated_daily_cost=1.0,
        price_per_1k_input_tokens=0.0,
        price_per_1k_output_tokens=0.0,
    )

    agent = LewisAgent(settings=settings, tools=None, budget=None, logger=None)

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        agent.run("hello")


def test_system_prompt_has_no_assumed_user_name_without_config(tmp_path):
    settings = Settings(
        deepseek_api_key=None,
        deepseek_model="test-model",
        deepseek_base_url="https://api.deepseek.com",
        debug=False,
        user_name="the user",
        user_name_configured=False,
        database_path=tmp_path / "test.sqlite3",
        documents_dir=Path("documents"),
        max_tool_calls_per_request=5,
        max_estimated_daily_cost=1.0,
        price_per_1k_input_tokens=0.0,
        price_per_1k_output_tokens=0.0,
    )

    prompt = build_system_prompt(settings)

    assert "".join(("Em", "ily")) not in prompt
    assert "the user's supervised personal assistant" in prompt
    assert "with no name beneath it" in prompt


def test_documents_dir_ignores_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LEWIS_DOCUMENTS_DIR", str(tmp_path / "outside"))

    settings = get_settings()

    assert settings.documents_dir == DOCUMENTS_DIR


def test_settings_default_code_model_profile(monkeypatch):
    monkeypatch.setattr("app.config.load_dotenv", lambda: None)
    monkeypatch.delenv("DEEPSEEK_CODE_MODEL", raising=False)

    settings = get_settings()

    assert settings.deepseek_model == "deepseek-v4-flash"
    assert settings.deepseek_code_model == "deepseek-v4-pro"


def test_blank_code_model_profile_falls_back_to_default(tmp_path):
    settings = Settings(
        deepseek_api_key=None,
        deepseek_model="test-default-model",
        deepseek_base_url="https://api.deepseek.com",
        debug=False,
        user_name="the user",
        user_name_configured=False,
        database_path=tmp_path / "test.sqlite3",
        documents_dir=Path("documents"),
        max_tool_calls_per_request=5,
        max_estimated_daily_cost=1.0,
        price_per_1k_input_tokens=0.0,
        price_per_1k_output_tokens=0.0,
        deepseek_code_model="",
    )

    profile = code_profile(settings)

    assert profile.model == "test-default-model"
