from pathlib import Path

from app.budget import BudgetManager
from app.config import Settings


def test_tool_call_limit_blocks_after_max():
    settings = Settings(
        deepseek_api_key=None,
        deepseek_model="test",
        deepseek_base_url="https://api.deepseek.com",
        debug=False,
        user_name="the user",
        user_name_configured=False,
        database_path=Path("test.sqlite3"),
        documents_dir=Path("documents"),
        max_tool_calls_per_request=5,
        max_estimated_daily_cost=1.0,
        price_per_1k_input_tokens=0.0,
        price_per_1k_output_tokens=0.0,
    )
    budget = BudgetManager(settings)

    assert budget.can_make_tool_call(4).allowed is True
    assert budget.can_make_tool_call(5).allowed is False


def test_cost_estimate_uses_configured_prices():
    settings = Settings(
        deepseek_api_key=None,
        deepseek_model="test",
        deepseek_base_url="https://api.deepseek.com",
        debug=False,
        user_name="the user",
        user_name_configured=False,
        database_path=Path("test.sqlite3"),
        documents_dir=Path("documents"),
        max_tool_calls_per_request=5,
        max_estimated_daily_cost=1.0,
        price_per_1k_input_tokens=0.10,
        price_per_1k_output_tokens=0.20,
    )

    assert BudgetManager(settings).estimate_cost(1000, 2000) == 0.5
