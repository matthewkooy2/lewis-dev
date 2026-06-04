from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    reason: str = ""


class BudgetManager:
    def __init__(self, settings: Settings):
        self.settings = settings

    def can_make_tool_call(self, current_tool_calls: int) -> BudgetDecision:
        if current_tool_calls >= self.settings.max_tool_calls_per_request:
            return BudgetDecision(
                allowed=False,
                reason=f"Tool call limit reached ({self.settings.max_tool_calls_per_request}).",
            )
        return BudgetDecision(allowed=True)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1000) * self.settings.price_per_1k_input_tokens
        output_cost = (output_tokens / 1000) * self.settings.price_per_1k_output_tokens
        return round(input_cost + output_cost, 8)

    def can_spend(self, estimated_daily_cost: float) -> BudgetDecision:
        if estimated_daily_cost >= self.settings.max_estimated_daily_cost:
            return BudgetDecision(
                allowed=False,
                reason=(
                    f"Daily estimated cost budget reached "
                    f"(${estimated_daily_cost:.4f}/${self.settings.max_estimated_daily_cost:.4f})."
                ),
            )
        return BudgetDecision(allowed=True)
