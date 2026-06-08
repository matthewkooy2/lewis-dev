from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.project_search_discipline import (
    MAX_PROJECT_SEARCHES_PER_TURN,
    has_useful_project_search_results,
    is_code_proposal_request,
    project_search_records,
)


PROJECT_SEARCH_TOOL = "search_project_files"
CODE_PROPOSAL_TOOL = "propose_code_change"


@dataclass(frozen=True)
class ToolStrategy:
    allowed_tool_names: frozenset[str] | None = None
    required_tool_name: str | None = None


def code_proposal_tool_strategy(
    user_input: str,
    tool_records: list[dict[str, Any]],
) -> ToolStrategy:
    searches = project_search_records(tool_records)
    if not searches and not _is_search_first_code_proposal(user_input):
        if is_code_proposal_request(user_input):
            return ToolStrategy(
                allowed_tool_names=frozenset({CODE_PROPOSAL_TOOL}),
                required_tool_name=CODE_PROPOSAL_TOOL,
            )
        return ToolStrategy()
    if searches and not is_code_proposal_request(user_input):
        if (
            has_useful_project_search_results(tool_records)
            or len(searches) >= MAX_PROJECT_SEARCHES_PER_TURN
        ):
            return ToolStrategy(allowed_tool_names=frozenset())
        return ToolStrategy(
            allowed_tool_names=frozenset({PROJECT_SEARCH_TOOL}),
            required_tool_name=PROJECT_SEARCH_TOOL,
        )
    if any(record.get("name") == CODE_PROPOSAL_TOOL for record in tool_records):
        return ToolStrategy(allowed_tool_names=frozenset())

    has_search_evidence = has_useful_project_search_results(tool_records)
    if len(searches) >= MAX_PROJECT_SEARCHES_PER_TURN and not has_search_evidence:
        return ToolStrategy(allowed_tool_names=frozenset())
    return ToolStrategy(
        allowed_tool_names=(
            frozenset({CODE_PROPOSAL_TOOL})
            if has_search_evidence
            else frozenset({PROJECT_SEARCH_TOOL})
        ),
        required_tool_name=CODE_PROPOSAL_TOOL
        if has_search_evidence
        else PROJECT_SEARCH_TOOL,
    )


def _is_search_first_code_proposal(user_input: str) -> bool:
    normalized = " ".join(user_input.lower().split())
    asks_for_code_proposal = "code proposal" in normalized
    asks_for_project_search = "project search" in normalized
    return asks_for_code_proposal and asks_for_project_search
