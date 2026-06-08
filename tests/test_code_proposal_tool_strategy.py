from app.code_proposal_tool_strategy import (
    CODE_PROPOSAL_TOOL,
    PROJECT_SEARCH_TOOL,
    code_proposal_tool_strategy,
)
from app.project_search_discipline import is_code_proposal_request


REQUEST = (
    "Make a code proposal to add a small label to the command center chat UI, "
    "using project search first."
)


def test_search_first_code_proposal_requires_search_without_exposing_other_tools():
    strategy = code_proposal_tool_strategy(REQUEST, [])

    assert strategy.allowed_tool_names == frozenset({PROJECT_SEARCH_TOOL})
    assert strategy.required_tool_name == PROJECT_SEARCH_TOOL


def test_safe_project_search_evidence_requires_code_proposal_next():
    strategy = code_proposal_tool_strategy(
        REQUEST,
        [
            {
                "name": PROJECT_SEARCH_TOOL,
                "result": {"ok": True, "results": [{"path": "app/web_components.py"}]},
            }
        ],
    )

    assert strategy.allowed_tool_names == frozenset({CODE_PROPOSAL_TOOL})
    assert strategy.required_tool_name == CODE_PROPOSAL_TOOL


def test_plain_ui_change_request_requires_code_proposal_after_search_evidence():
    strategy = code_proposal_tool_strategy(
        (
            "add a small UI change to chat so that when youre thinking "
            "its like an animated character"
        ),
        [
            {
                "name": PROJECT_SEARCH_TOOL,
                "result": {
                    "ok": True,
                    "results": [{"path": "app/web_components.py"}],
                },
            }
        ],
    )

    assert strategy.allowed_tool_names == frozenset({CODE_PROPOSAL_TOOL})
    assert strategy.required_tool_name == CODE_PROPOSAL_TOOL


def test_two_empty_project_searches_disable_more_tools_for_partial_answer():
    search_record = {
        "name": PROJECT_SEARCH_TOOL,
        "result": {"ok": True, "results": []},
    }

    strategy = code_proposal_tool_strategy(REQUEST, [search_record, search_record])

    assert strategy.allowed_tool_names == frozenset()
    assert strategy.required_tool_name is None


def test_completed_code_proposal_disables_more_tools_for_the_final_answer():
    strategy = code_proposal_tool_strategy(
        REQUEST,
        [{"name": CODE_PROPOSAL_TOOL, "result": {"ok": True, "status": "pending"}}],
    )

    assert strategy.allowed_tool_names == frozenset()
    assert strategy.required_tool_name is None


def test_other_requests_keep_normal_tool_availability():
    strategy = code_proposal_tool_strategy("Search my notes for Alpha.", [])

    assert strategy.allowed_tool_names is None
    assert strategy.required_tool_name is None


def test_self_inspection_with_useful_project_evidence_disables_more_tools():
    strategy = code_proposal_tool_strategy(
        "Inspect Lewis usage tracking methods.",
        [
            {
                "name": PROJECT_SEARCH_TOOL,
                "result": {"ok": True, "results": [{"path": "app/llm.py"}]},
            }
        ],
    )

    assert strategy.allowed_tool_names == frozenset()
    assert strategy.required_tool_name is None


def test_explicit_code_proposal_without_project_search_requires_only_propose_code_change():
    prompt = (
        "I want you to create a code proposal that makes the text blue "
        "for this chat instead of black"
    )
    assert is_code_proposal_request(prompt)
    strategy = code_proposal_tool_strategy(prompt, tool_records=[])
    assert strategy.allowed_tool_names == frozenset({CODE_PROPOSAL_TOOL})
    assert strategy.required_tool_name == CODE_PROPOSAL_TOOL
