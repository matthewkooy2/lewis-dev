from app.code_proposal_text_tool_fallback import (
    code_proposal_args_from_search_results,
    extract_project_search_queries,
)


def test_extract_project_search_queries_from_text_tool_blocks():
    text = """
    I'll search first.
    <search_project_files>
      <query>chat ui command center</query>
    </search_project_files>
    <search_project_files><query> chat interface </query></search_project_files>
    <search_project_files><query>chat interface</query></search_project_files>
    <search_project_files><query>command center ui chat</query></search_project_files>
    """

    assert extract_project_search_queries(text) == [
        "chat ui command center",
        "chat interface",
    ]


def test_code_proposal_args_use_search_result_paths_without_patch_or_execution():
    args = code_proposal_args_from_search_results(
        "Make a code proposal.",
        [
            {
                "name": "search_project_files",
                "result": {
                    "ok": True,
                    "results": [
                        {"path": "app/web_components.py"},
                        {"path": "app/web_components.py"},
                        {"path": "app/static/lewis.css"},
                    ],
                },
            }
        ],
    )

    assert args["target_surface"] == "command center web UI"
    assert args["relevant_files"] == ["app/web_components.py", "app/static/lewis.css"]
    assert "No files are edited" in args["proposed_approach"]
