from __future__ import annotations

import copy
import os
import sqlite3
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.approval_service import approve_approval, reject_approval
from app.approvals import ApprovalStore
from app.budget import BudgetManager
from app.config import Settings
from app.llm import LewisAgent, parse_token_usage
from app.logging_db import RunLogger
from app.memory_service import archive_memory, propose_memory
from app.memory_store import MemoryStore
from app.project_search import ProjectFileSearch
from app.rag import DocumentSearch
from app.tools import ToolRegistry
from app.web import create_app


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        deepseek_api_key="test-key",
        deepseek_model="test-model",
        deepseek_base_url="https://api.deepseek.com",
        debug=False,
        user_name="the user",
        user_name_configured=False,
        database_path=tmp_path / "test.sqlite3",
        documents_dir=tmp_path / "documents",
        max_tool_calls_per_request=5,
        max_estimated_daily_cost=1.0,
        price_per_1k_input_tokens=0.0,
        price_per_1k_output_tokens=0.0,
        deepseek_code_model="test-code-model",
    )


def make_agent_from_settings(settings: Settings) -> LewisAgent:
    settings.documents_dir.mkdir()
    approvals = ApprovalStore(settings.database_path)
    tools = ToolRegistry(DocumentSearch(settings.documents_dir), approvals)
    return LewisAgent(
        settings=settings,
        tools=tools,
        budget=BudgetManager(settings),
        logger=RunLogger(settings.database_path),
    )


def make_agent_with_project_search(
    settings: Settings,
    project_root: Path,
    project_file_search: ProjectFileSearch | None = None,
) -> LewisAgent:
    settings.documents_dir.mkdir()
    approvals = ApprovalStore(settings.database_path)
    tools = ToolRegistry(
        DocumentSearch(settings.documents_dir),
        approvals,
        project_file_search=project_file_search or ProjectFileSearch(project_root),
    )
    return LewisAgent(
        settings=settings,
        tools=tools,
        budget=BudgetManager(settings),
        logger=RunLogger(settings.database_path),
    )


def make_agent(tmp_path: Path) -> LewisAgent:
    return make_agent_from_settings(make_settings(tmp_path))


def assert_no_unanswered_tool_calls(messages: list[dict[str, object]]) -> None:
    index = 0
    while index < len(messages):
        message = messages[index]
        tool_calls = message.get("tool_calls") or []
        if message.get("role") != "assistant" or not tool_calls:
            index += 1
            continue
        expected_ids = [str(call["id"]) for call in tool_calls]
        following = messages[index + 1 : index + 1 + len(expected_ids)]
        assert [item.get("role") for item in following] == ["tool"] * len(expected_ids)
        assert [str(item.get("tool_call_id")) for item in following] == expected_ids
        index += 1 + len(expected_ids)


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.models: list[object] = []
        self.count = 0

    def create(self, **kwargs: object) -> object:
        self.count += 1
        self.calls.append(copy.deepcopy(kwargs["messages"]))
        self.models.append(kwargs["model"])
        message = SimpleNamespace(content=f"reply {self.count}", tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )


class FakeClient:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class FailIfCalledCompletions:
    def create(self, **kwargs: object) -> object:
        del kwargs
        raise AssertionError(
            "The deterministic search-first proposal flow called the model."
        )


class FailIfCalledClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FailIfCalledCompletions())


class RecordingProjectFileSearch(ProjectFileSearch):
    def __init__(self, project_root: Path) -> None:
        super().__init__(project_root)
        self.queries: list[str] = []

    def search(self, query: str, limit: int = 10):
        self.queries.append(query)
        return super().search(query, limit)


class ToolCallingCompletions:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.models: list[object] = []
        self.count = 0

    def create(self, **kwargs: object) -> object:
        self.count += 1
        self.calls.append(copy.deepcopy(kwargs["messages"]))
        self.models.append(kwargs["model"])
        if self.count == 1:
            function = SimpleNamespace(
                name="search_documents",
                arguments='{"query": "Alpha"}',
            )
            tool_call = SimpleNamespace(
                id="call_1",
                function=function,
            )
            message = ToolCallingMessage(
                content=None,
                tool_calls=[tool_call],
            )
        else:
            message = ToolCallingMessage(
                content="Alpha appears in note.md.",
                tool_calls=None,
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )


class ToolCallingMessage:
    def __init__(
        self,
        content: str | None,
        tool_calls: list[object] | None,
        reasoning_content: str | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content

    def model_dump(self) -> dict[str, object]:
        message: dict[str, object] = {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls or []
            ],
        }
        if self.reasoning_content is not None:
            message["reasoning_content"] = self.reasoning_content
        return message


class ToolCallingClient:
    def __init__(self) -> None:
        self.completions = ToolCallingCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class ReasoningMemoryToolCallingCompletions:
    reasoning_content = "INTERNAL_REASONING_DO_NOT_PERSIST"

    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.count = 0

    def create(self, **kwargs: object) -> object:
        self.count += 1
        messages = copy.deepcopy(kwargs["messages"])
        self.calls.append(messages)
        if self.count == 1:
            function = SimpleNamespace(
                name="propose_memory",
                arguments=(
                    '{"memory_type": "preference", '
                    '"content": "Blake prefers concise summaries.", '
                    '"source": "chat", '
                    '"tags": ["summary"]}'
                ),
            )
            message = ToolCallingMessage(
                content=None,
                tool_calls=[
                    SimpleNamespace(id="call_memory_reasoning", function=function)
                ],
                reasoning_content=self.reasoning_content,
            )
        else:
            reasoning_messages = [
                item
                for item in messages
                if item.get("reasoning_content") == self.reasoning_content
            ]
            assert len(reasoning_messages) == 1
            assert reasoning_messages[0]["tool_calls"][0]["id"] == (
                "call_memory_reasoning"
            )
            message = ToolCallingMessage(
                content=(
                    "Created a pending memory approval."
                    if self.count == 2
                    else "You're welcome."
                ),
                tool_calls=None,
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )


class ReasoningMemoryToolCallingClient:
    def __init__(self) -> None:
        self.completions = ReasoningMemoryToolCallingCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class CodeProposalToolCallingCompletions:
    def __init__(self, final_answer: str) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.models: list[object] = []
        self.count = 0
        self.final_answer = final_answer

    def create(self, **kwargs: object) -> object:
        self.count += 1
        self.calls.append(copy.deepcopy(kwargs["messages"]))
        self.models.append(kwargs["model"])
        if self.count == 1:
            function = SimpleNamespace(
                name="propose_code_change",
                arguments=(
                    '{"requested_goal": "Add a Command Center stronger model toggle.", '
                    '"relevant_files": ["app/llm.py", "app/config.py"], '
                    '"proposed_approach": "Add runtime model profile selection for code-related chat turns.", '
                    '"safety_notes": ["Do not expose secrets."], '
                    '"suggested_test_commands": ["python -m pytest tests/test_llm_session.py"]}'
                ),
            )
            tool_call = SimpleNamespace(
                id="call_code_1",
                function=function,
            )
            message = ToolCallingMessage(
                content=None,
                tool_calls=[tool_call],
            )
        else:
            message = ToolCallingMessage(
                content=self.final_answer,
                tool_calls=None,
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )


class CodeProposalToolCallingClient:
    def __init__(self, final_answer: str) -> None:
        self.completions = CodeProposalToolCallingCompletions(final_answer)
        self.chat = SimpleNamespace(completions=self.completions)


class SearchGroundedCodeProposalCompletions:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.models: list[object] = []
        self.tool_names: list[list[str]] = []
        self.tool_choices: list[object] = []
        self.count = 0

    def create(self, **kwargs: object) -> object:
        self.count += 1
        self.calls.append(copy.deepcopy(kwargs["messages"]))
        self.models.append(kwargs["model"])
        self.tool_names.append(
            [item["function"]["name"] for item in kwargs.get("tools", [])]
        )
        self.tool_choices.append(copy.deepcopy(kwargs.get("tool_choice")))
        if self.count == 1:
            message = self._tool_message(
                "call_search_1",
                "search_project_files",
                '{"query": "command center chat UI label"}',
            )
        elif self.count == 2:
            message = self._tool_message(
                "call_code_1",
                "propose_code_change",
                (
                    '{"requested_goal": "Add a small label to the command center chat UI.", '
                    '"target_surface": "command center web UI", '
                    '"relevant_files": ["app/web_components.py", "app/static/lewis.css"], '
                    '"proposed_approach": "Use the project search results to plan a small review-only label in the command center chat UI.", '
                    '"safety_notes": ["Pending proposal only."], '
                    '"suggested_test_commands": ["python -m pytest tests/test_web.py"]}'
                ),
            )
        else:
            message = ToolCallingMessage(
                content="Saved a pending local code proposal.",
                tool_calls=None,
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )

    @staticmethod
    def _tool_message(
        call_id: str,
        name: str,
        arguments: str,
    ) -> ToolCallingMessage:
        function = SimpleNamespace(name=name, arguments=arguments)
        tool_call = SimpleNamespace(id=call_id, function=function)
        return ToolCallingMessage(content=None, tool_calls=[tool_call])


class SearchGroundedCodeProposalClient:
    def __init__(self) -> None:
        self.completions = SearchGroundedCodeProposalCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class TextProjectSearchCompletions:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.count = 0

    def create(self, **kwargs: object) -> object:
        self.count += 1
        self.calls.append(copy.deepcopy(kwargs["messages"]))
        message = ToolCallingMessage(
            content=(
                "I'll start with project searches.\n\n"
                "<search_project_files>\n"
                "  <query>chat ui command center</query>\n"
                "</search_project_files>\n\n"
                "<search_project_files>\n"
                "  <query>chat interface</query>\n"
                "</search_project_files>"
            ),
            tool_calls=None,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )


class TextProjectSearchClient:
    def __init__(self) -> None:
        self.completions = TextProjectSearchCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class DsmlProjectSearchCompletions:
    def __init__(self, second_content: str) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.tool_names: list[list[str]] = []
        self.count = 0
        self.second_content = second_content

    def create(self, **kwargs: object) -> object:
        self.count += 1
        self.calls.append(copy.deepcopy(kwargs["messages"]))
        self.tool_names.append(
            [item["function"]["name"] for item in kwargs.get("tools", [])]
        )
        if self.count == 1:
            content = (
                "<｜｜DSML｜｜tool_calls>\n"
                '<｜｜DSML｜｜invoke name="search_project_files">\n'
                '<｜｜DSML｜｜parameter name="query">usage tracking methods'
                "</｜｜DSML｜｜parameter>\n"
                "</｜｜DSML｜｜invoke>\n"
                "</｜｜DSML｜｜tool_calls>"
            )
        else:
            content = self.second_content
        message = ToolCallingMessage(content=content, tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )


class DsmlProjectSearchClient:
    def __init__(self, second_content: str) -> None:
        self.completions = DsmlProjectSearchCompletions(second_content)
        self.chat = SimpleNamespace(completions=self.completions)


class DsmlReadFileOnlyCompletions:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.tool_names: list[list[str]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(copy.deepcopy(kwargs["messages"]))
        self.tool_names.append(
            [item["function"]["name"] for item in kwargs.get("tools", [])]
        )
        message = ToolCallingMessage(
            content=(
                "<｜｜DSML｜｜tool_calls>\n"
                '<｜｜DSML｜｜invoke name="read_file">\n'
                '<｜｜DSML｜｜parameter name="path">app/llm.py'
                "</｜｜DSML｜｜parameter>\n"
                "</｜｜DSML｜｜invoke>\n"
                "</｜｜DSML｜｜tool_calls>"
            ),
            tool_calls=None,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )


class DsmlReadFileOnlyClient:
    def __init__(self) -> None:
        self.completions = DsmlReadFileOnlyCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class UnregisteredStructuredToolCompletions:
    def __init__(self) -> None:
        self.tool_names: list[list[str]] = []

    def create(self, **kwargs: object) -> object:
        self.tool_names.append(
            [item["function"]["name"] for item in kwargs.get("tools", [])]
        )
        function = SimpleNamespace(
            name="read_file",
            arguments='{"path": "app/llm.py"}',
        )
        message = ToolCallingMessage(
            content=None,
            tool_calls=[SimpleNamespace(id="call_read_file", function=function)],
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )


class UnregisteredStructuredToolClient:
    def __init__(self) -> None:
        self.completions = UnregisteredStructuredToolCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class EmptySearchThenCodeProposalCompletions:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.tool_names: list[list[str]] = []
        self.tool_choices: list[object] = []
        self.count = 0

    def create(self, **kwargs: object) -> object:
        self.count += 1
        self.calls.append(copy.deepcopy(kwargs["messages"]))
        self.tool_names.append(
            [item["function"]["name"] for item in kwargs.get("tools", [])]
        )
        self.tool_choices.append(copy.deepcopy(kwargs.get("tool_choice")))
        tool_names = [item["function"]["name"] for item in kwargs.get("tools", [])]
        required_name = (
            tool_names[0] if kwargs.get("tool_choice") == "required" else None
        )
        if required_name == "search_project_files":
            message = SearchGroundedCodeProposalCompletions._tool_message(
                f"call_search_{self.count}",
                "search_project_files",
                f'{{"query": "missing command center label {self.count}"}}',
            )
        elif required_name == "propose_code_change":
            message = SearchGroundedCodeProposalCompletions._tool_message(
                "call_code_1",
                "propose_code_change",
                (
                    '{"requested_goal": "Add a small label to the command center chat UI.", '
                    '"target_surface": "command center web UI", '
                    '"relevant_files": [], '
                    '"proposed_approach": "Create a pending review-only plan for a small command center chat UI label, noting that project search found no matching files.", '
                    '"safety_notes": ["Pending proposal only."]}'
                ),
            )
        else:
            message = ToolCallingMessage(
                content="Saved a pending local code proposal.",
                tool_calls=None,
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )


class EmptySearchThenCodeProposalClient:
    def __init__(self) -> None:
        self.completions = EmptySearchThenCodeProposalCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class RepeatingProjectSearchCompletions:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.models: list[object] = []
        self.count = 0

    def create(self, **kwargs: object) -> object:
        self.count += 1
        self.calls.append(copy.deepcopy(kwargs["messages"]))
        self.models.append(kwargs["model"])
        function = SimpleNamespace(
            name="search_project_files",
            arguments=f'{{"query": "search {self.count}"}}',
        )
        tool_call = SimpleNamespace(id=f"call_search_{self.count}", function=function)
        message = ToolCallingMessage(content=None, tool_calls=[tool_call])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )


class RepeatingProjectSearchClient:
    def __init__(self) -> None:
        self.completions = RepeatingProjectSearchCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class DuplicateProjectSearchBatchCompletions:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.count = 0

    def create(self, **kwargs: object) -> object:
        self.count += 1
        self.calls.append(copy.deepcopy(kwargs["messages"]))
        if self.count == 1:
            calls = []
            for index, query in enumerate(
                ("usage tracking methods", "tracking methods usage"),
                start=1,
            ):
                function = SimpleNamespace(
                    name="search_project_files",
                    arguments=f'{{"query": "{query}"}}',
                )
                calls.append(
                    SimpleNamespace(id=f"call_search_{index}", function=function)
                )
            message = ToolCallingMessage(content=None, tool_calls=calls)
        else:
            message = ToolCallingMessage(
                content=(
                    "Lewis parses usage counters in app/llm.py and records the "
                    "aggregated token totals on each run."
                ),
                tool_calls=None,
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )


class DuplicateProjectSearchBatchClient:
    def __init__(self) -> None:
        self.completions = DuplicateProjectSearchBatchCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class UsageTrackingSelfInspectionCompletions:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.tool_names: list[list[str]] = []
        self.count = 0

    def create(self, **kwargs: object) -> object:
        self.count += 1
        self.calls.append(copy.deepcopy(kwargs["messages"]))
        self.tool_names.append(
            [item["function"]["name"] for item in kwargs.get("tools", [])]
        )
        if self.count == 1:
            function = SimpleNamespace(
                name="search_project_files",
                arguments='{"query": "usage tracking methods"}',
            )
            tool_call = SimpleNamespace(id="call_search_1", function=function)
            message = ToolCallingMessage(content=None, tool_calls=[tool_call])
        else:
            message = ToolCallingMessage(
                content=(
                    "Lewis parses token usage in app/llm.py, accumulates input, "
                    "output, total, cache, and reasoning counters, estimates cost, "
                    "and logs those totals with each run."
                ),
                tool_calls=None,
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )


class UsageTrackingSelfInspectionClient:
    def __init__(self) -> None:
        self.completions = UsageTrackingSelfInspectionCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class TooLargeToolBatchCompletions:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.count = 0

    def create(self, **kwargs: object) -> object:
        self.count += 1
        self.calls.append(copy.deepcopy(kwargs["messages"]))
        calls = []
        for index in range(2):
            function = SimpleNamespace(
                name="search_documents",
                arguments=f'{{"query": "Alpha {index}"}}',
            )
            calls.append(SimpleNamespace(id=f"call_{index}", function=function))
        message = ToolCallingMessage(content=None, tool_calls=calls)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )


class TooLargeToolBatchClient:
    def __init__(self) -> None:
        self.completions = TooLargeToolBatchCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class TaskToolCallingCompletions:
    def __init__(self, final_answer: str) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.count = 0
        self.final_answer = final_answer

    def create(self, **kwargs: object) -> object:
        self.count += 1
        self.calls.append(copy.deepcopy(kwargs["messages"]))
        if self.count == 1:
            function = SimpleNamespace(
                name="propose_task",
                arguments=(
                    '{"title": "Call dentist", '
                    '"description": "Schedule a cleaning.", '
                    '"due_date_text": "tomorrow", '
                    '"priority": "high", '
                    '"source": "chat"}'
                ),
            )
            tool_call = SimpleNamespace(
                id="call_task_1",
                function=function,
            )
            message = ToolCallingMessage(
                content=None,
                tool_calls=[tool_call],
            )
        else:
            message = ToolCallingMessage(
                content=self.final_answer,
                tool_calls=None,
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )


class TaskToolCallingClient:
    def __init__(self, final_answer: str) -> None:
        self.completions = TaskToolCallingCompletions(final_answer)
        self.chat = SimpleNamespace(completions=self.completions)


class CalendarToolCallingCompletions:
    def __init__(self, final_answer: str) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.count = 0
        self.final_answer = final_answer

    def create(self, **kwargs: object) -> object:
        self.count += 1
        self.calls.append(copy.deepcopy(kwargs["messages"]))
        if self.count == 1:
            function = SimpleNamespace(
                name="propose_calendar_event",
                arguments=(
                    '{"title": "Project sync", '
                    '"description": "Discuss the launch checklist.", '
                    '"date_text": "next Tuesday", '
                    '"start_time_text": "2:00 PM", '
                    '"end_time_text": "2:30 PM", '
                    '"location": "Zoom", '
                    '"attendees": ["Alex", "Jordan"], '
                    '"source": "chat", '
                    '"notes": "Local proposal only."}'
                ),
            )
            tool_call = SimpleNamespace(
                id="call_calendar_1",
                function=function,
            )
            message = ToolCallingMessage(
                content=None,
                tool_calls=[tool_call],
            )
        else:
            message = ToolCallingMessage(
                content=self.final_answer,
                tool_calls=None,
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )


class CalendarToolCallingClient:
    def __init__(self, final_answer: str) -> None:
        self.completions = CalendarToolCallingCompletions(final_answer)
        self.chat = SimpleNamespace(completions=self.completions)


class MemoryToolCallingCompletions:
    def __init__(self, final_answer: str) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.count = 0
        self.final_answer = final_answer

    def create(self, **kwargs: object) -> object:
        self.count += 1
        self.calls.append(copy.deepcopy(kwargs["messages"]))
        if self.count == 1:
            function = SimpleNamespace(
                name="propose_memory",
                arguments=(
                    '{"memory_type": "preference", '
                    '"content": "Blake prefers concise implementation summaries.", '
                    '"source": "chat", '
                    '"tags": ["summary"]}'
                ),
            )
            tool_call = SimpleNamespace(
                id="call_memory_1",
                function=function,
            )
            message = ToolCallingMessage(
                content=None,
                tool_calls=[tool_call],
            )
        else:
            message = ToolCallingMessage(
                content=self.final_answer,
                tool_calls=None,
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )


class MemoryToolCallingClient:
    def __init__(self, final_answer: str) -> None:
        self.completions = MemoryToolCallingCompletions(final_answer)
        self.chat = SimpleNamespace(completions=self.completions)


def test_session_history_is_preserved_across_chat_turns(tmp_path):
    agent = make_agent(tmp_path)
    fake_client = FakeClient()
    agent.client = fake_client

    agent.run("Remember this detail.")
    agent.run("What did I ask you to remember?")

    second_messages = fake_client.completions.calls[1]

    assert {"role": "user", "content": "Remember this detail."} in second_messages
    assert {"role": "assistant", "content": "reply 1"} in second_messages
    assert second_messages[-1] == {
        "role": "user",
        "content": "What did I ask you to remember?",
    }


def test_normal_chat_uses_default_chat_model(tmp_path):
    agent = make_agent(tmp_path)
    fake_client = FakeClient()
    agent.client = fake_client

    run = agent.run("How should Lewis think about tasks?")

    assert run.answer == "reply 1"
    assert fake_client.completions.models == ["test-model"]


def test_deepseek_reasoning_tool_turn_continues_without_exposing_or_persisting_it(
    tmp_path,
):
    agent = make_agent(tmp_path)
    fake_client = ReasoningMemoryToolCallingClient()
    agent.client = fake_client

    run = agent.run("Please remember that I prefer concise summaries.")
    followup = agent.run("Thanks.")

    secret = fake_client.completions.reasoning_content
    assert fake_client.completions.count == 3
    assert secret not in run.answer
    assert secret not in followup.answer
    assert secret not in repr(run.tool_calls)

    store = ApprovalStore(agent.settings.database_path)
    approval = store.list(status="pending")[0]
    assert approval.action_type == "propose_memory"
    assert secret not in repr(approval.payload)
    approve_approval(store, approval.id)
    memory = MemoryStore(agent.settings.database_path).get_by_approval_id(approval.id)
    assert memory is not None
    assert secret not in repr(memory)

    with sqlite3.connect(agent.settings.database_path) as conn:
        table_names = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        ]
        persisted_rows = [
            row
            for table_name in table_names
            for row in conn.execute(f'SELECT * FROM "{table_name}"').fetchall()
        ]
    assert secret not in repr(persisted_rows)


def test_code_proposal_flow_uses_code_model_for_llm_calls(tmp_path):
    project_root = tmp_path / "project"
    (project_root / "app" / "static").mkdir(parents=True)
    (project_root / "app" / "web_components.py").write_text(
        "def command_center_chat():\n    return 'chat ui'\n",
        encoding="utf-8",
    )
    agent = make_agent_with_project_search(make_settings(tmp_path), project_root)
    agent.client = FailIfCalledClient()

    run = agent.run(
        "Make a code proposal to add a stronger model toggle to the command center chat, "
        "using project search first."
    )

    assert run.tool_calls[-1]["name"] == "propose_code_change"
    assert run.tool_calls[-1]["result"]["ok"] is True


def test_code_model_falls_back_to_default_when_unset(tmp_path):
    settings = replace(make_settings(tmp_path), deepseek_code_model="")
    project_root = tmp_path / "project"
    (project_root / "app" / "static").mkdir(parents=True)
    (project_root / "app" / "web_components.py").write_text(
        "def command_center_chat():\n    return 'chat ui'\n",
        encoding="utf-8",
    )
    agent = make_agent_with_project_search(settings, project_root)
    agent.client = FailIfCalledClient()

    run = agent.run(
        "Make a code proposal to add a stronger model toggle to the command center chat, "
        "using project search first."
    )

    assert run.tool_calls[-1]["name"] == "propose_code_change"


def test_exact_search_first_code_proposal_creates_real_visible_approval_without_runtime_side_effects(
    tmp_path, monkeypatch
):
    project_root = tmp_path / "project"
    (project_root / "app").mkdir(parents=True)
    target = project_root / "app" / "web_components.py"
    target.write_text(
        "def command_center_chat():\n    return 'chat ui'\n",
        encoding="utf-8",
    )
    (project_root / "app" / "static").mkdir()
    (project_root / "app" / "static" / "chat.js").write_text(
        "function renderMessage() {}\n",
        encoding="utf-8",
    )
    (project_root / "README.md").write_text(
        "command center chat UI label\n",
        encoding="utf-8",
    )
    (project_root / "app" / "code_proposal_helper.py").write_text(
        "COMMAND_CENTER_CHAT = 'chat UI label'\n",
        encoding="utf-8",
    )
    before = {
        path.relative_to(project_root).as_posix(): path.read_bytes()
        for path in project_root.rglob("*")
        if path.is_file()
    }
    search = RecordingProjectFileSearch(project_root)
    settings = make_settings(tmp_path)
    agent = make_agent_with_project_search(settings, project_root, search)
    agent.client = FailIfCalledClient()

    def fail_runtime_action(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError(
            "Lewis runtime attempted a forbidden shell or file-list action."
        )

    monkeypatch.setattr(agent.tools.document_search, "list_files", fail_runtime_action)
    monkeypatch.setattr(subprocess, "run", fail_runtime_action)
    monkeypatch.setattr(subprocess, "Popen", fail_runtime_action)
    monkeypatch.setattr(os, "system", fail_runtime_action)

    request = (
        "Make a code proposal to add a small label to the command center chat UI, "
        "using project search first."
    )
    run = agent.run(request)

    names = [record["name"] for record in run.tool_calls]
    assert names[-1] == "propose_code_change"
    assert names.count("search_project_files") == len(search.queries)
    assert 1 <= len(search.queries) <= 2
    assert "list_local_files" not in names
    proposal_result = run.tool_calls[-1]["result"]
    approval_id = proposal_result["approval_id"]
    approval = ApprovalStore(settings.database_path).get(approval_id)
    assert approval is not None
    assert approval.status == "pending"
    assert approval.action_type == "propose_code_change"
    assert (
        approval.payload["requested_goal"]
        == "Add a small label to the command center chat UI."
    )
    assert approval.payload["relevant_files"]
    assert "app/web_components.py" in approval.payload["relevant_files"]
    assert "app/static/chat.js" in approval.payload["relevant_files"]
    assert "README.md" not in approval.payload["relevant_files"]
    assert not any(
        "code_proposal" in path for path in approval.payload["relevant_files"]
    )
    assert "app/web_components.py" in approval.payload["proposed_approach"]
    assert approval.payload["patch_preview"] == (
        "diff --git a/app/web_components.py b/app/web_components.py\n"
        "--- a/app/web_components.py\n"
        "+++ b/app/web_components.py\n"
        "@@ -2,1 +2,1 @@\n"
        "-    return 'chat ui'\n"
        "+    return 'Chat: chat ui'"
    )
    assert approval.payload["patch_preview_path_classifications"] == [
        {
            "path": "app/web_components.py",
            "classification": "existing project file",
        }
    ]
    assert approval.payload["patch_preview_implementation_ready"] is True
    assert "app/static/chat.js" not in approval.payload["patch_preview"]
    assert "app/static/lewis.css" not in approval.payload["patch_preview"]
    assert f"#{approval_id}" in run.answer
    assert set(approval.payload["relevant_files"]) <= set(before)
    assert "app/static/lewis.css" not in approval.payload["relevant_files"]
    assert {
        path.relative_to(project_root).as_posix(): path.read_bytes()
        for path in project_root.rglob("*")
        if path.is_file()
    } == before
    response = TestClient(create_app(settings)).get("/approvals")
    assert response.status_code == 200
    assert f"#{approval_id}" in response.text
    assert "propose_code_change" in response.text
    assert_no_unanswered_tool_calls(agent.messages)


def test_exact_search_first_code_proposal_uses_at_most_two_searches_and_invents_no_files(
    tmp_path,
):
    project_root = tmp_path / "project"
    (project_root / "app").mkdir(parents=True)
    unrelated_file = project_root / "app" / "web_routes.py"
    unrelated_file.write_text("def routes():\n    return []\n", encoding="utf-8")
    original = unrelated_file.read_bytes()
    search = RecordingProjectFileSearch(project_root)
    agent = make_agent_with_project_search(
        make_settings(tmp_path), project_root, search
    )
    agent.client = FailIfCalledClient()

    run = agent.run(
        "Make a code proposal to add a small label to the command center chat UI, "
        "using project search first."
    )

    assert [record["name"] for record in run.tool_calls] == [
        "search_project_files",
        "search_project_files",
    ]
    assert "More grounding is needed" in run.answer
    assert "pending" not in run.answer.lower()
    assert ApprovalStore(agent.settings.database_path).list(status="pending") == []
    assert len(search.queries) == 2
    assert unrelated_file.read_bytes() == original
    assert_no_unanswered_tool_calls(agent.messages)


def test_text_project_search_blocks_create_pending_code_proposal(tmp_path):
    project_root = tmp_path / "project"
    (project_root / "app").mkdir(parents=True)
    target = project_root / "app" / "web_components.py"
    original = "def command_center_chat():\n    return 'chat ui'\n"
    target.write_text(original, encoding="utf-8")
    agent = make_agent_with_project_search(make_settings(tmp_path), project_root)
    agent.client = TextProjectSearchClient()

    run = agent.run(
        "Plan a code proposal to add a small label to the command center chat UI "
        "with project search."
    )

    assert [record["name"] for record in run.tool_calls] == [
        "search_project_files",
        "propose_code_change",
    ]
    assert run.tool_calls[-1]["result"]["status"] == "pending"
    assert "<search_project_files>" not in run.answer
    assert "Saved pending code proposal" in run.answer
    approval = ApprovalStore(agent.settings.database_path).list(status="pending")[0]
    assert "app/web_components.py" in approval.payload["patch_preview"]
    assert approval.payload["patch_preview_path_classifications"] == [
        {
            "path": "app/web_components.py",
            "classification": "existing project file",
        }
    ]
    assert target.read_text(encoding="utf-8") == original
    assert_no_unanswered_tool_calls(agent.messages)


def test_text_project_search_fallback_preserves_zero_tool_call_cap(tmp_path):
    settings = replace(make_settings(tmp_path), max_tool_calls_per_request=0)
    project_root = tmp_path / "project"
    project_root.mkdir()
    agent = make_agent_with_project_search(settings, project_root)
    agent.client = TextProjectSearchClient()

    run = agent.run(
        "Plan a code proposal to add a small label to the command center chat UI "
        "with project search."
    )

    assert run.tool_calls == [
        {
            "name": "tool_limit",
            "result": "Tool call limit reached (0). I stopped before making more tool calls.",
        }
    ]
    assert ApprovalStore(agent.settings.database_path).list(status="pending") == []
    assert_no_unanswered_tool_calls(agent.messages)


def test_text_project_search_fallback_returns_partial_answer_on_weak_evidence(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    agent = make_agent_with_project_search(make_settings(tmp_path), project_root)
    agent.client = TextProjectSearchClient()

    run = agent.run(
        "Plan a code proposal to add a small label to the command center chat UI "
        "with project search."
    )

    assert [record["name"] for record in run.tool_calls] == [
        "search_project_files",
        "search_project_files",
    ]
    assert "did not find strong matching evidence" in run.answer
    assert "tool_limit" not in [record["name"] for record in run.tool_calls]
    assert ApprovalStore(agent.settings.database_path).list(status="pending") == []
    assert_no_unanswered_tool_calls(agent.messages)


def test_repeated_near_duplicate_project_searches_are_suppressed(tmp_path):
    project_root = tmp_path / "project"
    (project_root / "app").mkdir(parents=True)
    (project_root / "app" / "llm.py").write_text(
        "def usage_tracking_methods():\n    return ['input_tokens']\n",
        encoding="utf-8",
    )
    search = RecordingProjectFileSearch(project_root)
    agent = make_agent_with_project_search(
        make_settings(tmp_path),
        project_root,
        search,
    )
    agent.client = DuplicateProjectSearchBatchClient()

    run = agent.run("Inspect Lewis usage tracking methods.")

    assert search.queries == ["usage tracking methods"]
    assert [record["name"] for record in run.tool_calls] == [
        "search_project_files",
        "search_project_files",
    ]
    assert run.tool_calls[1]["result"]["suppressed"] is True
    assert "tool_limit" not in [record["name"] for record in run.tool_calls]
    assert "app/llm.py" in run.answer
    assert_no_unanswered_tool_calls(agent.messages)


def test_usage_tracking_self_inspection_does_not_hit_tool_limit(tmp_path):
    project_root = tmp_path / "project"
    (project_root / "app").mkdir(parents=True)
    (project_root / "app" / "llm.py").write_text(
        "def parse_token_usage(usage):\n    return usage.total_tokens\n",
        encoding="utf-8",
    )
    search = RecordingProjectFileSearch(project_root)
    agent = make_agent_with_project_search(
        make_settings(tmp_path),
        project_root,
        search,
    )
    agent.client = UsageTrackingSelfInspectionClient()

    run = agent.run("could you inspect your usage tracking methods")

    assert search.queries == ["usage tracking methods"]
    assert [record["name"] for record in run.tool_calls] == ["search_project_files"]
    assert "tool_limit" not in [record["name"] for record in run.tool_calls]
    assert "app/llm.py" in run.answer
    assert "Project search found this relevant evidence:" not in run.answer
    assert (
        "accumulates input, output, total, cache, and reasoning counters" in run.answer
    )
    assert agent.client.completions.count == 2
    assert "list_local_files" not in agent.client.completions.tool_names[0]
    assert agent.client.completions.tool_names[1] == []
    assert_no_unanswered_tool_calls(agent.messages)


def test_plain_ui_change_request_creates_pending_review_only_proposal_without_runtime_side_effects(
    tmp_path, monkeypatch
):
    project_root = tmp_path / "project"
    (project_root / "app" / "static").mkdir(parents=True)
    target = project_root / "app" / "web_components.py"
    target.write_text(
        "def command_center_chat():\n    return 'chat ui thinking state'\n",
        encoding="utf-8",
    )
    before = target.read_bytes()
    search = RecordingProjectFileSearch(project_root)
    agent = make_agent_with_project_search(
        make_settings(tmp_path), project_root, search
    )
    agent.client = FailIfCalledClient()

    def fail_runtime_action(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Lewis runtime attempted a forbidden shell action.")

    monkeypatch.setattr(subprocess, "run", fail_runtime_action)
    monkeypatch.setattr(subprocess, "Popen", fail_runtime_action)
    monkeypatch.setattr(os, "system", fail_runtime_action)

    run = agent.run(
        "add a small UI change to chat so that when youre thinking "
        "its like an animated character"
    )

    names = [record["name"] for record in run.tool_calls]
    assert names[-1] == "propose_code_change"
    assert names.count("search_project_files") == len(search.queries)
    assert "tool_limit" not in names
    assert 1 <= len(search.queries) <= 2
    assert target.read_bytes() == before
    approval = ApprovalStore(agent.settings.database_path).list(status="pending")[0]
    assert approval.action_type == "propose_code_change"
    assert approval.status == "pending"
    assert "review-only" in approval.payload["proposed_approach"]
    assert_no_unanswered_tool_calls(agent.messages)


def test_dsml_project_search_markup_is_executed_and_never_displayed(tmp_path):
    project_root = tmp_path / "project"
    (project_root / "app").mkdir(parents=True)
    (project_root / "app" / "llm.py").write_text(
        "def parse_token_usage(usage):\n    return usage.total_tokens\n",
        encoding="utf-8",
    )
    agent = make_agent_with_project_search(make_settings(tmp_path), project_root)
    agent.client = DsmlProjectSearchClient(
        "Lewis parses usage totals in app/llm.py and logs them for each run."
    )

    run = agent.run("Inspect Lewis usage tracking methods.")

    assert [record["name"] for record in run.tool_calls] == ["search_project_files"]
    assert "Lewis parses usage totals" in run.answer
    assert "DSML" not in run.answer
    assert "tool_calls" not in run.answer
    assert "<｜｜" not in run.answer
    assert "read_file" not in run.answer
    assert_no_unanswered_tool_calls(agent.messages)
    assistant_text = "\n".join(
        str(message.get("content"))
        for message in agent.messages
        if message.get("role") == "assistant"
    )
    assert "DSML" not in assistant_text
    assert "<｜｜" not in assistant_text


def test_dsml_read_file_after_project_search_returns_synthesized_answer(tmp_path):
    project_root = tmp_path / "project"
    (project_root / "app").mkdir(parents=True)
    (project_root / "app" / "llm.py").write_text(
        "def parse_token_usage(usage):\n"
        "    usage_totals = usage.total_tokens\n"
        "    return usage_totals\n",
        encoding="utf-8",
    )
    agent = make_agent_with_project_search(make_settings(tmp_path), project_root)
    agent.client = DsmlProjectSearchClient(
        "<｜｜DSML｜｜tool_calls>\n"
        '<｜｜DSML｜｜invoke name="read_file">\n'
        '<｜｜DSML｜｜parameter name="path">app/llm.py</｜｜DSML｜｜parameter>\n'
        "</｜｜DSML｜｜invoke>\n"
        "</｜｜DSML｜｜tool_calls>"
    )

    run = agent.run("Inspect Lewis usage tracking methods.")

    assert [record["name"] for record in run.tool_calls] == ["search_project_files"]
    assert "Based on the bounded read-only project search" in run.answer
    assert "app/llm.py" in run.answer
    assert "usage_totals" in run.answer
    assert "Project search found this relevant evidence:" not in run.answer
    assert "read_file" not in run.answer
    assert "DSML" not in run.answer
    assert_no_unanswered_tool_calls(agent.messages)


def test_unregistered_read_file_text_tool_is_not_exposed_or_executed(tmp_path):
    project_root = tmp_path / "project"
    (project_root / "app").mkdir(parents=True)
    (project_root / "app" / "llm.py").write_text("secret = False\n", encoding="utf-8")
    agent = make_agent_with_project_search(make_settings(tmp_path), project_root)
    agent.client = DsmlReadFileOnlyClient()

    tool_names = {schema["function"]["name"] for schema in agent.tools.schemas()}
    run = agent.run("Please read app/llm.py")

    assert "read_file" not in tool_names
    assert run.tool_calls == []
    assert "read_file" not in run.answer
    assert "DSML" not in run.answer
    assert "unavailable tool" in run.answer
    assert_no_unanswered_tool_calls(agent.messages)


def test_unregistered_structured_tool_call_is_not_exposed_or_executed(tmp_path):
    agent = make_agent(tmp_path)
    agent.client = UnregisteredStructuredToolClient()

    run = agent.run("Please inspect app/llm.py")

    assert "read_file" not in agent.client.completions.tool_names[0]
    assert run.tool_calls == []
    assert "read_file" not in run.answer
    assert "unavailable tool" in run.answer
    assert_no_unanswered_tool_calls(agent.messages)


def test_search_first_code_proposal_does_not_hit_tool_limit_and_stays_pending(
    tmp_path,
):
    project_root = tmp_path / "project"
    (project_root / "app" / "static").mkdir(parents=True)
    target = project_root / "app" / "web_components.py"
    target.write_text(
        "def command_center_chat():\n    return 'chat ui'\n",
        encoding="utf-8",
    )
    (project_root / "app" / "static" / "lewis.css").write_text(
        ".chat-ui {}\n",
        encoding="utf-8",
    )
    agent = make_agent_with_project_search(make_settings(tmp_path), project_root)
    agent.client = SearchGroundedCodeProposalClient()

    run = agent.run(
        "Plan a code proposal to add a small label to the command center chat UI "
        "with project search."
    )

    assert [record["name"] for record in run.tool_calls] == [
        "search_project_files",
        "propose_code_change",
    ]
    assert "tool_limit" not in [record["name"] for record in run.tool_calls]
    assert run.tool_calls[-1]["result"]["status"] == "pending"
    assert agent.client.completions.tool_names[:2] == [
        ["search_project_files"],
        ["propose_code_change"],
    ]
    assert target.read_text(encoding="utf-8") == (
        "def command_center_chat():\n    return 'chat ui'\n"
    )
    assert_no_unanswered_tool_calls(agent.messages)


def test_code_tool_limit_still_caps_repeated_project_searches(tmp_path):
    settings = replace(make_settings(tmp_path), max_tool_calls_per_request=1)
    project_root = tmp_path / "project"
    (project_root / "app").mkdir(parents=True)
    (project_root / "app" / "web_components.py").write_text(
        "command center chat UI",
        encoding="utf-8",
    )
    agent = make_agent_with_project_search(settings, project_root)
    fake_client = RepeatingProjectSearchClient()
    agent.client = fake_client

    run = agent.run(
        "Plan a code proposal to add a small label to the command center chat UI "
        "with project search."
    )

    assert [record["name"] for record in run.tool_calls] == [
        "search_project_files",
        "tool_limit",
    ]
    assert "Tool call limit reached (1)" in run.answer
    assert "search_project_files" in run.answer
    assert ApprovalStore(agent.settings.database_path).list(status="pending") == []
    assert_no_unanswered_tool_calls(agent.messages)


def test_tool_limit_exit_does_not_persist_unanswered_tool_call_batch(tmp_path):
    settings = replace(make_settings(tmp_path), max_tool_calls_per_request=1)
    agent = make_agent_from_settings(settings)
    agent.client = TooLargeToolBatchClient()

    run = agent.run("Search my notes for Alpha and summarize them.")

    assert run.tool_calls == [
        {
            "name": "tool_limit",
            "result": "Tool call limit reached (1). I stopped before making more tool calls.",
        }
    ]
    assert not any(message.get("tool_calls") for message in agent.messages)
    assert_no_unanswered_tool_calls(agent.messages)


def test_chat_model_context_includes_active_memories_transiently(tmp_path):
    agent = make_agent(tmp_path)
    store = ApprovalStore(agent.settings.database_path)
    active = propose_memory(
        memory_type="preference",
        content="Blake prefers concise implementation summaries.",
        source="test",
        tags=["summary"],
        db_path=agent.settings.database_path,
    )
    pending = propose_memory(
        memory_type="preference",
        content="Pending proposal should not enter chat context.",
        source="test",
        db_path=agent.settings.database_path,
    )
    rejected = propose_memory(
        memory_type="fact",
        content="Rejected proposal should not enter chat context.",
        source="test",
        db_path=agent.settings.database_path,
    )
    approve_approval(store, active.id)
    reject_approval(store, rejected.id)
    fake_client = FakeClient()
    agent.client = fake_client

    run = agent.run("What style should implementation summaries use?")

    messages = fake_client.completions.calls[0]
    memory_contexts = [
        item["content"]
        for item in messages
        if item["role"] == "system"
        and "Approved active local memories" in str(item["content"])
    ]
    assert len(memory_contexts) == 1
    assert "Blake prefers concise implementation summaries." in memory_contexts[0]
    assert pending.payload["content"] not in memory_contexts[0]
    assert rejected.payload["content"] not in memory_contexts[0]
    assert "Blake prefers concise implementation summaries." not in "\n".join(
        str(item["content"]) for item in agent.messages
    )
    assert run.memory_context_count == 1


def test_injected_memory_context_is_not_persisted_into_chat_history(tmp_path):
    agent = make_agent(tmp_path)
    item = propose_memory(
        memory_type="preference",
        content="Blake likes focused implementation notes.",
        source="test",
        db_path=agent.settings.database_path,
    )
    approve_approval(ApprovalStore(agent.settings.database_path), item.id)
    fake_client = FakeClient()
    agent.client = fake_client

    agent.run("What implementation note style should you use?")
    agent.run("Thanks.")

    durable_history = "\n".join(str(item["content"]) for item in agent.messages)
    assert "Approved active local memories" not in durable_history
    assert "Blake likes focused implementation notes." not in durable_history
    assert {"role": "assistant", "content": "reply 1"} in agent.messages
    assert fake_client.completions.calls[1][-1] == {
        "role": "user",
        "content": "Thanks.",
    }


def test_tool_limit_with_memory_context_does_not_persist_memory_metadata(tmp_path):
    settings = replace(make_settings(tmp_path), max_tool_calls_per_request=0)
    agent = make_agent_from_settings(settings)
    item = propose_memory(
        memory_type="preference",
        content="Blake likes focused implementation notes.",
        source="test",
        db_path=agent.settings.database_path,
    )
    approve_approval(ApprovalStore(agent.settings.database_path), item.id)
    agent.client = ToolCallingClient()

    run = agent.run("What implementation note style should you use?")

    assert run.memory_context_count == 1
    assert run.tool_calls == [
        {
            "name": "tool_limit",
            "result": "Tool call limit reached (0). I stopped before making more tool calls.",
        }
    ]
    with sqlite3.connect(agent.settings.database_path) as conn:
        user_input, tool_calls = conn.execute(
            "SELECT user_input, tool_calls FROM runs"
        ).fetchone()
    stored_run = f"{user_input}\n{tool_calls}"
    assert "memory_context_count" not in stored_run
    assert "Approved active local memories" not in stored_run
    assert "Blake likes focused implementation notes." not in stored_run


def test_archived_memory_is_not_injected_after_archiving(tmp_path):
    agent = make_agent(tmp_path)
    item = propose_memory(
        memory_type="project_context",
        content="Lewis used an older dashboard label.",
        source="test",
        db_path=agent.settings.database_path,
    )
    approve_approval(ApprovalStore(agent.settings.database_path), item.id)
    memory = MemoryStore(agent.settings.database_path).get_by_approval_id(item.id)
    archive_memory(agent.settings.database_path, memory.id, reason="Outdated")
    fake_client = FakeClient()
    agent.client = fake_client

    agent.run("What label should Lewis use?")

    model_context = "\n".join(
        str(message["content"]) for message in fake_client.completions.calls[0]
    )
    assert "Approved active local memories" not in model_context
    assert "older dashboard label" not in model_context
    assert agent.run("Thanks.").memory_context_count == 0


def test_chat_executes_search_documents_tool_call(tmp_path):
    agent = make_agent(tmp_path)
    docs = agent.settings.documents_dir
    (docs / "note.md").write_text("Project Alpha design note.", encoding="utf-8")
    fake_client = ToolCallingClient()
    agent.client = fake_client

    run = agent.run("Search my notes for Alpha")

    assert run.answer == "Alpha appears in note.md."
    assert run.tool_calls[0]["name"] == "search_documents"
    assert run.tool_calls[0]["result"]["results"][0]["path"] == "note.md"
    tool_message = fake_client.completions.calls[1][-1]
    assert tool_message["role"] == "tool"
    assert "Project Alpha" in tool_message["content"]


def test_explicit_remember_chat_creates_pending_memory_approval(tmp_path):
    agent = make_agent(tmp_path)
    agent.client = MemoryToolCallingClient("I've remembered that.")

    run = agent.run("Remember that Blake prefers concise implementation summaries.")

    approvals = [
        item
        for item in ApprovalStore(agent.settings.database_path).list(status="pending")
        if item.action_type == "propose_memory"
    ]
    assert len(approvals) == 1
    assert approvals[0].payload == {
        "memory_type": "preference",
        "content": "Blake prefers concise implementation summaries.",
        "source": "chat",
        "tags_json": ["summary"],
    }
    assert MemoryStore(agent.settings.database_path).list_active() == []
    assert run.tool_calls[0]["name"] == "propose_memory"
    assert run.tool_calls[0]["result"]["status"] == "pending"
    assert "pending memory approval" in run.answer
    assert "Approval is required" in run.answer
    assert "remembered" not in run.answer.lower()


def test_ordinary_chat_cannot_create_memory_proposal(tmp_path):
    agent = make_agent(tmp_path)
    agent.client = MemoryToolCallingClient("Got it.")

    run = agent.run("I prefer concise implementation summaries.")

    assert ApprovalStore(agent.settings.database_path).list(status="pending") == []
    assert MemoryStore(agent.settings.database_path).list_active() == []
    assert run.tool_calls[0]["name"] == "propose_memory"
    assert run.tool_calls[0]["result"]["ok"] is False
    assert "explicit user request" in run.tool_calls[0]["result"]["error"]


def test_parse_token_usage_includes_cache_fields():
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=25,
        total_tokens=125,
        prompt_cache_hit_tokens=80,
        prompt_cache_miss_tokens=20,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=7),
    )

    parsed = parse_token_usage(usage)

    assert parsed.prompt_tokens == 100
    assert parsed.completion_tokens == 25
    assert parsed.total_tokens == 125
    assert parsed.prompt_cache_hit_tokens == 80
    assert parsed.prompt_cache_miss_tokens == 20
    assert parsed.reasoning_tokens == 7


def test_parse_token_usage_omits_missing_cache_fields():
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
    }

    parsed = parse_token_usage(usage)

    assert parsed.prompt_tokens == 10
    assert parsed.completion_tokens == 4
    assert parsed.total_tokens == 14
    assert parsed.prompt_cache_hit_tokens is None
    assert parsed.prompt_cache_miss_tokens is None
    assert parsed.reasoning_tokens is None


def test_send_that_email_references_latest_email_approval(tmp_path):
    agent = make_agent(tmp_path)
    agent.latest_approval_id = 42
    agent.latest_email_draft_approval_id = 42

    run = agent.run("looks good send it")

    assert "#42" in run.answer
    assert "python -m app.main approvals" in run.answer
    assert "python -m app.main approve 42" in run.answer


def test_chat_execute_approved_request_explains_dry_run_only(tmp_path):
    agent = make_agent(tmp_path)
    fake_client = FakeClient()
    agent.client = fake_client

    run = agent.run("Execute approved item 12")

    assert fake_client.completions.count == 0
    assert "Real execution is not available yet" in run.answer
    assert "dry-run execution preview" in run.answer
    assert ".\\.venv\\Scripts\\python.exe -m app.main execute-approved 12" in run.answer
    assert "does not send email" in run.answer
    assert "touch external systems" in run.answer


def test_send_that_email_does_not_claim_email_was_sent(tmp_path):
    agent = make_agent(tmp_path)
    agent.latest_email_draft_approval_id = 7

    run = agent.run("send that email")

    assert "can't send emails directly" in run.answer
    assert "sent" not in run.answer.lower()
    assert "does not send externally" in run.answer


def test_remind_me_creates_pending_task_proposal_not_real_reminder(tmp_path):
    agent = make_agent(tmp_path)
    agent.client = TaskToolCallingClient("I'll notify you tomorrow.")

    run = agent.run("Remind me to call the dentist tomorrow.")

    tasks = [
        item
        for item in ApprovalStore(agent.settings.database_path).list(status="pending")
        if item.action_type == "propose_task"
    ]
    assert len(tasks) == 1
    assert tasks[0].payload == {
        "title": "Call dentist",
        "description": "Schedule a cleaning.",
        "due_date_text": "tomorrow",
        "priority": "high",
        "source": "chat",
    }
    assert "not a real reminder" in run.answer
    assert "pending task approval item" in run.answer
    assert "marks the task proposal as approved" in run.answer
    assert "will notify" not in run.answer.lower()
    assert "i'll notify" not in run.answer.lower()


def test_task_proposal_answer_gets_safety_note_without_notification_claim(tmp_path):
    agent = make_agent(tmp_path)
    agent.client = TaskToolCallingClient("I've saved that task for review.")

    run = agent.run("Add this to my todo list: submit the permit form.")

    assert "I've saved that task for review." in run.answer
    assert "not a real reminder" in run.answer
    assert "marks the task proposal as approved" in run.answer
    assert "No alerts" in run.answer


def test_task_proposal_answer_does_not_say_done_on_approval(tmp_path):
    agent = make_agent(tmp_path)
    agent.client = TaskToolCallingClient(
        "If you confirm, I'll mark it as done on my end. This is not a real reminder."
    )

    run = agent.run("Remind me to check DeepSeek usage tomorrow.")

    assert "done on my end" not in run.answer.lower()
    assert "mark it as done" not in run.answer.lower()
    assert "marks the task proposal as approved" in run.answer
    assert "does not mark the task done or completed" in run.answer
    assert "not a real reminder" in run.answer


def test_schedule_meeting_creates_pending_calendar_proposal_not_real_event(tmp_path):
    agent = make_agent(tmp_path)
    agent.client = CalendarToolCallingClient(
        "I've scheduled the meeting and sent the invite."
    )

    run = agent.run("Schedule a meeting with Alex and Jordan next Tuesday at 2.")

    proposals = [
        item
        for item in ApprovalStore(agent.settings.database_path).list(status="pending")
        if item.action_type == "propose_calendar_event"
    ]
    assert len(proposals) == 1
    assert proposals[0].payload == {
        "title": "Project sync",
        "description": "Discuss the launch checklist.",
        "date_text": "next Tuesday",
        "start_time_text": "2:00 PM",
        "end_time_text": "2:30 PM",
        "location": "Zoom",
        "attendees": ["Alex", "Jordan"],
        "source": "chat",
        "notes": "Local proposal only.",
    }
    assert "pending local calendar proposal" in run.answer
    assert "not a real calendar event" in run.answer
    assert "does not create an Outlook" in run.answer
    assert "send invites" in run.answer
    assert "notify attendees" in run.answer
    assert "scheduled the meeting" not in run.answer.lower()
    assert "sent the invite" not in run.answer.lower()


def test_calendar_proposal_answer_keeps_safe_local_only_note(tmp_path):
    agent = make_agent(tmp_path)
    agent.client = CalendarToolCallingClient(
        "I've saved that as a pending local calendar proposal."
    )

    run = agent.run("Block off time for launch planning next Tuesday.")

    assert "pending local calendar proposal" in run.answer
    assert "created a real calendar event" not in run.answer.lower()
    assert "notified attendees" not in run.answer.lower()


def test_chat_uses_list_tasks_for_task_inventory(tmp_path):
    agent = make_agent(tmp_path)
    store = ApprovalStore(agent.settings.database_path)
    item = store.create(
        action_type="propose_task",
        title="Submit form",
        payload={"title": "Submit form", "description": "Submit the permit form."},
    )
    fake_client = FakeClient()
    agent.client = fake_client

    run = agent.run("what tasks do I have?")

    assert fake_client.completions.count == 0
    assert run.tool_calls[0]["name"] == "list_tasks"
    assert run.tool_calls[0]["result"]["tasks"][0]["approval_id"] == item.id
    assert "Submit form" in run.answer
    assert "invented" not in run.answer


def test_chat_uses_list_calendar_proposals_for_meeting_inventory(tmp_path):
    agent = make_agent(tmp_path)
    store = ApprovalStore(agent.settings.database_path)
    item = store.create(
        action_type="propose_calendar_event",
        title="Project sync",
        payload={
            "title": "Project sync",
            "description": "Discuss launch.",
            "date_text": "Tuesday",
            "start_time_text": "2:00 PM",
            "end_time_text": "2:30 PM",
            "location": "Zoom",
            "attendees": ["Alex"],
        },
    )
    fake_client = FakeClient()
    agent.client = fake_client

    run = agent.run("what meetings are pending approval?")

    assert fake_client.completions.count == 0
    assert run.tool_calls[0]["name"] == "list_calendar_proposals"
    assert (
        run.tool_calls[0]["result"]["calendar_proposals"][0]["approval_id"] == item.id
    )
    assert "Project sync" in run.answer
    assert "invented" not in run.answer


def test_chat_uses_pending_approvals_tool_without_inventing(tmp_path):
    agent = make_agent(tmp_path)
    store = ApprovalStore(agent.settings.database_path)
    email = store.create(
        action_type="propose_email_draft",
        title="Email Alex",
        payload={"to": "Alex", "body": "Hello"},
    )
    rejected = store.create(
        action_type="propose_task",
        title="Old task",
        payload={"title": "Old task", "description": "Ignore this."},
    )
    store.update_status(rejected.id, "rejected")
    fake_client = FakeClient()
    agent.client = fake_client

    run = agent.run("what approvals are pending?")

    assert fake_client.completions.count == 0
    assert run.tool_calls[0]["name"] == "list_pending_approvals"
    assert run.tool_calls[0]["result"]["approvals"][0]["approval_id"] == email.id
    assert "Email Alex" in run.answer
    assert "Old task" not in run.answer


def test_chat_uses_dashboard_summary_for_dashboard_request(tmp_path):
    agent = make_agent(tmp_path)
    store = ApprovalStore(agent.settings.database_path)
    store.create(
        action_type="propose_email_draft",
        title="Draft Alex",
        payload={"to": "Alex", "body": "Hello"},
    )
    store.create(
        action_type="propose_task",
        title="Submit form",
        payload={"title": "Submit form", "description": "Submit the permit form."},
    )
    store.create(
        action_type="propose_calendar_event",
        title="Project sync",
        payload={"title": "Project sync", "description": "Discuss launch."},
    )
    fake_client = FakeClient()
    agent.client = fake_client

    run = agent.run("give me a dashboard")

    assert fake_client.completions.count == 0
    assert run.tool_calls[0]["name"] == "get_dashboard_summary"
    assert "total pending: 3" in run.answer
    assert "pending drafts: 1" in run.answer
    assert "pending tasks: 1" in run.answer
    assert "pending calendar events: 1" in run.answer
    assert "approved tasks: 0" in run.answer
    assert "completed tasks: 0" in run.answer
    assert "approved calendar events: 0" in run.answer
    assert "total rejected: 0" in run.answer
    assert "pending approvals:" not in run.answer
    assert "task proposals:" not in run.answer
    assert "calendar proposals:" not in run.answer


def test_natural_language_buzzword_messages_use_normal_chat(tmp_path):
    agent = make_agent(tmp_path)
    fake_client = FakeClient()
    agent.client = fake_client

    messages = [
        "How should Lewis think about tasks?",
        "Why is the dashboard showing this?",
        "Why are pending approvals blocked?",
        "How should I think about calendar planning?",
        "Can you explain memories and history?",
    ]

    for index, message in enumerate(messages, start=1):
        run = agent.run(message)

        assert fake_client.completions.count == index
        assert run.answer == f"reply {index}"
        assert run.tool_calls == []


class FunctionCallsMarkupCompletions:
    """Fake LLM that returns <function_calls> text markup (not DSML)."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.count = 0

    def create(self, **kwargs: object) -> object:
        self.count += 1
        self.calls.append(copy.deepcopy(kwargs["messages"]))
        message = ToolCallingMessage(
            content=(
                "<function_calls>\n"
                '<invoke name="propose_code_change">\n'
                '<parameter name="description">change chat text to blue</parameter>\n'
                "</invoke>\n"
                "</function_calls>"
            ),
            tool_calls=None,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )


class FunctionCallsMarkupClient:
    def __init__(self) -> None:
        self.completions = FunctionCallsMarkupCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_explicit_code_proposal_creates_real_pending_approval_deterministically(
    tmp_path,
):
    """The exact failing prompt must create a real approval without calling the LLM."""
    settings = make_settings(tmp_path)
    agent = make_agent_from_settings(settings)
    agent.client = FailIfCalledClient()

    prompt = (
        "I want you to create a code proposal that makes the text blue "
        "for this chat instead of black"
    )
    run = agent.run(prompt)

    names = [record["name"] for record in run.tool_calls]
    assert names[-1] == "propose_code_change"
    assert 1 <= names.count("search_project_files") <= 2

    # A real pending approval was created
    proposal_result = run.tool_calls[-1]["result"]
    assert proposal_result["ok"] is True
    approval_id = proposal_result["approval_id"]
    assert isinstance(approval_id, int)

    # Approval can be loaded from ApprovalStore
    approval = ApprovalStore(settings.database_path).get(approval_id)
    assert approval is not None
    assert approval.status == "pending"
    assert approval.action_type == "propose_code_change"
    assert (
        approval.payload.get("requested_goal", "")
        == "Make the text blue for this chat instead of black."
    )
    assert "app/static/lewis.css" in approval.payload["relevant_files"]
    assert "app/static/lewis.css" in approval.payload["proposed_approach"]
    assert "global ink color" in approval.payload["proposed_approach"]
    assert approval.payload["proposed_approach"] != (
        "Create a pending review-only plan. No files are modified and no commands are run."
    )

    # Response includes the real approval ID
    assert f"#{approval_id}" in run.answer

    # Response is NOT just "Project search found ..."
    assert "Project search found" not in run.answer
    assert "Project search did not find" not in run.answer

    # No <function_calls> or DSML markup in answer
    assert "<function_calls>" not in run.answer
    assert "<invoke" not in run.answer
    assert "DSML" not in run.answer
    assert "<search_project_files>" not in run.answer

    # No unregistered read_file exposed
    assert "read_file" not in run.answer
    assert ".." not in run.answer

    assert_no_unanswered_tool_calls(agent.messages)


def test_code_proposal_animation_prompt_creates_real_cleaned_approval(
    tmp_path,
):
    """Prompt #2: animation proposal must create a real approval with cleaned goal."""
    settings = make_settings(tmp_path)
    agent = make_agent_from_settings(settings)
    agent.client = FailIfCalledClient()

    prompt = (
        "I want you to create a code proposal that adds a new animation "
        'in this chat when you are "thinking"'
    )
    run = agent.run(prompt)

    names = [record["name"] for record in run.tool_calls]
    assert names[-1] == "propose_code_change"
    assert 1 <= names.count("search_project_files") <= 2

    proposal_result = run.tool_calls[-1]["result"]
    assert proposal_result["ok"] is True
    approval_id = proposal_result["approval_id"]
    assert isinstance(approval_id, int)

    approval = ApprovalStore(settings.database_path).get(approval_id)
    assert approval is not None
    assert approval.status == "pending"
    assert approval.action_type == "propose_code_change"
    assert (
        approval.payload.get("requested_goal", "")
        == "Add a new animation in this chat when Lewis is thinking."
    )
    assert "app/static/chat.js" in approval.payload["relevant_files"]
    assert "app/static/lewis.css" in approval.payload["relevant_files"]
    assert "app/static/chat.js" in approval.payload["proposed_approach"]
    assert "app/static/lewis.css" in approval.payload["proposed_approach"]
    assert "thinking-bubble lifecycle" in approval.payload["proposed_approach"]

    assert f"#{approval_id}" in run.answer
    assert "Project search found" not in run.answer
    assert "Project search did not find" not in run.answer
    assert "<function_calls>" not in run.answer
    assert "<invoke" not in run.answer
    assert "DSML" not in run.answer
    assert "read_file" not in run.answer
    assert ".." not in run.answer
    assert_no_unanswered_tool_calls(agent.messages)


def test_proposal_involving_animated_character_creates_real_approval(
    tmp_path,
):
    """Prompt #3: 'create a proposal involving...' must create a real approval,
    not an evidence-only response."""
    settings = make_settings(tmp_path)
    agent = make_agent_from_settings(settings)
    agent.client = FailIfCalledClient()

    prompt = (
        "this time I want you to create a proposal involving adding an "
        "animated character thinking, everytime that you are thinking"
    )
    run = agent.run(prompt)

    names = [record["name"] for record in run.tool_calls]
    assert names[-1] == "propose_code_change"
    assert 1 <= names.count("search_project_files") <= 2

    proposal_result = run.tool_calls[-1]["result"]
    assert proposal_result["ok"] is True
    approval_id = proposal_result["approval_id"]
    assert isinstance(approval_id, int)

    approval = ApprovalStore(settings.database_path).get(approval_id)
    assert approval is not None
    assert approval.status == "pending"
    assert approval.action_type == "propose_code_change"
    assert (
        approval.payload.get("requested_goal", "")
        == "Add an animated character thinking every time Lewis is thinking."
    )
    assert approval.payload["relevant_files"]
    assert "app/static/chat.js" in approval.payload["proposed_approach"]
    assert "app/static/lewis.css" in approval.payload["proposed_approach"]
    assert "thinking-bubble lifecycle" in approval.payload["proposed_approach"]

    assert f"#{approval_id}" in run.answer
    assert "Project search found" not in run.answer
    assert "Project search did not find" not in run.answer
    assert "<function_calls>" not in run.answer
    assert "<invoke" not in run.answer
    assert "DSML" not in run.answer
    assert "read_file" not in run.answer
    assert ".." not in run.answer
    assert_no_unanswered_tool_calls(agent.messages)


def test_non_code_dinner_proposal_does_not_create_code_approval(tmp_path):
    settings = make_settings(tmp_path)
    agent = make_agent_from_settings(settings)
    agent.client = FakeClient()

    run = agent.run("create a proposal for dinner")

    assert run.answer == "reply 1"
    assert run.tool_calls == []
    assert [
        item
        for item in ApprovalStore(settings.database_path).list(status="pending")
        if item.action_type == "propose_code_change"
    ] == []


def test_function_calls_markup_is_never_displayed(tmp_path):
    """When the LLM emits <function_calls><invoke> markup, Lewis must strip it."""
    project_root = tmp_path / "project"
    (project_root / "app").mkdir(parents=True)
    (project_root / "app" / "llm.py").write_text(
        "def usage_tracking():\n    pass\n",
        encoding="utf-8",
    )
    agent = make_agent_with_project_search(make_settings(tmp_path), project_root)
    agent.client = FunctionCallsMarkupClient()

    run = agent.run("could you inspect your usage tracking methods")

    assert "<function_calls>" not in run.answer
    assert "<invoke" not in run.answer
    assert "<parameter" not in run.answer
    assert "change chat text to blue" not in run.answer
    assert "DSML" not in run.answer
    assert run.tool_calls == []
    assert_no_unanswered_tool_calls(agent.messages)
