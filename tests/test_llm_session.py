from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

from app.approval_service import approve_approval, reject_approval
from app.approvals import ApprovalStore
from app.budget import BudgetManager
from app.config import Settings
from app.llm import LewisAgent, parse_token_usage
from app.logging_db import RunLogger
from app.memory_service import archive_memory, propose_memory
from app.memory_store import MemoryStore
from app.rag import DocumentSearch
from app.tools import ToolRegistry


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
    )


def make_agent(tmp_path: Path) -> LewisAgent:
    settings = make_settings(tmp_path)
    settings.documents_dir.mkdir()
    approvals = ApprovalStore(settings.database_path)
    tools = ToolRegistry(DocumentSearch(settings.documents_dir), approvals)
    return LewisAgent(
        settings=settings,
        tools=tools,
        budget=BudgetManager(settings),
        logger=RunLogger(settings.database_path),
    )


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.count = 0

    def create(self, **kwargs: object) -> object:
        self.count += 1
        self.calls.append(copy.deepcopy(kwargs["messages"]))
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


class ToolCallingCompletions:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.count = 0

    def create(self, **kwargs: object) -> object:
        self.count += 1
        self.calls.append(copy.deepcopy(kwargs["messages"]))
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
    def __init__(self, content: str | None, tool_calls: list[object] | None) -> None:
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self) -> dict[str, object]:
        return {
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


class ToolCallingClient:
    def __init__(self) -> None:
        self.completions = ToolCallingCompletions()
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
