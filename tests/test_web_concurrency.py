import asyncio
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import httpx
import pytest

from app.config import Settings
from app.web import create_app


def _app(tmp_path: Path):
    return create_app(
        Settings(
            deepseek_api_key="test-key",
            deepseek_model="test-model",
            deepseek_base_url="https://example.test",
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
    )


class BlockingAgent:
    def __init__(self):
        self.started = Event()
        self.release = Event()
        self.second_started = Event()
        self.calls = []

    def run(self, message):
        self.calls.append(message)
        if message == "first":
            self.started.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("Test did not release the first model call")
        else:
            self.second_started.set()
        return SimpleNamespace(answer=f"Reply to {message}", memory_context_count=0)


async def _wait_started(event):
    assert await asyncio.to_thread(event.wait, 2), "Worker did not start"


def _transcript(app):
    return [(item["role"], item["content"]) for item in app.state.chat_messages]


def test_dashboard_and_chat_page_remain_responsive_during_model_call(tmp_path):
    app = _app(tmp_path)
    agent = BlockingAgent()
    app.state.chat_agent = agent

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8787",
            headers={"Origin": "http://127.0.0.1:8787"},
        ) as client:
            pending = asyncio.create_task(
                client.post("/chat", json={"message": "first"})
            )
            try:
                await _wait_started(agent.started)
                dashboard = await asyncio.wait_for(client.get("/dashboard"), 2)
                chat = await asyncio.wait_for(client.get("/chat"), 2)
                assert dashboard.status_code == chat.status_code == 200
                assert not pending.done(), "Model call blocked the event loop"
                assert _transcript(app) == [], "Only complete turns should be published"
            finally:
                agent.release.set()
                response = await pending
            assert response.json()["answer"] == "Reply to first"

    asyncio.run(scenario())


@pytest.mark.parametrize("block_initialization", [False, True])
def test_overlapping_chat_requests_share_one_agent_and_ordered_turns(
    tmp_path, monkeypatch, block_initialization
):
    app = _app(tmp_path)
    agent = BlockingAgent()
    factory_started = Event()
    release_factory = Event()
    creations = []

    def build_agent(settings):
        creations.append(settings)
        factory_started.set()
        if block_initialization:
            assert release_factory.wait(timeout=5), (
                "Test did not release initialization"
            )
        return agent

    monkeypatch.setattr("app.web_routes.build_agent", build_agent)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8787",
            headers={"Origin": "http://127.0.0.1:8787"},
        ) as client:
            first = asyncio.create_task(client.post("/chat", json={"message": "first"}))
            second = None
            try:
                await _wait_started(
                    factory_started if block_initialization else agent.started
                )
                second = asyncio.create_task(
                    client.post("/chat", json={"message": "second"})
                )
                # Hold the first turn open so a broken lock exposes a concurrent run.
                assert not await asyncio.to_thread(agent.second_started.wait, 0.1)
                assert len(creations) == 1
            finally:
                release_factory.set()
                agent.release.set()
                tasks = [first] + ([second] if second is not None else [])
                responses = await asyncio.gather(*tasks)
            assert [r.json()["answer"] for r in responses] == [
                "Reply to first",
                "Reply to second",
            ]

    asyncio.run(scenario())
    assert len(creations) == 1
    assert agent.calls == ["first", "second"]
    assert _transcript(app) == [
        ("user", "first"),
        ("assistant", "Reply to first"),
        ("user", "second"),
        ("assistant", "Reply to second"),
    ]


def test_cancelled_request_keeps_agent_serialized_until_worker_finishes(tmp_path):
    app = _app(tmp_path)
    agent = BlockingAgent()
    app.state.chat_agent = agent

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8787",
            headers={"Origin": "http://127.0.0.1:8787"},
        ) as client:
            first = asyncio.create_task(client.post("/chat", json={"message": "first"}))
            second = None
            try:
                await _wait_started(agent.started)
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                second = asyncio.create_task(
                    client.post("/chat", json={"message": "second"})
                )
                assert not await asyncio.to_thread(agent.second_started.wait, 0.1)
                assert _transcript(app) == []
            finally:
                agent.release.set()
                await asyncio.gather(first, return_exceptions=True)
                if second is not None:
                    await second
            assert second is not None
            assert second.result().json()["answer"] == "Reply to second"

    asyncio.run(scenario())
    assert agent.calls == ["first", "second"]
    assert _transcript(app) == [
        ("user", "first"),
        ("assistant", "Reply to first"),
        ("user", "second"),
        ("assistant", "Reply to second"),
    ]


def test_agent_error_does_not_block_the_next_turn(tmp_path):
    app = _app(tmp_path)

    class FailingOnceAgent:
        def run(self, message):
            if message == "first":
                raise RuntimeError("Synthetic model failure")
            return SimpleNamespace(answer="Recovered", memory_context_count=0)

    app.state.chat_agent = FailingOnceAgent()

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8787",
            headers={"Origin": "http://127.0.0.1:8787"},
        ) as client:
            first = await client.post("/chat", json={"message": "first"})
            assert "Synthetic model failure" in first.json()["answer"]
            second = await asyncio.wait_for(
                client.post("/chat", json={"message": "second"}), 2
            )
            assert second.json()["answer"] == "Recovered"

    asyncio.run(scenario())
    assert [role for role, _ in _transcript(app)] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
