"""The local dashboard must not accept writes from another browser origin."""

from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

from app.approvals import ApprovalStore
from app.config import Settings
from app.web import create_app, run_web_server


@pytest.fixture
def dashboard(tmp_path):
    settings = Settings(
        deepseek_api_key=None,
        deepseek_model="test-model",
        deepseek_base_url="https://example.test",
        debug=False,
        user_name="Test",
        user_name_configured=False,
        database_path=tmp_path / "test.sqlite3",
        documents_dir=tmp_path / "documents",
        max_tool_calls_per_request=5,
        max_estimated_daily_cost=1.0,
        price_per_1k_input_tokens=0.0,
        price_per_1k_output_tokens=0.0,
    )
    store = ApprovalStore(settings.database_path)
    item = store.create(action_type="propose_email_draft", title="Test", payload={})
    return (
        TestClient(create_app(settings), base_url="http://127.0.0.1:8787"),
        store,
        item.id,
    )


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Origin": "https://attacker.example"},
        {"Origin": "null"},
        {"Origin": "http://127.0.0.1:9999"},
        {"Origin": "http://127.0.0.1:8787.attacker.example"},
        {"Origin": "http://127.0.0.1:8787", "Sec-Fetch-Site": "cross-site"},
        {"Origin": "http://127.0.0.1:8787", "Sec-Fetch-Site": "same-site"},
        {"Referer": "https://attacker.example/form"},
    ],
)
def test_untrusted_post_cannot_approve(dashboard, headers):
    client, store, item_id = dashboard
    response = client.post(f"/approvals/{item_id}/approve", headers=headers)
    assert response.status_code == 403
    assert store.get(item_id).status == "pending"


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "http://127.0.0.1:8787"},
        {"Referer": "http://127.0.0.1:8787/approvals"},
        {"Origin": "http://127.0.0.1:8787", "Sec-Fetch-Site": "same-origin"},
    ],
)
def test_same_origin_post_can_approve(dashboard, headers):
    client, store, item_id = dashboard
    response = client.post(f"/approvals/{item_id}/approve", headers=headers)
    assert response.status_code == 200
    assert store.get(item_id).status == "approved"


def test_foreign_host_cannot_read_or_write(dashboard):
    client, store, item_id = dashboard
    headers = {"Host": "attacker.example", "Origin": "http://attacker.example"}
    assert client.get("/dashboard", headers=headers).status_code == 400
    assert (
        client.post(f"/approvals/{item_id}/approve", headers=headers).status_code == 400
    )
    assert store.get(item_id).status == "pending"


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.2", "::"])
def test_server_rejects_nonlocal_binding(host):
    with patch("uvicorn.run") as run:
        with pytest.raises(ValueError, match="loopback"):
            run_web_server(host=host)
        run.assert_not_called()
