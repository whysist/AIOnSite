import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "echo"
    assert "calculator" in body["tools"]


def test_task_lifecycle(client):
    created = client.post("/tasks", json={"task": "Compare 120 bar vs 100 bar limit and recommend."})
    assert created.status_code == 200
    payload = created.json()
    eid = payload["execution_id"]
    assert payload["status"] in ("completed", "needs_review")

    got = client.get(f"/tasks/{eid}").json()
    assert got["execution_id"] == eid
    assert got["task"].startswith("Compare")
    assert got["node_results"]

    pipe = client.get(f"/tasks/{eid}/pipeline").json()
    node_ids = [n["id"] for n in pipe["pipeline"]["nodes"]]
    assert "verify_final" in node_ids

    audit = client.get(f"/tasks/{eid}/audit").json()
    event_types = {e["event_type"] for e in audit["events"]}
    assert {"task_received", "plan_created", "execution_completed"} <= event_types


def test_unknown_execution_returns_404(client):
    assert client.get("/tasks/does-not-exist").status_code == 404


def test_task_validation_error(client):
    assert client.post("/tasks", json={"task": ""}).status_code == 422
