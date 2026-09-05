"""End-to-end: POST /tasks/stream returns immediately; GET .../events
streams the real pipeline's steps live, ending with the final answer.

Run inside a worker thread with a hard wall-clock bound so a genuine
deadlock (the background task never getting scheduled) fails the test
loudly instead of hanging the suite -- no new test dependency required.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from src.api.app import _sse_event_stream, app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _run_streaming_task(client: TestClient, task_text: str) -> tuple[list[str], dict]:
    created = client.post("/tasks/stream", json={"task": task_text})
    assert created.status_code == 200
    payload = created.json()
    assert payload["status"] == "running"
    assert payload["task_status"] == "pending"
    execution_id = payload["execution_id"]

    event_types: list[str] = []
    terminal: dict = {}
    with client.stream("GET", f"/tasks/{execution_id}/events") as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = json.loads(line[len("data:") :].strip())
            event_types.append(data.get("event_type"))
            if data.get("event_type") == "stream_end":
                terminal = data
                break
    return event_types, terminal


def test_streaming_task_shows_live_steps_and_final_answer(client):
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            _run_streaming_task, client,
            "Compare 120 bar vs 100 bar limit and recommend.",
        )
        event_types, terminal = future.result(timeout=15)

    assert "task_classified" in event_types
    assert "plan_created" in event_types
    assert "node_started" in event_types
    assert "execution_completed" in event_types
    assert terminal["status"] in ("completed", "needs_review")
    assert terminal["final_answer"] is not None


async def test_sse_stream_sends_keepalive_comments_during_long_gaps():
    """Regression test: found by hand while live-testing against a slow
    local model -- a real gap of a minute-plus between pipeline events
    (normal for CPU inference) previously sent zero bytes on the wire the
    whole time, which is exactly the situation a client or proxy with its
    own idle-read timeout tears the connection down over. The generator
    must emit an SSE comment (": ...") whenever the gap since the last
    real event exceeds the keep-alive interval.
    """
    queue: asyncio.Queue = asyncio.Queue()
    lines = []
    gen = _sse_event_stream(queue, keepalive_seconds=0.05)

    # Nothing published yet -- two keep-alive comments should appear
    # before any real event, since the interval (0.05s) is tiny.
    lines.append(await gen.__anext__())
    lines.append(await gen.__anext__())
    assert all(line == ": keep-alive\n\n" for line in lines)

    # Now a real event arrives -- must be delivered as a real data line,
    # not swallowed by the keep-alive path.
    queue.put_nowait({"event_type": "node_started", "message": "step_1"})
    real_line = await gen.__anext__()
    assert real_line.startswith("data: ")
    assert "node_started" in real_line

    queue.put_nowait({"event_type": "stream_end", "final_answer": "done"})
    end_line = await gen.__anext__()
    assert "stream_end" in end_line
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()  # generator must terminate after stream_end


def test_late_subscriber_still_gets_full_replay_after_completion(client):
    """Start a task without immediately streaming it, wait for it to show
    up via the existing polling endpoint, then open the event stream --
    it must still replay the full history rather than showing nothing."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        created = client.post("/tasks/stream", json={"task": "Tell me about P-101."})
        execution_id = created.json()["execution_id"]

        def _wait_and_stream():
            import time

            for _ in range(100):
                got = client.get(f"/tasks/{execution_id}")
                if got.status_code == 200 and got.json()["status"] != "pending":
                    break
                time.sleep(0.05)
            event_types = []
            with client.stream("GET", f"/tasks/{execution_id}/events") as response:
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = json.loads(line[len("data:") :].strip())
                    event_types.append(data.get("event_type"))
                    if data.get("event_type") == "stream_end":
                        break
            return event_types

        future = pool.submit(_wait_and_stream)
        event_types = future.result(timeout=15)

    assert "task_classified" in event_types
    assert "stream_end" in event_types
