"""Live per-execution event fan-out, for streaming a running task's steps
to a client as they happen.

Deliberately separate from :class:`~src.audit.trail.AuditTrail` (the
durable, execution-scoped event log used for post-hoc audit/debugging):
this is a small in-memory pub/sub layer whose only job is getting events
from "just recorded" to "an open SSE connection" with low latency. It is
wired to the audit trail via ``AuditTrail(on_event=bus.publish)`` in
``api/app.py`` -- every audit event (task classified, node started, tool
called, tool cache hit, node completed, verification completed, ...)
becomes a live step in the client's view for free, with no duplicate
instrumentation.
"""

from __future__ import annotations

import asyncio
from typing import Any


class ExecutionEventBus:
    def __init__(self, *, max_buffer: int = 2000) -> None:
        self._buffers: dict[str, list[dict[str, Any]]] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._max_buffer = max_buffer

    def create(self, execution_id: str) -> None:
        """Pre-register *execution_id* before its background task starts,
        so ``publish`` has a buffer to append to even if no one has
        subscribed yet."""
        self._buffers.setdefault(execution_id, [])
        self._subscribers.setdefault(execution_id, [])

    def publish(self, execution_id: str, payload: dict[str, Any]) -> None:
        buf = self._buffers.setdefault(execution_id, [])
        buf.append(payload)
        if len(buf) > self._max_buffer:
            del buf[: len(buf) - self._max_buffer]
        for q in self._subscribers.get(execution_id, []):
            q.put_nowait(payload)

    def subscribe(self, execution_id: str) -> asyncio.Queue:
        """Return a queue that first replays everything buffered so far
        (so a subscriber connecting slightly late, or after the task
        already finished, still sees the full history), then receives new
        events live."""
        q: asyncio.Queue = asyncio.Queue()
        for payload in self._buffers.get(execution_id, []):
            q.put_nowait(payload)
        self._subscribers.setdefault(execution_id, []).append(q)
        return q

    def unsubscribe(self, execution_id: str, q: asyncio.Queue) -> None:
        subs = self._subscribers.get(execution_id)
        if subs and q in subs:
            subs.remove(q)
