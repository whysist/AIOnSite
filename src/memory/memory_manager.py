"""Backward-compatible synchronous memory facade.

New code should use :class:`src.memory.base.MemoryStore` /
:class:`src.memory.in_memory.InMemoryStore`.  ``MemoryManager`` is kept
because the original skeleton and early tests reference it; it now simply
wraps an :class:`InMemoryStore`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .in_memory import InMemoryStore


class MemoryManager:
    """Short-term memory keyed by session id (sync wrapper)."""

    def __init__(self, store: InMemoryStore | None = None) -> None:
        self._store = store or InMemoryStore()

    def _run(self, coro: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError(
            "MemoryManager sync API cannot be used inside a running event loop; "
            "use the async InMemoryStore directly."
        )

    def get(self, session_id: str) -> Any | None:
        return self._run(self._store.get(session_id))

    def set(self, session_id: str, value: Any) -> None:
        self._run(self._store.set(session_id, value))

    def delete(self, session_id: str) -> None:
        self._run(self._store.delete(session_id))

    @property
    def store(self) -> InMemoryStore:
        return self._store
