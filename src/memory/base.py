"""Memory abstraction.

Agents/orchestrator depend only on :class:`MemoryStore`.  The default
implementation is in-process; SQLite / Postgres / vector-store backends
can be added later without touching callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MemoryStore(ABC):
    @abstractmethod
    async def get(self, key: str) -> Any | None: ...

    @abstractmethod
    async def set(self, key: str, value: Any, *, ttl_seconds: float | None = None) -> None: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...

    @abstractmethod
    async def keys(self, *, prefix: str = "") -> list[str]: ...

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k in keys:
            value = await self.get(k)
            if value is not None:
                out[k] = value
        return out
