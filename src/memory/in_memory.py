"""In-process :class:`MemoryStore` with optional per-key TTL."""

from __future__ import annotations

import time
from typing import Any

from .base import MemoryStore


class InMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._data: dict[str, tuple[Any, float | None]] = {}

    def _expired(self, key: str) -> bool:
        item = self._data.get(key)
        if item is None:
            return True
        _, expires = item
        return expires is not None and expires < time.time()

    async def get(self, key: str) -> Any | None:
        if self._expired(key):
            self._data.pop(key, None)
            return None
        return self._data[key][0]

    async def set(self, key: str, value: Any, *, ttl_seconds: float | None = None) -> None:
        expires = time.time() + ttl_seconds if ttl_seconds else None
        self._data[key] = (value, expires)

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def keys(self, *, prefix: str = "") -> list[str]:
        return sorted(
            k for k in self._data if k.startswith(prefix) and not self._expired(k)
        )

    async def clear(self) -> None:
        self._data.clear()
