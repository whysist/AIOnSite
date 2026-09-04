"""Audit trail: append-only event capture for one or many executions.

The trail redacts obvious secrets from event metadata before storing, so
callers can pass context dicts without hand-scrubbing every field.
"""

from __future__ import annotations

from typing import Any, Callable
from collections.abc import Iterable

from ..core.logging import get_logger
from .events import AuditEvent, AuditEventType

_log = get_logger("audit")
_SECRET_HINTS = ("key", "token", "secret", "password", "authorization", "api_key")


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: ("***redacted***" if any(h in k.lower() for h in _SECRET_HINTS) else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if isinstance(value, str) and len(value) > 2000:
        return value[:2000] + "...(truncated)"
    return value


class AuditTrail:
    """In-memory append-only event log.

    A durable sink (file / DB) can be added by passing ``on_event``.
    """

    def __init__(self, on_event: callable | None = None) -> None:
        self._events: list[AuditEvent] = []
        self._on_event = on_event

    def record(
        self,
        execution_id: str,
        event_type: AuditEventType,
        *,
        component: str,
        status: str = "ok",
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        event = AuditEvent(
            execution_id=execution_id,
            event_type=event_type,
            component=component,
            status=status,
            message=message,
            metadata=_redact(metadata or {}),
        )
        self._events.append(event)
        _log.info(
            "audit_event",
            execution_id=execution_id,
            event_type=event_type.value,
            component=component,
            status=status,
        )
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception as exc:  # noqa: BLE001 - sink must not break the run
                _log.warning("audit_sink_failed", error=str(exc))
        return event

    def events(self, execution_id: str | None = None) -> list[AuditEvent]:
        if execution_id is None:
            return list(self._events)
        return [e for e in self._events if e.execution_id == execution_id]

    def as_dicts(self, execution_id: str | None = None) -> list[dict[str, Any]]:
        return [e.model_dump(mode="json") for e in self.events(execution_id)]

    def extend(self, events: Iterable[AuditEvent]) -> None:
        self._events.extend(events)
