"""Audit layer: append-only execution event capture."""

from .events import AuditEvent, AuditEventType
from .trail import AuditTrail

__all__ = ["AuditEvent", "AuditEventType", "AuditTrail"]
