"""Reusable prompt templates."""

from __future__ import annotations

import json
from typing import Any


def format_inspection_prompt(context: dict) -> str:
    return f"Inspection context: {json.dumps(context, default=str, indent=2)}"


def format_context_block(outputs: dict[str, Any]) -> str:
    """Render upstream step outputs as a labelled context block."""
    if not outputs:
        return ""
    parts = ["CONTEXT FROM PREVIOUS STEPS:"]
    for key, value in outputs.items():
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        parts.append(f"[{key}]\n{text}")
    return "\n".join(parts)
