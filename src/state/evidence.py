"""First-class evidence records.

Before this module, a tool's output only ever existed as free text embedded
in an agent's final ``output`` string (see ``AgentResult.output`` in
``src/pipeline/models.py``) -- there was no queryable record of *what was
actually retrieved*, so a downstream node or the verifier could not tell a
grounded claim from an invented one. ``Evidence`` is a minimal, tool-agnostic
record of one piece of retrieved information; it is populated generically
from any successful tool call (see ``PipelineExecutor._run_node``) rather
than being tied to any specific tool implementation, since tool and
retrieval-source implementations are owned elsewhere in this project.
"""

from __future__ import annotations

import enum
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class EvidenceStatus(str, enum.Enum):
    RETRIEVED = "retrieved"
    MISSING = "missing"
    INVALID = "invalid"


class Evidence(BaseModel):
    id: str = Field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:12]}")
    source: str
    source_type: str = "tool_output"
    content: Any = None
    reference: str | None = None
    retrieved_at: float = Field(default_factory=time.time)
    producer_tool: str | None = None
    producer_node: str | None = None
    status: EvidenceStatus = EvidenceStatus.RETRIEVED
    metadata: dict[str, Any] = Field(default_factory=dict)
