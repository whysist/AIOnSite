"""Agent pipeline: models, DAG, builder, execution engine and run state.

``builder`` and ``executor`` are exposed lazily (module ``__getattr__``) so
that importing :mod:`src.pipeline.models` from the agents layer does not
pull the executor -- which imports the agents layer -- back in and create
an import cycle.
"""

from __future__ import annotations

from typing import Any

from .graph import PipelineGraph
from .models import (
    AgentResult,
    ExecutionStatus,
    NodeStatus,
    NodeType,
    Pipeline,
    PipelineEdge,
    PipelineNode,
    RetryPolicy,
    ToolResult,
)
from .state import ExecutionContext, PipelineState

__all__ = [
    "build_pipeline",
    "PipelineExecutor",
    "PipelineGraph",
    "AgentResult",
    "ExecutionStatus",
    "NodeStatus",
    "NodeType",
    "Pipeline",
    "PipelineEdge",
    "PipelineNode",
    "RetryPolicy",
    "ToolResult",
    "ExecutionContext",
    "PipelineState",
]


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute
    if name == "build_pipeline":
        from .builder import build_pipeline

        return build_pipeline
    if name == "PipelineExecutor":
        from .executor import PipelineExecutor

        return PipelineExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
