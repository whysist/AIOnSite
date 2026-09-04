"""Turn a validated :class:`~src.agents.planner.Plan` into a real DAG.

Step ids are preserved as node ids so a plan step and its pipeline node
share an identifier end-to-end (planner -> pipeline -> audit -> frontend).
A terminal ``verifier`` node is appended by default.
"""

from __future__ import annotations

from ..agents.planner import Plan
from ..core.config import Settings, get_settings
from .graph import PipelineGraph
from .models import (
    NodeType,
    Pipeline,
    PipelineEdge,
    PipelineNode,
    RetryPolicy,
)

_AGENT_TO_NODETYPE = {
    "researcher": NodeType.RESEARCHER,
    "analyst": NodeType.ANALYST,
    "executor": NodeType.EXECUTOR,
    "summarizer": NodeType.SUMMARIZER,
    "verifier": NodeType.VERIFIER,
    "custom": NodeType.CUSTOM,
}


def build_pipeline(
    plan: Plan,
    *,
    settings: Settings | None = None,
    add_verifier: bool = True,
    require_local: bool = False,
) -> Pipeline:
    settings = settings or get_settings()
    retry = RetryPolicy(max_retries=settings.max_retries)

    nodes: list[PipelineNode] = []
    edges: list[PipelineEdge] = []

    for step in plan.steps:
        node = PipelineNode(
            id=step.id,
            name=step.description[:60] or step.id,
            type=_AGENT_TO_NODETYPE.get(step.agent, NodeType.CUSTOM),
            description=step.description,
            agent=step.agent,
            depends_on=list(step.depends_on),
            tools=list(step.tools),
            require_local=require_local or settings.sovereign_mode,
            retry_policy=retry,
            timeout_seconds=settings.node_timeout_seconds,
        )
        nodes.append(node)
        for dep in step.depends_on:
            edges.append(PipelineEdge(source=dep, target=step.id))

    if add_verifier:
        leaves = _leaf_ids(nodes)
        verifier = PipelineNode(
            id="verify_final",
            name="Verify final result",
            type=NodeType.VERIFIER,
            description="Check the final result for correctness, completeness and consistency.",
            agent="verifier",
            depends_on=leaves,
            require_local=require_local or settings.sovereign_mode,
            retry_policy=RetryPolicy(max_retries=0),
            timeout_seconds=settings.node_timeout_seconds,
        )
        nodes.append(verifier)
        edges.extend(PipelineEdge(source=leaf, target="verify_final") for leaf in leaves)

    pipeline = Pipeline(goal=plan.goal, nodes=nodes, edges=edges,
                        metadata={"source": "planner"})
    PipelineGraph(pipeline).validate()
    return pipeline


def _leaf_ids(nodes: list[PipelineNode]) -> list[str]:
    referenced: set[str] = set()
    for n in nodes:
        referenced.update(n.depends_on)
    leaves = [n.id for n in nodes if n.id not in referenced]
    return leaves or [nodes[-1].id]
