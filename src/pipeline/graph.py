"""DAG utilities for pipelines: cycle detection and execution layering.

Kept separate from :mod:`models` so the graph algorithms can be unit
tested in isolation and reused by both the builder (validation) and the
executor (scheduling).
"""

from __future__ import annotations

from collections import defaultdict, deque

from ..core.exceptions import PipelineExecutionError
from .models import Pipeline


class PipelineGraph:
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self._deps: dict[str, set[str]] = {
            n.id: set(n.depends_on) for n in pipeline.nodes
        }
        # dependencies expressed only as edges also count
        for edge in pipeline.edges:
            self._deps.setdefault(edge.target, set()).add(edge.source)
        self._dependents: dict[str, set[str]] = defaultdict(set)
        for node_id, deps in self._deps.items():
            for dep in deps:
                self._dependents[dep].add(node_id)

    # ------------------------------------------------------------------
    def dependencies(self, node_id: str) -> set[str]:
        return set(self._deps.get(node_id, set()))

    def dependents(self, node_id: str) -> set[str]:
        return set(self._dependents.get(node_id, set()))

    def detect_cycle(self) -> list[str] | None:
        """Return a node-id cycle if one exists, else ``None``."""
        WHITE, GREY, BLACK = 0, 1, 2
        colour = {n.id: WHITE for n in self.pipeline.nodes}
        stack: list[str] = []

        def visit(node: str) -> list[str] | None:
            colour[node] = GREY
            stack.append(node)
            for nxt in self._deps.get(node, ()):  # walk toward dependencies
                if colour.get(nxt, BLACK) == GREY:
                    idx = stack.index(nxt)
                    return stack[idx:] + [nxt]
                if colour.get(nxt, BLACK) == WHITE:
                    found = visit(nxt)
                    if found:
                        return found
            stack.pop()
            colour[node] = BLACK
            return None

        for node in colour:
            if colour[node] == WHITE:
                cycle = visit(node)
                if cycle:
                    return cycle
        return None

    def topological_order(self) -> list[str]:
        indegree = {n.id: len(self._deps.get(n.id, ())) for n in self.pipeline.nodes}
        queue = deque(sorted(nid for nid, d in indegree.items() if d == 0))
        order: list[str] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for dependent in sorted(self._dependents.get(nid, ())):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    queue.append(dependent)
        if len(order) != len(self.pipeline.nodes):
            raise PipelineExecutionError(
                "pipeline contains a cycle", details={"cycle": self.detect_cycle()}
            )
        return order

    def execution_layers(self) -> list[list[str]]:
        """Group node ids into layers that can run in parallel.

        Layer *k* contains every node whose dependencies are all satisfied
        by layers ``0..k-1``.
        """
        remaining = {n.id: set(self._deps.get(n.id, ())) for n in self.pipeline.nodes}
        done: set[str] = set()
        layers: list[list[str]] = []
        while remaining:
            ready = sorted(nid for nid, deps in remaining.items() if deps <= done)
            if not ready:
                raise PipelineExecutionError(
                    "pipeline contains a cycle", details={"cycle": self.detect_cycle()}
                )
            layers.append(ready)
            done.update(ready)
            for nid in ready:
                remaining.pop(nid)
        return layers

    def validate(self) -> None:
        cycle = self.detect_cycle()
        if cycle:
            raise PipelineExecutionError(
                "pipeline contains a cycle", details={"cycle": cycle}
            )
