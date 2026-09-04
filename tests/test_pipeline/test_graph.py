import pytest

from src.core.exceptions import PipelineExecutionError
from src.pipeline.graph import PipelineGraph
from src.pipeline.models import Pipeline, PipelineNode


def _pipeline(edges):
    ids = {i for pair in edges for i in pair}
    nodes = [PipelineNode(id=i, name=i) for i in sorted(ids)]
    by_id = {n.id: n for n in nodes}
    for src, dst in edges:
        by_id[dst].depends_on.append(src)
    return Pipeline(nodes=nodes)


def test_topological_order_respects_dependencies():
    g = PipelineGraph(_pipeline([("a", "b"), ("b", "c")]))
    assert g.topological_order() == ["a", "b", "c"]


def test_execution_layers_group_parallel_nodes():
    # a -> b, a -> c, (b,c) -> d
    g = PipelineGraph(_pipeline([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]))
    layers = g.execution_layers()
    assert layers[0] == ["a"]
    assert set(layers[1]) == {"b", "c"}
    assert layers[2] == ["d"]


def test_cycle_detected():
    nodes = [PipelineNode(id="a", name="a"), PipelineNode(id="b", name="b")]
    nodes[0].depends_on.append("b")
    nodes[1].depends_on.append("a")
    g = PipelineGraph(Pipeline(nodes=nodes))
    assert g.detect_cycle() is not None
    with pytest.raises(PipelineExecutionError):
        g.execution_layers()


def test_pipeline_model_rejects_unknown_dependency():
    with pytest.raises(Exception):
        Pipeline(nodes=[PipelineNode(id="a", name="a", depends_on=["ghost"])])
