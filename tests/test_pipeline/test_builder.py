from src.agents.planner import Plan, PlanStep
from src.core.config import Settings
from src.pipeline.builder import build_pipeline
from src.pipeline.graph import PipelineGraph
from src.pipeline.models import NodeType


def _settings(**kw):
    return Settings(_env_file=None, **kw)


def test_builder_preserves_step_ids_and_adds_verifier():
    plan = Plan(goal="g", steps=[
        PlanStep(id="step_1", agent="researcher", description="r", depends_on=[]),
        PlanStep(id="step_2", agent="summarizer", description="s", depends_on=["step_1"]),
    ])
    pipe = build_pipeline(plan, settings=_settings())
    ids = [n.id for n in pipe.nodes]
    assert ids[:2] == ["step_1", "step_2"]
    assert "verify_final" in ids
    assert pipe.nodes[-1].type is NodeType.VERIFIER
    PipelineGraph(pipe).validate()


def test_builder_marks_nodes_local_under_sovereign():
    plan = Plan(goal="g", steps=[PlanStep(id="a", agent="analyst", description="x")])
    pipe = build_pipeline(plan, settings=_settings(llm_provider="ollama", sovereign_mode=True))
    assert all(n.require_local for n in pipe.nodes)


def test_builder_edges_match_dependencies():
    plan = Plan(goal="g", steps=[
        PlanStep(id="a", agent="researcher", description="x"),
        PlanStep(id="b", agent="analyst", description="y", depends_on=["a"]),
    ])
    pipe = build_pipeline(plan, settings=_settings(), add_verifier=False)
    assert ("a", "b") in [(e.source, e.target) for e in pipe.edges]
