import pytest

from src.agents.planner import Plan, Planner, PlanStep
from src.llm.base import BaseLLM
from src.llm.schemas import LLMResponse


class _ScriptedLLM(BaseLLM):
    provider_name = "scripted"

    def __init__(self, replies):
        super().__init__(model="scripted")
        self._replies = list(replies)

    async def _complete(self, request):
        return LLMResponse(content=self._replies.pop(0), model="scripted", provider="scripted")


def test_plan_model_rejects_duplicate_ids():
    with pytest.raises(Exception):
        Plan(goal="g", steps=[PlanStep(id="a"), PlanStep(id="a")])


def test_plan_model_rejects_unknown_dependency():
    with pytest.raises(Exception):
        Plan(goal="g", steps=[PlanStep(id="a", depends_on=["missing"])])


def test_plan_model_rejects_forward_dependency():
    with pytest.raises(Exception):
        Plan(goal="g", steps=[PlanStep(id="a", depends_on=["b"]), PlanStep(id="b")])


def test_unknown_agent_coerced_to_custom():
    step = PlanStep(id="a", agent="wizard")
    assert step.agent == "custom"


async def test_planner_parses_valid_json():
    good = '{"goal":"do it","steps":[{"id":"s1","agent":"researcher","description":"x","depends_on":[]},{"id":"s2","agent":"summarizer","description":"y","depends_on":["s1"]}]}'
    planner = Planner(_ScriptedLLM([good]))
    plan = await planner.plan("task")
    assert plan.goal == "do it"
    assert [s.id for s in plan.steps] == ["s1", "s2"]


async def test_planner_recovers_from_malformed_then_valid():
    good = '{"goal":"g","steps":[{"id":"s1","agent":"analyst","description":"x","depends_on":[]}]}'
    planner = Planner(_ScriptedLLM(["not json at all", good]), max_attempts=2)
    plan = await planner.plan("task")
    assert plan.steps[0].agent == "analyst"


async def test_planner_falls_back_after_exhausting_attempts():
    planner = Planner(_ScriptedLLM(["garbage", "still garbage"]), max_attempts=2)
    plan = await planner.plan("Analyze something")
    # deterministic fallback plan
    assert [s.agent for s in plan.steps] == ["researcher", "analyst", "summarizer"]


async def test_planner_rejects_cyclic_plan_and_falls_back():
    cyclic = '{"goal":"g","steps":[{"id":"s1","agent":"analyst","description":"x","depends_on":["s2"]},{"id":"s2","agent":"analyst","description":"y","depends_on":["s1"]}]}'
    planner = Planner(_ScriptedLLM([cyclic, cyclic]), max_attempts=2)
    plan = await planner.plan("task")
    assert len(plan.steps) == 3  # fallback
