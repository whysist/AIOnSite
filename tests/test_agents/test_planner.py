import pytest

from src.agents.planner import Plan, Planner, PlanStep
from src.core.exceptions import ProviderConnectionError, ProviderTimeoutError
from src.llm.base import BaseLLM
from src.llm.schemas import LLMResponse


class _ScriptedLLM(BaseLLM):
    provider_name = "scripted"

    def __init__(self, replies):
        super().__init__(model="scripted")
        self._replies = list(replies)

    async def _complete(self, request):
        return LLMResponse(content=self._replies.pop(0), model="scripted", provider="scripted")


class _FlakyLLM(BaseLLM):
    """Raises a provider infrastructure error a fixed number of times before
    (optionally) succeeding -- simulates Ollama being briefly unreachable."""

    provider_name = "flaky"

    def __init__(self, *, fail_times, error_cls=ProviderConnectionError, then=None):
        super().__init__(model="flaky")
        self._fail_times = fail_times
        self._error_cls = error_cls
        self._then = then
        self.calls = 0
        self.sleep_calls: list[float] = []

    async def _complete(self, request):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._error_cls(f"attempt {self.calls} unreachable")
        return LLMResponse(content=self._then, model="flaky", provider="flaky")


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


# ---------------------------------------------------------------------
# Degraded-state propagation for provider-unavailable planning.


async def test_planner_server_unavailable_exhausts_and_falls_back_degraded():
    llm = _FlakyLLM(fail_times=99, error_cls=ProviderConnectionError)
    planner = Planner(llm, max_attempts=2, backoff_seconds=0)
    plan = await planner.plan("Analyze something")
    assert plan.degraded is True
    assert plan.degraded_reason is not None
    assert [s.agent for s in plan.steps] == ["researcher", "analyst", "summarizer"]


async def test_planner_slow_response_eventually_succeeds_without_fallback():
    good = '{"goal":"g","steps":[{"id":"s1","agent":"analyst","description":"x","depends_on":[]}]}'
    llm = _FlakyLLM(fail_times=1, error_cls=ProviderTimeoutError, then=good)
    planner = Planner(llm, max_attempts=2, backoff_seconds=0)
    plan = await planner.plan("task")
    assert plan.degraded is False
    assert plan.steps[0].agent == "analyst"


async def test_planner_backs_off_between_infra_failures():
    llm = _FlakyLLM(fail_times=2, error_cls=ProviderConnectionError)
    planner = Planner(llm, max_attempts=2, backoff_seconds=0.01)
    plan = await planner.plan("task")
    assert plan.degraded is True
    assert llm.calls == 2  # bounded, not unbounded retry


async def test_planner_normal_plan_is_not_degraded():
    good = '{"goal":"g","steps":[{"id":"s1","agent":"analyst","description":"x","depends_on":[]}]}'
    planner = Planner(_ScriptedLLM([good]))
    plan = await planner.plan("task")
    assert plan.degraded is False
    assert plan.degraded_reason is None


# ---------------------------------------------------------------------
# Requirements extraction (feeds task-completeness tracking).


def test_extract_requirements_prefers_explicit_plan_requirements():
    from src.pipeline.requirements import extract_requirements

    plan = Plan(
        goal="g",
        steps=[PlanStep(id="s1", description="do a thing")],
        requirements=["Pressure must be compared", "SOP must be retrieved"],
    )
    reqs = extract_requirements(plan)
    assert [r.description for r in reqs] == ["Pressure must be compared", "SOP must be retrieved"]
    assert all(r.required for r in reqs)


def test_extract_requirements_falls_back_to_one_per_step():
    from src.pipeline.requirements import extract_requirements

    plan = Plan(
        goal="g",
        steps=[
            PlanStep(id="s1", description="gather facts"),
            PlanStep(id="s2", description="summarise", depends_on=["s1"]),
        ],
    )
    reqs = extract_requirements(plan)
    assert [r.source_step_id for r in reqs] == ["s1", "s2"]
    assert [r.description for r in reqs] == ["gather facts", "summarise"]


def test_extract_requirements_nonempty_for_fallback_plan():
    from src.pipeline.requirements import extract_requirements

    plan = Planner.fallback_plan("Analyze something")
    reqs = extract_requirements(plan)
    assert len(reqs) == 3
    assert all(r.required for r in reqs)
