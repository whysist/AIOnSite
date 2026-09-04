
from src.llm.base import BaseLLM
from src.llm.schemas import LLMResponse
from src.pipeline.models import ToolResult
from src.pipeline.requirements import TaskRequirement
from src.state.evidence import Evidence
from src.verification.schemas import Recommendation
from src.verification.verifier import ResultVerifier


async def test_pass_on_clean_result():
    v = ResultVerifier(None)
    res = await v.verify("task", "The pressure of 120 bar exceeds the 100 bar limit. Source: SOP-12 page 3.")
    assert res.passed is True
    assert res.recommendation is Recommendation.ACCEPT


async def test_fail_on_empty_result_triggers_replan():
    res = await ResultVerifier(None).verify("task", "")
    assert res.passed is False
    assert res.recommendation is Recommendation.REPLAN


async def test_fail_on_error_marker_triggers_retry():
    res = await ResultVerifier(None).verify("task", "Error: could not compute deviation")
    assert res.passed is False
    assert res.recommendation in (Recommendation.RETRY, Recommendation.REPLAN)


async def test_missing_evidence_flagged_when_required():
    res = await ResultVerifier(None).verify(
        "task", "The equipment is fine.", require_evidence=True
    )
    assert res.passed is False
    assert any(i.code == "missing_evidence" for i in res.issues)


async def test_llm_critique_merged_conservatively():
    class Critic(BaseLLM):
        provider_name = "critic"

        async def _complete(self, request):
            return LLMResponse(
                content='{"passed": false, "score": 0.2, "issues": [{"code":"c","message":"m","severity":"major"}], "recommendation": "retry"}',
                provider="critic",
            )

    res = await ResultVerifier(Critic()).verify(
        "task", "Looks fine with Source: X page 1"
    )
    assert res.passed is False
    assert res.checker == "rule-based+llm"


async def test_llm_critique_failure_degrades_to_rules():
    class Broken(BaseLLM):
        provider_name = "broken"

        async def _complete(self, request):
            raise RuntimeError("model down")

    res = await ResultVerifier(Broken()).verify(
        "task", "Clean result. Source: doc page 2."
    )
    assert res.checker == "rule-based"
    assert res.passed is True


# ---------------------------------------------------------------------
# Completeness / evidence-grounding: this is the direct fix for the
# reported bug where a plausible-looking but incomplete answer scored
# passed=True, score=1.0 despite missing required work.


async def test_plausible_answer_with_missing_requirement_fails_verification():
    reqs = [
        TaskRequirement(id="req_1", description="Retrieve the relevant SOP", source_step_id="s1"),
    ]
    # No node_outcomes entry for "s1" (never ran) and the text never
    # mentions the SOP -- a clean-looking answer must still fail.
    res = await ResultVerifier(None).verify(
        "task", "The pressure of 120 bar exceeds the 100 bar limit.",
        requirements=reqs,
    )
    assert res.passed is False
    assert res.score < 1.0
    assert any(m.requirement_id == "req_1" for m in res.missing_requirements)
    assert res.requirement_statuses["req_1"] != "satisfied"


async def test_score_cannot_be_perfect_when_mandatory_work_failed():
    reqs = [TaskRequirement(id="req_1", description="Review maintenance history", source_step_id="s1")]
    res = await ResultVerifier(None).verify(
        "task", "Everything looks fine and is fully verified.",
        requirements=reqs, node_outcomes={"s1": "blocked"},
    )
    assert res.passed is False
    assert res.score <= 0.4
    assert res.requirement_statuses["req_1"] == "blocked"


async def test_requirement_satisfied_when_producing_step_succeeded():
    reqs = [TaskRequirement(id="req_1", description="Compute the deviation", source_step_id="s1")]
    res = await ResultVerifier(None).verify(
        "task", "The deviation is 20 bar.",
        requirements=reqs, node_outcomes={"s1": "satisfied"},
    )
    assert res.requirement_statuses["req_1"] == "satisfied"
    assert not any(m.requirement_id == "req_1" for m in res.missing_requirements)


async def test_missing_evidence_blocks_recommendation_even_with_clean_text():
    res = await ResultVerifier(None).verify(
        "task", "The equipment is within limits and the inspection is complete.",
        require_evidence=True, evidence=[],
    )
    assert res.passed is False
    assert res.missing_evidence


async def test_evidence_present_satisfies_the_evidence_requirement():
    ev = [Evidence(source="read_file", producer_tool="read_file", producer_node="n1")]
    res = await ResultVerifier(None).verify(
        "task", "The equipment is within limits.", require_evidence=True, evidence=ev,
    )
    assert not res.missing_evidence


async def test_unresolved_tool_failure_forces_verification_failure():
    tool_failures = [
        ToolResult(tool="calculate_deviation", ok=False, error_kind="validation", error="bad args")
    ]
    res = await ResultVerifier(None).verify(
        "task", "The deviation is 20 bar (20%).", tool_failures=tool_failures,
    )
    assert res.passed is False
    assert any(i.code == "unresolved_tool_failure" for i in res.issues)


async def test_llm_critique_cannot_override_a_completeness_gap_back_to_pass():
    class OverOptimisticCritic(BaseLLM):
        provider_name = "critic"

        async def _complete(self, request):
            return LLMResponse(
                content='{"passed": true, "score": 1.0, "issues": [], "recommendation": "accept"}',
                provider="critic",
            )

    reqs = [TaskRequirement(id="req_1", description="Retrieve the relevant SOP", source_step_id="s1")]
    res = await ResultVerifier(OverOptimisticCritic()).verify(
        "task", "Final verified answer: accept.",
        requirements=reqs, node_outcomes={"s1": "failed"},
    )
    assert res.passed is False
    assert any(m.requirement_id == "req_1" for m in res.missing_requirements)
