
from src.llm.base import BaseLLM
from src.llm.schemas import LLMResponse
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
