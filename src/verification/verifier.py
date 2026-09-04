"""Result verifier.

Two layers, cheap first:

1. **Rule-based checks** (always run, no model): empty output, error markers,
   truncation, unmet "must reference evidence" expectations, contradiction
   keywords, malformed JSON when structured output was required.
2. **Optional LLM critique** (when an :class:`BaseLLM` is supplied): asks the
   model to judge correctness / completeness / consistency and return a
   structured verdict.  Failures here degrade gracefully to the rule-based
   score rather than blowing up the run.

The verifier never turns "no evidence" into a pass, and it never loops:
callers enforce ``max_retries`` / ``max_replans``.
"""

from __future__ import annotations

import json
import re

from ..core.exceptions import VerificationError
from ..core.logging import get_logger
from ..llm.base import BaseLLM
from ..pipeline.models import ToolResult
from ..pipeline.requirements import TaskRequirement
from ..state.evidence import Evidence
from .schemas import (
    MissingEvidence,
    MissingRequirement,
    Recommendation,
    Severity,
    VerificationIssue,
    VerificationResult,
)

_log = get_logger("verification")

_ERROR_MARKERS = ("traceback (most recent call last)", "exception:", "error:")
_CONTRADICTION_MARKERS = ("however, this contradicts", "conflicting", "cannot be both")

_CRITIQUE_SYSTEM = (
    "You are a strict verification agent. Given a TASK and a RESULT, judge the "
    "result for correctness, completeness, consistency and unsupported claims. "
    "Respond ONLY with JSON: {\"passed\": bool, \"score\": 0..1, "
    "\"issues\": [{\"code\": str, \"message\": str, \"severity\": "
    "\"info|minor|major|critical\"}], \"recommendation\": "
    "\"accept|retry|correct|replan|escalate|reject\"}."
)


class ResultVerifier:
    def __init__(self, llm: BaseLLM | None = None, *, min_score: float = 0.5) -> None:
        self._llm = llm
        self._min_score = min_score

    # ------------------------------------------------------------------
    async def verify(
        self,
        task: str,
        result: str,
        *,
        require_evidence: bool = False,
        require_json: bool = False,
        structured: dict | None = None,
        requirements: list[TaskRequirement] | None = None,
        evidence: list[Evidence] | None = None,
        tool_failures: list[ToolResult] | None = None,
        node_outcomes: dict[str, str] | None = None,
    ) -> VerificationResult:
        issues: list[VerificationIssue] = []
        missing_requirements: list[MissingRequirement] = []
        missing_evidence: list[MissingEvidence] = []
        requirement_statuses: dict[str, str] = {}
        text = (result or "").strip()

        if not text and not structured:
            issues.append(
                VerificationIssue(
                    code="empty_output",
                    message="The result is empty.",
                    severity=Severity.CRITICAL,
                )
            )

        lowered = text.lower()
        if any(m in lowered for m in _ERROR_MARKERS):
            issues.append(
                VerificationIssue(
                    code="error_marker",
                    message="The result contains an error/traceback marker.",
                    severity=Severity.MAJOR,
                )
            )
        if any(m in lowered for m in _CONTRADICTION_MARKERS):
            issues.append(
                VerificationIssue(
                    code="contradiction",
                    message="The result appears to contain an unresolved contradiction.",
                    severity=Severity.MAJOR,
                )
            )
        if require_json:
            try:
                json.loads(text)
            except (json.JSONDecodeError, TypeError):
                if structured is None:
                    issues.append(
                        VerificationIssue(
                            code="malformed_json",
                            message="Structured output was required but the result is not valid JSON.",
                            severity=Severity.MAJOR,
                        )
                    )
        evidence = evidence or []
        requirements = requirements or []
        tool_failures = tool_failures or []

        if require_evidence and not evidence and not _looks_evidenced(text):
            missing_evidence.append(
                MissingEvidence(description="no retrieved evidence was attached to this execution")
            )
            issues.append(
                VerificationIssue(
                    code="missing_evidence",
                    message="No citation/evidence reference found although evidence was required.",
                    severity=Severity.MAJOR,
                )
            )

        if tool_failures:
            failed_names = sorted({tr.tool for tr in tool_failures})
            issues.append(
                VerificationIssue(
                    code="unresolved_tool_failure",
                    message=(
                        f"{len(failed_names)} tool(s) failed and were not recovered by "
                        "repair: " + ", ".join(failed_names)
                    ),
                    severity=Severity.CRITICAL,
                )
            )

        # Completeness: every *required* TaskRequirement must be traceable
        # either to a plan step that actually completed successfully, or
        # (for requirements with no single producing step) to a mention in
        # the final text. This is what stops "the DAG finished" from being
        # reported as "the task was done" -- see TaskStatus in
        # ``src/pipeline/task_status.py`` for how this feeds the final gate.
        for req in requirements:
            outcome = (node_outcomes or {}).get(req.source_step_id or "")
            if outcome == "satisfied":
                status = "satisfied"
            elif outcome in ("blocked", "failed"):
                status = "blocked"
            elif outcome == "degraded":
                status = "unsatisfied"
            elif _requirement_mentioned(req.description, text):
                status = "satisfied"
            else:
                status = "unsatisfied"
            requirement_statuses[req.id] = status
            if req.required and status != "satisfied":
                reason = (
                    f"the step meant to satisfy this did not complete successfully "
                    f"(outcome={outcome!r})" if outcome
                    else "no supporting evidence or completed step found"
                )
                missing_requirements.append(
                    MissingRequirement(
                        requirement_id=req.id, description=req.description, reason=reason,
                    )
                )

        rule_result = _score_from_issues(issues)
        if missing_requirements or missing_evidence:
            rule_result.passed = False
            rule_result.score = min(rule_result.score, 0.4)
            if rule_result.recommendation is Recommendation.ACCEPT:
                rule_result.recommendation = Recommendation.REPLAN
        rule_result.missing_requirements = missing_requirements
        rule_result.missing_evidence = missing_evidence
        rule_result.requirement_statuses = requirement_statuses

        llm = self._llm
        if llm is None:
            return rule_result

        try:
            critique = await self._llm_critique(llm, task, text or json.dumps(structured))
        except Exception as exc:  # noqa: BLE001 - never fail the run on critique error
            _log.warning("llm_critique_failed", error=str(exc))
            return rule_result

        merged = _merge(rule_result, critique, self._min_score)
        merged.missing_requirements = missing_requirements
        merged.missing_evidence = missing_evidence
        merged.requirement_statuses = requirement_statuses
        if missing_requirements or missing_evidence:
            # A completeness gap is a fact about execution, not a matter of
            # LLM-critique opinion -- the critique can only ever agree that
            # this blocks passing, never override it back to a pass.
            merged.passed = False
        return merged

    # ------------------------------------------------------------------
    async def _llm_critique(
        self, llm: BaseLLM, task: str, result: str
    ) -> VerificationResult:
        raw = await llm.generate_json(
            [
                BaseLLM.system(_CRITIQUE_SYSTEM),
                BaseLLM.user(f"TASK:\n{task}\n\nRESULT:\n{result}"),
            ],
            metadata={"aionsite_kind": "verify"},
            temperature=0.0,
        )
        if not isinstance(raw, dict):
            raise VerificationError("critique did not return an object")
        issues = [
            VerificationIssue(
                code=str(i.get("code", "llm_issue")),
                message=str(i.get("message", "")),
                severity=_coerce_severity(i.get("severity")),
            )
            for i in raw.get("issues", [])
            if isinstance(i, dict)
        ]
        return VerificationResult(
            passed=bool(raw.get("passed", False)),
            score=_clamp(float(raw.get("score", 0.0))),
            recommendation=_coerce_reco(raw.get("recommendation")),
            issues=issues,
            checker="llm",
        )


# ----------------------------------------------------------------------
def _looks_evidenced(text: str) -> bool:
    markers = ("source:", "page", "[", "http://", "https://", "citation", "evidence")
    return any(m in text.lower() for m in markers)


def _requirement_mentioned(description: str, text: str) -> bool:
    """Weak fallback check: does the final text plausibly address *description*?

    Only used when there is no ``node_outcomes`` signal for the requirement's
    producing step (e.g. a plan-level requirement with no single source
    step). This is intentionally a coarse keyword-overlap heuristic, not a
    semantic check -- it exists to avoid false negatives for requirements
    that have no step to point at, not to replace the step-outcome check.
    """
    words = re.findall(r"[a-zA-Z]{4,}", description.lower())[:8]
    if not words:
        return True
    lowered = text.lower()
    hits = sum(1 for w in words if w in lowered)
    return hits >= max(1, len(words) // 2)


def _score_from_issues(issues: list[VerificationIssue]) -> VerificationResult:
    penalty = {
        Severity.INFO: 0.0,
        Severity.MINOR: 0.1,
        Severity.MAJOR: 0.4,
        Severity.CRITICAL: 1.0,
    }
    score = max(0.0, 1.0 - sum(penalty[i.severity] for i in issues))
    blocking = any(i.severity in (Severity.MAJOR, Severity.CRITICAL) for i in issues)
    passed = not blocking and score >= 0.5
    if passed:
        reco = Recommendation.ACCEPT
    elif any(i.severity is Severity.CRITICAL for i in issues):
        reco = Recommendation.REPLAN
    else:
        reco = Recommendation.RETRY
    return VerificationResult(
        passed=passed, score=round(score, 3), recommendation=reco, issues=issues
    )


def _merge(
    rule: VerificationResult, llm: VerificationResult, min_score: float
) -> VerificationResult:
    issues = rule.issues + llm.issues
    score = round(min(rule.score, llm.score), 3)
    passed = rule.passed and llm.passed and score >= min_score
    if passed:
        reco = Recommendation.ACCEPT
    else:
        # take the more conservative of the two recommendations
        order = list(Recommendation)
        reco = max((rule.recommendation, llm.recommendation), key=order.index)
    return VerificationResult(
        passed=passed,
        score=score,
        recommendation=reco,
        issues=issues,
        checker="rule-based+llm",
    )


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _coerce_severity(value: object) -> Severity:
    try:
        return Severity(str(value).lower())
    except ValueError:
        return Severity.MINOR


def _coerce_reco(value: object) -> Recommendation:
    try:
        return Recommendation(str(value).lower())
    except ValueError:
        return Recommendation.RETRY
