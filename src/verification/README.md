# Verification Layer

**Implemented.**  See `schemas.py` and `verifier.py` (exported from
`src.verification`).

- `VerificationResult` — `passed`, `score` (0..1), `recommendation`
  (`accept | retry | correct | replan | escalate | reject`), `issues`
  (`VerificationIssue` with `severity`), `checker`.
- `ResultVerifier.verify(task, result, ...)` runs:
  1. **Rule-based checks** (no model): empty output, error/traceback
     markers, unresolved contradictions, malformed JSON when structured
     output was required, missing evidence when `require_evidence=True`.
  2. **Optional LLM critique** (when a `BaseLLM` is injected): structured
     correctness / completeness / consistency judgement, merged
     conservatively with the rule-based score. Critique failure degrades
     gracefully to the rule-based result.

A terminal `verifier` node is appended to every pipeline by the builder.
The verifier never converts "no evidence" into a pass; retry/replan counts
are bounded by `MAX_RETRIES` / `MAX_REPLANS`.
