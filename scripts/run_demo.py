"""End-to-end demonstration of the agentic workbench.

    User task
      -> Planner        (structured plan)
      -> Pipeline DAG   (builder)
      -> Router + Agents (per-node provider/model, tool use)
      -> Verifier       (terminal node)
      -> Final answer
      -> Audit trail

Runs fully offline with the deterministic ``echo`` provider by default so
it works from a clean checkout with no model server.  Point it at a real
local model with, e.g.::

    LLM_PROVIDER=ollama LLM_MODEL=qwen2.5:7b-instruct python scripts/run_demo.py

Usage: ``python scripts/run_demo.py ["custom task text"]``
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Default to the offline provider unless the caller chose one.
os.environ.setdefault("LLM_PROVIDER", "echo")

from src.core.config import reload_settings  # noqa: E402
from src.core.logging import configure_logging  # noqa: E402
from src.orchestrator import Orchestrator  # noqa: E402

DEFAULT_TASK = (
    "Analyze the V-101 inspection package: compare the observed operating "
    "pressure of 120 bar against the approved limit of 100 bar, review the "
    "maintenance history, calculate the deviation, retrieve the relevant SOP, "
    "verify the evidence, and produce a final inspection recommendation."
)


def _rule(title: str) -> None:
    print("\n" + "=" * 78 + f"\n  {title}\n" + "=" * 78)


async def main() -> int:
    settings = reload_settings()
    configure_logging(settings.log_level, json_output=False)
    task = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TASK

    _rule(f"TASK  (provider={settings.llm_provider.value}, sovereign={settings.sovereign_mode})")
    print(task)

    orch = Orchestrator(settings)
    try:
        ctx = await orch.run_task(task)
    finally:
        await orch.aclose()

    _rule("PLAN")
    for step in (ctx.plan or {}).get("steps", []):
        print(f"  {step['id']:<8} {step['agent']:<11} <- {step['depends_on'] or '[]'}")
        print(f"           {step['description']}")

    _rule("PIPELINE (DAG)")
    for node in ctx.pipeline.nodes:  # type: ignore[union-attr]
        print(f"  [{node.status.value:<9}] {node.id:<13} {node.type.value:<11} "
              f"provider={node.provider or '-'} model={node.model or '-'}")

    _rule("NODE RESULTS")
    for nid, res in ctx.state.node_results.items():
        preview = res.output.replace("\n", " ")[:160]
        print(f"  {nid}: ({len(res.tool_results)} tool call(s)) {preview}")

    _rule("VERIFICATION")
    fv = ctx.final_verification
    if fv:
        print(f"  passed={fv.passed}  score={fv.score}  recommendation={fv.recommendation.value}")
        for issue in fv.issues:
            print(f"   - [{issue.severity.value}] {issue.code}: {issue.message}")
    else:
        print("  (no verification result)")

    _rule("FINAL ANSWER")
    print(ctx.final_answer or "(none)")

    _rule("AUDIT TRAIL")
    for event in orch.audit.events(ctx.execution_id):
        print(f"  {event.event_type.value:<26} {event.component:<12} {event.status}")

    _rule(f"RESULT: {ctx.status.value.upper()}  (execution_id={ctx.execution_id})")
    return 0 if ctx.status.value in ("completed", "needs_review") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
