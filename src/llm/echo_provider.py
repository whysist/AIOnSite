"""Deterministic offline provider.

``EchoProvider`` performs no network I/O.  It exists so the full pipeline
(planner -> builder -> executor -> verifier -> audit) can be exercised in
tests and demos on an air-gapped machine with no model server running.

It is *not* a language model: it returns structured, rule-based output.
Agents/planner/verifier tag their requests via
``metadata["aionsite_kind"]`` so this provider can emit a shape they can
parse; real providers ignore that metadata entirely.
"""

from __future__ import annotations

import json
import textwrap

from .base import BaseLLM
from .schemas import LLMRequest, LLMResponse, Role, UsageMetadata


class EchoProvider(BaseLLM):
    provider_name = "echo"

    def __init__(self, *, model: str | None = "echo-1", **_: object) -> None:
        super().__init__(model=model or "echo-1", is_local=True)

    async def _complete(self, request: LLMRequest) -> LLMResponse:
        kind = str(request.metadata.get("aionsite_kind", "")).lower()
        user_text = self._last_user_text(request)

        if kind == "plan":
            content = self._plan(user_text)
        elif kind == "verify":
            content = self._verdict()
        else:
            content = self._answer(kind, user_text)

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider_name,
            finish_reason="stop",
            usage=UsageMetadata(
                prompt_tokens=sum(len(m.content.split()) for m in request.messages),
                completion_tokens=len(content.split()),
                total_tokens=0,
            ),
            metadata={"deterministic": True, "kind": kind or "answer"},
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _last_user_text(request: LLMRequest) -> str:
        for message in reversed(request.messages):
            if message.role is Role.USER:
                return message.content
        return request.messages[-1].content if request.messages else ""

    @staticmethod
    def _plan(task: str) -> str:
        goal = task.strip().splitlines()[0][:200] if task.strip() else "Complete the task"
        plan = {
            "goal": goal,
            "steps": [
                {
                    "id": "step_1",
                    "agent": "researcher",
                    "description": f"Gather the information needed to address: {goal}",
                    "depends_on": [],
                },
                {
                    "id": "step_2",
                    "agent": "analyst",
                    "description": "Analyse the gathered information and compute any required results.",
                    "depends_on": ["step_1"],
                },
                {
                    "id": "step_3",
                    "agent": "summarizer",
                    "description": "Synthesise a final answer from the analysis.",
                    "depends_on": ["step_2"],
                },
            ],
        }
        return json.dumps(plan, indent=2)

    @staticmethod
    def _verdict() -> str:
        return json.dumps(
            {
                "passed": True,
                "score": 0.8,
                "issues": [],
                "recommendation": "accept",
                "rationale": "Deterministic offline verifier: no contradictions detected in the provided result.",
            },
            indent=2,
        )

    @staticmethod
    def _answer(kind: str, user_text: str) -> str:
        snippet = textwrap.shorten(user_text.replace("\n", " ").strip(), width=600,
                                   placeholder=" ...")
        role = kind or "assistant"
        return (
            f"[echo:{role}] Based on the provided context, here is a structured response.\n\n"
            f"Input considered: {snippet}\n\n"
            "Conclusion: the request has been processed deterministically by the offline "
            "provider. Replace the 'echo' provider with 'ollama'/'vllm'/'local' for "
            "real model output."
        )
