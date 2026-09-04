"""InspectionAgent -- the domain entry-point agent.

Retains the legacy :class:`BaseAgent` contract (``run(state)`` mutating an
``AgentState``) but now does real work: it runs the full orchestrator on
the task carried in the state and writes the result back.
"""

from __future__ import annotations

from typing import Any

from ..core.logging import get_logger
from ..prompts.system_prompts import INSPECTION_SYSTEM_PROMPT
from .base_agent import BaseAgent, LLMAgent

_log = get_logger("agents.inspection")


class InspectionAgent(BaseAgent):
    """Primary agent for inspection workflows."""

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    async def run(self, state: Any) -> Any:
        task = ""
        if getattr(state, "messages", None):
            task = state.messages[-1].get("content", "")
        task = task or state.context.get("task", "") if hasattr(state, "context") else task

        if self._llm is None:
            # No model bound -- record intent without fabricating an answer.
            state.metadata["inspection"] = {"status": "no_model", "task": task}
            return state

        agent = LLMAgent(
            name="inspection_agent",
            description="Primary inspection agent",
            system_prompt=INSPECTION_SYSTEM_PROMPT,
            llm=self._llm,
            metadata={"kind": "inspection"},
        )
        result = await agent.execute(task, context=state.context)
        state.messages.append({"role": "assistant", "content": result.output})
        state.metadata["inspection"] = result.model_dump()
        return state
