from uuid import uuid4

from .schemas import AgentState


def create_session(context: dict | None = None) -> AgentState:
    return AgentState(
        session_id=str(uuid4()),
        context=context or {},
    )
