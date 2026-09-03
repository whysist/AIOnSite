from pydantic import BaseModel, Field


class AgentState(BaseModel):
    session_id: str

    messages: list[dict] = Field(default_factory=list)
    context: dict = Field(default_factory=dict)

    tool_results: list[dict] = Field(default_factory=list)

    metadata: dict = Field(default_factory=dict)
