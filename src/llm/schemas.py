from pydantic import BaseModel, Field


class LLMRequest(BaseModel):
    messages: list[dict]
    temperature: float = Field(default=0.2, ge=0, le=2)


class LLMResult(BaseModel):
    content: str
    model: str | None = None
    metadata: dict = Field(default_factory=dict)
