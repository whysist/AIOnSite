"""Provider-neutral request / response models for the LLM layer.

Nothing outside ``src/llm/`` should ever see a provider-specific response
object (an ``openai`` SDK model, a raw Ollama dict, ...).  Providers must
translate their native output into :class:`LLMResponse`.
"""

from __future__ import annotations

import enum
import time
from typing import Any

from pydantic import BaseModel, Field


class Role(str, enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    role: Role
    content: str
    name: str | None = None

    def as_dict(self) -> dict[str, str]:
        d = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


def _coerce_messages(messages: list[Any]) -> list[Message]:
    out: list[Message] = []
    for m in messages:
        if isinstance(m, Message):
            out.append(m)
        elif isinstance(m, dict):
            out.append(Message(role=Role(m["role"]), content=m.get("content", ""),
                               name=m.get("name")))
        else:  # pragma: no cover - defensive
            raise TypeError(f"Unsupported message type: {type(m)!r}")
    return out


class LLMRequest(BaseModel):
    """A single chat-completion request, independent of any provider."""

    messages: list[Message]
    model: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, gt=0)
    stop: list[str] | None = None
    # When set, the provider is asked to return JSON (best-effort for
    # providers without native structured-output support).
    json_mode: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_messages(cls, messages: list[Any], **kwargs: Any) -> "LLMRequest":
        return cls(messages=_coerce_messages(messages), **kwargs)

    def wire_messages(self) -> list[dict[str, str]]:
        return [m.as_dict() for m in self.messages]


class UsageMetadata(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "UsageMetadata") -> "UsageMetadata":
        return UsageMetadata(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class LLMResponse(BaseModel):
    """Normalised model output returned to agents."""

    content: str
    model: str | None = None
    provider: str | None = None
    finish_reason: str | None = None
    usage: UsageMetadata = Field(default_factory=UsageMetadata)
    latency_ms: float | None = None
    created_at: float = Field(default_factory=time.time)
    raw: dict[str, Any] | None = Field(default=None, repr=False)
    metadata: dict[str, Any] = Field(default_factory=dict)


# Backwards-compatible alias: the original skeleton called this ``LLMResult``.
LLMResult = LLMResponse
