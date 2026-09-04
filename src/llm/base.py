"""Provider-neutral LLM interface.

Agent / planner / verifier code depends only on :class:`BaseLLM` and the
schemas in :mod:`src.llm.schemas`.  Concrete providers live next to this
file and are constructed via :func:`src.llm.factory.create_llm`.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any

from ..core.exceptions import LLMError
from .schemas import LLMRequest, LLMResponse, Message, Role

_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


class BaseLLM(ABC):
    """Abstract chat-completion model.

    Subclasses implement :meth:`_complete`; callers use :meth:`generate`
    (accepts loose ``list[dict]`` messages for convenience) or
    :meth:`complete` (typed :class:`LLMRequest`).
    """

    #: provider identifier, e.g. ``"ollama"`` -- set by subclasses
    provider_name: str = "base"

    def __init__(self, *, model: str | None = None, is_local: bool = True) -> None:
        self.model = model
        #: whether this provider keeps all traffic on-premise
        self.is_local = is_local

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def generate(self, messages: list[Any], **kwargs: Any) -> LLMResponse:
        """Convenience wrapper accepting ``list[dict]`` or ``list[Message]``."""
        request = LLMRequest.from_messages(
            messages,
            model=kwargs.pop("model", self.model),
            temperature=kwargs.pop("temperature", 0.2),
            max_tokens=kwargs.pop("max_tokens", 2000),
            stop=kwargs.pop("stop", None),
            json_mode=kwargs.pop("json_mode", False),
            metadata=kwargs.pop("metadata", {}),
        )
        return await self.complete(request)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        request = request.model_copy(update={"model": request.model or self.model})
        started = time.perf_counter()
        try:
            response = await self._complete(request)
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise provider errors
            raise LLMError(
                f"{self.provider_name} completion failed: {exc}",
                details={"provider": self.provider_name},
            ) from exc
        if response.latency_ms is None:
            response.latency_ms = (time.perf_counter() - started) * 1000
        response.provider = response.provider or self.provider_name
        return response

    async def generate_json(self, messages: list[Any], **kwargs: Any) -> Any:
        """Generate and parse a JSON object/array from the model output.

        Tolerant of models that wrap JSON in prose or code fences.  Raises
        :class:`LLMError` if nothing parseable is found.
        """
        kwargs.setdefault("json_mode", True)
        response = await self.generate(messages, **kwargs)
        return self.extract_json(response.content)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def extract_json(text: str) -> Any:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = re.sub(r"^(json|JSON)\s*", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = _JSON_BLOCK.search(text)
            if not match:
                raise LLMError("Model output contained no JSON") from None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise LLMError(f"Model output was not valid JSON: {exc}") from exc

    @staticmethod
    def system(content: str) -> Message:
        return Message(role=Role.SYSTEM, content=content)

    @staticmethod
    def user(content: str) -> Message:
        return Message(role=Role.USER, content=content)

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------
    @abstractmethod
    async def _complete(self, request: LLMRequest) -> LLMResponse:
        """Perform one completion and return a normalised response."""
        raise NotImplementedError

    async def health_check(self) -> bool:
        """Best-effort reachability probe.  Overridden by network providers."""
        return True

    async def aclose(self) -> None:  # pragma: no cover - overridden where needed
        """Release any held resources (HTTP clients, ...)."""
        return None
