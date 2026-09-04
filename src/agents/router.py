"""Routing.

* :class:`AgentRouter` -- unchanged tiny helper: name -> agent instance.
* :class:`ModelRouter` -- decides which provider/model handles a given
  pipeline node, subject to policy:

    - sovereign mode / ``require_local`` / confidential task  => local only
    - explicit ``node.provider`` / ``node.model``            => honoured if allowed
    - otherwise                                              => configured default,
      with an optional complexity-based model map from ``configs/models.yaml``

  Cloud routes are refused (not silently downgraded) when policy forbids them.
"""

from __future__ import annotations

from typing import Any

from ..core.config import LLMProvider, Settings, get_settings
from ..core.exceptions import SovereigntyError
from ..core.logging import get_logger
from ..llm.base import BaseLLM
from ..llm.factory import create_llm
from ..pipeline.models import PipelineNode

_log = get_logger("router")

_LOCAL = {"local", "ollama", "vllm", "echo"}


class AgentRouter:
    """Routes tasks to the appropriate agent or workflow."""

    def __init__(self, agents: dict) -> None:
        self.agents = agents

    def get_agent(self, name: str):
        return self.agents.get(name)


class RouteDecision:
    __slots__ = ("provider", "model", "reason", "is_local")

    def __init__(self, provider: str, model: str | None, reason: str, is_local: bool) -> None:
        self.provider = provider
        self.model = model
        self.reason = reason
        self.is_local = is_local

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "reason": self.reason,
            "is_local": self.is_local,
        }


class ModelRouter:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model_config: dict[str, Any] | None = None,
        confidential: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.confidential = confidential
        self._config = model_config or {}
        self._cache: dict[tuple[str, str | None], BaseLLM] = {}

    # ------------------------------------------------------------------
    def _local_only(self, node: PipelineNode) -> bool:
        return bool(
            self.settings.sovereign_mode or self.confidential or node.require_local
        )

    def _default_provider(self) -> str:
        return getattr(self.settings.llm_provider, "value", str(self.settings.llm_provider))

    def _model_for(self, node: PipelineNode, provider: str) -> str | None:
        if node.model:
            return node.model
        routing = self._config.get("routing", {})
        complexity = str(node.metadata.get("complexity", "")) if hasattr(node, "metadata") else ""
        by_complexity = routing.get(complexity) if isinstance(routing, dict) else None
        if isinstance(by_complexity, dict) and by_complexity.get("model"):
            return by_complexity["model"]
        prov_cfg = self._config.get(provider) if isinstance(self._config, dict) else None
        if isinstance(prov_cfg, dict) and prov_cfg.get("model"):
            return prov_cfg["model"]
        return self.settings.llm_model

    def route(self, node: PipelineNode) -> RouteDecision:
        local_only = self._local_only(node)
        requested = node.provider or self._default_provider()
        requested = requested.lower()

        if local_only and requested not in _LOCAL:
            if not self.settings.provider_allowed(LLMProvider(requested)):
                raise SovereigntyError(
                    f"node {node.id!r} would use cloud provider {requested!r} but "
                    "sovereign / confidential policy requires a local provider."
                )
            fallback = self._first_local_provider()
            _log.warning("route_forced_local", node=node.id, requested=requested, chosen=fallback)
            requested = fallback

        model = self._model_for(node, requested)
        reason = (
            "policy: local-only" if local_only
            else ("node override" if node.provider else "configured default")
        )
        return RouteDecision(
            provider=requested,
            model=model,
            reason=reason,
            is_local=requested in _LOCAL,
        )

    def _first_local_provider(self) -> str:
        for candidate in ("ollama", "local", "vllm", "echo"):
            return candidate
        return "echo"  # pragma: no cover

    def get_llm(self, node: PipelineNode) -> tuple[BaseLLM, RouteDecision]:
        decision = self.route(node)
        key = (decision.provider, decision.model)
        if key not in self._cache:
            self._cache[key] = create_llm(
                self.settings, provider=decision.provider, model=decision.model
            )
        return self._cache[key], decision

    async def aclose(self) -> None:
        for llm in self._cache.values():
            await llm.aclose()
        self._cache.clear()
