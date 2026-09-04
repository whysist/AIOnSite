"""Project-specific exception hierarchy.

A single base class (:class:`AIOnSiteError`) makes it easy for the API layer
to catch "anything we raised on purpose" and turn it into a clean response
without leaking stack traces in production.
"""

from __future__ import annotations


class AIOnSiteError(Exception):
    """Base class for all deliberate errors raised by this project."""

    #: short machine-readable code, surfaced in API error payloads
    code: str = "aionsite_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, object]:
        return {"error": self.code, "message": self.message, "details": self.details}


class ConfigurationError(AIOnSiteError):
    """Invalid or missing configuration."""

    code = "configuration_error"


class LLMError(AIOnSiteError):
    """Generic failure in the LLM abstraction layer."""

    code = "llm_error"


class ProviderError(LLMError):
    """An LLM provider (OpenAI / Ollama / vLLM / ...) failed or is unreachable."""

    code = "provider_error"


class ProviderConnectionError(ProviderError):
    """The provider's endpoint refused the connection or could not be resolved.

    Distinct from :class:`ProviderTimeoutError`: this means the server is not
    there to talk to at all (e.g. ``ollama serve`` is not running), whereas a
    timeout means it *is* there but did not respond in time (e.g. the model is
    still loading). Conflating the two produces a misleading "is it running?"
    message even when the server is up and just slow.
    """

    code = "provider_connection_error"


class ProviderTimeoutError(ProviderError):
    """The provider did not respond within the configured timeout.

    Callers should not treat this identically to a connection failure: a slow
    local model load is expected behaviour, not evidence the server is down.
    """

    code = "provider_timeout_error"


class ProviderMalformedResponseError(ProviderError):
    """The provider responded, but the response body could not be parsed."""

    code = "provider_malformed_response_error"


class SovereigntyError(AIOnSiteError):
    """An operation was blocked because it would violate sovereign / local-only policy."""

    code = "sovereignty_error"


class AgentExecutionError(AIOnSiteError):
    """An agent failed while executing its task."""

    code = "agent_execution_error"


class PlanningError(AgentExecutionError):
    """The planner could not produce a valid structured plan."""

    code = "planning_error"


class PipelineExecutionError(AIOnSiteError):
    """The pipeline execution engine failed."""

    code = "pipeline_execution_error"


class ToolExecutionError(AIOnSiteError):
    """A tool raised an error or returned an invalid result."""

    code = "tool_execution_error"


class ToolNotFoundError(ToolExecutionError):
    """A requested tool is not present in the registry."""

    code = "tool_not_found"


class VerificationError(AIOnSiteError):
    """The verification layer failed to evaluate a result."""

    code = "verification_error"
