class AIOnSiteError(Exception):
    """Base project exception."""


class LLMError(AIOnSiteError):
    pass


class ToolExecutionError(AIOnSiteError):
    pass


class VerificationError(AIOnSiteError):
    pass
