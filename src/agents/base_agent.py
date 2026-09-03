from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Common interface for all AIOnSite agents."""

    @abstractmethod
    async def run(self, state):
        raise NotImplementedError
