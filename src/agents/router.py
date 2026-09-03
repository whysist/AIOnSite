class AgentRouter:
    """Routes tasks to the appropriate agent or workflow."""

    def __init__(self, agents: dict):
        self.agents = agents

    def get_agent(self, name: str):
        return self.agents.get(name)
