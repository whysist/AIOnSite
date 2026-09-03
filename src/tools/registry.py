class ToolRegistry:
    """Central registry for tools available to agents."""

    def __init__(self):
        self._tools = {}

    def register(self, tool):
        self._tools[tool.name] = tool

    def get(self, name):
        return self._tools.get(name)

    def all(self):
        return list(self._tools.values())
