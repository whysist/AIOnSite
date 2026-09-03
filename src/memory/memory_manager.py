class MemoryManager:
    """Initial short-term memory abstraction."""

    def __init__(self):
        self._memory = {}

    def get(self, session_id):
        return self._memory.get(session_id)

    def set(self, session_id, value):
        self._memory[session_id] = value
