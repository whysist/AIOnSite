"""Tool registry.

Agents look tools up by name here rather than importing implementations.
The registry can be constructed with an *allow-list* of permissions so a
deployment can, for example, forbid every network tool.
"""

from __future__ import annotations

from ..core.exceptions import ToolExecutionError, ToolNotFoundError
from ..core.logging import get_logger
from .base_tool import BaseTool, ToolPermission

_log = get_logger("tools.registry")


class ToolRegistry:
    def __init__(
        self, *, allowed_permissions: set[ToolPermission] | None = None
    ) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._allowed = allowed_permissions

    # ------------------------------------------------------------------
    def register(self, tool: BaseTool, *, replace: bool = False) -> None:
        if tool.name in self._tools and not replace:
            raise ToolExecutionError(f"tool {tool.name!r} already registered")
        if self._allowed is not None:
            missing = set(tool.permissions) - self._allowed
            if missing:
                raise ToolExecutionError(
                    f"tool {tool.name!r} needs disallowed permissions: "
                    + ", ".join(sorted(p.value for p in missing))
                )
        self._tools[tool.name] = tool
        _log.info("tool_registered", tool=tool.name)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(f"unknown tool: {name!r}") from None

    def has(self, name: str) -> bool:
        return name in self._tools

    def all(self) -> list[BaseTool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict]:
        return [t.spec() for t in self._tools.values()]

    async def call(self, name: str, **kwargs: object):
        return await self.get(name).run(**kwargs)


def default_registry(
    *, allow_network: bool = False, allow_sandbox: bool = False, allow_write_filesystem: bool = False,
) -> ToolRegistry:
    """Registry pre-loaded with the safe built-in tools.

    ``allow_sandbox`` gates ``ToolPermission.SANDBOXED_EXEC`` and
    ``allow_write_filesystem`` gates ``ToolPermission.WRITE_FILESYSTEM`` --
    without them, ``SandboxedPythonTool`` / ``GenerateWordDocumentTool`` are
    skipped at registration (their permission is not in ``allowed``), the
    same way a network tool is skipped unless ``allow_network=True``. This
    keeps arbitrary-code-execution and filesystem-write capability strictly
    opt-in.
    """
    from .builtin import BUILTIN_TOOLS

    allowed = {ToolPermission.PURE, ToolPermission.READ_FILESYSTEM}
    if allow_network:
        allowed.add(ToolPermission.NETWORK)
    if allow_sandbox:
        allowed.add(ToolPermission.SANDBOXED_EXEC)
    if allow_write_filesystem:
        allowed.add(ToolPermission.WRITE_FILESYSTEM)
    registry = ToolRegistry(allowed_permissions=allowed)
    for tool_cls in BUILTIN_TOOLS:
        try:
            registry.register(tool_cls())
        except ToolExecutionError:
            # permission not granted in this deployment -- skip quietly
            continue
    return registry
