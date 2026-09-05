from src.sandbox.executor import SandboxExecutor
from src.tools.base_tool import ToolPermission
from src.tools.builtin.code_exec import SandboxedPythonTool
from src.tools.registry import default_registry


async def test_execute_python_runs_and_returns_stdout():
    tool = SandboxedPythonTool(SandboxExecutor(use_docker=False, timeout_seconds=5.0))
    result = await tool.run(code="print(21 * 2)")
    assert result.ok is True
    assert result.output["stdout"].strip() == "42"


async def test_execute_python_rejects_disallowed_code_as_validation_style_failure():
    tool = SandboxedPythonTool(SandboxExecutor(use_docker=False, timeout_seconds=5.0))
    result = await tool.run(code="import os")
    assert result.ok is False
    assert result.error_kind == "execution"
    assert "os" in result.error


def test_sandbox_disabled_by_default_tool_not_registered():
    registry = default_registry(allow_network=False, allow_sandbox=False)
    assert not registry.has("execute_python")


def test_sandbox_enabled_registers_the_tool():
    registry = default_registry(allow_network=False, allow_sandbox=True)
    assert registry.has("execute_python")
    assert ToolPermission.SANDBOXED_EXEC in registry.get("execute_python").permissions


def test_execute_python_not_cacheable():
    assert SandboxedPythonTool.cacheable is False
