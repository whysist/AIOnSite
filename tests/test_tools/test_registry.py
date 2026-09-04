import pytest

from src.core.exceptions import ToolExecutionError, ToolNotFoundError
from src.tools.base_tool import ToolPermission
from src.tools.builtin import CalculatorTool, DeviationTool
from src.tools.registry import ToolRegistry, default_registry


async def test_registry_register_and_call():
    reg = ToolRegistry(allowed_permissions={ToolPermission.PURE})
    reg.register(CalculatorTool())
    assert reg.has("calculator")
    result = await reg.call("calculator", expression="2 * (3 + 4)")
    assert result.ok is True
    assert result.output["result"] == 14.0


def test_registry_unknown_tool_raises():
    with pytest.raises(ToolNotFoundError):
        ToolRegistry().get("nope")


def test_registry_rejects_disallowed_permissions():
    reg = ToolRegistry(allowed_permissions={ToolPermission.PURE})

    from src.tools.builtin.fs_reader import FileReadTool

    with pytest.raises(ToolExecutionError):
        reg.register(FileReadTool())


def test_default_registry_has_builtins_but_no_network_tool():
    reg = default_registry(allow_network=False)
    assert "calculator" in reg.names()
    assert "calculate_deviation" in reg.names()


async def test_calculator_rejects_names_and_calls():
    result = await CalculatorTool().run(expression="__import__('os').system('x')")
    assert result.ok is False


async def test_calculator_division_by_zero_is_handled():
    result = await CalculatorTool().run(expression="1/0")
    assert result.ok is False
    assert "zero" in result.error.lower()


async def test_deviation_tool_math():
    result = await DeviationTool().run(actual=120, limit=100)
    assert result.ok is True
    assert result.output["absolute_deviation"] == 20
    assert result.output["percent_deviation"] == 20.0
    assert result.output["within_limit"] is False


async def test_tool_validates_arguments():
    result = await DeviationTool().run(actual="not-a-number", limit=1)
    assert result.ok is False
    assert "invalid arguments" in result.error


async def test_tool_missing_required_field_is_a_validation_error_with_schema():
    result = await DeviationTool().run(actual=120)  # missing 'limit'
    assert result.ok is False
    assert result.error_kind == "validation"
    assert result.expected_schema is not None
    assert set(result.expected_schema["required"]) == {"actual", "limit"}


async def test_tool_wrong_field_names_is_a_validation_error():
    """Reproduces the reported failure: the model guessed field names that
    don't match the tool's real schema (e.g. 'observed_pressure'/'approved_limit'
    instead of 'actual'/'limit')."""
    result = await DeviationTool().run(observed_pressure=120, approved_limit=100)
    assert result.ok is False
    assert result.error_kind == "validation"
    assert "actual" in result.expected_schema["required"]
    assert "limit" in result.expected_schema["required"]


async def test_tool_execution_failure_has_execution_error_kind():
    result = await CalculatorTool().run(expression="1/0")
    assert result.ok is False
    assert result.error_kind == "execution"
    assert result.expected_schema is None
