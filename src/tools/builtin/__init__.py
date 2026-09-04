"""Built-in deterministic tools."""

from src.tools.builtin.calculator import CalculatorTool, DeviationTool
from src.tools.builtin.fs_reader import FileReadTool
from src.tools.builtin.json_tool import JsonParseTool
from src.tools.builtin.text_stats import TextStatsTool
from src.tools.builtin.industrial import (
    ParameterComparisonTool,
    RiskScoreTool,
)

BUILTIN_TOOLS = [
    CalculatorTool,
    DeviationTool,
    FileReadTool,
    JsonParseTool,
    TextStatsTool,
    ParameterComparisonTool,
    RiskScoreTool,
]

__all__ = [
    "CalculatorTool",
    "DeviationTool",
    "FileReadTool",
    "JsonParseTool",
    "TextStatsTool",
    "ParameterComparisonTool",
    "RiskScoreTool",
    "BUILTIN_TOOLS",
]