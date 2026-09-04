"""Built-in, safe-by-default tools."""

from .calculator import CalculatorTool, DeviationTool
from .fs_reader import FileReadTool
from .json_tool import JsonParseTool
from .text_stats import TextStatsTool

BUILTIN_TOOLS = [
    CalculatorTool,
    DeviationTool,
    TextStatsTool,
    JsonParseTool,
    FileReadTool,
]

__all__ = [
    "BUILTIN_TOOLS",
    "CalculatorTool",
    "DeviationTool",
    "TextStatsTool",
    "JsonParseTool",
    "FileReadTool",
]
