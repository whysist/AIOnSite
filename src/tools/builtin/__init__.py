"""Built-in, safe-by-default tools."""

from collections.abc import Callable
from typing import Any

from ..base_tool import BaseTool
from .calculator import CalculatorTool, DeviationTool
from .db_tool import EquipmentLookupTool
from .document import ProcessDocumentTool
from .fs_reader import FileReadTool
from .json_tool import JsonParseTool
from .kb import ExtractStructuredEvidenceTool, KnowledgeBaseSearchTool
from .text_stats import TextStatsTool

#: Zero-argument factories for the safe built-in tools.  Typed as callables
#: (not ``type[BaseTool]``) so callers may instantiate them directly -- each
#: entry is a concrete, non-abstract ``BaseTool`` subclass.
BUILTIN_TOOLS: list[Callable[[], BaseTool[Any]]] = [
    CalculatorTool,
    DeviationTool,
    TextStatsTool,
    JsonParseTool,
    FileReadTool,
    EquipmentLookupTool,
    ProcessDocumentTool,
    KnowledgeBaseSearchTool,
    ExtractStructuredEvidenceTool,
]

__all__ = [
    "BUILTIN_TOOLS",
    "CalculatorTool",
    "DeviationTool",
    "TextStatsTool",
    "JsonParseTool",
    "FileReadTool",
    "EquipmentLookupTool",
    "ProcessDocumentTool",
    "KnowledgeBaseSearchTool",
    "ExtractStructuredEvidenceTool",
]
