"""Built-in, safe-by-default tools."""

from collections.abc import Callable
from typing import Any

from ..base_tool import BaseTool
from .calculator import CalculatorTool, DeviationTool
from .db_tool import EquipmentLookupTool
from .document import ProcessDocumentTool
from .docx_export import DocxExportTool
from .fs_reader import FileReadTool
from .json_tool import JsonParseTool
from .kb import ExtractStructuredEvidenceTool, KnowledgeBaseSearchTool
from .text_stats import TextStatsTool
from .vision_tool import AnalyzeImageTool

#: Zero-argument factories for the safe built-in tools.  Typed as callables
#: (not ``type[BaseTool]``) so callers may instantiate them directly -- each
#: entry is a concrete, non-abstract ``BaseTool`` subclass.
#:
#: This list is always extended, never replaced wholesale -- a past PR once
#: replaced this file entirely and deleted every existing tool, breaking the
#: whole orchestrator (see PROJECT_CONTEXT.md sec 6).
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
    AnalyzeImageTool,
    DocxExportTool,
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
    "AnalyzeImageTool",
    "DocxExportTool",
]
