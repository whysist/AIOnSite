"""Built-in, safe-by-default tools."""

from collections.abc import Callable
from typing import Any

from ..base_tool import BaseTool
from .calculator import CalculatorTool, DeviationTool
from .code_exec import SandboxedPythonTool
from .db_tool import EquipmentLookupTool
from .docx_writer import GenerateWordDocumentTool
from .document import ProcessDocumentTool
from .fs_reader import FileReadTool
from .json_tool import JsonParseTool
from .kb import ExtractStructuredEvidenceTool, KnowledgeBaseSearchTool
from .text_stats import TextStatsTool

#: Zero-argument factories for the safe built-in tools.  Typed as callables
#: (not ``type[BaseTool]``) so callers may instantiate them directly -- each
#: entry is a concrete, non-abstract ``BaseTool`` subclass.
#:
#: ``SandboxedPythonTool`` and ``GenerateWordDocumentTool`` are listed
#: unconditionally like every other tool here -- they are excluded at
#: *registration* time, not at this list, because they need
#: ``ToolPermission.SANDBOXED_EXEC`` / ``ToolPermission.WRITE_FILESYSTEM``,
#: which ``default_registry`` only grants when explicitly asked
#: (``allow_sandbox=True`` / ``allow_write_filesystem=True``), the same
#: opt-in pattern already used for ``ToolPermission.NETWORK``.
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
    SandboxedPythonTool,
    GenerateWordDocumentTool,
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
    "SandboxedPythonTool",
    "GenerateWordDocumentTool",
]
