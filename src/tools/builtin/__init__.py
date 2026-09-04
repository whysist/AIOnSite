from .db_tool import EquipmentLookupTool, CalculateDeviationTool
from .kb import KnowledgeBaseSearchTool, ExtractStructuredEvidenceTool
from .document import ProcessDocumentTool

# Backward-compatibility aliases
RAGSearchTool = KnowledgeBaseSearchTool
OCRTool = ProcessDocumentTool

__all__ = [
    "EquipmentLookupTool",
    "CalculateDeviationTool",
    "KnowledgeBaseSearchTool",
    "ExtractStructuredEvidenceTool",
    "ProcessDocumentTool",
    "RAGSearchTool",
    "OCRTool"
]
