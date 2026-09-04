import os
import logging
from typing import Dict, Any, Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    try:
        import pymupdf as fitz
    except ImportError:
        fitz = None
        logging.warning("PyMuPDF (fitz) is not installed. PDF parsing will fail.")

logger = logging.getLogger(__name__)


def extract_native_text(page: Any) -> str:
    """Use PyMuPDF for born-digital pages."""
    if page is None:
        return ""
    try:
        if hasattr(page, "get_text"):
            return page.get_text("text").strip()
    except Exception as e:
        logger.error(f"Failed to extract native text: {e}")
    return ""


def get_pdf_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata from PDF document."""
    metadata = {
        "title": "",
        "author": "",
        "total_pages": 0,
        "format": "PDF",
        "file_name": os.path.basename(file_path),
        "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
    }
    if not fitz or not os.path.exists(file_path):
        return metadata

    try:
        doc = fitz.open(file_path)
        metadata["total_pages"] = len(doc)
        if doc.metadata:
            metadata["title"] = doc.metadata.get("title", "")
            metadata["author"] = doc.metadata.get("author", "")
        doc.close()
    except Exception as e:
        logger.error(f"Error reading PDF metadata from {file_path}: {e}")

    return metadata
