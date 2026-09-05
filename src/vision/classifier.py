import logging
from typing import Any

from .schemas import ProcessingMode

logger = logging.getLogger(__name__)


def classify_page(page: Any) -> ProcessingMode:
    """Choose native text, OCR, or mixed processing from page characteristics.
    
    Heuristic:
    - If page has substantial extractable digital text (> 50 chars) and no large raster images -> "native"
    - If page has very little or no text (< 30 chars) and has image objects -> "ocr"
    - If page has both text and large images/diagrams -> "mixed"
    """
    if page is None:
        return "native"

    try:
        # Check if page is a PyMuPDF Page
        if hasattr(page, "get_text"):
            text = page.get_text("text").strip()
            text_len = len(text)
            
            # Check for embedded images
            image_list = page.get_images() if hasattr(page, "get_images") else []
            has_images = len(image_list) > 0

            if text_len < 30 and has_images:
                return "ocr"
            elif text_len >= 50 and not has_images:
                return "native"
            elif text_len >= 30 and has_images:
                return "mixed"
            elif text_len >= 30:
                return "native"
            else:
                return "ocr"
    except Exception as e:
        logger.debug(f"classify_page exception: {e}")

    return "native"
