import os
import tempfile
import logging
from typing import List, Any
from .schemas import ImageResult

logger = logging.getLogger(__name__)


def extract_images(page: Any, page_num: int = 1) -> List[ImageResult]:
    """Capture image references from a page for OCR/VLM only when needed."""
    images: List[ImageResult] = []
    if page is None:
        return images

    try:
        if hasattr(page, "get_images") and hasattr(page, "parent"):
            doc = page.parent
            for img_idx, img in enumerate(page.get_images(full=True), 1):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image")
                image_ext = base_image.get("ext", "png")
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
                
                images.append(ImageResult(
                    image_id=f"img_p{page_num}_{img_idx}",
                    page=page_num,
                    format=image_ext,
                    width=width,
                    height=height,
                    image_bytes=image_bytes
                ))
    except Exception as e:
        logger.debug(f"Image extraction non-fatal on page {page_num}: {e}")

    return images


def render_page_to_temp_image(page: Any, scale: float = 2.0) -> str:
    """Render a PyMuPDF page to a high-resolution temporary PNG file for OCR."""
    try:
        import fitz
    except ImportError:
        import pymupdf as fitz

    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    pix.save(tmp_path)
    return tmp_path
