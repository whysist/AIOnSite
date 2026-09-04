import os
import logging
from typing import List, Optional, Any
from .schemas import OCRBlock, OCRResult

logger = logging.getLogger(__name__)

# Global cached engines
_ACTIVE_ENGINE = None
_PADDLE_INSTANCE = None
_TESSERACT_AVAILABLE = False


def _get_engine(requested: str = "auto") -> str:
    global _ACTIVE_ENGINE, _PADDLE_INSTANCE, _TESSERACT_AVAILABLE
    if _ACTIVE_ENGINE is not None:
        return _ACTIVE_ENGINE

    if requested in ("auto", "paddleocr"):
        try:
            from paddleocr import PaddleOCR
            _PADDLE_INSTANCE = PaddleOCR(use_angle_cls=True, lang="en")
            _ACTIVE_ENGINE = "paddleocr"
            return _ACTIVE_ENGINE
        except Exception as e:
            logger.debug(f"PaddleOCR not available: {e}")

    if requested in ("auto", "tesseract"):
        try:
            import pytesseract
            from PIL import Image
            _TESSERACT_AVAILABLE = True
            _ACTIVE_ENGINE = "tesseract"
            return _ACTIVE_ENGINE
        except Exception as e:
            logger.debug(f"Tesseract not available: {e}")

    _ACTIVE_ENGINE = "fallback"
    return _ACTIVE_ENGINE


def ocr_page(image_path: str, engine: str = "auto") -> OCRResult:
    """Run local OCR and preserve method, confidence, bounding blocks, and warnings."""
    active = _get_engine(engine)
    warnings = []
    blocks: List[OCRBlock] = []

    if not os.path.exists(image_path):
        return OCRResult(
            text="",
            confidence=0.0,
            blocks=[],
            engine=active,
            warnings=[f"Image path not found: {image_path}"]
        )

    if active == "paddleocr" and _PADDLE_INSTANCE:
        try:
            res = _PADDLE_INSTANCE.ocr(image_path, cls=True)
            if res and res[0]:
                for line in res[0]:
                    box = line[0]
                    text = line[1][0]
                    conf = float(line[1][1])
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                    blocks.append(OCRBlock(text=text, confidence=conf, bbox=bbox))
                full_text = " ".join([b.text for b in blocks])
                avg_conf = sum([b.confidence for b in blocks]) / len(blocks) if blocks else 1.0
                return OCRResult(text=full_text, confidence=avg_conf, blocks=blocks, engine="paddleocr")
        except Exception as e:
            warnings.append(f"PaddleOCR execution error: {e}")

    elif active == "tesseract" and _TESSERACT_AVAILABLE:
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(image_path)
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            for i in range(len(data["text"])):
                txt = data["text"][i].strip()
                if txt:
                    conf = float(data["conf"][i]) / 100.0 if data["conf"][i] != "-1" else 0.5
                    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    blocks.append(OCRBlock(text=txt, confidence=conf, bbox=[x, y, x + w, y + h]))
            full_text = " ".join([b.text for b in blocks])
            avg_conf = sum([b.confidence for b in blocks]) / len(blocks) if blocks else 1.0
            return OCRResult(text=full_text, confidence=avg_conf, blocks=blocks, engine="tesseract")
        except Exception as e:
            warnings.append(f"Tesseract execution error: {e}")

    # Fallback mode
    warnings.append("Local OCR engine not installed (install paddleocr or pytesseract for raw raster OCR).")
    return OCRResult(
        text="",
        confidence=0.0,
        blocks=[],
        engine="fallback",
        warnings=warnings
    )
