import os
import logging
import tempfile
from typing import List, Optional
from .schemas import OCRBlock, DocumentPage

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logging.warning("PIL is not installed. Some image operations may fail.")

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    logging.warning("PyMuPDF (fitz) is not installed. Rendering PDF pages to image will fail.")

class OCREngine:
    def __init__(self, engine: str = 'auto'):
        """engine: 'paddleocr', 'tesseract', or 'auto' (try paddle first, fallback to tesseract)"""
        self._engine = engine
        self._active_engine = "fallback"
        self._paddle = None
        self._tesseract = None
        
        self._initialize_engines()

    def _initialize_engines(self):
        try_paddle = self._engine in ('auto', 'paddleocr')
        try_tesseract = self._engine in ('auto', 'tesseract')
        
        if try_paddle:
            try:
                from paddleocr import PaddleOCR
                self._paddle = PaddleOCR(use_angle_cls=True, lang='en')
                self._active_engine = "paddleocr"
                return
            except ImportError:
                logging.warning("PaddleOCR is not installed or failed to import.")
            except Exception as e:
                logging.warning(f"PaddleOCR initialization failed: {e}")
                
        if try_tesseract:
            try:
                import pytesseract
                # Test pytesseract
                if not HAS_PIL:
                    logging.warning("PIL is required for pytesseract.")
                else:
                    self._tesseract = pytesseract
                    self._active_engine = "tesseract"
                    return
            except ImportError:
                logging.warning("pytesseract is not installed.")
                
        if self._engine != 'fallback':
            logging.warning(f"Failed to initialize requested engine '{self._engine}'. Using fallback mode.")
            self._active_engine = "fallback"

    def ocr_image(self, image_path: str) -> List[OCRBlock]:
        """OCR a single image file."""
        if self._active_engine == "paddleocr":
            try:
                result = self._paddle.ocr(image_path, cls=True)
                blocks = []
                if result and result[0]:
                    for line in result[0]:
                        box = line[0] # [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        text = line[1][0]
                        confidence = line[1][1]
                        
                        # Simplified bbox: [min_x, min_y, max_x, max_y]
                        xs = [p[0] for p in box]
                        ys = [p[1] for p in box]
                        bbox = [min(xs), min(ys), max(xs), max(ys)]
                        
                        blocks.append(OCRBlock(text=text, confidence=confidence, bbox=bbox))
                return blocks
            except Exception as e:
                logging.error(f"PaddleOCR failed on {image_path}: {e}")
                
        elif self._active_engine == "tesseract":
            try:
                if not HAS_PIL:
                    return []
                img = Image.open(image_path)
                data = self._tesseract.image_to_data(img, output_type=self._tesseract.Output.DICT)
                blocks = []
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    if text:
                        conf = float(data['conf'][i]) / 100.0 if data['conf'][i] != '-1' else 0.0
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        blocks.append(OCRBlock(text=text, confidence=conf, bbox=[x, y, x+w, y+h]))
                return blocks
            except Exception as e:
                logging.error(f"Tesseract failed on {image_path}: {e}")
                
        # Fallback
        logging.warning("Using fallback OCR (no real OCR performed). Returning placeholder.")
        return [OCRBlock(text="[OCR Fallback: Engine not available]", confidence=0.0, bbox=[0,0,0,0])]

    def ocr_pdf_page(self, file_path: str, page_num: int) -> DocumentPage:
        """Render a PDF page to image and OCR it."""
        if not fitz:
            return DocumentPage(
                page_number=page_num,
                text="[PDF Rendering failed: PyMuPDF not installed]",
                extraction_method="ocr",
                confidence=0.0
            )
            
        try:
            doc = fitz.open(file_path)
            if not (0 < page_num <= len(doc)):
                raise ValueError(f"Invalid page number {page_num}")
                
            page = doc[page_num - 1]
            # Render to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # 2x scale for better OCR
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                temp_img_path = tmp.name
            pix.save(temp_img_path)
            
            try:
                blocks = self.ocr_image(temp_img_path)
            finally:
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                
            # Combine text
            full_text = " ".join([b.text for b in blocks])
            avg_conf = sum([b.confidence for b in blocks]) / len(blocks) if blocks else 0.0
            
            return DocumentPage(
                page_number=page_num,
                text=full_text,
                extraction_method='ocr',
                confidence=avg_conf,
                ocr_blocks=blocks
            )
            
        except Exception as e:
            logging.error(f"Error OCRing PDF page {file_path}:{page_num}: {e}")
            return DocumentPage(page_number=page_num, text="", extraction_method="ocr")

    def ocr_document(self, file_path: str) -> List[DocumentPage]:
        """OCR all pages of a scanned PDF."""
        pages = []
        if not fitz:
            return pages
            
        try:
            doc = fitz.open(file_path)
            for page_num in range(1, len(doc) + 1):
                page = self.ocr_pdf_page(file_path, page_num)
                pages.append(page)
        except Exception as e:
            logging.error(f"Error OCRing document {file_path}: {e}")
            
        return pages

    @property
    def engine_name(self) -> str:
        """Return which engine is active."""
        return self._active_engine
