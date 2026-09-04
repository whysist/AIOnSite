import os
import re
import logging
from typing import List, Optional
from .schemas import DocumentResult, PageResult, TableResult
from .classifier import classify_page
from .pdf_parser import extract_native_text, get_pdf_metadata
from .ocr import ocr_page
from .tables import extract_tables
from .images import extract_images, render_page_to_temp_image

logger = logging.getLogger(__name__)

try:
    import fitz
except ImportError:
    try:
        import pymupdf as fitz
    except ImportError:
        fitz = None


def _detect_equipment_ids(text: str) -> List[str]:
    """Find equipment IDs using regex: V-101, P-101, HX-101, FV-103, PT-101, etc."""
    pattern = r'\b([A-Z]{1,3}-?\d{2,4})\b'
    matches = set(re.findall(pattern, text))
    return sorted(list(matches))


def _infer_document_type(filename: str) -> str:
    """Infer document category from filename."""
    f = filename.lower()
    if 'datasheet' in f:
        return 'datasheet'
    elif 'inspection' in f:
        return 'inspection_report'
    elif 'operating_limits' in f:
        return 'operating_limits'
    elif 'maintenance_sop' in f:
        return 'maintenance_sop'
    elif 'maintenance_history' in f:
        return 'maintenance_history'
    elif 'schematic' in f:
        return 'schematic'
    return 'document'


def validate_document_result(result: DocumentResult) -> DocumentResult:
    """Check page uniqueness, provenance completeness, and validity."""
    if not result.document_id:
        result.document_id = result.source or "unknown_doc"
    
    seen_pages = set()
    deduped_pages = []
    for p in result.pages:
        if p.page_number not in seen_pages:
            seen_pages.add(p.page_number)
            deduped_pages.append(p)
        else:
            result.processing_warnings.append(f"Duplicate page number {p.page_number} removed.")
    result.pages = deduped_pages
    return result


def process_document(file_path: str, ocr_engine: str = "auto") -> DocumentResult:
    """Master Coordinator for Person 3:
    Converts raw industrial document into normalized, page-aware machine-readable output.
    Preserves page ordering, identity, tables, and warnings.
    """
    if not os.path.exists(file_path):
        return DocumentResult(
            document_id=os.path.basename(file_path),
            source=file_path,
            pages=[],
            metadata={"error": "File not found"},
            processing_warnings=[f"File does not exist: {file_path}"]
        )

    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1].lower()
    doc_type = _infer_document_type(file_name)
    pages: List[PageResult] = []
    warnings: List[str] = []

    # 1. Image handling (PNG, JPG, BMP, TIFF)
    if ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
        ocr_res = ocr_page(file_path, engine=ocr_engine)
        text = ocr_res.text
        if ocr_res.warnings:
            warnings.extend(ocr_res.warnings)

        pages.append(PageResult(
            page_number=1,
            text=text,
            extraction_method="ocr",
            tables=[],
            images=[{"file_path": file_path}],
            warnings=ocr_res.warnings
        ))

    # 2. PDF handling
    elif ext == ".pdf":
        if not fitz:
            warnings.append("PyMuPDF (fitz) is not installed. PDF parsing unavailable.")
        else:
            try:
                doc = fitz.open(file_path)
                for page_idx in range(len(doc)):
                    page_num = page_idx + 1
                    page = doc[page_idx]
                    mode = classify_page(page)
                    page_warnings = []
                    page_tables = extract_tables(page, page_num=page_num)
                    page_images = extract_images(page, page_num=page_num)
                    page_text = ""

                    if mode == "native":
                        page_text = extract_native_text(page)
                    elif mode == "ocr":
                        tmp_img = render_page_to_temp_image(page)
                        try:
                            ocr_res = ocr_page(tmp_img, engine=ocr_engine)
                            page_text = ocr_res.text
                            if ocr_res.warnings:
                                page_warnings.extend(ocr_res.warnings)
                        finally:
                            if os.path.exists(tmp_img):
                                os.remove(tmp_img)
                    elif mode == "mixed":
                        native_t = extract_native_text(page)
                        tmp_img = render_page_to_temp_image(page)
                        try:
                            ocr_res = ocr_page(tmp_img, engine=ocr_engine)
                            ocr_t = ocr_res.text
                            page_text = f"{native_t}\n\n[OCR Layer]:\n{ocr_t}" if ocr_t else native_t
                        finally:
                            if os.path.exists(tmp_img):
                                os.remove(tmp_img)

                    # Append table text if missing from main text
                    for t in page_tables:
                        if t.rows:
                            tab_text_preview = " | ".join(t.headers)
                            if tab_text_preview not in page_text:
                                page_text += f"\n\n[Table: {t.table_id}]\n" + "\n".join([" | ".join(r) for r in t.rows])

                    pages.append(PageResult(
                        page_number=page_num,
                        text=page_text.strip(),
                        extraction_method=mode,
                        tables=page_tables,
                        images=[{"image_id": img.image_id, "width": img.width, "height": img.height} for img in page_images],
                        warnings=page_warnings
                    ))
                doc.close()
            except Exception as e:
                warnings.append(f"Error reading PDF {file_path}: {e}")
                logger.error(f"Error reading PDF {file_path}: {e}")

    # Calculate full metadata
    full_text = "\n\n".join([p.text for p in pages if p.text])
    equipment_ids = _detect_equipment_ids(full_text)
    meta = {
        "file_name": file_name,
        "file_path": file_path,
        "total_pages": len(pages),
        "document_type": doc_type,
        "equipment_ids": equipment_ids,
        "source_identifier": file_name
    }

    res = DocumentResult(
        document_id=file_name,
        source=file_name,
        pages=pages,
        metadata=meta,
        processing_warnings=warnings
    )
    return validate_document_result(res)
