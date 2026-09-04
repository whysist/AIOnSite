import os
import re
from typing import List, Optional
from .schemas import ProcessedDocument, DocumentMetadata, DocumentPage, DocumentResult, TableData
from .pdf_parser import PDFParser
from .ocr_engine import OCREngine


class DocumentIntelligence:
    def __init__(self, ocr_engine: str = 'auto'):
        self.pdf_parser = PDFParser()
        self.ocr_engine = OCREngine(engine=ocr_engine)
    
    def process(self, file_path: str) -> DocumentResult:
        """Process any document (PDF or image).
        For PDFs: auto-detect digital vs scanned, use appropriate method.
        For images: use OCR directly.
        Extracts equipment IDs from text.
        Infers document_type from filename.
        Preserves page numbers and tables.
        """
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()
        
        pages: List[DocumentPage] = []
        all_tables: List[TableData] = []
        is_pdf = ext == '.pdf'
        is_scanned = False
        
        if is_pdf:
            is_scanned = self.pdf_parser.is_scanned(file_path)
            if is_scanned:
                pages = self.ocr_engine.ocr_document(file_path)
            else:
                pages = self.pdf_parser.extract_text(file_path)
                for p in pages:
                    if p.tables:
                        all_tables.extend(p.tables)
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            blocks = self.ocr_engine.ocr_image(file_path)
            full_text = " ".join([b.text for b in blocks])
            avg_conf = sum([b.confidence for b in blocks]) / len(blocks) if blocks else 0.0
            pages = [DocumentPage(
                page_number=1,
                text=full_text,
                extraction_method='ocr',
                confidence=avg_conf,
                ocr_blocks=blocks,
                tables=[]
            )]
            is_scanned = True
        
        full_text = "\n\n".join([p.text for p in pages])
        equipment_ids = self._detect_equipment_ids(full_text)
        doc_type = self._infer_document_type(filename)
        
        metadata = DocumentMetadata(
            file_path=file_path,
            file_name=filename,
            total_pages=len(pages),
            document_type=doc_type,
            equipment_ids=equipment_ids,
            extraction_method='ocr' if is_scanned else 'digital',
            source_identifier=filename
        )
        
        return DocumentResult(
            metadata=metadata,
            pages=pages,
            full_text=full_text,
            tables=all_tables
        )
    
    def process_directory(self, dir_path: str, extensions: Optional[List[str]] = None) -> List[DocumentResult]:
        """Process all documents in a directory."""
        if extensions is None:
            extensions = ['.pdf', '.png', '.jpg', '.jpeg']
        extensions = [ext.lower() for ext in extensions]
        
        processed_docs = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in extensions:
                    file_path = os.path.join(root, file)
                    doc = self.process(file_path)
                    processed_docs.append(doc)
                    
        return processed_docs
    
    def _detect_equipment_ids(self, text: str) -> List[str]:
        """Find equipment IDs in text using regex: V-101, P-101, HX-101, FV-103, etc."""
        pattern = r'\b([A-Z]{1,3}-?\d{2,4})\b'
        matches = set(re.findall(pattern, text))
        return sorted(list(matches))
    
    def _infer_document_type(self, filename: str) -> str:
        """Infer type from filename patterns."""
        filename_lower = filename.lower()
        if 'datasheet' in filename_lower:
            return 'datasheet'
        elif 'inspection' in filename_lower:
            return 'inspection_report'
        elif 'operating_limits' in filename_lower:
            return 'operating_limits'
        elif 'maintenance_sop' in filename_lower:
            return 'maintenance_sop'
        elif 'maintenance_history' in filename_lower:
            return 'maintenance_history'
        elif 'schematic' in filename_lower:
            return 'schematic'
        return 'unknown'


# Top-level functional contract P3 -> P2
def process_document(file_path: str, ocr_engine: str = 'auto') -> DocumentResult:
    """Master Contract P3 -> P2: process_document(file_path) -> DocumentResult"""
    di = DocumentIntelligence(ocr_engine=ocr_engine)
    return di.process(file_path)
