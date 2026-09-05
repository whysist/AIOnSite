import logging
from typing import Any

from .schemas import TableResult

logger = logging.getLogger(__name__)


def extract_tables(page: Any, page_num: int = 1) -> list[TableResult]:
    """Extract machine-readable tables from a page.
    Returns headers/rows and page identity.
    """
    tables: list[TableResult] = []
    if page is None:
        return tables

    try:
        # Check PyMuPDF table finder
        if hasattr(page, "find_tables"):
            tabs = page.find_tables()
            if tabs and hasattr(tabs, "tables"):
                for idx, t in enumerate(tabs.tables, 1):
                    extracted = t.extract()
                    if extracted and len(extracted) > 0:
                        headers = [str(c or "").strip() for c in extracted[0]]
                        rows = [[str(c or "").strip() for c in r] for r in extracted[1:]]
                        table_id = f"table_p{page_num}_{idx}"
                        bbox = list(t.bbox) if hasattr(t, "bbox") else None
                        
                        tables.append(TableResult(
                            table_id=table_id,
                            page=page_num,
                            headers=headers,
                            rows=rows,
                            bbox=bbox
                        ))
    except Exception as e:
        logger.debug(f"Table extraction non-fatal exception on page {page_num}: {e}")

    return tables
