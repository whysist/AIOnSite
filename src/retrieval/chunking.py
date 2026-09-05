import hashlib
import re

from src.vision.schemas import DocumentResult

from .schemas import Chunk


def _infer_authority(doc_type: str, text: str) -> str:
    """Infer authority tier to distinguish limits/SOPs from observations."""
    dt = doc_type.lower()
    if "operating_limits" in dt or "datasheet" in dt:
        return "authoritative_limit"
    elif "sop" in dt:
        return "sop"
    elif "inspection" in dt:
        return "observation"
    elif "history" in dt:
        return "history"
    return "general"


def _detect_revision(text: str, doc_name: str) -> str | None:
    """Detect document revision tag (e.g. 'Revision B' or 'Rev A')."""
    m = re.search(r'\b(?:Rev|Revision)\s*[:\-\s]?\s*([A-Z0-9]+)\b', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m_name = re.search(r'_Rev_?([A-Z0-9]+)', doc_name, re.IGNORECASE)
    if m_name:
        return m_name.group(1).upper()
    return None


def _get_primary_equipment_id(filename: str, text: str) -> list[str]:
    """Determine the primary equipment ID that this document or chunk belongs to."""
    fn_upper = filename.upper()
    if "V101" in fn_upper or "V-101" in fn_upper:
        return ["V-101"]
    elif "P101" in fn_upper or "P-101" in fn_upper:
        return ["P-101"]
    elif "P102" in fn_upper or "P-102" in fn_upper:
        return ["P-102"]
    elif "HX101" in fn_upper or "HX-101" in fn_upper:
        return ["HX-101"]
    elif "FV103" in fn_upper or "FV-103" in fn_upper:
        return ["FV-103"]

    header_area = text[:200]
    pattern = r'\b([A-Z]{1,3}-\d{2,4})\b'
    matches = re.findall(pattern, header_area)
    if matches:
        return [matches[0]]

    return sorted(set(re.findall(pattern, text)))


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into distinct narrative paragraphs or logical units."""
    raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    cleaned = []
    for p in raw_paras:
        # If paragraph is very long, split into sentences
        if len(p.split()) > 150:
            sentences = re.split(r'(?<=[.!?])\s+', p)
            curr: list[str] = []
            curr_len = 0
            for s in sentences:
                s_len = len(s.split())
                if curr_len + s_len > 100 and curr:
                    cleaned.append(" ".join(curr))
                    curr = [s]
                    curr_len = s_len
                else:
                    curr.append(s)
                    curr_len += s_len
            if curr:
                cleaned.append(" ".join(curr))
        else:
            cleaned.append(p)
    return cleaned


def chunk_document(
    document: DocumentResult,
    child_chunk_size: int = 100,
    create_parent_chunks: bool = True
) -> list[Chunk]:
    """Hierarchical (Parent-Child) Chunking Algorithm:
    1. Creates a Large Parent Chunk per page (for full LLM narrative context).
    2. Creates Small Child Chunks (for high-precision vector search):
       - Narrative paragraph / sentence chunks.
       - Atomic table-row chunks (preserving row-level facts).
    3. Links every Child Chunk to its Parent via `parent_id` and `parent_text`.
    """
    all_chunks: list[Chunk] = []
    doc_id = document.document_id
    doc_source = document.source
    doc_type = document.metadata.get("document_type", "document")
    doc_equipment = _get_primary_equipment_id(doc_id, document.full_text)
    global_chunk_idx = 0

    for page in document.pages:
        page_num = page.page_number
        page_text = page.text
        if not page_text:
            continue

        page_rev = _detect_revision(page_text, doc_id)
        page_authority = _infer_authority(doc_type, page_text)

        # 1. Create the PARENT Chunk (Full Page Context)
        parent_id_str = f"par_{doc_id}_p{page_num}"
        parent_hash = hashlib.md5(parent_id_str.encode("utf-8")).hexdigest()[:16]
        parent_chunk_id = f"par_{parent_hash}"

        parent_chunk = Chunk(
            chunk_id=parent_chunk_id,
            text=page_text,
            document_id=doc_id,
            source=doc_source,
            page_number=page_num,
            chunk_index=global_chunk_idx,
            section="Full Page Context",
            revision=page_rev,
            authority=page_authority,
            equipment_ids=doc_equipment,
            document_type=doc_type,
            parent_id=None,
            parent_text=None,
            is_parent=True,
            chunk_type="parent",
            metadata={
                "page": page_num,
                "document_id": doc_id,
                "extraction_method": page.extraction_method,
                "is_parent": True
            }
        )
        if create_parent_chunks:
            all_chunks.append(parent_chunk)
            global_chunk_idx += 1

        # 2. Create CHILD Chunks from Narrative Paragraphs
        paragraphs = _split_into_paragraphs(page_text)
        for p_idx, para in enumerate(paragraphs):
            # Skip pure table block strings if extracted separately
            if para.startswith("[Table:"):
                continue

            child_id_str = f"chd_{doc_id}_p{page_num}_para{p_idx}"
            child_hash = hashlib.md5(child_id_str.encode("utf-8")).hexdigest()[:16]
            child_chunk_id = f"chk_{child_hash}"

            sec_match = re.search(r'^(?:[0-9\.]+\s+)?([A-Z][A-Za-z0-9\s\-_]{3,40})(?:\n|:)', para)
            section = sec_match.group(1).strip() if sec_match else None

            all_chunks.append(Chunk(
                chunk_id=child_chunk_id,
                text=para,
                document_id=doc_id,
                source=doc_source,
                page_number=page_num,
                chunk_index=global_chunk_idx,
                section=section,
                revision=page_rev,
                authority=page_authority,
                equipment_ids=doc_equipment,
                document_type=doc_type,
                parent_id=parent_chunk_id,
                parent_text=page_text,
                is_parent=False,
                chunk_type="child",
                metadata={
                    "page": page_num,
                    "document_id": doc_id,
                    "parent_id": parent_chunk_id,
                    "extraction_method": page.extraction_method
                }
            ))
            global_chunk_idx += 1

        # 3. Create ATOMIC CHILD Chunks from Structured Tables
        for table in page.tables:
            header_str = " | ".join(table.headers) if table.headers else "Parameter | Value | Unit"
            for r_idx, row in enumerate(table.rows):
                row_str = " | ".join(row)
                atomic_table_text = f"Table: {table.table_id} (Page {page_num})\nHeaders: {header_str}\nRow: {row_str}"

                t_child_str = f"tab_{doc_id}_p{page_num}_{table.table_id}_r{r_idx}"
                t_child_hash = hashlib.md5(t_child_str.encode("utf-8")).hexdigest()[:16]
                t_chunk_id = f"tab_{t_child_hash}"

                all_chunks.append(Chunk(
                    chunk_id=t_chunk_id,
                    text=atomic_table_text,
                    document_id=doc_id,
                    source=doc_source,
                    page_number=page_num,
                    chunk_index=global_chunk_idx,
                    section=f"Table {table.table_id}",
                    revision=page_rev,
                    authority=page_authority,
                    equipment_ids=doc_equipment,
                    document_type=doc_type,
                    parent_id=parent_chunk_id,
                    parent_text=page_text,
                    is_parent=False,
                    chunk_type="table_row",
                    metadata={
                        "page": page_num,
                        "document_id": doc_id,
                        "table_id": table.table_id,
                        "parent_id": parent_chunk_id
                    }
                ))
                global_chunk_idx += 1

    return all_chunks
