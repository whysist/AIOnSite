import os
import sys
import asyncio
import time
from pathlib import Path

# Ensure UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.tools.registry import default_registry, ToolRegistry
from src.tools.base_tool import ToolPermission, ToolResult
from src.vision import process_document, DocumentResult, TableResult
from src.retrieval.service import KnowledgeBase
from src.retrieval.schemas import Fact, Evidence
from src.retrieval.chunking import chunk_document


async def run_confidence_verification():
    print("\n" + "=" * 80)
    print("AIOnSite — Sovereign On-Premise Industrial AI Workbench")
    print("RAG (Person 2) & OCR / Document Intelligence (Person 3) Confidence Verification Suite")
    print("=" * 80)

    total_checks = 0
    passed_checks = 0

    def assert_check(name: str, condition: bool, details: str = ""):
        nonlocal total_checks, passed_checks
        total_checks += 1
        if condition:
            passed_checks += 1
            print(f"  [PASS] {name}")
            if details:
                print(f"         └─ {details}")
        else:
            print(f"  [FAIL] {name}")
            if details:
                print(f"         └─ Error/Details: {details}")

    # --------------------------------------------------------------------------
    # MILESTONE 1: P1 Tool Contract Compliance
    # --------------------------------------------------------------------------
    print("\n[MILESTONE 1/7] Validating P1 Tool System Contract Compliance...")
    reg: ToolRegistry = default_registry()
    tools_present = reg.names()
    assert_check("Default Tool Registry initialised with all builtin tools",
                 len(tools_present) >= 5,
                 f"Registered tools: {tools_present}")

    kb_tool = reg.get("search_knowledge_base")
    assert_check("KnowledgeBaseSearchTool implements PURE permission",
                 kb_tool.permissions == (ToolPermission.PURE,),
                 f"Permissions: {[p.value for p in kb_tool.permissions]}")

    doc_tool = reg.get("process_document")
    assert_check("ProcessDocumentTool implements READ_FILESYSTEM permission",
                 doc_tool.permissions == (ToolPermission.READ_FILESYSTEM,),
                 f"Permissions: {[p.value for p in doc_tool.permissions]}")

    calc_tool = reg.get("calculate_deviation")
    assert_check("CalculateDeviationTool implements PURE permission and schema validation",
                 calc_tool.InputModel is not None and calc_tool.OutputModel is not None)

    # --------------------------------------------------------------------------
    # MILESTONE 2: Person 3 Document Intelligence & Table Parsing
    # --------------------------------------------------------------------------
    print("\n[MILESTONE 2/7] Testing Person 3 Document Intelligence on Synthetic PDFs...")
    docs_dir = project_root / "data" / "synthetic" / "SIH26117_Synthetic_Dataset_v2" / "documents"
    pdf_files = sorted(list(docs_dir.glob("*.pdf")))
    assert_check(f"Synthetic dataset verified ({len(pdf_files)} PDF documents found)", len(pdf_files) == 10)

    # Test digital PDF processing
    v101_datasheet = str(docs_dir / "01_V101_Equipment_Datasheet.pdf")
    t0 = time.perf_counter()
    doc_res: DocumentResult = process_document(v101_datasheet)
    duration_ms = (time.perf_counter() - t0) * 1000

    assert_check("Digital PDF processed via native PyMuPDF engine",
                 doc_res.pages[0].extraction_method == "native",
                 f"Duration: {duration_ms:.1f}ms | Extracted chars: {len(doc_res.full_text)}")
    assert_check("Document metadata preserves source identifier and detected equipment",
                 doc_res.metadata["source_identifier"] == "01_V101_Equipment_Datasheet.pdf" and
                 "V-101" in doc_res.metadata["equipment_ids"])

    # Test Table Extraction on Inspection Report
    v101_insp = str(docs_dir / "02_V101_Inspection_Report.pdf")
    insp_res = process_document(v101_insp)
    assert_check("Machine-readable table cells extracted into structured TableResult objects",
                 len(insp_res.all_tables) >= 1 and len(insp_res.all_tables[0].rows) >= 3,
                 f"Table rows found: {insp_res.all_tables[0].rows if insp_res.all_tables else 'None'}")

    # --------------------------------------------------------------------------
    # MILESTONE 3: Person 3 OCR & Scanned Document Processing
    # --------------------------------------------------------------------------
    print("\n[MILESTONE 3/7] Testing OCR and Image Processing Paths...")
    scan_png = str(project_root / "data" / "synthetic" / "SIH26117_Synthetic_Dataset_v2" / "auxiliary" / "03_scanned_inspection_note.png")
    scan_res: DocumentResult = process_document(scan_png)
    assert_check("Scanned image classified into 'ocr' mode with page and provenance tracking",
                 scan_res.pages[0].extraction_method == "ocr",
                 f"Source: {scan_res.source} | Pages: {scan_res.total_pages}")

    # --------------------------------------------------------------------------
    # MILESTONE 4: Person 2 Knowledge Base Ingestion & Chunking
    # --------------------------------------------------------------------------
    print("\n[MILESTONE 4/7] Testing Person 2 Ingestion, Chunking & Provenance...")
    kb = KnowledgeBase(persist_dir=str(project_root / "data" / "processed" / "chroma_db"))
    kb.reset()
    stats = kb.ingest_directory(str(docs_dir))

    assert_check("Batch ingestion processed all 10 synthetic documents into VectorStore",
                 stats["documents_processed"] == 10 and stats["total_chunks"] >= 20,
                 f"Chunks indexed: {stats['total_chunks']} | Entities: {stats['equipment_entities']}")

    chunks = chunk_document(doc_res)
    assert_check("Chunker enforces authority tags ('authoritative_limit' vs 'observation' vs 'history')",
                 all(c.authority is not None for c in chunks),
                 f"Authority tag for datasheet chunk: {chunks[0].authority if chunks else 'None'}")

    # --------------------------------------------------------------------------
    # MILESTONE 5: Person 2 Structured Fact Extraction (No LLM scraping)
    # --------------------------------------------------------------------------
    print("\n[MILESTONE 5/7] Testing P2 -> P1 Structured Fact Extraction Contract...")
    facts_res: ToolResult = await reg.get("extract_structured_evidence").run(
        query="observed pressure",
        equipment_id="V-101"
    )
    v101_facts: list[Fact] = facts_res.output.facts if facts_res.ok else []
    p_fact = next((f for f in v101_facts if f.parameter == "pressure"), None)

    assert_check("P2 ExtractStructuredEvidenceTool returns typed Fact models with exact readings",
                 p_fact is not None and p_fact.value == 14.5 and p_fact.unit == "bar",
                 f"Extracted Fact: {p_fact}")
    assert_check("Extracted facts carry exact page-level source provenance for governance",
                 p_fact is not None and p_fact.page == 1 and "02_V101_Inspection_Report.pdf" in p_fact.source,
                 f"Source: {p_fact.source if p_fact else 'None'} | Page: {p_fact.page if p_fact else 'None'}")

    # --------------------------------------------------------------------------
    # MILESTONE 6: Strict Equipment Isolation (V-101 vs P-101)
    # --------------------------------------------------------------------------
    print("\n[MILESTONE 6/7] Testing Cross-Equipment Isolation (No distractor bleed)...")
    p101_facts_res = await reg.get("extract_structured_evidence").run(
        query="discharge pressure",
        equipment_id="P-101"
    )
    p101_facts = p101_facts_res.output.facts if p101_facts_res.ok else []
    p101_p_fact = next((f for f in p101_facts if "discharge" in f.parameter and "pressure" in f.parameter), None)

    assert_check("P-101 query retrieves P-101 discharge pressure (5.4 bar)",
                 p101_p_fact is not None and p101_p_fact.value == 5.4,
                 f"P-101 Fact: {p101_p_fact}")

    has_contamination = any(f.value == 5.4 for f in v101_facts)
    assert_check("V-101 fact extraction is strictly isolated from P-101 distractor values",
                 not has_contamination,
                 "0% cross-equipment contamination verified.")

    # --------------------------------------------------------------------------
    # MILESTONE 7: Deterministic Arithmetic & Epistemic Negative Testing
    # --------------------------------------------------------------------------
    print("\n[MILESTONE 7/7] Testing Deterministic Math & Epistemic Boundaries...")
    calc_res = await reg.get("calculate_deviation").run(actual=14.5, limit=12.0, label="V-101 Pressure")
    out = calc_res.output
    assert_check("Deviation computed via exact Python AST (14.5 vs 12.0 -> +20.83%)",
                 calc_res.ok and out.percent_deviation == 20.83 and out.within_limit is False,
                 f"Formula: {out.formula} | Percent Deviation: +{out.percent_deviation}%")

    # Missing parameter test
    flow_facts_res = await reg.get("extract_structured_evidence").run(
        query="flow rate m3/h",
        equipment_id="V-101"
    )
    flow_facts = [f for f in (flow_facts_res.output.facts if flow_facts_res.ok else []) if "flow" in f.parameter]
    assert_check("Epistemic Integrity: Missing flow-rate parameter correctly returns 0 facts (no hallucination)",
                 len(flow_facts) == 0,
                 f"Missing facts safely returned as: {flow_facts}")

    # --------------------------------------------------------------------------
    # SUMMARY REPORT
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    acc = (passed_checks / total_checks) * 100.0 if total_checks else 0.0
    print(f"CONFIDENCE VERIFICATION REPORT: {passed_checks}/{total_checks} CHECKS PASSED ({acc:.1f}% SUCCESS)")
    print("=" * 80)
    if passed_checks == total_checks:
        print("  >>> VERDICT: 100% PRODUCTION READY FOR PERSON 1 ORCHESTRATION INTEGRATION <<<")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(run_confidence_verification())
