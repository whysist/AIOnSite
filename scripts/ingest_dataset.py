import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.service import KnowledgeBase
from src.retrieval.evaluation import evaluate_retrieval


def ingest_dataset():
    """Process documents and populate the knowledge base."""
    docs_dir = project_root / "data" / "synthetic" / "SIH26117_Synthetic_Dataset_v2" / "documents"
    persist_dir = project_root / "data" / "processed" / "chroma_db"
    
    print("=" * 70)
    print("AIOnSite — Sovereign Document Intelligence & Knowledge Base Ingestion")
    print("=" * 70)
    print(f"Source Directory : {docs_dir}")
    print(f"Persist Directory: {persist_dir}\n")
    
    if not docs_dir.exists():
        print(f"Error: Documents directory not found at {docs_dir}")
        print("Please run 'python scripts/setup_dataset.py' first.")
        sys.exit(1)
        
    kb = KnowledgeBase(persist_dir=str(persist_dir))
    kb.reset()
    print("[1/3] Cleared existing vector store collection.")
    
    print("[2/3] Batch processing and ingesting all documents with Person 3 pipeline...")
    stats = kb.ingest_directory(str(docs_dir))
    
    print(f"      -> Documents Ingested : {stats['documents_processed']}")
    print(f"      -> Chunks Indexed     : {stats['total_chunks']}")
    print(f"      -> Entities Tagged    : {stats['equipment_entities']}")

    print("\n[3/3] Running automated retrieval evaluation cases...")
    report = evaluate_retrieval(kb)
    for c in report.results:
        status_icon = "[PASS]" if c.passed else "[FAIL]"
        print(f"      {status_icon} {c.case_id}: {c.description}")
        print(f"             Details: {c.details}")

    print("\n" + "=" * 70)
    print(f"Ingestion & Evaluation Summary: {report.passed_cases}/{report.total_cases} passed ({report.accuracy_score:.1f}% accuracy)")
    print("=" * 70)


if __name__ == "__main__":
    ingest_dataset()
