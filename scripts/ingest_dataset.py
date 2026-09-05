"""One-time setup: extract the synthetic dataset and ingest its documents
into the local knowledge base so agents can retrieve real evidence instead
of finding nothing under ``data/``.

Usage::

    python scripts/ingest_dataset.py

Safe to re-run: extraction is skipped if already done, and ingestion
resets the collection first so re-running never duplicates chunks.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.service import KnowledgeBase  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = PROJECT_ROOT / "SIH26117_Synthetic_Dataset_v2.zip"
DATASET_DIR = PROJECT_ROOT / "data" / "synthetic" / "SIH26117_Synthetic_Dataset_v2"
DOCS_DIR = DATASET_DIR / "documents"


def extract_dataset() -> None:
    if DATASET_DIR.exists():
        print(f"[1/2] Dataset already extracted at {DATASET_DIR}")
        return
    if not ZIP_PATH.exists():
        print(f"ERROR: {ZIP_PATH} not found.")
        sys.exit(1)
    print(f"[1/2] Extracting {ZIP_PATH.name} -> {DATASET_DIR.parent}")
    DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATASET_DIR.parent)


def ingest() -> None:
    if not DOCS_DIR.exists():
        print(f"ERROR: {DOCS_DIR} not found after extraction.")
        sys.exit(1)
    print(f"[2/2] Ingesting documents from {DOCS_DIR}")
    kb = KnowledgeBase()
    kb.reset()
    stats = kb.ingest_directory(str(DOCS_DIR))
    print(
        f"      documents={stats['documents_processed']} "
        f"chunks={stats['total_chunks']} "
        f"equipment={stats['equipment_entities']}"
    )


if __name__ == "__main__":
    extract_dataset()
    ingest()
    print("\nDone. Try: python scripts/run_demo.py")
