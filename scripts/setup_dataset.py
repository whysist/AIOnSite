import os
import sys
import zipfile
import shutil
from pathlib import Path

def setup_dataset():
    """Extract synthetic dataset and verify database setup."""
    project_root = Path(__file__).resolve().parent.parent
    zip_path = project_root / "SIH26117_Synthetic_Dataset_v2.zip"
    dest_dir = project_root / "data" / "synthetic"
    
    print(f"Project root: {project_root}")
    
    if zip_path.exists():
        print(f"Extracting {zip_path.name} to {dest_dir}...")
        dest_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        print("Extraction complete.")
    else:
        print(f"Zip file {zip_path} not found. Checking if already extracted...")
        
    dataset_dir = dest_dir / "SIH26117_Synthetic_Dataset_v2"
    if not dataset_dir.exists():
        print(f"Error: Dataset directory {dataset_dir} does not exist.")
        sys.exit(1)
        
    db_src = dataset_dir / "database" / "equipment.db"
    db_fallback = project_root / "data" / "equipment.db"
    if db_src.exists():
        shutil.copy2(db_src, db_fallback)
        print(f"Copied {db_src} -> {db_fallback}")
        
    # Verify DB tables
    import sqlite3
    conn = sqlite3.connect(db_src)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]
    conn.close()
    
    print(f"Database tables verified: {tables}")
    print("Dataset setup successful.")

if __name__ == "__main__":
    setup_dataset()
