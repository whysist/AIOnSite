import sqlite3
from contextlib import contextmanager
from pathlib import Path


def get_db_path() -> str:
    """
    Resolves the path to equipment.db.
    Checks primary location first, then falls back to secondary.
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    
    primary_path = project_root / "data" / "synthetic" / "SIH26117_Synthetic_Dataset_v2" / "database" / "equipment.db"
    if primary_path.exists():
        return str(primary_path)
        
    fallback_path = project_root / "data" / "equipment.db"
    return str(fallback_path)

@contextmanager
def get_connection(db_path: str | None = None):
    """
    Returns a sqlite3 connection context manager.
    """
    if db_path is None:
        db_path = get_db_path()
        
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
