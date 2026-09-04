# src/database/connection.py
import sqlite3
from contextlib import contextmanager

class DatabaseConnection:
    def __init__(self, db_path: str = "industrial_data.db"):
        self.db_path = db_path

    @contextmanager
    def get_cursor(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        finally:
            conn.close()