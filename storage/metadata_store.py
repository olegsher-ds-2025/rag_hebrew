import sqlite3

class MetadataStore:
    def __init__(self, db_path="metadata.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            text TEXT,
            source TEXT
        )
        """)

    def insert(self, text, source):
        self.conn.execute(
            "INSERT INTO documents (text, source) VALUES (?, ?)",
            (text, source)
        )
        self.conn.commit()