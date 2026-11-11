import sqlite3
import unittest
from pathlib import Path
from app.services.audit import init_db, write_audit, get_connection
from app.core.config import get_settings

class Phase4Tests(unittest.TestCase):

    def test_audit_sqlite_wal_mode(self):
        init_db()
        with get_connection() as con:
            cur = con.cursor()
            cur.execute("PRAGMA journal_mode;")
            mode = cur.fetchone()[0]
            self.assertEqual(mode.lower(), "wal")

    def test_audit_write_and_read(self):
        init_db()
        question = "What is the MFA policy?"
        source_used = "private_kb"
        trace = ["Step 1", "Step 2"]

        write_audit(question, source_used, trace)

        with get_connection() as con:
            cur = con.cursor()
            cur.execute(
                "SELECT question, source_used, trace_json FROM query_audit WHERE question = ? ORDER BY id DESC LIMIT 1",
                (question,)
            )
            row = cur.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], question)
            self.assertEqual(row[1], source_used)
            self.assertIn("Step 1", row[2])

    def test_env_example_file(self):
        env_example = Path(".env.example")
        self.assertTrue(env_example.exists())
        content = env_example.read_text(encoding="utf-8")
        self.assertIn("GEMINI_API_KEY=", content)
        self.assertIn("TAVILY_API_KEY=", content)
        self.assertIn("PINECONE_API_KEY=", content)
        # Verify no actual live keys leaked
        self.assertNotIn("AQ.Ab8", content)
        self.assertNotIn("tvly-dev", content)
        self.assertNotIn("pcsk_", content)

if __name__ == "__main__":
    unittest.main()

