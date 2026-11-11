import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

def get_connection() -> sqlite3.Connection:
    db_path = Path(settings.audit_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), timeout=30.0)
    con.execute("PRAGMA journal_mode=WAL;")
    return con

def init_db() -> None:
    try:
        with get_connection() as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS query_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    question TEXT NOT NULL,
                    source_used TEXT NOT NULL,
                    trace_json TEXT NOT NULL
                )"""
            )
    except Exception as exc:
        logger.error("Failed to initialize audit database: %s", exc)

def write_audit(question: str, source_used: str, trace: list[str]) -> None:
    try:
        with get_connection() as con:
            con.execute(
                "INSERT INTO query_audit(created_at, question, source_used, trace_json) VALUES (?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), question, source_used, json.dumps(trace)),
            )
    except Exception as exc:
        logger.error("Failed to write query audit: %s", exc)