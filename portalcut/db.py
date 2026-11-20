import json
import sqlite3
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "portalcut.db"

FINAL = ("ready", "failed")


def _conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init() -> None:
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                status TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0,
                error TEXT,
                meta TEXT,
                created_at REAL NOT NULL
            )"""
        )
        # A job left mid-ingest by a server restart will never finish; say so instead of spinning forever.
        c.execute(
            "UPDATE jobs SET status='failed', error='interrupted by server restart' "
            f"WHERE status NOT IN {FINAL}"
        )


def create(job_id: str, filename: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO jobs (id, filename, status, created_at) VALUES (?, ?, 'uploaded', ?)",
            (job_id, filename, time.time()),
        )


def update(job_id: str, **fields) -> None:
    if "meta" in fields:
        fields["meta"] = json.dumps(fields["meta"])
    cols = ", ".join(f"{k}=?" for k in fields)
    with _conn() as c:
        c.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["meta"] = json.loads(d["meta"]) if d["meta"] else None
    return d


def get(job_id: str) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _row(r) if r else None


def delete(job_id: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM jobs WHERE id=?", (job_id,))


def all_jobs() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    return [_row(r) for r in rows]
