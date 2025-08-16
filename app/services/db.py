from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import time

from app.config import settings


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS merchants (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS receipts (
  id INTEGER PRIMARY KEY,
  stored_name TEXT NOT NULL UNIQUE,
  original_name TEXT NOT NULL,
  date TEXT,
  total REAL,
    subtotal REAL,
    tax REAL,
    tip REAL,
    discounts REAL,
    fees REAL,
    method TEXT,
    last4 TEXT,
  merchant_id INTEGER,
  memo TEXT,
  created_at INTEGER NOT NULL,
  FOREIGN KEY (merchant_id) REFERENCES merchants(id)
);

CREATE TABLE IF NOT EXISTS line_items (
  id INTEGER PRIMARY KEY,
  receipt_id INTEGER NOT NULL,
    description TEXT, -- interpreted item name
    ocr_text TEXT,
    line_index INTEGER,
  qty REAL,
  unit_price REAL,
  amount REAL,
    confidence REAL,
  created_at INTEGER NOT NULL,
  FOREIGN KEY (receipt_id) REFERENCES receipts(id)
);

-- Track LLM extraction performance over time
CREATE TABLE IF NOT EXISTS llm_runs (
    id INTEGER PRIMARY KEY,
    stored_name TEXT NOT NULL,
    model TEXT,
    prompt_chars INTEGER,
    prompt_lines INTEGER,
    indexed_lines INTEGER,
    include_full_text INTEGER,
    long_input INTEGER,
    full_text_sent_chars INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    wall_sec REAL,
    total_sec REAL,
    prompt_eval_sec REAL,
    eval_sec REAL,
    tps_completion REAL,
    tps_end_to_end REAL,
    prep_sec REAL,
    created_at INTEGER NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    Path(settings.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA_SQL)
        # Migrations for existing DBs
        _ensure_column(conn, "receipts", "subtotal", "REAL")
        _ensure_column(conn, "receipts", "tax", "REAL")
        _ensure_column(conn, "receipts", "tip", "REAL")
        _ensure_column(conn, "receipts", "discounts", "REAL")
        _ensure_column(conn, "receipts", "fees", "REAL")
        _ensure_column(conn, "receipts", "method", "TEXT")
        _ensure_column(conn, "receipts", "last4", "TEXT")
        _ensure_column(conn, "line_items", "ocr_text", "TEXT")
        _ensure_column(conn, "line_items", "line_index", "INTEGER")
        _ensure_column(conn, "line_items", "confidence", "REAL")
    # Ensure llm_runs table exists (already created by SCHEMA_SQL); add columns if we extend later
        conn.commit()


def upsert_merchant(name: str) -> int:
    name = (name or "").strip()
    if not name:
        return 0
    now = int(time.time())
    with _connect() as conn:
        cur = conn.execute("INSERT OR IGNORE INTO merchants(name, created_at) VALUES(?, ?)", (name, now))
        if cur.lastrowid:
            mid = cur.lastrowid
        else:
            row = conn.execute("SELECT id FROM merchants WHERE name=?", (name,)).fetchone()
            mid = int(row["id"]) if row else 0
        conn.commit()
        return mid


def insert_receipt(
    stored_name: str,
    original_name: str,
    date: str,
    total: float,
    merchant_name: str,
    memo: str,
    payment: Optional[dict] = None,
) -> int:
    now = int(time.time())
    mid = upsert_merchant(merchant_name)
    p = payment or {}
    def _f(x):
        try:
            return float(x)
        except Exception:
            return 0.0
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO receipts(stored_name, original_name, date, total, subtotal, tax, tip, discounts, fees, method, last4, merchant_id, memo, created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(stored_name) DO UPDATE SET
              date=excluded.date,
              total=excluded.total,
              subtotal=excluded.subtotal,
              tax=excluded.tax,
              tip=excluded.tip,
              discounts=excluded.discounts,
              fees=excluded.fees,
              method=excluded.method,
              last4=excluded.last4,
              merchant_id=excluded.merchant_id,
              memo=excluded.memo
            """,
            (
                stored_name,
                original_name,
                date,
                float(total or 0),
                _f(p.get("subtotal")),
                _f(p.get("tax")),
                _f(p.get("tip")),
                _f(p.get("discounts")),
                _f(p.get("fees")),
                str(p.get("method") or ""),
                str(p.get("last4") or ""),
                mid if mid else None,
                memo,
                now,
            ),
        )
        row = conn.execute("SELECT id FROM receipts WHERE stored_name=?", (stored_name,)).fetchone()
        rid = int(row["id"]) if row else 0
        conn.commit()
        return rid


def clear_line_items(receipt_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM line_items WHERE receipt_id=?", (receipt_id,))
        conn.commit()


def insert_line_items(receipt_id: int, items: Sequence[dict]) -> int:
    if not receipt_id or not items:
        return 0
    now = int(time.time())
    rows = [
        (
            receipt_id,
            (it.get("name") or it.get("description") or "").strip(),
            (it.get("ocr_text") or "").strip(),
            int(it.get("line_index") or -1),
            float(it.get("qty") or 0),
            float(it.get("unit_price") or 0),
            float(it.get("amount") or 0),
            float(it.get("confidence") or 0),
            now,
        )
        for it in items
    ]
    with _connect() as conn:
        conn.executemany(
            "INSERT INTO line_items(receipt_id, description, ocr_text, line_index, qty, unit_price, amount, confidence, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        return len(rows)


def insert_llm_run(stored_name: str, model: str | None, metrics: dict) -> None:
    if not stored_name or not metrics:
        return
    p = metrics.get("prompt", {}) or {}
    r = metrics.get("response", {}) or {}
    t = metrics.get("timing", {}) or {}
    now = int(time.time())
    def _i(x):
        try:
            return int(x)
        except Exception:
            return None
    def _f(x):
        try:
            return float(x)
        except Exception:
            return None
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO llm_runs(
              stored_name, model,
              prompt_chars, prompt_lines, indexed_lines, include_full_text, long_input, full_text_sent_chars,
              prompt_tokens, completion_tokens,
              wall_sec, total_sec, prompt_eval_sec, eval_sec,
              tps_completion, tps_end_to_end, prep_sec,
              created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                stored_name,
                str(model or ""),
                _i(p.get("chars")),
                _i(p.get("lines")),
                _i(p.get("indexed_lines")),
                1 if p.get("include_full_text") else 0,
                1 if p.get("long_input") else 0,
                _i((p.get("cleanup") or {}).get("full_text_sent_chars")),
                _i(r.get("prompt_tokens")),
                _i(r.get("completion_tokens")),
                _f(t.get("wall_sec")),
                _f(t.get("total_sec")),
                _f(t.get("prompt_eval_sec")),
                _f(t.get("eval_sec")),
                _f(t.get("tps_completion")),
                _f(t.get("tps_end_to_end")),
                _f(t.get("prep_sec")),
                now,
            ),
        )
        conn.commit()


def get_llm_stats() -> list[dict]:
    """Return basic aggregates of llm_runs grouped by model."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT model,
                   COUNT(*) AS n,
                   AVG(wall_sec) AS avg_wall,
                   AVG(tps_end_to_end) AS avg_tps,
                   AVG(prompt_chars) AS avg_prompt_chars,
                   AVG(indexed_lines) AS avg_indexed_lines,
                   AVG(include_full_text) AS full_text_rate
            FROM llm_runs
            GROUP BY model
            ORDER BY n DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]
