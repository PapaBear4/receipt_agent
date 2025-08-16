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
