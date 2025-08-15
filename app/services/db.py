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
  merchant_id INTEGER,
  memo TEXT,
  created_at INTEGER NOT NULL,
  FOREIGN KEY (merchant_id) REFERENCES merchants(id)
);

CREATE TABLE IF NOT EXISTS line_items (
  id INTEGER PRIMARY KEY,
  receipt_id INTEGER NOT NULL,
  description TEXT,
  qty REAL,
  unit_price REAL,
  amount REAL,
  created_at INTEGER NOT NULL,
  FOREIGN KEY (receipt_id) REFERENCES receipts(id)
);
"""


def _connect() -> sqlite3.Connection:
    Path(settings.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA_SQL)
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


def insert_receipt(stored_name: str, original_name: str, date: str, total: float, merchant_name: str, memo: str) -> int:
    now = int(time.time())
    mid = upsert_merchant(merchant_name)
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO receipts(stored_name, original_name, date, total, merchant_id, memo, created_at)
            VALUES(?,?,?,?,?,?,?)
            """,
            (stored_name, original_name, date, total, mid if mid else None, memo, now),
        )
        if cur.lastrowid:
            rid = cur.lastrowid
        else:
            row = conn.execute("SELECT id FROM receipts WHERE stored_name=?", (stored_name,)).fetchone()
            rid = int(row["id"]) if row else 0
        conn.commit()
        return rid


def insert_line_items(receipt_id: int, items: Sequence[dict]) -> int:
    if not receipt_id or not items:
        return 0
    now = int(time.time())
    rows = [
        (
            receipt_id,
            (it.get("description") or "").strip(),
            float(it.get("qty") or 0),
            float(it.get("unit_price") or 0),
            float(it.get("amount") or 0),
            now,
        )
        for it in items
    ]
    with _connect() as conn:
        conn.executemany(
            "INSERT INTO line_items(receipt_id, description, qty, unit_price, amount, created_at) VALUES(?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        return len(rows)
