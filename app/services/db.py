from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, List, Dict, Any
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

-- Grocery catalog: abstract items, specific variants, and price captures
CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS abstract_items (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE, -- canonical name, e.g., 'peanut butter'
    category_id INTEGER,
    notes TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories(id)
);

CREATE TABLE IF NOT EXISTS item_variants (
    id INTEGER PRIMARY KEY,
    abstract_item_id INTEGER NOT NULL,
    brand TEXT,                       -- e.g., 'Jif'
    name TEXT NOT NULL,               -- full product name as seen, e.g., 'Jif creamy peanut butter 8oz organic'
    size_value REAL,                  -- e.g., 8.0
    size_unit TEXT,                   -- e.g., 'oz'
    upc TEXT,
    organic INTEGER,
    gluten_free INTEGER,
    attributes TEXT,                  -- JSON string for flexible attributes
    created_at INTEGER NOT NULL,
    updated_at INTEGER,
    FOREIGN KEY (abstract_item_id) REFERENCES abstract_items(id)
);

CREATE TABLE IF NOT EXISTS price_captures (
    id INTEGER PRIMARY KEY,
    item_variant_id INTEGER NOT NULL,
    merchant_id INTEGER NOT NULL,
    price REAL NOT NULL,
    currency TEXT,
    unit_price REAL,                  -- price per unit (derived) if known
    unit TEXT,                        -- unit basis for unit_price (e.g., 'oz')
    promo INTEGER,                    -- 1 if promotional price
    captured_at INTEGER NOT NULL,     -- unix time
    receipt_id INTEGER,               -- optional link back
    line_item_id INTEGER,             -- optional link back
    created_at INTEGER NOT NULL,
    FOREIGN KEY (item_variant_id) REFERENCES item_variants(id),
    FOREIGN KEY (merchant_id) REFERENCES merchants(id),
    FOREIGN KEY (receipt_id) REFERENCES receipts(id),
    FOREIGN KEY (line_item_id) REFERENCES line_items(id)
);

CREATE INDEX IF NOT EXISTS idx_item_variants_abstract ON item_variants(abstract_item_id);
CREATE INDEX IF NOT EXISTS idx_price_captures_variant ON price_captures(item_variant_id);
CREATE INDEX IF NOT EXISTS idx_price_captures_merchant ON price_captures(merchant_id);
CREATE INDEX IF NOT EXISTS idx_price_captures_captured ON price_captures(captured_at);
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


# ---- Grocery catalog helpers ----
def upsert_category(name: str) -> int:
    name = (name or "").strip()
    if not name:
        return 0
    now = int(time.time())
    with _connect() as conn:
        conn.execute("INSERT OR IGNORE INTO categories(name, created_at) VALUES(?, ?)", (name, now))
        row = conn.execute("SELECT id FROM categories WHERE name=?", (name,)).fetchone()
        conn.commit()
        return int(row[0]) if row else 0


def upsert_abstract_item(name: str, category: str | None = None, notes: str | None = None) -> int:
    nm = (name or "").strip()
    if not nm:
        return 0
    now = int(time.time())
    cat_id = upsert_category(category) if category else None
    with _connect() as conn:
        # insert or ignore by unique name, then update optional fields
        conn.execute("INSERT OR IGNORE INTO abstract_items(name, category_id, notes, created_at) VALUES(?,?,?,?)", (nm, cat_id, notes or None, now))
        conn.execute("UPDATE abstract_items SET category_id=COALESCE(?, category_id), notes=COALESCE(?, notes), updated_at=? WHERE name=?", (cat_id, notes or None, now, nm))
        row = conn.execute("SELECT id FROM abstract_items WHERE name=?", (nm,)).fetchone()
        conn.commit()
        return int(row[0]) if row else 0


def upsert_item_variant(abstract_item_id: int, name: str, brand: str | None = None, size_value: float | None = None, size_unit: str | None = None, upc: str | None = None, attributes: dict | None = None, organic: bool | None = None, gluten_free: bool | None = None) -> int:
    if not abstract_item_id or not name:
        return 0
    now = int(time.time())
    attrs = None
    try:
        import json as _json
        attrs = _json.dumps(attributes) if attributes else None
    except Exception:
        attrs = None
    with _connect() as conn:
        # Variants are not globally unique by name since different abstract items could share a name.
        # We'll consider (abstract_item_id, name) as a soft-unique pair.
        conn.execute(
            """
            INSERT INTO item_variants(abstract_item_id, brand, name, size_value, size_unit, upc, organic, gluten_free, attributes, created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO NOTHING
            """,
            (abstract_item_id, brand, name, size_value, size_unit, upc, 1 if organic else 0 if organic is not None else None, 1 if gluten_free else 0 if gluten_free is not None else None, attrs, now),
        )
        # Fetch by pair
        row = conn.execute("SELECT id FROM item_variants WHERE abstract_item_id=? AND name=?", (abstract_item_id, name)).fetchone()
        if row:
            vid = int(row[0])
            conn.execute(
                "UPDATE item_variants SET brand=COALESCE(?, brand), size_value=COALESCE(?, size_value), size_unit=COALESCE(?, size_unit), upc=COALESCE(?, upc), attributes=COALESCE(?, attributes), updated_at=? WHERE id=?",
                (brand, size_value, size_unit, upc, attrs, now, vid),
            )
            conn.commit()
            return vid
        conn.commit()
        # In rare race, re-select
        row2 = conn.execute("SELECT id FROM item_variants WHERE abstract_item_id=? AND name=?", (abstract_item_id, name)).fetchone()
        return int(row2[0]) if row2 else 0


def insert_price_capture(item_variant_id: int, merchant_id: int, price: float, captured_at: int, receipt_id: int | None = None, line_item_id: int | None = None, currency: str | None = None, unit_price: float | None = None, unit: str | None = None, promo: bool | None = None) -> int:
    if not item_variant_id or not merchant_id:
        return 0
    now = int(time.time())
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO price_captures(item_variant_id, merchant_id, price, currency, unit_price, unit, promo, captured_at, receipt_id, line_item_id, created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (item_variant_id, merchant_id, float(price or 0), currency, unit_price, unit, 1 if promo else 0 if promo is not None else None, int(captured_at or now), receipt_id, line_item_id, now),
        )
        conn.commit()
        return int(cur.lastrowid or 0)


def get_receipt(receipt_id: int) -> dict | None:
    if not receipt_id:
        return None
    with _connect() as conn:
        row = conn.execute("SELECT * FROM receipts WHERE id=?", (receipt_id,)).fetchone()
        return dict(row) if row else None


def list_abstract_items(limit: int = 100, offset: int = 0) -> list[dict]:
    with _connect() as conn:
        cur = conn.execute(
            """
            SELECT *
            FROM abstract_items
            ORDER BY (updated_at IS NULL) ASC, updated_at DESC, created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        cols = [c[0] for c in cur.description or []]
        return [ {k: row[idx] for idx,k in enumerate(cols)} for row in cur.fetchall() ]


def list_variants_for_abstract(abstract_item_id: int) -> list[dict]:
    with _connect() as conn:
        cur = conn.execute(
            """
            SELECT *
            FROM item_variants
            WHERE abstract_item_id=?
            ORDER BY (updated_at IS NULL) ASC, updated_at DESC, created_at DESC
            """,
            (abstract_item_id,),
        )
        cols = [c[0] for c in cur.description or []]
        return [ {k: row[idx] for idx,k in enumerate(cols)} for row in cur.fetchall() ]


def recent_prices_for_variant(item_variant_id: int, limit: int = 20) -> list[dict]:
    with _connect() as conn:
        cur = conn.execute(
            """
            SELECT pc.*, m.name AS merchant_name
            FROM price_captures pc
            LEFT JOIN merchants m ON m.id = pc.merchant_id
            WHERE pc.item_variant_id=?
            ORDER BY pc.captured_at DESC
            LIMIT ?
            """,
            (item_variant_id, limit),
        )
        cols = [c[0] for c in cur.description or []]
        return [ {k: row[idx] for idx,k in enumerate(cols)} for row in cur.fetchall() ]


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


def get_line_items_for_receipt(receipt_id: int) -> list[dict]:
    if not receipt_id:
        return []
    with _connect() as conn:
        cur = conn.execute(
            "SELECT id, description, ocr_text, line_index, qty, unit_price, amount, confidence FROM line_items WHERE receipt_id=?",
            (receipt_id,),
        )
        cols = [c[0] for c in cur.description or []]
        return [ {k: row[idx] for idx,k in enumerate(cols)} for row in cur.fetchall() ]


# ---- Read-only DB inspection helpers ----
def list_tables() -> List[str]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [str(r[0]) for r in rows]


def get_table_columns(table: str) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return [dict(r) for r in rows]


def get_table_count(table: str) -> int:
    with _connect() as conn:
        row = conn.execute(f"SELECT COUNT(1) AS c FROM {table}").fetchone()
        return int(row[0]) if row else 0


def get_table_rows(table: str, offset: int, limit: int) -> List[Dict[str, Any]]:
    if offset < 0:
        offset = 0
    if limit <= 0:
        limit = 50
    with _connect() as conn:
        cur = conn.execute(f"SELECT * FROM {table} LIMIT ? OFFSET ?", (int(limit), int(offset)))
        cols = [c[0] for c in cur.description or []]
        out: List[Dict[str, Any]] = []
        for row in cur.fetchall():
            out.append({k: row[idx] for idx, k in enumerate(cols)})
        return out


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
