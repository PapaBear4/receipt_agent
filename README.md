# Receipt Agent MVP

Minimal FastAPI app to OCR receipt images, extract fields with a local LLM (Ollama), and export to a YNAB‑style CSV. Supports single and batch uploads with background processing and a cache‑first review UI.

## Features
- Upload single or multiple images (JPG/PNG); duplicates deduped by SHA‑1.
- Background jobs: OCR → LLM → artifacts; Jobs page with auto‑refresh.
- Review UI with overlay highlights, zoom/toggle, hotkeys, and LLM model badge.
- Persistent artifacts under `data/processed/`: `overlay_<stem>.jpg`, `ocr_<stem>.json`, `fields_<stem>.json`.
- Cache‑first review (instant reload from artifacts; `?rerun=1` to recompute).
- CSV append with validation and error handling; download at `/download/csv`.
- Admin: Clear all data (uploads, processed, CSV) with guard against running jobs.

## Architecture overview
- FastAPI app in `app/main.py`
- Services in `app/services/`:
  - `ocr.py` (Tesseract/OpenCV), `llm.py` (Ollama), `jobs.py` (background processing), `db.py` (SQLite),
  - `ynab.py` (raw YNAB API client) and `ynab_service.py` (route-facing logic for settings/meta/push)
- Templates and static assets under `app/templates/` and `app/static/`

## Requirements
- Python 3.10+ (tested on 3.12)
- System packages:
  - Tesseract OCR (`tesseract-ocr`)
  - OpenCV runtime deps (Ubuntu/Debian: `libgl1`)
- Python packages (examples): FastAPI, Uvicorn, Jinja2, Pillow, pytesseract, opencv-python, requests
- Ollama running locally with at least one instruction model (e.g. `llama3.1:8b-instruct-q4_0`)

## Quick start
1. Create and activate a virtualenv.
2. Install Python deps:
   - fastapi, uvicorn[standard], jinja2, pillow, pytesseract, opencv-python, requests
3. Ensure Tesseract is installed and reachable in PATH.
4. Start Ollama and make sure your model is available (`/health/llm` will verify and select a fallback if needed).
5. Run the app:
   - `python run.py`
6. Open http://localhost:8000

## Configuration (env vars)
- General
  - `APP_HOST` (default `0.0.0.0`)
  - `APP_PORT` (default `8000`)
  - `DATA_DIR` (default `<repo>/data`)
  - `CSV_PATH` (default `<DATA_DIR>/ynab_receipts.csv`)
- LLM
  - `OLLAMA_ENDPOINT` (default `http://127.0.0.1:11434`)
  - `OLLAMA_MODEL` (default `llama3.1:8b-instruct-q4_0`)
  - `OLLAMA_TIMEOUT` (default `1200` seconds)
- OCR tuning (sane defaults)
  - `OCR_RAW_ONLY` (default `true`)
  - `OCR_PSMS`, `OCR_OEM`, `OCR_USE_WHITELIST`, `OCR_DISABLE_DICTIONARY`, `OCR_PRESERVE_SPACES`, `OCR_USER_DPI`
  - `OCR_USE_THRESH`, `OCR_ADAPTIVE_BLOCK`, `OCR_ADAPTIVE_C`, `OCR_MEDIAN_BLUR`
- YNAB
  - `YNAB_API_BASE` (default `https://api.ynab.com/v1`)
  - `YNAB_TOKEN` (required for YNAB features)
  - `YNAB_BUDGET_ID` (optional default budget)
  - `YNAB_DEFAULT_ACCOUNT_ID` (optional default account for pushes)
- Debug
  - `DEBUG` (default `false`)

## Endpoints
- `/` — Upload UI (single + batch) with links to Jobs and Processed.
- `/review?file=<stored_name>[&rerun=1]` — Review a receipt; cache‑first with optional recompute.
- `/jobs` — Background jobs with status and auto‑refresh.
- `/processed` — Gallery of processed items with Review/OCR links.
- `/health`, `/health/llm`, `/health/llm/test` — Health and LLM info.
- `/download/csv` — Download CSV.
- `POST /admin/clear` — Clear all data (guarded if jobs are running).
- YNAB
  - `GET /settings/ynab` — Browse budgets, accounts, categories, payees (via `ynab_service`).
  - `POST /settings/ynab` — Save selected budget to `data/user_config.json`.
  - `GET /ynab/meta` — Summary view for budgets/accounts/categories.
  - `POST /ynab/push` — Push a single transaction to YNAB.
  - `POST /ynab/mapping/account` — Persist account mapping for a card last4.

## Data locations
- `data/uploads/` — Original uploaded files (timestamped prefix).
- `data/processed/` — Generated overlays and JSON artifacts, plus optional copied images.
- `data/ynab_receipts.csv` — Appended CSV.

## Database schema (SQLite)
The schema is created and migrated by `app/services/db.py`. Current DDL:

```sql
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

CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS abstract_items (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    category_id INTEGER,
    notes TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories(id)
);

CREATE TABLE IF NOT EXISTS item_variants (
    id INTEGER PRIMARY KEY,
    abstract_item_id INTEGER NOT NULL,
    brand TEXT,
    name TEXT NOT NULL,
    size_value REAL,
    size_unit TEXT,
    upc TEXT,
    organic INTEGER,
    gluten_free INTEGER,
    attributes TEXT,
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
    unit_price REAL,
    unit TEXT,
    promo INTEGER,
    captured_at INTEGER NOT NULL,
    receipt_id INTEGER,
    line_item_id INTEGER,
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

CREATE TABLE IF NOT EXISTS ynab_mappings (
    id INTEGER PRIMARY KEY,
    budget_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    chosen_id TEXT NOT NULL,
    chosen_name TEXT,
    updated_at INTEGER NOT NULL,
    UNIQUE (budget_id, kind, key)
);

CREATE TABLE IF NOT EXISTS ynab_suggestions (
    id INTEGER PRIMARY KEY,
    budget_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    guess_id TEXT,
    guess_name TEXT,
    confidence REAL,
    model TEXT,
    prompt_hash TEXT,
    expires_at INTEGER NOT NULL,
    created_at INTEGER NOT NULL
);
```

## Troubleshooting
- Overlay looks inverted: we prefer drawing on the original color image when sizes match. If you still see inversions, ensure images load correctly and try `?rerun=1`.
- LLM unavailable: visit `/health/llm` to see endpoint/model status; app will choose a fallback if the preferred model is missing.
- OpenCV errors on Linux: install `libgl1` (or equivalent) for headless environments.

## License
Private MVP. Add a license if you plan to distribute.
