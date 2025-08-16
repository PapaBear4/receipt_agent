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

## Data locations
- `data/uploads/` — Original uploaded files (timestamped prefix).
- `data/processed/` — Generated overlays and JSON artifacts, plus optional copied images.
- `data/ynab_receipts.csv` — Appended CSV.

## Troubleshooting
- Overlay looks inverted: we prefer drawing on the original color image when sizes match. If you still see inversions, ensure images load correctly and try `?rerun=1`.
- LLM unavailable: visit `/health/llm` to see endpoint/model status; app will choose a fallback if the preferred model is missing.
- OpenCV errors on Linux: install `libgl1` (or equivalent) for headless environments.

## License
Private MVP. Add a license if you plan to distribute.
