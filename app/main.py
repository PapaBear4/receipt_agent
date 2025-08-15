from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, cast, List
from fastapi import FastAPI, UploadFile, File, Form, Request, Query
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
import logging
import json
import hashlib
import shutil
from app.services.ocr import ocr_image, ocr_image_detailed, draw_annotated_overlay
from app.services.ocr import ocr_on_cropped_image, draw_overlay_on_image
from app.services.image_preproc import crop_receipt, make_receipt_preview
from app.services.llm import extract_fields_from_text, ollama_health, select_model
from app.services.jobs import job_manager, Job
from app.services.db import init_db
from app.utils.csv_writer import CSVWriter
from app.utils.date_utils import normalize_date_to_mmddyyyy
from app.services.db import insert_receipt

app = FastAPI(title="Receipt Agent MVP")

# Module logger
logger = logging.getLogger(__name__)
if settings.DEBUG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

# Static mount for uploaded/processed images for preview
app.mount("/data", StaticFiles(directory=str(settings.DATA_DIR)), name="data")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

csv_writer = CSVWriter(settings.CSV_PATH)

# Start background job worker
job_manager.start()

# Initialize database schema (idempotent)
try:
    init_db()
    logger.info("Database initialized at %s", settings.DB_PATH)
except Exception as e:
    logger.exception("Failed to initialize database: %s", e)


def _migrate_legacy_overlays() -> None:
    """Copy legacy overlay_<full>.jpg files to the normalized overlay_<stem>.jpg.
    This makes old overlays visible in the Processed gallery.
    """
    try:
        for p in settings.PROCESSED_DIR.glob("overlay_*.jpg"):
            tail = p.name[len("overlay_"):]
            target_name = f"overlay_{Path(tail).stem}.jpg"
            if p.name == target_name:
                continue
            target_path = settings.PROCESSED_DIR / target_name
            if not target_path.exists():
                try:
                    target_path.write_bytes(p.read_bytes())
                    logger.info("Migrated legacy overlay: %s -> %s", p.name, target_name)
                except Exception as e:
                    logger.warning("Failed to migrate overlay %s: %s", p.name, e)
    except Exception as e:
        logger.debug("Overlay migration skipped/failed: %s", e)


_migrate_legacy_overlays()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    logger.debug("Render upload page")
    return templates.TemplateResponse("upload.html", {"request": request})


@app.get("/health")
async def health():
    # Basic liveness; optionally could check Ollama port reachability
    logger.debug("Health check OK")
    return {"status": "ok"}


@app.get("/health/llm")
async def health_llm():
    info = ollama_health()
    # Also report the model that would actually be used (with fallback selection)
    model_value = str(info.get("model", ""))
    info["used_model"] = select_model(model_value)
    # Log concise summary
    logger.info(
        "LLM health: endpoint_ok=%s model_ok=%s model=%s",
        info.get("endpoint_ok"),
        info.get("model_ok"),
        info.get("model"),
    )
    return info


@app.get("/health/llm/test")
async def health_llm_test(q: str = "Walmart 06/09/2025 Total $14.23"):
    used_model = select_model()
    fields = extract_fields_from_text(q)
    return {"model": used_model, "fields": fields}


def _build_review_context(
    request: Request,
    src_path: Path,
    original_filename: str,
) -> Dict[str, object]:
    """Run the OCR + LLM pipeline and construct a template context for review.html."""
    filename = src_path.name
    # Crop (placeholder/no-op currently)
    preview_url = None
    cropped_for_ocr = None
    try:
        import importlib
        cv2 = importlib.import_module("cv2")
        logger.info("Cropping/warping receipt: %s", src_path)
        cropped = crop_receipt(str(src_path))
        cropped_for_ocr = cropped
    except Exception as e:
        if settings.DEBUG:
            logger.exception("Failed to crop/preview: %s", e)
        cropped_for_ocr = None

    # OCR
    annotated_url = None
    ocr_json_url = None
    detailed = None
    raw_text = ""
    try:
        import importlib
        cv2 = importlib.import_module("cv2")
        if cropped_for_ocr is None:
            logger.info("Crop failed; using original image for OCR")
            cropped_for_ocr = cv2.imread(str(src_path))
        logger.info("Starting OCR on cropped image")
        debug_base = Path(filename).stem
        detailed = ocr_on_cropped_image(cropped_for_ocr, debug_basename=debug_base)
        raw_text = detailed.get("text", "")
        logger.info("OCR complete: %d chars, %d lines", len(raw_text or ""), len(detailed.get("lines", [])))
    except Exception:
        logger.exception("OCR failed for %s", src_path)
        raw_text = ""
        detailed = None

    # LLM extraction
    logger.info("Starting LLM field extraction")
    fields = extract_fields_from_text(raw_text)
    used_model = select_model()
    if settings.DEBUG:
        logger.debug("LLM fields: keys=%s", list(fields.keys()))

    # Defaults for review form
    date_val = normalize_date_to_mmddyyyy(fields.get("date", ""))
    payee_val = fields.get("payee", "")
    outflow_val = "" if not fields.get("total") else f"{float(fields['total']):.2f}"
    memo_val = f"Generated by Agent Fineas from {original_filename}"

    # Associate fields to OCR lines
    associated_lines: Dict[str, Optional[tuple]] = {"date": None, "payee": None, "total": None}
    lines = []
    if detailed:
        lines = detailed.get("lines", [])
        lids = detailed.get("line_ids", [])

        def find_lid(value: str):
            if not value:
                return None
            v = value.strip().lower()
            best = None
            for lid, line in zip(lids, lines):
                if v and v in line.lower():
                    best = lid
                    break
            return best

        associated_lines["payee"] = find_lid(payee_val)
        associated_lines["date"] = find_lid(date_val)
        associated_lines["total"] = find_lid(outflow_val)

        try:
            logger.info("Creating overlay for %s", src_path)
            highlight_map = {k: v for k, v in associated_lines.items() if v is not None}
            highlight_map = cast(Dict[str, tuple], highlight_map)
            proc_image = detailed.get("proc_image")
            if proc_image is None:
                import importlib
                cv2 = importlib.import_module("cv2")
                proc_image = cropped_for_ocr if cropped_for_ocr is not None else cv2.imread(str(src_path))
            # Prefer drawing overlay on the original color image if dimensions match,
            # so the preview doesn't look inverted/thresholded while keeping alignment.
            import importlib
            cv2 = importlib.import_module("cv2")
            orig_img = cv2.imread(str(src_path))
            def _wh(img):
                if img is None:
                    return (-1, -1)
                if len(img.shape) == 2:
                    h, w = img.shape
                else:
                    h, w = img.shape[:2]
                return (w, h)
            base_img = proc_image
            if orig_img is not None and _wh(orig_img) == _wh(proc_image):
                base_img = orig_img
            overlay = draw_overlay_on_image(base_img, detailed["words"], detailed["line_ids"], highlight_map)
            overlay_name = f"overlay_{Path(filename).stem}.jpg"
            overlay_path = settings.PROCESSED_DIR / overlay_name
            import importlib
            cv2 = importlib.import_module("cv2")
            cv2.imwrite(str(overlay_path), overlay)
            annotated_url = f"/data/processed/{overlay_name}"
            # OCR JSON dump for download and for client-side overlay
            try:
                serial = {
                    "text": detailed.get("text", ""),
                    "lines": detailed.get("lines", []),
                    "words": detailed.get("words", []),
                    "line_ids": detailed.get("line_ids", []),
                    "size": list(detailed.get("size", []))
                    if isinstance(detailed.get("size"), (list, tuple))
                    else detailed.get("size"),
                }
                ocr_json_name = f"ocr_{Path(filename).stem}.json"
                ocr_json_path = settings.PROCESSED_DIR / ocr_json_name
                ocr_json_path.write_text(json.dumps(serial, indent=2))
                ocr_json_url = f"/data/processed/{ocr_json_name}"
            except Exception as e:
                if settings.DEBUG:
                    logger.exception("Failed to write OCR JSON: %s", e)
        except Exception as e:
            if settings.DEBUG:
                logger.exception("Failed to create overlay: %s", e)
            annotated_url = None

        if settings.DEBUG:
            try:
                debug_dump = {
                    "assoc": associated_lines,
                    "lines": lines,
                    "lids": [list(l) for l in lids],
                }
                dbg_path = settings.PROCESSED_DIR / f"debug_{filename}.json"
                dbg_path.write_text(json.dumps(debug_dump, indent=2))
                logger.debug("Wrote debug dump: %s", dbg_path)
            except Exception:
                logger.exception("Failed to write debug dump for %s", src_path)

    # Assoc CSV strings for front-end JS without JSON filters
    def lid_to_csv(lid: Optional[tuple]) -> str:
        if not lid:
            return ""
        try:
            return ",".join(str(int(x)) for x in lid)
        except Exception:
            return ""

    assoc_csv = {k: lid_to_csv(v) for k, v in associated_lines.items()}

    # Persist fields JSON for cache-first review
    try:
        stem = Path(filename).stem
        fields_doc = {
            "fields": fields,
            "used_model": used_model,
            "stored_name": filename,
            "original_name": original_filename,
            "created_at": datetime.now().isoformat(),
        }
        fields_path = settings.PROCESSED_DIR / f"fields_{stem}.json"
        fields_path.write_text(json.dumps(fields_doc, indent=2))
    except Exception as e:
        if settings.DEBUG:
            logger.exception("Failed to write fields JSON: %s", e)

    ctx: Dict[str, object] = {
        "request": request,
        "image_url": f"/data/uploads/{filename}",
        "annotated_url": annotated_url,
        "ocr_json_url": ocr_json_url,
        "raw_text": raw_text,
        "date": date_val,
        "payee": payee_val,
        "outflow": outflow_val,
        "memo": memo_val,
        "stored_filename": filename,
        "original_filename": original_filename,
        "assoc": associated_lines,
        "assoc_csv": assoc_csv,
        "ocr_lines": lines,
        "used_model": used_model,
        "debug": settings.DEBUG,
    }
    return ctx


@app.post("/upload")
async def upload_receipt(request: Request, file: UploadFile = File(...)):
    # Save upload
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    dest_path = settings.UPLOADS_DIR / filename
    logger.info("Upload received: %s -> %s", file.filename, dest_path)
    with dest_path.open("wb") as f:
        f.write(await file.read())

    ctx = _build_review_context(request, dest_path, file.filename or "upload")
    logger.info("Rendering review page: image_url=%s annotated=%s", ctx.get("image_url"), bool(ctx.get("annotated_url")))
    return templates.TemplateResponse("review.html", ctx)


def _load_cached_review_context(request: Request, stored_name: str, original_name: str) -> Optional[Dict[str, object]]:
    """Return a template context using cached artifacts if available; otherwise None."""
    try:
        stem = Path(stored_name).stem
        fields_path = settings.PROCESSED_DIR / f"fields_{stem}.json"
        ocr_path = settings.PROCESSED_DIR / f"ocr_{stem}.json"
        overlay_primary = settings.PROCESSED_DIR / f"overlay_{stem}.jpg"
        overlay_legacy = settings.PROCESSED_DIR / f"overlay_{stored_name}.jpg"  # backward-compat (double extension)
        if not (fields_path.exists() and ocr_path.exists()):
            return None
        fields_doc = json.loads(fields_path.read_text())
        ocr_doc = json.loads(ocr_path.read_text())
        fields = fields_doc.get("fields", {})
        used_model = fields_doc.get("used_model", "")
        # Pre-fill values
        date_val = normalize_date_to_mmddyyyy(str(fields.get("date", "")))
        payee_val = str(fields.get("payee", ""))
        total = fields.get("total", 0) or 0
        try:
            outflow_val = f"{float(total):.2f}" if total else ""
        except Exception:
            outflow_val = ""
        # Associate lines using cached OCR lines
        lines = ocr_doc.get("lines", []) or []
        lids = ocr_doc.get("line_ids", []) or []
        def find_lid(value: str):
            if not value:
                return None
            v = value.strip().lower()
            for lid, line in zip(lids, lines):
                if v and v in str(line).lower():
                    return lid
            return None
        associated_lines: Dict[str, Optional[tuple]] = {
            "date": find_lid(date_val),
            "payee": find_lid(payee_val),
            "total": find_lid(outflow_val),
        }
        def lid_to_csv(lid: Optional[tuple]) -> str:
            if not lid:
                return ""
            try:
                return ",".join(str(int(x)) for x in lid)
            except Exception:
                return ""
        assoc_csv = {k: lid_to_csv(v) for k, v in associated_lines.items()}

        return {
            "request": request,
            "image_url": f"/data/uploads/{stored_name}",
            "annotated_url": (
                f"/data/processed/{overlay_primary.name}" if overlay_primary.exists() else (
                    f"/data/processed/{overlay_legacy.name}" if overlay_legacy.exists() else None
                )
            ),
            "ocr_json_url": f"/data/processed/ocr_{stem}.json",
            "raw_text": ocr_doc.get("text", ""),
            "date": date_val,
            "payee": payee_val,
            "outflow": outflow_val,
            "memo": f"Generated by Agent Fineas from {original_name}",
            "stored_filename": stored_name,
            "original_filename": original_name,
            "assoc": associated_lines,
            "assoc_csv": assoc_csv,
            "ocr_lines": lines,
            "used_model": used_model,
            "debug": settings.DEBUG,
        }
    except Exception as e:
        if settings.DEBUG:
            logger.exception("Cache load failed for %s: %s", stored_name, e)
        return None


@app.get("/review", response_class=HTMLResponse)
async def review_existing(request: Request, file: str = Query(..., alias="file"), rerun: int = Query(0)):
    """Review a previously uploaded file by filename (under uploads).
    Cache-first: if cached fields + OCR exist and rerun=0, load them.
    """
    src_path = settings.UPLOADS_DIR / file
    if not src_path.exists():
        return HTMLResponse(status_code=404, content=f"File not found: {file}")
    # Use original filename as the tail after timestamp underscore if present
    try:
        original = file.split("_", 1)[1]
    except Exception:
        original = file
    if not rerun:
        cached = _load_cached_review_context(request, file, original)
        if cached:
            return templates.TemplateResponse("review.html", cached)
    ctx = _build_review_context(request, src_path, original)
    logger.info("Rendering review page for existing upload: %s", file)
    return templates.TemplateResponse("review.html", ctx)


@app.post("/upload/batch", response_class=HTMLResponse)
async def upload_batch(request: Request, files: List[UploadFile] = File(...)):
    """Accept multiple files, dedupe by sha1, save, and provide links to review."""
    index_path = settings.DATA_DIR / "uploads_index.json"
    try:
        existing_index: Dict[str, str] = json.loads(index_path.read_text()) if index_path.exists() else {}
    except Exception:
        existing_index = {}

    results: List[Dict[str, str]] = []
    for file in files:
        try:
            content = await file.read()
            sha1 = hashlib.sha1(content).hexdigest()
            if sha1 in existing_index:
                # duplicate
                existing_name = existing_index[sha1]
                results.append({
                    "filename": file.filename or "",
                    "status": "duplicate",
                    "stored": existing_name,
                    "review_url": f"/review?file={existing_name}",
                })
                continue
            # save new
            stored_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            dest_path = settings.UPLOADS_DIR / stored_name
            dest_path.write_bytes(content)
            existing_index[sha1] = stored_name
            results.append({
                "filename": file.filename or "",
                "status": "saved",
                "stored": stored_name,
                "review_url": f"/review?file={stored_name}",
            })
        except Exception as e:
            logger.exception("Batch upload failed for %s: %s", file.filename, e)
            results.append({
                "filename": file.filename or "",
                "status": "error",
                "stored": "",
                "review_url": "",
            })

    # persist index
    try:
        index_path.write_text(json.dumps(existing_index, indent=2))
    except Exception:
        logger.exception("Failed to persist uploads index: %s", index_path)

    # Enqueue background processing jobs for all saved (non-duplicate) files
    for r in results:
        if r.get("status") == "saved" and r.get("stored"):
            jid = f"job_{r['stored']}"
            job_manager.enqueue(Job(id=jid, stored_name=r["stored"], original_name=r["filename"]))

    return RedirectResponse(url="/jobs", status_code=303)


@app.get("/jobs", response_class=HTMLResponse)
async def list_jobs(request: Request):
    jobs = job_manager.all()
    # Prepare a light view model
    items = []
    for jid, j in sorted(jobs.items(), key=lambda kv: kv[1].created_at, reverse=True):
        items.append({
            "id": jid,
            "stored": j.stored_name,
            "original": j.original_name,
            "status": j.status,
            "error": j.error or "",
            "review_url": f"/review?file={j.stored_name}" if j.status == "done" else "",
        })
    return templates.TemplateResponse("jobs.html", {"request": request, "jobs": items})


@app.get("/processed", response_class=HTMLResponse)
async def list_processed(request: Request):
    # List processed overlays/fields pairs by reading the processed folder
    entries = []
    for p in sorted(settings.PROCESSED_DIR.glob("fields_*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        stem = p.stem.replace("fields_", "")
        overlay = settings.PROCESSED_DIR / f"overlay_{stem}.jpg"
        ocrj = settings.PROCESSED_DIR / f"ocr_{stem}.json"
        # stored filename includes the original extension; stem is <timestamp>_<orig_name_without_ext>
        # The uploaded stored_name is typically <timestamp>_<orig_name> so reconstruct by scanning uploads.
        stored_name_guess = None
        # Primary: find any upload whose stem matches our stem prefix
        try:
            for up in settings.UPLOADS_DIR.glob(f"{stem}.*"):
                stored_name_guess = up.name
                break
        except Exception:
            stored_name_guess = None
        entries.append({
            "stem": stem,
            "overlay_url": f"/data/processed/overlay_{stem}.jpg" if overlay.exists() else "",
            "ocr_json_url": f"/data/processed/ocr_{stem}.json" if ocrj.exists() else "",
            "review_url": f"/review?file={stored_name_guess}" if stored_name_guess else "",
        })
    return templates.TemplateResponse("processed.html", {"request": request, "items": entries})


@app.post("/admin/clear", response_class=HTMLResponse)
async def admin_clear(request: Request):
    """Clear uploads, processed artifacts, and reset CSV. Guard if jobs are running."""
    if job_manager.any_busy():
        return HTMLResponse(status_code=409, content="Jobs are running. Try again later.")

    # Best-effort deletion, recreate directories and CSV headers
    try:
        # Clear uploads and processed
        for p in settings.UPLOADS_DIR.glob("*"):
            try:
                p.unlink()
            except IsADirectoryError:
                shutil.rmtree(p, ignore_errors=True)
        for p in settings.PROCESSED_DIR.glob("*"):
            try:
                p.unlink()
            except IsADirectoryError:
                shutil.rmtree(p, ignore_errors=True)
        # Reset CSV
        if settings.CSV_PATH.exists():
            settings.CSV_PATH.unlink()
        # Recreate CSV with headers
        _ = CSVWriter(settings.CSV_PATH)
        # Clear uploads index
        idx = settings.DATA_DIR / "uploads_index.json"
        if idx.exists():
            idx.unlink()
        # Clear job records
        job_manager.clear_jobs()
        logger.info("All data cleared by admin request")
    except Exception as e:
        logger.exception("Admin clear failed: %s", e)
        return HTMLResponse(status_code=500, content="Failed to clear data.")

    # Redirect back to upload page
    return RedirectResponse(url="/", status_code=303)


@app.post("/save")
async def save_receipt(
    request: Request,
    stored_filename: str = Form(...),
    original_filename: str = Form(...),
    date: str = Form(...),
    payee: str = Form(...),
    outflow: str = Form(...),
    memo: str = Form(...),
):
    # Validate
    logger.info("Saving receipt: stored=%s original=%s", stored_filename, original_filename)
    norm_date = normalize_date_to_mmddyyyy(date)
    errors = []
    if not norm_date:
        errors.append("Invalid or missing date")
    payee = (payee or "").strip()
    if not payee:
        errors.append("Payee is required")
    amount: float = 0.0
    try:
        amount = float(outflow)
        if amount <= 0:
            errors.append("Outflow must be > 0")
    except Exception:
        errors.append("Outflow must be a number")

    if errors:
        logger.warning("Validation errors: %s", errors)
        # Re-render review with errors
        image_url = f"/data/uploads/{stored_filename}"
        raw_text, _ = ocr_image(str(settings.UPLOADS_DIR / stored_filename))
        return templates.TemplateResponse(
            "review.html",
            {
                "request": request,
                "errors": errors,
                "image_url": image_url,
                "raw_text": raw_text,
                "date": date,
                "payee": payee,
                "outflow": outflow,
                "memo": memo,
                "stored_filename": stored_filename,
                "original_filename": original_filename,
            },
        )

    # Append to CSV (handle failures and do not proceed if write fails)
    try:
        csv_writer.append_row([norm_date, payee, "", memo, f"{amount:.2f}", ""])
        logger.info("Appended to CSV: %s", settings.CSV_PATH)
    except Exception as e:
        logger.exception("CSV write failed for %s: %s", settings.CSV_PATH, e)
        # Re-render review with error, do not copy to processed
        image_url = f"/data/uploads/{stored_filename}"
        raw_text, _ = ocr_image(str(settings.UPLOADS_DIR / stored_filename))
        return templates.TemplateResponse(
            "review.html",
            {
                "request": request,
                "errors": ["Failed to write to CSV. Please try again."],
                "image_url": image_url,
                "raw_text": raw_text,
                "date": norm_date,
                "payee": payee,
                "outflow": f"{amount:.2f}",
                "memo": memo,
                "stored_filename": stored_filename,
                "original_filename": original_filename,
            },
        )

    # Copy to processed
    src = settings.UPLOADS_DIR / stored_filename
    timestamped = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{stored_filename}"
    dst = settings.PROCESSED_DIR / timestamped
    try:
        dst.write_bytes(src.read_bytes())
        logger.info("Copied to processed: %s", dst)
    except Exception:
        logger.exception("Failed to copy to processed: %s -> %s", src, dst)

    # Persist to DB (best-effort; does not block redirect)
    try:
        rid = insert_receipt(stored_filename, original_filename, norm_date, amount, payee, memo)
        logger.info("Saved receipt to DB: id=%s stored=%s", rid, stored_filename)
    except Exception:
        logger.exception("Failed to save receipt in DB: stored=%s", stored_filename)

    # Redirect to upload page (flash messages could be added later)
    logger.info("Redirecting to upload page")
    return RedirectResponse(url="/", status_code=303)


@app.get("/download/csv")
async def download_csv():
    logger.info("CSV download requested: %s", settings.CSV_PATH)
    return FileResponse(str(settings.CSV_PATH), media_type="text/csv", filename="ynab_receipts.csv")

