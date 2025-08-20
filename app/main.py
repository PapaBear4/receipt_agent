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
from app.services.ocr import ocr_image
from app.services.llm import extract_fields_from_text, ollama_health
from app.services.jobs import job_manager, Job
from app.services.db import init_db
from app.services.ynab import YNABClient
from app.utils.csv_writer import CSVWriter
from app.utils.date_utils import normalize_date_to_mmddyyyy
from app.utils import parse_size, normalize_abstract_name, parse_date_to_unix
from app.services.db import (
    insert_receipt, insert_line_items, clear_line_items,
    insert_llm_run, get_llm_stats,
    list_tables, get_table_columns, get_table_count, get_table_rows,
    list_abstract_items, list_variants_for_abstract, recent_prices_for_variant,
    get_line_items_for_receipt, upsert_abstract_item, upsert_item_variant, insert_price_capture, upsert_merchant, get_receipt,
    get_mapping, upsert_mapping, clear_table, clear_all_tables,
)

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
USER_CONF_PATH = settings.DATA_DIR / "user_config.json"

def _load_user_conf() -> dict:
    try:
        if USER_CONF_PATH.exists():
            return json.loads(USER_CONF_PATH.read_text())
    except Exception:
        pass
    return {}

def _save_user_conf(d: dict) -> None:
    try:
        USER_CONF_PATH.write_text(json.dumps(d, indent=2))
    except Exception:
        pass

csv_writer = CSVWriter(settings.CSV_PATH)

@app.on_event("startup")
async def _startup() -> None:
    try:
        init_db()
    except Exception as e:
        logger.exception("DB init failed: %s", e)
    try:
        job_manager.start()
        logger.info("Background JobManager started")
    except Exception as e:
        logger.exception("Failed to start JobManager: %s", e)


@app.post("/settings/ynab")
async def settings_ynab_save(budget_id: str = Form("")):
    conf = _load_user_conf()
    conf["ynab_budget_id"] = budget_id
    _save_user_conf(conf)
    return RedirectResponse(url="/settings/ynab", status_code=303)


@app.post("/ynab/push")
async def ynab_push(
    stored_filename: str = Form(...),
    date: str = Form(...),
    payee: str = Form(""),
    outflow: str = Form(...),
    memo: str = Form(""),
    account_id: str = Form("") ,
    category_id: str = Form("") ,
):
    client = YNABClient()
    if not client.is_configured():
        return {"ok": False, "error": "YNAB not configured"}
    bid = _load_user_conf().get("ynab_budget_id") or settings.YNAB_BUDGET_ID
    if not bid:
        return {"ok": False, "error": "YNAB_BUDGET_ID not set"}
    acc = account_id or (settings.YNAB_DEFAULT_ACCOUNT_ID or "")
    if not acc:
        return {"ok": False, "error": "account_id required"}
    try:
        amt = float(outflow)
    except Exception:
        return {"ok": False, "error": "invalid amount"}
    try:
        iid = YNABClient.make_import_id(date, amt, stored_filename)
        data = client.create_transaction(
            bid,
            acc,
            date=date,
            amount=amt,
            payee_name=payee,
            memo=memo,
            category_id=(category_id or None),
            import_id=iid,
        )
        tx = (data.get("transaction") or {}) if isinstance(data, dict) else {}
        return {"ok": True, "transaction": tx}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/metrics/llm.json")
async def llm_metrics_json():
    try:
        stats = get_llm_stats()
        return stats
    except Exception as e:
        logger.debug("Failed to load llm stats json: %s", e)
        return []


@app.get("/admin/db", response_class=HTMLResponse)
async def db_browser(request: Request, table: Optional[str] = None, page: int = 1, size: int = 50):
    try:
        tables = list_tables()
    except Exception as e:
        tables = []
        logger.debug("list_tables failed: %s", e)
    if not table and tables:
        table = tables[0]
    rows = []
    cols = []
    total = 0
    pages = 1
    page = max(1, int(page or 1))
    size = max(1, min(500, int(size or 50)))
    if table:
        try:
            cols = get_table_columns(table)
            total = get_table_count(table)
            pages = max(1, (total + size - 1) // size)
            offset = (page - 1) * size
            rows = get_table_rows(table, offset, size)
        except Exception as e:
            logger.debug("db view error for %s: %s", table, e)
    ctx = {
        "request": request,
        "tables": tables,
        "table": table or "",
        "cols": cols,
        "rows": rows,
        "page": page,
        "pages": pages,
        "size": size,
        "total": total,
    }
    return templates.TemplateResponse("db_browser.html", ctx)


@app.post("/admin/db/clear")
async def db_clear_all(confirm: str = Form("")):
    if confirm != "yes":
        return RedirectResponse(url="/admin/db", status_code=303)
    # Exclude sqlite internal tables (already filtered) and nothing else by default
    _ = clear_all_tables()
    return RedirectResponse(url="/admin/db", status_code=303)


@app.post("/admin/db/clear_table")
async def db_clear_table(table: str = Form(...)):
    if not table:
        return RedirectResponse(url="/admin/db", status_code=303)
    try:
        clear_table(table)
    except Exception:
        pass
    return RedirectResponse(url=f"/admin/db?table={table}", status_code=303)


@app.get("/admin/items", response_class=HTMLResponse)
async def admin_items(request: Request, abstract_id: Optional[int] = None):
    items = list_abstract_items(limit=200)
    variants = []
    prices = []
    if abstract_id:
        try:
            variants = list_variants_for_abstract(int(abstract_id))
            if variants:
                first_id = variants[0].get("id")
                vid = None
                try:
                    vid = int(first_id) if first_id is not None else None
                except Exception:
                    vid = None
                if vid:
                    prices = recent_prices_for_variant(vid, limit=20)
        except Exception as e:
            logger.debug("Failed loading variants/prices: %s", e)
    return templates.TemplateResponse("items_admin.html", {"request": request, "items": items, "variants": variants, "abstract_id": abstract_id or 0, "prices": prices})


"""
All inline OCR/LLM processing for review has been removed to ensure LLM prompts run only in background jobs.
"""






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
        payment = (fields.get("payment") or {})
        items = (fields.get("items") or [])
        metrics = fields.get("metrics")
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
            "payment": payment,
            "items": items,
            "stored_filename": stored_name,
            "original_filename": original_name,
            "assoc": associated_lines,
            "assoc_csv": assoc_csv,
            "ocr_lines": lines,
            "used_model": used_model,
            "metrics": metrics,
            "debug": settings.DEBUG,
            "config": settings,
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
    if rerun:
        # Enqueue a background re-process job with force=true, then redirect
        jid = f"job_reprocess_{file}"
        job_manager.enqueue(Job(id=jid, stored_name=file, original_name=original, force=True))
        return RedirectResponse(url=f"/jobs", status_code=303)
    # Cache-first path
    cached = _load_cached_review_context(request, file, original)
    if cached:
        # Attach mapping suggestion for account by card_last4 if available
        try:
            budget_id = (_load_user_conf().get("ynab_budget_id") or settings.YNAB_BUDGET_ID or "").strip()
            last4 = ""
            if isinstance(cached, dict):
                pay_obj = cached.get("payment")
                if isinstance(pay_obj, dict):
                    try:
                        last4 = str(pay_obj.get("last4") or "").strip()
                    except Exception:
                        last4 = ""
            sugg = {}
            if budget_id and last4:
                m = get_mapping(budget_id, "account", f"card_last4:{last4}")
                if m:
                    sugg["account_id"] = m.get("chosen_id")
                    sugg["account_name"] = m.get("chosen_name")
            cached["suggestions"] = sugg
        except Exception:
            cached["suggestions"] = {}
        return templates.TemplateResponse("review.html", cached)
    # Fallback: no cache → queue background processing and redirect to jobs
    jid = f"job_build_{file}"
    job_manager.enqueue(Job(id=jid, stored_name=file, original_name=original, force=False))
    return RedirectResponse(url="/jobs", status_code=303)


@app.post("/ynab/mapping/account")
async def ynab_save_account_mapping(
    key_last4: str = Form(...),
    chosen_id: str = Form(...),
    chosen_name: str = Form("")
):
    budget_id = (_load_user_conf().get("ynab_budget_id") or settings.YNAB_BUDGET_ID or "").strip()
    if not budget_id:
        return {"ok": False, "error": "no budget selected"}
    if not key_last4 or not chosen_id:
        return {"ok": False, "error": "missing inputs"}
    key = f"card_last4:{key_last4.strip()}"
    try:
        _ = upsert_mapping(budget_id, "account", key, chosen_id.strip(), (chosen_name or "").strip())
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


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
    # Optional payment fields
    subtotal: Optional[str] = Form(None),
    tax: Optional[str] = Form(None),
    tip: Optional[str] = Form(None),
    discounts: Optional[str] = Form(None),
    fees: Optional[str] = Form(None),
    method: Optional[str] = Form(None),
    last4: Optional[str] = Form(None),
    # Items JSON from the form (stringified JSON array)
    items_json: Optional[str] = Form(None),
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
        payment = {
            "subtotal": subtotal or 0,
            "tax": tax or 0,
            "tip": tip or 0,
            "discounts": discounts or 0,
            "fees": fees or 0,
            "method": (method or "").strip(),
            "last4": (last4 or "").strip(),
        }
        rid = insert_receipt(stored_filename, original_filename, norm_date, amount, payee, memo, payment)
        # Items
        if items_json:
            try:
                items = json.loads(items_json)
                clear_line_items(rid)
                n = insert_line_items(rid, items)
                logger.info("Saved %d line items to DB for receipt id=%s", n, rid)
                # Auto-seed abstract items, variants, and price captures
                try:
                    # Merchant id from DB (upsert again to get id)
                    mid = upsert_merchant(payee)
                    # Capture timestamp from normalized date
                    from app.utils import parse_date_to_unix as _parse_unix
                    captured_at = _parse_unix(norm_date)
                    line_rows = get_line_items_for_receipt(rid)
                    for lr in line_rows:
                        desc = (lr.get("description") or lr.get("ocr_text") or "").strip()
                        if not desc:
                            continue
                        abstract = normalize_abstract_name(desc)
                        if not abstract:
                            continue
                        aid = upsert_abstract_item(abstract)
                        # Parse size/unit heuristically
                        size_v, size_u = parse_size(desc)
                        vid = upsert_item_variant(aid, name=desc, brand=None, size_value=size_v, size_unit=size_u)
                        price_amt = None
                        try:
                            price_amt = float(lr.get("amount") or 0)
                        except Exception:
                            price_amt = None
                        if vid and mid and (price_amt is not None) and price_amt > 0:
                            unit_price = None
                            if size_v and size_v > 0 and size_u:
                                try:
                                    unit_price = price_amt / float(size_v)
                                except Exception:
                                    unit_price = None
                            insert_price_capture(vid, mid, price_amt, captured_at or int(datetime.now().timestamp()), receipt_id=rid, line_item_id=int(lr.get("id") or 0), unit_price=unit_price, unit=size_u)
                except Exception:
                    logger.debug("Auto-seed items/variants/prices failed for receipt id=%s", rid)
            except Exception:
                logger.exception("Failed parsing/saving items for receipt id=%s", rid)
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


@app.get("/docs/flow", response_class=HTMLResponse)
async def docs_flow(request: Request):
    """Render a page that explains the end-to-end data flow and key functions."""
    return templates.TemplateResponse("flow.html", {"request": request})

