import json
import time
import requests
from typing import Dict, Optional, List, Any
import logging

from app.config import settings

from typing import Any

# Two-pass prompts
META_PROMPT = (
        """
You are a receipt extraction assistant. Extract only HEADER fields and do not list items.
Return ONLY valid JSON in this exact shape:
{
    "date": "",
    "payee": "",
    "total": 0.0,
    "payment": {
        "subtotal": 0.0,
        "tax": 0.0,
        "tip": 0.0,
        "discounts": 0.0,
        "fees": 0.0,
        "method": "",
        "last4": ""
    }
}
Guidance:
- Identify merchant/store name (payee).
- Parse purchase date (prefer MM/DD/YYYY; best-effort otherwise).
- Determine final total charged; fill payment breakouts if present.
- Payment method and last 4 digits of the card when available.
Return only JSON.
"""
)

ITEMS_PROMPT_PREAMBLE = (
        """
You are extracting ITEM LINES from a receipt using OCR LINES (indexed).
Use this context to avoid misclassifying totals as items:
PAYEE: {payee}
DATE: {date}
TOTAL_HINT: {total}
PAYMENT: method={method} last4={last4}

Return ONLY valid JSON in this exact shape:
{
    "items": [
        {"line_index": -1, "ocr_text": "", "name": "", "qty": 0.0, "unit_price": 0.0, "amount": 0.0, "confidence": 0.0}
    ]
}

Rules for items:
- Map each item to the appropriate line_index from OCR LINES; set ocr_text to the related text (you may combine adjacent lines when an item spans multiple lines).
- Merge multi-line items where quantity/weight/unit-price are on separate lines (common for produce and weighed goods).
- Handle multi-quantity patterns (e.g., "x3", "3 @ 0.99", or "3 for 2.99"). Compute qty, unit_price, and amount accordingly.
- For multi-line items return only the first line_index.
- Ignore non-item lines like headers, SUBTOTAL, TAX, TOTAL, and payment details.
- Provide your guess confidence for each item. Confidence is a sliding range from 0.0–1.0.
"""
)


def _http_post(url: str, payload: dict, timeout: int) -> requests.Response:
    """Wrapper for POST with small retry on connection errors."""
    try:
        return requests.post(url, json=payload, timeout=timeout)
    except requests.ConnectionError as e:
        # brief retry once
        time.sleep(0.3)
        return requests.post(url, json=payload, timeout=timeout)

def ollama_health() -> Dict[str, object]:
    """Check Ollama endpoint and model availability.

    Returns a dict with keys: ok, endpoint, model, endpoint_ok, model_ok, error(optional)
    """
    info: Dict[str, object] = {
        "ok": False,
        "endpoint": settings.OLLAMA_ENDPOINT,
        "model": settings.OLLAMA_MODEL,
        "endpoint_ok": False,
        "model_ok": False,
    }
    try:
        # Basic endpoint reachability via /api/tags (list models)
        url = f"{settings.OLLAMA_ENDPOINT}/api/tags"
        r = requests.get(url, timeout=min(settings.OLLAMA_TIMEOUT, 10))
        r.raise_for_status()
        info["endpoint_ok"] = True
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        # tags returns models as {models:[{name:..}, ...]}
        names = {m.get("name") for m in (data.get("models") or [])}
        info["available_models"] = sorted([n for n in names if n])
        if settings.OLLAMA_MODEL in names:
            info["model_ok"] = True
        info["ok"] = bool(info["endpoint_ok"] and info["model_ok"])
    except Exception as e:
        info["error"] = str(e)
    return info

def extract_fields_from_text(ocr_text: str, ocr_lines: Optional[List[str]] = None) -> Dict:
    """Two-pass LLM extraction: pass 1 (meta), pass 2 (items), then combine.

    On any failure, returns empty defaults so the UI can allow manual entry.
    """
    try:
        # Prep and cleanup
        tprep0 = time.perf_counter()
        raw = (ocr_text or "")
        raw_no_cr = raw.replace("\r", "")
        raw_lines_list = raw_no_cr.split("\n")
        ocr_clean = "\n".join([" ".join(part.split()) for part in raw_lines_list]).strip()
        cleaned_text_lines_list = ocr_clean.split("\n") if ocr_clean else []

        def _looks_useful(line: str) -> bool:
            if not line:
                return False
            s = str(line).strip()
            if len(s) <= 1:
                return False
            return any(c.isdigit() for c in s) or any(ch in s for ch in "$€£%") or len(s) >= 4

        ocr_lines_clean = [ln for ln in (ocr_lines or []) if _looks_useful(ln)]

        max_chars = int(getattr(settings, "LLM_MAX_CHARS", 0))
        if max_chars and max_chars > 0 and len(ocr_clean) > max_chars:
            ocr_snippet = ocr_clean[:max_chars]
        else:
            ocr_snippet = ocr_clean

        used_model = settings.OLLAMA_MODEL
        max_lines = int(getattr(settings, "LLM_MAX_LINES", 250))
        include_full_text = bool(getattr(settings, "LLM_INCLUDE_FULL_TEXT", True))
        only_indexed_when_long = bool(getattr(settings, "LLM_ONLY_INDEXED_WHEN_LONG", True))

        # Build OCR lines preview block
        lines_block = ""
        included_lines_count = 0
        if ocr_lines_clean:
            preview_lines = [f"{i}: {str(line)}" for i, line in enumerate(ocr_lines_clean)]
            if max_lines and max_lines > 0 and len(preview_lines) > max_lines:
                preview_lines = preview_lines[:max_lines]
            included_lines_count = len(preview_lines)
            lines_block = "\nOCR LINES (indexed):\n" + "\n".join(preview_lines)

        long_input = (max_chars and len(ocr_clean) >= max_chars) or (max_lines and len(ocr_lines_clean) >= max_lines)

        # Pass 1: META
        prompt_parts_meta: List[str] = [META_PROMPT]
        if include_full_text and not (only_indexed_when_long and long_input):
            prompt_parts_meta.append("\nOCR TEXT:\n" + (ocr_snippet or ""))
        if lines_block:
            prompt_parts_meta.append(lines_block)
        prompt_meta = "\n\n".join([p for p in prompt_parts_meta if p]) + "\n"
        meta_chars = len(prompt_meta)
        meta_lines = prompt_meta.count("\n") + 1
        prep_sec = max(0.0, time.perf_counter() - tprep0)

        url = f"{settings.OLLAMA_ENDPOINT}/api/generate"
        t0 = time.perf_counter()
        r1 = _http_post(url, {"model": used_model, "prompt": prompt_meta, "stream": False, "format": "json"}, settings.OLLAMA_TIMEOUT)
        r1.raise_for_status()
        d1 = r1.json()
        t1 = time.perf_counter()
        text_meta = (d1.get("response", "") or "").strip()
        # Robust JSON extract
        try:
            s = text_meta.find("{")
            e = text_meta.rfind("}")
            meta_obj = json.loads(text_meta[s:e+1] if s != -1 and e != -1 else text_meta)
        except Exception:
            meta_obj = {}

        def _f(x) -> float:
            try:
                return float(x)
            except Exception:
                return 0.0

        date = str(meta_obj.get("date", "") or "").strip()
        payee = str(meta_obj.get("payee", "") or "").strip()
        total = _f(meta_obj.get("total", 0))
        psrc = meta_obj.get("payment", {}) or {}
        payment = {
            "subtotal": _f(psrc.get("subtotal", 0)),
            "tax": _f(psrc.get("tax", 0)),
            "tip": _f(psrc.get("tip", 0)),
            "discounts": _f(psrc.get("discounts", 0)),
            "fees": _f(psrc.get("fees", 0)),
            "method": str(psrc.get("method", "") or ""),
            "last4": str(psrc.get("last4", "") or ""),
        }

        # Pass 2: ITEMS
        # Build items preamble manually to avoid str.format brace conflicts with JSON example
        items_preamble = (
            "You are extracting ITEM LINES from a receipt using OCR LINES (indexed).\n"
            "Use this context to avoid misclassifying totals as items:\n"
            f"PAYEE: {payee}\n"
            f"DATE: {date}\n"
            f"TOTAL_HINT: {total}\n"
            f"PAYMENT: method={payment.get('method','')} last4={payment.get('last4','')}\n\n"
            "Return ONLY valid JSON in this exact shape:\n"
            "{\n"
            '    "items": [\n'
            '        {"line_index": -1, "ocr_text": "", "name": "", "qty": 0.0, "unit_price": 0.0, "amount": 0.0, "confidence": 0.0}\n'
            "    ]\n"
            "}\n\n"
            "Rules for items:\n"
            "- Map each item to the appropriate line_index from OCR LINES; set ocr_text to the related text (you may combine adjacent lines when an item spans multiple lines).\n"
            "- Merge multi-line items where quantity/weight/unit-price are on separate lines (common for produce and weighed goods).\n"
            '- Handle multi-quantity patterns (e.g., "x3", "3 @ 0.99", or "3 for 2.99"). Compute qty, unit_price, and amount accordingly.\n'
            "- For multi-line items return only the first line_index.\n"
            "- Ignore non-item lines like headers, SUBTOTAL, TAX, TOTAL, and payment details.\n"
            "- Provide your guess confidence for each item. Confidence is a sliding range from 0.0–1.0.\n"
        )
        prompt_parts_items: List[str] = [items_preamble]
        if include_full_text and not (only_indexed_when_long and long_input):
            prompt_parts_items.append("\nOCR TEXT:\n" + (ocr_snippet or ""))
        if lines_block:
            prompt_parts_items.append(lines_block)
        prompt_items = "\n\n".join([p for p in prompt_parts_items if p]) + "\n"

        t2 = time.perf_counter()
        r2 = _http_post(url, {"model": used_model, "prompt": prompt_items, "stream": False, "format": "json"}, settings.OLLAMA_TIMEOUT)
        r2.raise_for_status()
        d2 = r2.json()
        t3 = time.perf_counter()
        text_items = (d2.get("response", "") or "").strip()
        try:
            s2 = text_items.find("{")
            e2 = text_items.rfind("}")
            items_obj = json.loads(text_items[s2:e2+1] if s2 != -1 and e2 != -1 else text_items)
        except Exception:
            items_obj = {}
        items_raw = items_obj.get("items") or []
        items: List[Dict[str, Any]] = []
        if isinstance(items_raw, list):
            for it in items_raw:
                if not isinstance(it, dict):
                    continue
                items.append({
                    "line_index": int(it.get("line_index", -1) or -1),
                    "ocr_text": str(it.get("ocr_text", "") or ""),
                    "name": str(it.get("name", "") or ""),
                    "qty": float(it.get("qty", 0.0) or 0.0),
                    "unit_price": float(it.get("unit_price", 0.0) or 0.0),
                    "amount": float(it.get("amount", 0.0) or 0.0),
                    "confidence": float(it.get("confidence", 0.0) or 0.0),
                })

        # Metrics (aggregate light-weight)
        def _dur(ns):
            try:
                return (float(ns)/1e9) if ns else None
            except Exception:
                return None

        metrics = {
            "prompt": {
                "chars": len(prompt_items) + meta_chars,
                "lines": meta_lines,
                "include_full_text": bool(include_full_text and not (only_indexed_when_long and long_input)),
                "long_input": bool(long_input),
                "indexed_lines": included_lines_count,
                "max_chars": max_chars,
                "max_lines": max_lines,
                "cleanup": {
                    "raw_chars": len(raw_no_cr),
                    "raw_lines": len(raw_lines_list),
                    "cleaned_chars": len(ocr_clean),
                    "cleaned_lines": len(cleaned_text_lines_list),
                    "provided_lines": len(ocr_lines or []),
                    "provided_lines_cleaned": len(ocr_lines_clean),
                    "full_text_sent_chars": (len(ocr_snippet) if (include_full_text and not (only_indexed_when_long and long_input)) else 0),
                },
            },
            "response": {
                "prompt_tokens": d1.get("prompt_eval_count"),
                "completion_tokens": d1.get("eval_count"),
                "items": len(items),
            },
            "timing": {
                "prep_sec": max(0.0, time.perf_counter() - tprep0),
                "meta_wall_sec": max(0.0, t1 - t0),
                "items_wall_sec": max(0.0, t3 - t2),
                "meta_total_sec": _dur(d1.get("total_duration")),
                "meta_prompt_eval_sec": _dur(d1.get("prompt_eval_duration")),
                "meta_eval_sec": _dur(d1.get("eval_duration")),
                "items_total_sec": _dur(d2.get("total_duration")),
                "items_prompt_eval_sec": _dur(d2.get("prompt_eval_duration")),
                "items_eval_sec": _dur(d2.get("eval_duration")),
            },
        }

        return {"date": date, "payee": payee, "total": total, "payment": payment, "items": items, "metrics": metrics}
    except requests.exceptions.RequestException as e:
        logging.error("LLM HTTP error: %s", e)
        return {"date": "", "payee": "", "total": 0.0, "payment": {}, "items": []}
    except Exception as e:
        logging.error("LLM extraction failed: %s", e)
        return {"date": "", "payee": "", "total": 0.0, "payment": {}, "items": []}

def extract_meta_from_text(ocr_text: str, ocr_lines: Optional[List[str]] = None) -> Dict:
    """Extract only header/meta fields quickly (date, payee, total, payment).

    Returns keys: date, payee, total, payment, metrics(optional).
    """
    try:
        tprep0 = time.perf_counter()
        raw = (ocr_text or "")
        raw_no_cr = raw.replace("\r", "")
        raw_lines_list = raw_no_cr.split("\n")
        ocr_clean = "\n".join([" ".join(part.split()) for part in raw_lines_list]).strip()
        def _looks_useful(line: str) -> bool:
            if not line:
                return False
            s = str(line).strip()
            if len(s) <= 1:
                return False
            return any(c.isdigit() for c in s) or any(ch in s for ch in "$€£%") or len(s) >= 4
        ocr_lines_clean = [ln for ln in (ocr_lines or []) if _looks_useful(ln)]
        max_chars = int(getattr(settings, "LLM_MAX_CHARS", 0))
        if max_chars and max_chars > 0 and len(ocr_clean) > max_chars:
            ocr_snippet = ocr_clean[:max_chars]
        else:
            ocr_snippet = ocr_clean
        used_model = settings.OLLAMA_MODEL
        max_lines = int(getattr(settings, "LLM_MAX_LINES", 250))
        include_full_text = bool(getattr(settings, "LLM_INCLUDE_FULL_TEXT", True))
        only_indexed_when_long = bool(getattr(settings, "LLM_ONLY_INDEXED_WHEN_LONG", True))
        # Lines block
        lines_block = ""
        included_lines_count = 0
        if ocr_lines_clean:
            preview_lines = [f"{i}: {str(line)}" for i, line in enumerate(ocr_lines_clean)]
            if max_lines and max_lines > 0 and len(preview_lines) > max_lines:
                preview_lines = preview_lines[:max_lines]
            included_lines_count = len(preview_lines)
            lines_block = "\nOCR LINES (indexed):\n" + "\n".join(preview_lines)
        # Build prompt
        long_input = (max_chars and len(ocr_clean) >= max_chars) or (max_lines and len(ocr_lines_clean) >= max_lines)
        prompt_parts_meta: List[str] = [META_PROMPT]
        if include_full_text and not (only_indexed_when_long and long_input):
            prompt_parts_meta.append("\nOCR TEXT:\n" + (ocr_snippet or ""))
        if lines_block:
            prompt_parts_meta.append(lines_block)
        prompt_meta = "\n\n".join([p for p in prompt_parts_meta if p]) + "\n"
        include_full_text_effective = bool(include_full_text and not (only_indexed_when_long and long_input))
        prompt_chars = len(prompt_meta)
        prompt_line_count = prompt_meta.count("\n") + 1
        prep_sec = max(0.0, time.perf_counter() - tprep0)
        # Call LLM
        url = f"{settings.OLLAMA_ENDPOINT}/api/generate"
        t0 = time.perf_counter()
        resp = _http_post(url, {"model": used_model, "prompt": prompt_meta, "stream": False, "format": "json"}, settings.OLLAMA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        t1 = time.perf_counter()
        text = (data.get("response", "") or "").strip()
        # Parse
        obj: Dict[str, Any] = {}
        try:
            s = text.find("{")
            e = text.rfind("}")
            obj = json.loads(text[s:e+1] if s != -1 and e != -1 else text)
        except Exception:
            obj = {}
        def _f(x) -> float:
            try:
                return float(x)
            except Exception:
                return 0.0
        date = str(obj.get("date", "") or "").strip()
        payee = str(obj.get("payee", "") or "").strip()
        total = _f(obj.get("total", 0))
        psrc = obj.get("payment", {}) or {}
        payment = {
            "subtotal": _f(psrc.get("subtotal", 0)),
            "tax": _f(psrc.get("tax", 0)),
            "tip": _f(psrc.get("tip", 0)),
            "discounts": _f(psrc.get("discounts", 0)),
            "fees": _f(psrc.get("fees", 0)),
            "method": str(psrc.get("method", "") or ""),
            "last4": str(psrc.get("last4", "") or ""),
        }
        # Metrics (minimal)
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        total_ns = data.get("total_duration") or 0
        eval_ns = data.get("eval_duration") or 0
        prompt_eval_ns = data.get("prompt_eval_duration") or 0
        def _dur(ns):
            try:
                return (float(ns)/1e9) if ns else None
            except Exception:
                return None
        metrics = {
            "prompt": {
                "chars": prompt_chars,
                "lines": prompt_line_count,
                "include_full_text": include_full_text_effective,
                "indexed_lines": included_lines_count,
            },
            "response": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            "timing": {
                "wall_sec": max(0.0, t1 - t0),
                "total_sec": _dur(total_ns),
                "prompt_eval_sec": _dur(prompt_eval_ns),
                "eval_sec": _dur(eval_ns),
            },
        }
        return {"date": date, "payee": payee, "total": total, "payment": payment, "metrics": metrics}
    except Exception as e:
        logging.error("LLM meta extraction failed: %s", e)
        return {"date": "", "payee": "", "total": 0.0, "payment": {}}
