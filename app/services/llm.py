import json
import time
import requests
from typing import Dict, Optional, Tuple, List, Any
import logging

from app.config import settings


EXTRACTION_PROMPT = (
        """
You are a receipt extraction assistant. Extract purchase date, merchant name, total, item lines, and payment details.
Return ONLY valid JSON in this exact shape (omit nothing; use defaults when unknown):
{
    "date": "",                // purchase date as MM/DD/YYYY or best-effort string
    "payee": "",               // merchant/store name
    "total": 0.0,               // final total charged
    "payment": {
        "subtotal": 0.0,
        "tax": 0.0,
        "tip": 0.0,
        "discounts": 0.0,
        "fees": 0.0,
        "method": "",            // e.g., Visa, Mastercard, Cash
        "last4": ""              // last 4 digits if present
    },
    "items": [
        {
            "line_index": -1,       // index of the OCR LINES entry this came from (or -1 if N/A)
            "ocr_text": "",         // the original OCR line text
            "name": "",             // your best interpreted item name
            "qty": 0.0,
            "unit_price": 0.0,
            "amount": 0.0,
            "confidence": 0.0       // 0.0–1.0 confidence in your interpretation
        }
    ]
}

Guidance:
- Use OCR LINES (with indices) to reference exact item rows. If a line isn’t an item, skip it.
- Prefer numeric amounts that look like prices for amount/unit_price.
- If qty is absent, use 1.0 by default.
- The sum of item amounts + tax + tip - discounts + fees should roughly match total.
- If a field is unknown, keep the default value.
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
        # Suggest a fallback model if preferred isn't available
        info["suggested_model"] = _select_model_internal(settings.OLLAMA_MODEL, info.get("available_models", []))
        info["ok"] = bool(info["endpoint_ok"] and info["model_ok"])
    except Exception as e:
        info["error"] = str(e)
    return info


from typing import Any, Iterable


def _select_model_internal(preferred: str, available_models: Any) -> str:
    try:
        names = list(available_models or [])  # type: ignore[arg-type]
    except Exception:
        names = []
    if preferred and preferred in names:
        return preferred
    # Prefer any llama3.* instruct model
    for n in names:
        s = str(n)
        if "llama3" in s and "instruct" in s:
            return s
    # Otherwise prefer any instruct model
    for n in names:
        if "instruct" in str(n):
            return str(n)
    # Fallback to first available
    return names[0] if names else preferred


def select_model(preferred: str | None = None) -> str:
    """Return a usable model name. If preferred isn't available, choose a reasonable fallback."""
    preferred = preferred or settings.OLLAMA_MODEL
    info = ollama_health()
    available = info.get("available_models", [])
    return _select_model_internal(preferred, available)


def ollama_pull(model: str, stream: bool = False) -> Dict[str, object]:
    """Request Ollama to pull a model. Returns result dict or error."""
    try:
        url = f"{settings.OLLAMA_ENDPOINT}/api/pull"
        payload = {"name": model, "stream": stream}
        r = requests.post(url, json=payload, timeout=max(settings.OLLAMA_TIMEOUT, 60))
        r.raise_for_status()
        # When stream=False, Ollama returns a final JSON object
        return {"ok": True, "result": r.json()}
    except Exception as e:
        logging.error("Failed to pull model '%s': %s", model, e)
        return {"ok": False, "error": str(e)}


def extract_fields_from_text(ocr_text: str, ocr_lines: Optional[List[str]] = None) -> Dict:
    """Call local Ollama to extract minimal fields. Returns dict with keys date, payee, total.

    On any failure, returns empty defaults so the UI can allow manual entry.
    """
    try:
        # Truncate overly long OCR text to keep prompt reasonable
        ocr_snippet = (ocr_text or "")
        max_chars = 8000
        if len(ocr_snippet) > max_chars:
            ocr_snippet = ocr_snippet[:max_chars]

        used_model = select_model(settings.OLLAMA_MODEL)
        if used_model != settings.OLLAMA_MODEL:
            logging.warning("Configured model '%s' not available; using fallback '%s'", settings.OLLAMA_MODEL, used_model)

        # Prepare OCR lines block with indices for line correlation
        lines_block = ""
        if ocr_lines:
            try:
                preview_lines = [f"{i}: {str(line)}" for i, line in enumerate(ocr_lines)]
                # Truncate lines block if very large
                max_lines = 200
                if len(preview_lines) > max_lines:
                    preview_lines = preview_lines[:max_lines]
                lines_block = "\nOCR LINES (indexed):\n" + "\n".join(preview_lines)
            except Exception:
                lines_block = ""

        payload = {
            "model": used_model,
            "prompt": f"{EXTRACTION_PROMPT}\n\nOCR TEXT:\n{ocr_snippet}{lines_block}\n",
            "stream": False,
            # Ask Ollama to constrain output to JSON if supported
            "format": "json",
        }
        url = f"{settings.OLLAMA_ENDPOINT}/api/generate"
        resp = _http_post(url, payload, settings.OLLAMA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns { 'response': '...model text...' }
        text = (data.get("response", "") or "").strip()

        # Attempt to parse JSON directly first
        obj: Dict[str, Any] = {}
        try:
            # Trim to outermost braces if extra text present
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text_to_parse = text[start : end + 1]
            else:
                text_to_parse = text
            obj = json.loads(text_to_parse)
        except Exception as e:
            logging.warning("LLM JSON parse failed: %s", e)
            obj = {}

        # Normalize keys and defaults
        def _f(x) -> float:
            try:
                return float(x)
            except Exception:
                return 0.0

        date = str(obj.get("date", "")).strip()
        payee = str(obj.get("payee", "")).strip()
        total = _f(obj.get("total", 0))
        payment_src = obj.get("payment", {}) or {}
        payment = {
            "subtotal": _f(payment_src.get("subtotal", 0)),
            "tax": _f(payment_src.get("tax", 0)),
            "tip": _f(payment_src.get("tip", 0)),
            "discounts": _f(payment_src.get("discounts", 0)),
            "fees": _f(payment_src.get("fees", 0)),
            "method": str(payment_src.get("method", "") or ""),
            "last4": str(payment_src.get("last4", "") or ""),
        }
        items_src = obj.get("items", []) or []
        items: List[Dict[str, Any]] = []
        for it in items_src:
            try:
                items.append({
                    "line_index": int(it.get("line_index", -1)),
                    "ocr_text": str(it.get("ocr_text", "") or ""),
                    "name": str(it.get("name", "") or ""),
                    "qty": _f(it.get("qty", 0)),
                    "unit_price": _f(it.get("unit_price", 0)),
                    "amount": _f(it.get("amount", 0)),
                    "confidence": max(0.0, min(1.0, _f(it.get("confidence", 0)))),
                })
            except Exception:
                continue

        return {"date": date, "payee": payee, "total": total, "payment": payment, "items": items}
    except requests.exceptions.RequestException as e:
        logging.error("LLM HTTP error: %s", e)
        return {"date": "", "payee": "", "total": 0.0}
    except Exception as e:
        logging.error("LLM extraction failed: %s", e)
        return {"date": "", "payee": "", "total": 0.0}
