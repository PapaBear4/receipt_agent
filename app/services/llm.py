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
        # Clean OCR text (whitespace and obvious boilerplate) before sizing decisions
        raw = (ocr_text or "")
        # Normalize whitespace and strip repeated separators to reduce noise
        ocr_clean = "\n".join(
            [
                " ".join(part.split())
                for part in (raw.replace("\r", "").split("\n"))
            ]
        ).strip()
        # Optional: remove ultra-short junk lines (e.g., single symbols) but keep prices and dates
        def _looks_useful(line: str) -> bool:
            if not line:
                return False
            s = line.strip()
            if len(s) <= 1:
                return False
            # Keep lines with digits or currency-like characters
            keep_chars = any(c.isdigit() for c in s) or any(ch in s for ch in "$€£%")
            return keep_chars or len(s) >= 4
        ocr_lines_clean = [ln for ln in (ocr_lines or []) if _looks_useful(str(ln))]

        # Apply configurable truncation/limits
        max_chars = int(getattr(settings, "LLM_MAX_CHARS", 0))
        if max_chars and max_chars > 0 and len(ocr_clean) > max_chars:
            ocr_snippet = ocr_clean[:max_chars]
        else:
            ocr_snippet = ocr_clean

        used_model = select_model(settings.OLLAMA_MODEL)
        if used_model != settings.OLLAMA_MODEL:
            logging.warning("Configured model '%s' not available; using fallback '%s'", settings.OLLAMA_MODEL, used_model)

        # Prepare OCR lines block with indices for line correlation
        lines_block = ""
        max_lines = int(getattr(settings, "LLM_MAX_LINES", 250))
        include_full_text = bool(getattr(settings, "LLM_INCLUDE_FULL_TEXT", True))
        only_indexed_when_long = bool(getattr(settings, "LLM_ONLY_INDEXED_WHEN_LONG", True))

        if ocr_lines_clean:
            try:
                preview_lines = [f"{i}: {str(line)}" for i, line in enumerate(ocr_lines_clean)]
                if max_lines and len(preview_lines) > max_lines:
                    preview_lines = preview_lines[:max_lines]
                lines_block = "\nOCR LINES (indexed):\n" + "\n".join(preview_lines)
            except Exception:
                lines_block = ""

        # Decide whether to include full OCR text alongside indexed lines
        prompt_parts: List[str] = [EXTRACTION_PROMPT]
        long_input = (max_chars and len(ocr_clean) >= max_chars) or (max_lines and len(ocr_lines_clean) >= max_lines)
        if include_full_text and not (only_indexed_when_long and long_input):
            prompt_parts.append("\nOCR TEXT:\n" + (ocr_snippet or ""))
        # Always include indexed lines if available (more structured)
        if lines_block:
            prompt_parts.append(lines_block)
        prompt_text = "\n\n".join([p for p in prompt_parts if p]) + "\n"
        # Prompt metrics
        try:
            included_lines_count = len(preview_lines) if ocr_lines_clean else 0  # type: ignore[name-defined]
        except Exception:
            included_lines_count = 0
        include_full_text_effective = bool(include_full_text and not (only_indexed_when_long and long_input))
        prompt_chars = len(prompt_text)
        prompt_line_count = prompt_text.count("\n") + 1

        payload = {
            "model": used_model,
            "prompt": prompt_text,
            "stream": False,
            # Ask Ollama to constrain output to JSON if supported
            "format": "json",
        }
        url = f"{settings.OLLAMA_ENDPOINT}/api/generate"
        t0 = time.perf_counter()
        resp = _http_post(url, payload, settings.OLLAMA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        t1 = time.perf_counter()
        # Ollama returns { 'response': '...model text...' }
        text = (data.get("response", "") or "").strip()
        # Extract optional token/duration metrics from Ollama
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        total_ns = data.get("total_duration") or 0
        load_ns = data.get("load_duration") or 0
        prompt_eval_ns = data.get("prompt_eval_duration") or 0
        eval_ns = data.get("eval_duration") or 0
        wall_sec = max(0.0, t1 - t0)
        total_sec = (float(total_ns) / 1e9) if total_ns else None
        eval_sec = (float(eval_ns) / 1e9) if eval_ns else None
        prompt_eval_sec = (float(prompt_eval_ns) / 1e9) if prompt_eval_ns else None

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
        # Response metrics
        response_chars = len(text)
        tps_completion = None
        tps_end_to_end = None
        try:
            if completion_tokens is not None and eval_sec and eval_sec > 0:
                tps_completion = float(completion_tokens) / float(eval_sec)
        except Exception:
            tps_completion = None
        try:
            if total_sec and total_sec > 0 and (prompt_tokens or completion_tokens):
                all_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)
                tps_end_to_end = float(all_tokens) / float(total_sec)
        except Exception:
            tps_end_to_end = None

        metrics = {
            "prompt": {
                "chars": prompt_chars,
                "lines": prompt_line_count,
                "include_full_text": include_full_text_effective,
                "indexed_lines": included_lines_count,
                "max_chars": max_chars,
                "max_lines": max_lines,
            },
            "response": {
                "chars": response_chars,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "items": len(items),
            },
            "timing": {
                "wall_sec": wall_sec,
                "total_sec": total_sec,
                "load_sec": (float(load_ns) / 1e9) if load_ns else None,
                "prompt_eval_sec": prompt_eval_sec,
                "eval_sec": eval_sec,
                "tps_completion": tps_completion,
                "tps_end_to_end": tps_end_to_end,
            },
        }

        return {"date": date, "payee": payee, "total": total, "payment": payment, "items": items, "metrics": metrics}
    except requests.exceptions.RequestException as e:
        logging.error("LLM HTTP error: %s", e)
        return {"date": "", "payee": "", "total": 0.0}
    except Exception as e:
        logging.error("LLM extraction failed: %s", e)
        return {"date": "", "payee": "", "total": 0.0}
