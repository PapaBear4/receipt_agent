import json
import requests
from typing import Dict
import logging

from app.config import settings


EXTRACTION_PROMPT = (
    """
You are a receipt extraction assistant. Extract purchase date, merchant name, and total amount from the
provided OCR text of a retail receipt. Respond with ONLY valid JSON matching this schema:
{"date": "", "payee": "", "total": 0.0}.
If a field is unknown, use empty string for date/payee or 0.
"""
)


def extract_fields_from_text(ocr_text: str) -> Dict:
    """Call local Ollama to extract minimal fields. Returns dict with keys date, payee, total.

    On any failure, returns empty defaults so the UI can allow manual entry.
    """
    try:
        # Truncate overly long OCR text to keep prompt reasonable
        ocr_snippet = (ocr_text or "")
        max_chars = 8000
        if len(ocr_snippet) > max_chars:
            ocr_snippet = ocr_snippet[:max_chars]

        payload = {
            "model": settings.OLLAMA_MODEL,
            "prompt": f"{EXTRACTION_PROMPT}\n\nOCR TEXT:\n{ocr_snippet}\n",
            "stream": False,
            # Ask Ollama to constrain output to JSON if supported
            "format": "json",
        }
        url = f"{settings.OLLAMA_ENDPOINT}/api/generate"
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns { 'response': '...model text...' }
        text = (data.get("response", "") or "").strip()

        # Attempt to parse JSON directly first
        obj = {}
        try:
            # Trim to outermost braces if extra text present
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start : end + 1]
            obj = json.loads(text)
        except Exception as e:
            logging.warning("LLM JSON parse failed: %s", e)
            obj = {}

        # Normalize keys and defaults
        date = str(obj.get("date", "")).strip()
        payee = str(obj.get("payee", "")).strip()
        total = obj.get("total", 0)
        try:
            total = float(total)
        except Exception:
            total = 0.0

        return {"date": date, "payee": payee, "total": total}
    except Exception as e:
        logging.error("LLM extraction failed: %s", e)
        return {"date": "", "payee": "", "total": 0.0}
