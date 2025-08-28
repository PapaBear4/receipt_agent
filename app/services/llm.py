import json
import time
import requests
from typing import Dict, Optional, List, Any, Tuple
import logging
from pathlib import Path

from app.config import settings

from pydantic import BaseModel, Field, ValidationError
from app.services.enrichment import get_enricher

# ---- Models ----

class Payment(BaseModel):
        subtotal: float = 0.0
        tax: float = 0.0
        tip: float = 0.0
        discounts: float = 0.0
        fees: float = 0.0
        method: str = ""
        last4: str = ""


class Meta(BaseModel):
        date: str = ""
        payee: str = ""
        total: float = 0.0
        payment: Payment = Field(default_factory=Payment)


class Item(BaseModel):
        line_index: int = -1
        ocr_text: str = ""
        name: str = ""
        qty: float = 0.0
        unit_price: float = 0.0
        amount: float = 0.0
        confidence: float = 0.0


class PipelineContext(BaseModel):
    raw_text: str = ""
    ocr_lines: List[str] = Field(default_factory=list)
    meta: Meta = Field(default_factory=Meta)
    items: List[Item] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    debug: Dict[str, Any] = Field(default_factory=dict)


# ---- HTTP helpers ----

# HTTP POST wrapper with a single quick retry on connection errors.
def _http_post(url: str, payload: dict, timeout: int) -> requests.Response:
    """Wrapper for POST with small retry on connection errors."""
    try:
        return requests.post(url, json=payload, timeout=timeout)
    except requests.ConnectionError as e:
        # brief retry once
        time.sleep(0.3)
        return requests.post(url, json=payload, timeout=timeout)

# Prompt file loader (file-only; no inline defaults)
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

def _read_prompt(filename: str) -> str:
    """Read a prompt file; raise if missing or unreadable (file-only prompts)."""
    p = _PROMPTS_DIR / filename
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    try:
        return p.read_text()
    except Exception as e:
        raise IOError(f"Failed to read prompt file {p}: {e}")

# Extract JSON object from a possibly-noisy text response.
def _parse_json_object(text: str) -> dict:
    s = text.find("{")
    e = text.rfind("}")
    snippet = text[s:e+1] if (s != -1 and e != -1 and e > s) else text
    try:
        return json.loads(snippet)
    except Exception:
        return {}

# Basic cleanup of OCR lines to drop noise.
def _filter_useful_lines(lines: Optional[List[str]]) -> List[str]:
    def _looks_useful(line: str) -> bool:
        if not line:
            return False
        s = str(line).strip()
        if len(s) <= 1:
            return False
        return any(c.isdigit() for c in s) or any(ch in s for ch in "$€£%") or len(s) >= 4

    return [ln for ln in (lines or []) if _looks_useful(ln)]


# ---- LLM client wrapper ----

class LlmClient:
    def __init__(self, model: str, endpoint: str, timeout: int) -> None:
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def generate_json(self, prompt: str) -> Tuple[str, dict]:
        url = f"{self.endpoint}/api/generate"
        t0 = time.perf_counter()
        resp = _http_post(url, {"model": self.model, "prompt": prompt, "stream": False, "format": "json"}, self.timeout)
        resp.raise_for_status()
        data = resp.json()
        t1 = time.perf_counter()
        # attach wall clock
        data.setdefault("_client_wall_sec", max(0.0, t1 - t0))
        return (str(data.get("response", "") or "").strip(), data)


# ---- Experts ----

class BaseExpert:
    name = "base"

    def run(self, client: LlmClient, ctx: PipelineContext, lines_block: str) -> Tuple[PipelineContext, dict]:
        raise NotImplementedError


class NormalizeExpert(BaseExpert):
    name = "normalize"

    def run(self, client: LlmClient, ctx: PipelineContext, lines_block: str) -> Tuple[PipelineContext, dict]:
        """Ask the LLM to normalize indexed OCR lines (fix common OCR errors, strip junk).

        Expects a JSON like {"lines":[{"index":0,"text":"..."}, ...]} and updates ctx.ocr_lines in-place.
        """
        try:
            tmpl = _read_prompt("normalize_system.txt")
        except Exception as e:
            # If prompt missing, skip normalization but capture warning
            try:
                ctx.warnings.append("normalize-prompt-missing")
                ctx.debug.setdefault("normalize", {})
                ctx.debug["normalize"]["error"] = str(e)
            except Exception:
                pass
            return ctx, {"_client_wall_sec": 0.0}

        prompt = f"{tmpl}\n\n{lines_block}\n"
        text, data = client.generate_json(prompt)
        obj = _parse_json_object(text)

        # Build a mapping of index -> text from response
        mapping = {}
        if isinstance(obj, dict):
            raw = obj.get("lines") or []
            if isinstance(raw, list):
                for it in raw:
                    if not isinstance(it, dict):
                        continue
                    try:
                        idx = int(it.get("index", -1) or -1)
                        val = str(it.get("text", "") or "")
                        mapping[idx] = val
                    except Exception:
                        continue

        # Apply mapping to current ctx.ocr_lines, preserving length/order
        try:
            normalized = list(ctx.ocr_lines)
            for k, v in mapping.items():
                if 0 <= k < len(normalized):
                    normalized[k] = v
            ctx.ocr_lines = normalized
        except Exception:
            # Keep original lines on any failure
            ctx.warnings.append("normalize-apply-failed")

        # Capture debug details
        try:
            ctx.debug.setdefault("normalize", {})
            ctx.debug["normalize"].update({
                "prompt": tmpl,
                "prompt_full": prompt,
                "response_raw": text,
                "response_json": obj,
                "output_lines": list(ctx.ocr_lines),
                "stats": {
                    "prompt_tokens": data.get("prompt_eval_count"),
                    "completion_tokens": data.get("eval_count"),
                },
            })
        except Exception:
            pass

        return ctx, data


class MetaExpert(BaseExpert):
    name = "meta"

    def run(self, client: LlmClient, ctx: PipelineContext, lines_block: str) -> Tuple[PipelineContext, dict]:
        meta_tmpl = _read_prompt("meta_system.txt")
        prompt = f"{meta_tmpl}\n\n{lines_block}\n"
        text, data = client.generate_json(prompt)
        obj = _parse_json_object(text)
        try:
            ctx.meta = Meta.model_validate(obj)
        except ValidationError:
            # Keep defaults on failure
            ctx.warnings.append("meta-validate-failed")
        # Capture debug info including any non-modeled fields like 'sources'
        try:
            ctx.debug.setdefault("meta", {})
            ctx.debug["meta"].update({
                "prompt_full": prompt,
                "prompt": meta_tmpl,
                "response_raw": text,
                "response_json": obj,
                "sources": obj.get("sources") if isinstance(obj, dict) else None,
                "stats": {
                    "prompt_tokens": data.get("prompt_eval_count"),
                    "completion_tokens": data.get("eval_count"),
                },
            })
        except Exception:
            pass
        return ctx, data


class ItemsExpert(BaseExpert):
    name = "items"

    def run(self, client: LlmClient, ctx: PipelineContext, lines_block: str) -> Tuple[PipelineContext, dict]:
        items_tmpl = _read_prompt("items_system.txt")
        pre = items_tmpl.format(
            payee=ctx.meta.payee,
            date=ctx.meta.date,
            total=ctx.meta.total,
            method=ctx.meta.payment.method,
            last4=ctx.meta.payment.last4,
        )
        prompt = f"{pre}\n\n{lines_block}\n"
        text, data = client.generate_json(prompt)
        obj = _parse_json_object(text)
        raw_items = (obj.get("items") or []) if isinstance(obj, dict) else []
        items: List[Item] = []
        if isinstance(raw_items, list):
            for it in raw_items:
                if not isinstance(it, dict):
                    continue
                try:
                    items.append(Item(
                        line_index=int(it.get("line_index", -1) or -1),
                        ocr_text=str(it.get("ocr_text", "") or ""),
                        name=str(it.get("name", "") or ""),
                        qty=float(it.get("qty", 0.0) or 0.0),
                        unit_price=float(it.get("unit_price", 0.0) or 0.0),
                        amount=float(it.get("amount", 0.0) or 0.0),
                        confidence=float(it.get("confidence", 0.0) or 0.0),
                    ))
                except Exception:
                    continue
        ctx.items = items

        # Optional enrichment pass (non-blocking if disabled)
        try:
            enricher = get_enricher()
            enriched = []
            vendor = (ctx.meta.payee or "").strip()
            for it in items[:10]:  # cap quick pass
                q = (it.name or it.ocr_text).strip()
                if not q:
                    continue
                ei = enricher.enrich(vendor, q)
                if ei:
                    enriched.append({
                        "line_index": it.line_index,
                        "query": q,
                        "result": ei.__dict__,
                    })
            if enriched:
                ctx.debug.setdefault("items", {})
                ctx.debug["items"]["enrichment"] = enriched
        except Exception:
            # Keep enrichment best-effort
            pass
        # Capture debug including any per-item 'sources'
        try:
            ctx.debug.setdefault("items", {})
            ctx.debug["items"].update({
                "prompt": items_tmpl,
                "prompt_filled": pre,
                "prompt_full": prompt,
                "response_raw": text,
                "response_json": obj,
                "stats": {
                    "prompt_tokens": data.get("prompt_eval_count"),
                    "completion_tokens": data.get("eval_count"),
                },
            })
        except Exception:
            pass
        return ctx, data


class ReconcileExpert(BaseExpert):
    name = "reconcile"

    def run(self, client: LlmClient, ctx: PipelineContext, lines_block: str) -> Tuple[PipelineContext, dict]:
        # No LLM call; deterministic checks
        sum_items = float(sum(max(0.0, float(i.amount or 0.0)) for i in ctx.items)) if ctx.items else 0.0
        # If subtotal is missing but items exist, set subtotal
        if ctx.items and (ctx.meta.payment.subtotal or 0.0) <= 0.0:
            ctx.meta.payment.subtotal = round(sum_items, 2)
        # Compute implied total from parts
        implied_total = round((ctx.meta.payment.subtotal or 0.0)
                              + (ctx.meta.payment.tax or 0.0)
                              + (ctx.meta.payment.tip or 0.0)
                              + (ctx.meta.payment.fees or 0.0)
                              - (ctx.meta.payment.discounts or 0.0), 2)
        # If header total missing but implied exists, fill it
        if (ctx.meta.total or 0.0) <= 0.0 and implied_total > 0.0:
            ctx.meta.total = implied_total
        # If header total present and implied diff is significant, warn
        try:
            if ctx.meta.total and abs(implied_total - ctx.meta.total) > 0.01 and (ctx.meta.payment.subtotal or 0.0) > 0.0:
                ctx.warnings.append("reconcile-mismatch")
        except Exception:
            pass
        return ctx, {"_client_wall_sec": 0.0}


# ---- Health ----
# Probe Ollama endpoint/model availability and return a status summary dict.
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

# Expert pipeline using Ollama: meta -> items -> reconcile
def extract_fields_from_text(ocr_text: str, ocr_lines: Optional[List[str]] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict:
    """Sequential experts pipeline: Meta -> Items -> Reconcile.

    Returns fields dict compatible with previous API and includes metrics.
    """
    try:
        # Prep and cleanup
        tprep0 = time.perf_counter()
        raw = (ocr_text or "")
        raw_no_cr = raw.replace("\r", "")
        raw_lines_list = raw_no_cr.split("\n")
        ocr_clean = "\n".join([" ".join(part.split()) for part in raw_lines_list]).strip()
        cleaned_text_lines_list = ocr_clean.split("\n") if ocr_clean else []

        def cfg(name: str, default: Any = None) -> Any:
            if overrides is not None and name in overrides:
                return overrides[name]
            return getattr(settings, name, default)

        used_model = str(cfg("OLLAMA_MODEL", settings.OLLAMA_MODEL))
        timeout = int(cfg("OLLAMA_TIMEOUT", settings.OLLAMA_TIMEOUT))
        max_lines = int(cfg("LLM_MAX_LINES", 250))

        # Line block (raw cleaned lines)
        ocr_lines_clean = _filter_useful_lines(ocr_lines)
        preview_lines = [f"{i}: {str(line)}" for i, line in enumerate(ocr_lines_clean)]
        if max_lines and max_lines > 0 and len(preview_lines) > max_lines:
            preview_lines = preview_lines[:max_lines]
        included_lines_count = len(preview_lines)
        lines_block = ("\nOCR LINES (indexed):\n" + "\n".join(preview_lines)) if preview_lines else ""

        ctx = PipelineContext(raw_text=raw_no_cr, ocr_lines=ocr_lines_clean)
        client = LlmClient(model=used_model, endpoint=settings.OLLAMA_ENDPOINT, timeout=timeout)
        # Record shared debug inputs
        try:
            ctx.debug.update({
                "model": used_model,
                "endpoint": settings.OLLAMA_ENDPOINT,
                "lines_block_original": lines_block,
                "ocr_lines": ocr_lines_clean,
            })
        except Exception:
            pass

        # Run experts sequentially
        normalize_expert = NormalizeExpert()
        meta_expert = MetaExpert()
        items_expert = ItemsExpert()
        recon_expert = ReconcileExpert()

        # 1) Normalize lines with LLM
        ctx, d0 = normalize_expert.run(client, ctx, lines_block)

        # Recompute lines_block from normalized lines for downstream experts
        preview_lines = [f"{i}: {str(line)}" for i, line in enumerate(ctx.ocr_lines)]
        if max_lines and max_lines > 0 and len(preview_lines) > max_lines:
            preview_lines = preview_lines[:max_lines]
        included_lines_count = len(preview_lines)
        lines_block = ("\nOCR LINES (indexed):\n" + "\n".join(preview_lines)) if preview_lines else ""
        try:
            ctx.debug["lines_block"] = lines_block
        except Exception:
            pass

        # 2) Extract meta, then 3) items, then 4) reconcile
        ctx, d1 = meta_expert.run(client, ctx, lines_block)
        ctx, d2 = items_expert.run(client, ctx, lines_block)
        ctx, _ = recon_expert.run(client, ctx, lines_block)

        # Build outputs
        payment_dict = ctx.meta.payment.model_dump()
        items_out = [i.model_dump() for i in ctx.items]

        # Metrics
        def _dur(ns):
            try:
                return (float(ns)/1e9) if ns else None
            except Exception:
                return None

        # Aggregate metrics across all LLM calls (normalize + meta + items)
        def gi(d, k):
            try:
                return d.get(k)
            except Exception:
                return None

        prompt_tokens = int((gi(d0, "prompt_eval_count") or 0)) + int((gi(d1, "prompt_eval_count") or 0)) + int((gi(d2, "prompt_eval_count") or 0))
        completion_tokens = int((gi(d0, "eval_count") or 0)) + int((gi(d1, "eval_count") or 0)) + int((gi(d2, "eval_count") or 0))
        total_sec = (
            ( _dur(gi(d0, "total_duration")) or 0.0 ) +
            ( _dur(gi(d1, "total_duration")) or 0.0 ) +
            ( _dur(gi(d2, "total_duration")) or 0.0 )
        )
        prompt_eval_sec = (
            ( _dur(gi(d0, "prompt_eval_duration")) or 0.0 ) +
            ( _dur(gi(d1, "prompt_eval_duration")) or 0.0 ) +
            ( _dur(gi(d2, "prompt_eval_duration")) or 0.0 )
        )
        eval_sec = (
            ( _dur(gi(d0, "eval_duration")) or 0.0 ) +
            ( _dur(gi(d1, "eval_duration")) or 0.0 ) +
            ( _dur(gi(d2, "eval_duration")) or 0.0 )
        )
        wall_sec = max(0.0,
            max(gi(d0, "_client_wall_sec") or 0.0, 0.0)
            + max(gi(d1, "_client_wall_sec") or 0.0, 0.0)
            + max(gi(d2, "_client_wall_sec") or 0.0, 0.0)
        )

        # Prompt sizes from captured full prompts
        norm_pf = str(((ctx.debug.get("normalize") or {}).get("prompt_full") or ""))
        meta_pf = str(((ctx.debug.get("meta") or {}).get("prompt_full") or ""))
        items_pf = str(((ctx.debug.get("items") or {}).get("prompt_full") or ""))
        prompt_chars = (len(norm_pf) + len(meta_pf) + len(items_pf)) if (norm_pf or meta_pf or items_pf) else None
        prompt_lines = None
        try:
            pl0 = norm_pf.count("\n") + (1 if norm_pf else 0)
            pl1 = meta_pf.count("\n") + (1 if meta_pf else 0)
            pl2 = items_pf.count("\n") + (1 if items_pf else 0)
            prompt_lines = pl0 + pl1 + pl2 if (pl0 or pl1 or pl2) else None
        except Exception:
            prompt_lines = None

        # Throughputs
        tps_completion = (float(completion_tokens) / float(eval_sec)) if eval_sec and eval_sec > 0 else None
        tps_end_to_end = (float(completion_tokens) / float(wall_sec)) if wall_sec and wall_sec > 0 else None

        metrics = {
            "prompt": {
                "chars": prompt_chars,
                "lines": prompt_lines,
                "include_full_text": False,
                "long_input": bool(max_lines and len(ocr_lines_clean) >= max_lines),
                "indexed_lines": included_lines_count,
                "max_lines": max_lines,
                "max_chars": 0,
                "cleanup": {
                    "raw_chars": len(raw_no_cr),
                    "raw_lines": len(raw_lines_list),
                    "cleaned_chars": len(ocr_clean),
                    "cleaned_lines": len(cleaned_text_lines_list),
                    "provided_lines": len(ocr_lines or []),
                    "provided_lines_cleaned": len(ocr_lines_clean),
                    "full_text_sent_chars": 0,
                },
            },
            "response": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "items": len(items_out),
            },
            "timing": {
                "prep_sec": max(0.0, time.perf_counter() - tprep0),
                "wall_sec": wall_sec,
                "total_sec": total_sec,
                "prompt_eval_sec": prompt_eval_sec,
                "eval_sec": eval_sec,
                "tps_completion": tps_completion,
                "tps_end_to_end": tps_end_to_end,
            },
        }

        out = {
            "date": ctx.meta.date,
            "payee": ctx.meta.payee,
            "total": ctx.meta.total,
            "payment": payment_dict,
            "items": items_out,
            "metrics": metrics,
        }
        # Include debug when DEBUG is enabled
        try:
            if getattr(settings, "DEBUG", False):
                out["debug"] = ctx.debug
        except Exception:
            pass
        return out
    except requests.exceptions.RequestException as e:
        logging.error("LLM HTTP error: %s", e)
        return {"date": "", "payee": "", "total": 0.0, "payment": {}, "items": []}
    except Exception as e:
        logging.error("LLM extraction failed: %s", e)
        return {"date": "", "payee": "", "total": 0.0, "payment": {}, "items": []}

