from __future__ import annotations
from typing import Dict, List, Any, Tuple
from jinja2 import Template
from .client import LlmClient
from .types import PickResult, ExtractResult
from . import filters
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

def _read_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def _render(name: str, **ctx) -> str:
    return Template(_read_prompt(name)).render(**ctx)


def pick_index(client: LlmClient, target: str, lines: List[str], cand_idx: List[int]) -> Tuple[PickResult, dict, str, str]:
    tmpl = _render("pick_index.txt", target=target, candidates=[{"index": i, "text": lines[i]} for i in cand_idx])
    text, data = client.generate_json(tmpl, temperature=0.2, num_predict=96)
    import json
    try:
        obj = json.loads(text[text.find("{"): text.rfind("}")+1])
    except Exception:
        obj = {}
    pr = PickResult(index=int(obj.get("index", -1) or -1), confidence=float(obj.get("confidence", 0) or 0), candidates=cand_idx)
    # Fallback if model didn't choose from provided candidates
    if pr.index not in cand_idx:
        if cand_idx:
            pr.index = cand_idx[0]
            pr.confidence = max(pr.confidence, 0.6)
    return pr, data, tmpl, text


def extract_value(client: LlmClient, kind: str, idx: int, line: str) -> Tuple[ExtractResult, dict, str, str]:
    name = {
        "date": "extract_value_date.txt",
        "total": "extract_value_total.txt",
        "payee": "extract_value_payee.txt",
        "last4": "extract_value_last4.txt",
    }[kind]
    tmpl = _render(name, index=idx, text=line)
    text, data = client.generate_json(tmpl, temperature=0.2, num_predict=96)
    import json
    try:
        obj = json.loads(text[text.find("{"): text.rfind("}")+1])
    except Exception:
        obj = {}
    val = obj.get("value", "")
    conf = float(obj.get("confidence", 0) or 0)
    er = ExtractResult(index=idx, value=val, confidence=conf)
    return er, data, tmpl, text


def find_date(client: LlmClient, lines: List[str]) -> Tuple[ExtractResult, Dict[str, Any]]:
    cand = filters.top_k_date(lines)
    pr, d1, p1, r1 = pick_index(client, "date", lines, cand)
    dbg: Dict[str, Any] = {"pick": d1, "p1": p1, "r1": r1}
    if pr.index < 0:
        return ExtractResult(index=-1, value="", confidence=0.0), dbg
    # Guardrail: ensure the chosen line actually contains a date pattern; otherwise prefer a candidate that does
    try:
        chosen_txt = lines[pr.index] if 0 <= pr.index < len(lines) else ""
        if not filters.DATE_RE.search(chosen_txt or ""):
            for j in cand:
                if 0 <= j < len(lines) and filters.DATE_RE.search(lines[j] or ""):
                    dbg["override"] = {"reason": "no-date-pattern", "from": pr.index, "to": j}
                    pr.index = j
                    pr.confidence = max(pr.confidence, 0.85)
                    break
    except Exception:
        pass
    er, d2, p2, r2 = extract_value(client, "date", pr.index, lines[pr.index])
    dbg.update({"extract": d2, "p2": p2, "r2": r2})
    return er, dbg


def find_total(client: LlmClient, lines: List[str]) -> Tuple[ExtractResult, Dict[str, Any]]:
    cand = filters.top_k_total(lines)
    pr, d1, p1, r1 = pick_index(client, "total", lines, cand)
    if pr.index < 0:
        return ExtractResult(index=-1, value=0.0, confidence=0.0), {"pick": d1, "p1": p1, "r1": r1}
    er, d2, p2, r2 = extract_value(client, "total", pr.index, lines[pr.index])
    return er, {"pick": d1, "extract": d2, "p1": p1, "p2": p2, "r1": r1, "r2": r2}


def find_payee(client: LlmClient, lines: List[str]) -> Tuple[ExtractResult, Dict[str, Any]]:
    cand = filters.top_k_payee(lines)
    pr, d1, p1, r1 = pick_index(client, "payee", lines, cand)
    dbg: Dict[str, Any] = {"pick": d1, "p1": p1, "r1": r1}
    if pr.index < 0:
        return ExtractResult(index=-1, value="", confidence=0.0), dbg
    # Guardrail: downrank obviously bad payee lines and prefer brand-like candidates
    def looks_slogan(up: str) -> bool:
        return "YOUR NEIGHBORHOOD" in up
    def looks_staff(up: str) -> bool:
        return any(t in up for t in ("CASHIER", "CHECKOUT", "LANE", "REGISTER"))
    def looks_address(up: str) -> bool:
        return ("," in up) or any(tok in f" {up} " for tok in filters.ADDR_TOKENS)
    def digit_heavy(up: str) -> bool:
        return sum(ch.isdigit() for ch in up) >= 4
    def brandy(s: str) -> float:
        parts = [p for p in s.split() if p.isalpha()]
        if len(parts) in (1, 2) and all(p[:1].isupper() and p[1:].islower() for p in parts if len(p) > 1):
            return 1.0
        return 0.0
    try:
        chosen = lines[pr.index] if 0 <= pr.index < len(lines) else ""
        up = (chosen or "").strip().upper()
        # New: if this line looks like a pure money/amount line, override to a better candidate
        import re as _re
        if filters.MONEY_RE.search(chosen or "") or ("$" in chosen):
            for j in cand:
                if 0 <= j < len(lines):
                    s2 = lines[j]
                    if not (filters.MONEY_RE.search(s2 or "") or ("$" in (s2 or ""))):
                        dbg["override_amount"] = {"from": pr.index, "to": j}
                        pr.index = j
                        pr.confidence = max(pr.confidence, 0.85)
                        chosen = lines[pr.index]
                        up = (chosen or "").strip().upper()
                        break
        bad = looks_slogan(up) or looks_staff(up) or looks_address(up) or digit_heavy(up)
        if bad:
            # find best alternative among candidates by brandiness
            best = pr.index
            best_b = brandy(chosen)
            for j in cand:
                if 0 <= j < len(lines):
                    b = brandy(lines[j])
                    if b > best_b:
                        best = j; best_b = b
            if best != pr.index:
                dbg["override"] = {"reason": "bad-payee-line", "from": pr.index, "to": best}
                pr.index = best
                pr.confidence = max(pr.confidence, 0.85)
    except Exception:
        pass
    er, d2, p2, r2 = extract_value(client, "payee", pr.index, lines[pr.index])
    dbg.update({"extract": d2, "p2": p2, "r2": r2})
    return er, dbg


def find_last4(client: LlmClient, lines: List[str]) -> Tuple[ExtractResult, Dict[str, Any]]:
    cand = filters.top_k_last4(lines)
    pr, d1, p1, r1 = pick_index(client, "last4", lines, cand)
    if pr.index < 0:
        return ExtractResult(index=-1, value="", confidence=0.0), {"pick": d1, "p1": p1, "r1": r1}
    er, d2, p2, r2 = extract_value(client, "last4", pr.index, lines[pr.index])
    # strict last4 enforcement
    import re
    s = str(er.value or "").strip()
    er.value = s if re.fullmatch(r"\d{4}", s) else ""
    return er, {"pick": d1, "extract": d2, "p1": p1, "p2": p2, "r1": r1, "r2": r2}
