from __future__ import annotations
import re
from typing import List, Tuple

DATE_RE = re.compile(r"\b(\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}|\d{4}[\-/]\d{1,2}[\-/]\d{1,2})\b")
MONEY_RE = re.compile(r"\$\s*\d{1,3}(?:[\,\s]?\d{3})*(?:\.\d{2})?\b")
LAST4_RE = re.compile(r"(\*{2,}|X{2,}|x{2,}|#){0,}\s*(\d{4})\b")

KEYS_TOTAL = ("TOTAL", "AMOUNT DUE", "BALANCE DUE", "AMT DUE", "AMT PD", "TOTAL DUE")
KEYS_SUBTOTAL = ("SUBTOTAL",)
KEYS_TAX_TIP = ("TAX", "TIP", "GRATUITY")
KEYS_PAYEE_HINT = ("STORE", "MART", "MARKET", "GROCERY", "PHARMACY")
ADDR_TOKENS = (" RD ", " ROAD ", " ST ", " STREET ", " AVE ", " AVENUE ", " BLVD ", " DRIVE ", " DR ", " HWY ", " CT ", " LN ", " PKWY ", " SUITE ", " STE ")
KEYS_CARD = ("CREDIT", "DEBIT", "CARD", "ACCOUNT", "ACCT", "VISA", "MASTERCARD", "MC", "AMEX", "DISCOVER")


def top_k_date(lines: List[str], k: int = 5) -> List[int]:
    scored: List[Tuple[int, float]] = []
    for i, s in enumerate(lines[:80]):
        score = 0.0
        if DATE_RE.search(s):
            score += 1.0
        if "date" in s.lower():
            score += 0.5
        # slightly prefer early lines
        score += max(0.0, 0.4 - (i * 0.004))
        if score > 0:
            scored.append((i, score))
    scored.sort(key=lambda t: t[1], reverse=True)
    return [i for i, _ in scored[:k]]


def top_k_total(lines: List[str], k: int = 5) -> List[int]:
    scored: List[Tuple[int, float]] = []
    for i, s in enumerate(lines):
        up = s.upper()
        score = 0.0
        if MONEY_RE.search(s):
            score += 0.6
        if any(k in up for k in KEYS_TOTAL):
            score += 0.7
        if any(k in up for k in KEYS_SUBTOTAL):
            score -= 0.6
        if any(k in up for k in KEYS_TAX_TIP):
            score -= 0.3
        score += min(0.5, i * 0.002)  # totals trend to bottom
        if score > 0:
            scored.append((i, score))
    scored.sort(key=lambda t: t[1], reverse=True)
    return [i for i, _ in scored[:k]]


def top_k_payee(lines: List[str], k: int = 5) -> List[int]:
    scored: List[Tuple[int, float]] = []
    for i, s in enumerate(lines[:40]):
        up = s.strip().upper()
        score = 0.0
        if len(up) >= 3 and not up.startswith("HTTP"):
            score += 0.3
        if any(h in up for h in KEYS_PAYEE_HINT):
            score += 0.2
        # penalize slogans / staff / lane / checkout cues
        if "YOUR NEIGHBORHOOD" in up:
            score -= 0.7
        if any(tok in up for tok in ("CASHIER", "CHECKOUT", "LANE", "REGISTER")):
            score -= 0.6
        # penalize address-like lines
        if any(tok in f" {up} " for tok in ADDR_TOKENS) or "," in up:
            score -= 0.5
        # penalize digit-heavy lines (likely store numbers, phone, etc.)
        digits = sum(ch.isdigit() for ch in up)
        if digits >= 4:
            score -= 0.3
        # slight bonus for Title Case two-word brands (e.g., Harris Teeter)
        parts = [p for p in s.split() if p.isalpha()]
        if len(parts) in (1, 2) and all(p[:1].isupper() and p[1:].islower() for p in parts if len(p) > 1):
            score += 0.3
        # prefer top portion
        score += max(0.0, 0.5 - (i * 0.02))
        if score > 0:
            scored.append((i, score))
    scored.sort(key=lambda t: t[1], reverse=True)
    return [i for i, _ in scored[:k]]


def top_k_last4(lines: List[str], k: int = 5) -> List[int]:
    scored: List[Tuple[int, float]] = []
    for i, s in enumerate(lines):
        up = s.upper()
        score = 0.0
        if LAST4_RE.search(up):
            score += 0.8
        if any(k in up for k in KEYS_CARD):
            score += 0.5
        # penalize likely non-payment IDs
        if any(tag in up for tag in ("ORDER", "INVOICE", "STORE", "TERMINAL", "AUTH", "TXN")):
            score -= 0.6
        if score > 0:
            scored.append((i, score))
    scored.sort(key=lambda t: t[1], reverse=True)
    return [i for i, _ in scored[:k]]
