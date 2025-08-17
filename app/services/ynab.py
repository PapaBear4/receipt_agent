from __future__ import annotations

from typing import Optional, Dict, Any
import hashlib
import requests

from app.config import settings


class YNABClient:
    def __init__(self, token: Optional[str] = None, base_url: Optional[str] = None):
        self.base = (base_url or settings.YNAB_API_BASE).rstrip("/")
        self.token = token or settings.YNAB_TOKEN or ""

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def is_configured(self) -> bool:
        return bool(self.token)

    def budgets(self) -> Dict[str, Any]:
        r = requests.get(f"{self.base}/budgets", headers=self._headers(), timeout=20)
        r.raise_for_status()
        return r.json().get("data", {})

    def accounts(self, budget_id: str) -> Dict[str, Any]:
        r = requests.get(f"{self.base}/budgets/{budget_id}/accounts", headers=self._headers(), timeout=20)
        r.raise_for_status()
        return r.json().get("data", {})

    def categories(self, budget_id: str) -> Dict[str, Any]:
        r = requests.get(f"{self.base}/budgets/{budget_id}/categories", headers=self._headers(), timeout=20)
        r.raise_for_status()
        return r.json().get("data", {})

    def payees(self, budget_id: str) -> Dict[str, Any]:
        r = requests.get(f"{self.base}/budgets/{budget_id}/payees", headers=self._headers(), timeout=20)
        r.raise_for_status()
        return r.json().get("data", {})

    @staticmethod
    def to_milliunits(amount: float) -> int:
        # YNAB uses milliunits; negative for outflows
        try:
            return int(round(float(amount) * -1000))
        except Exception:
            return 0

    @staticmethod
    def make_import_id(date: str, amount: float, stored_name: str) -> str:
        # Stable id: RA:YYYY-MM-DD:amount_milli:hash6
        milli = YNABClient.to_milliunits(amount)
        h = hashlib.sha1(stored_name.encode("utf-8")).hexdigest()[:6]
        return f"RA:{date}:{milli}:{h}"

    def create_transaction(
        self,
        budget_id: str,
        account_id: str,
        date: str,
        amount: float,
        payee_name: Optional[str] = None,
        memo: Optional[str] = None,
        category_id: Optional[str] = None,
        import_id: Optional[str] = None,
        cleared: str = "cleared",
        approved: bool = True,
    ) -> Dict[str, Any]:
        tx = {
            "account_id": account_id,
            "date": date,
            "amount": self.to_milliunits(amount),
            "payee_name": (payee_name or None),
            "memo": (memo or None),
            "category_id": (category_id or None),
            "cleared": cleared,
            "approved": approved,
        }
        if import_id:
            tx["import_id"] = import_id
        payload = {"transaction": tx}
        r = requests.post(f"{self.base}/budgets/{budget_id}/transactions", json=payload, headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json().get("data", {})
