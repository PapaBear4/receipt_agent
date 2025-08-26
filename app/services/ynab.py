from __future__ import annotations

from typing import Optional, Dict, Any
import hashlib
import requests

from app.config import settings


class YNABClient:
    # Minimal HTTP client for YNAB API v1 with helpers for transactions and metadata.
    def __init__(self, token: Optional[str] = None, base_url: Optional[str] = None):
        self.base = (base_url or settings.YNAB_API_BASE).rstrip("/")
        self.token = token or settings.YNAB_TOKEN or ""

    # Standard auth/content headers for YNAB requests.
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # True if a token is present and API calls can be attempted.
    def is_configured(self) -> bool:
        return bool(self.token)

    # List available budgets for the current user.
    def budgets(self) -> Dict[str, Any]:
        r = requests.get(f"{self.base}/budgets", headers=self._headers(), timeout=20)
        r.raise_for_status()
        return r.json().get("data", {})

    # List accounts in a budget (filters happen in service layer).
    def accounts(self, budget_id: str) -> Dict[str, Any]:
        r = requests.get(f"{self.base}/budgets/{budget_id}/accounts", headers=self._headers(), timeout=20)
        r.raise_for_status()
        return r.json().get("data", {})

    # List category groups and categories in a budget.
    def categories(self, budget_id: str) -> Dict[str, Any]:
        r = requests.get(f"{self.base}/budgets/{budget_id}/categories", headers=self._headers(), timeout=20)
        r.raise_for_status()
        return r.json().get("data", {})

    #List payees for a budget.
    def payees(self, budget_id: str) -> Dict[str, Any]:
        r = requests.get(f"{self.base}/budgets/{budget_id}/payees", headers=self._headers(), timeout=20)
        r.raise_for_status()
        return r.json().get("data", {})

    # Convert a float amount to YNAB milliunits (negative for outflows).
    @staticmethod
    def to_milliunits(amount: float) -> int:
        try:
            return int(round(float(amount) * -1000))
        except Exception:
            return 0

    # Build a stable import_id for idempotency: RA:YYYY-MM-DD:milli:hash6.
    @staticmethod
    def make_import_id(date: str, amount: float, stored_name: str) -> str:
        milli = YNABClient.to_milliunits(amount)
        h = hashlib.sha1(stored_name.encode("utf-8")).hexdigest()[:6]
        return f"RA:{date}:{milli}:{h}"

    # Create a single transaction in YNAB for the given budget/account.
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
