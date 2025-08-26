from __future__ import annotations

from typing import Any, Dict, Optional
import json
import logging

from fastapi import Request

from app.config import settings
from app.services.ynab import YNABClient

logger = logging.getLogger(__name__)

USER_CONF_PATH = settings.DATA_DIR / "user_config.json"


# Load persisted user config (selected budget id, etc.) from data dir.
def _load_user_conf() -> dict:
    try:
        if USER_CONF_PATH.exists():
            return json.loads(USER_CONF_PATH.read_text())
    except Exception:
        pass
    return {}


# Persist user config JSON to data dir.
def _save_user_conf(d: dict) -> None:
    try:
        USER_CONF_PATH.write_text(json.dumps(d, indent=2))
    except Exception:
        pass


# Build template context for YNAB settings page (budgets/accounts/categories/payees/errors).
def get_settings_context(request: Request) -> Dict[str, Any]:
    """Build context for the YNAB settings page by calling YNAB APIs.
    Returns only the template context; the route will render.
    """
    client = YNABClient()
    configured = client.is_configured()
    budgets: list = []
    accounts: list = []
    category_groups: list = []
    payees: list = []
    error: Optional[str] = None

    selected_budget_id = (
        _load_user_conf().get("ynab_budget_id")
        or settings.YNAB_BUDGET_ID
        or ""
    ).strip()

    try:
        if configured:
            try:
                data = client.budgets()
                budgets = data.get("budgets", []) if isinstance(data, dict) else []
            except Exception as e:
                error = f"Failed to load budgets: {e}"
            if not selected_budget_id and budgets:
                try:
                    selected_budget_id = str(budgets[0].get("id") or "")
                except Exception:
                    selected_budget_id = ""
            if selected_budget_id:
                try:
                    ad = client.accounts(selected_budget_id)
                    raw_accounts = ad.get("accounts", []) if isinstance(ad, dict) else []
                    accounts = [a for a in raw_accounts if not a.get("closed")]
                except Exception:
                    pass
                try:
                    cd = client.categories(selected_budget_id)
                    raw_groups = cd.get("category_groups", []) if isinstance(cd, dict) else []
                    for g in raw_groups:
                        cats = [
                            c
                            for c in (g.get("categories", []) or [])
                            if not c.get("hidden") and not c.get("deleted")
                        ]
                        if cats:
                            category_groups.append({"name": g.get("name"), "categories": cats})
                except Exception:
                    pass
                try:
                    pd = client.payees(selected_budget_id)
                    payees = pd.get("payees", []) if isinstance(pd, dict) else []
                except Exception:
                    pass
    except Exception as e:
        error = str(e)

    return {
        "request": request,
        "configured": configured,
        "budgets": budgets,
        "selected_budget_id": selected_budget_id,
        "accounts": accounts,
        "category_groups": category_groups,
        "payees": payees,
        "error": error,
    }


# Build template context for YNAB metadata page (budgets/accounts/categories) with error fields.
def get_meta_context(request: Request) -> Dict[str, Any]:
    """Build context for the YNAB metadata page by calling YNAB APIs."""
    client = YNABClient()
    configured = client.is_configured()
    budgets: list = []
    accounts: list = []
    category_groups: list = []
    error_budgets: Optional[str] = None
    error_accounts: Optional[str] = None
    error_categories: Optional[str] = None
    active_budget_id = (
        _load_user_conf().get("ynab_budget_id")
        or settings.YNAB_BUDGET_ID
        or ""
    ).strip()
    default_account_id = settings.YNAB_DEFAULT_ACCOUNT_ID or ""

    if configured:
        try:
            bd = client.budgets()
            budgets = bd.get("budgets", []) if isinstance(bd, dict) else []
        except Exception as e:
            error_budgets = str(e)
        if not active_budget_id and budgets:
            try:
                active_budget_id = str(budgets[0].get("id") or "")
            except Exception:
                active_budget_id = ""
        if active_budget_id:
            try:
                ad = client.accounts(active_budget_id)
                raw_accounts = ad.get("accounts", []) if isinstance(ad, dict) else []
                accounts = [a for a in raw_accounts if not a.get("closed")]
            except Exception as e:
                error_accounts = str(e)
            try:
                cd = client.categories(active_budget_id)
                raw_groups = cd.get("category_groups", []) if isinstance(cd, dict) else []
                for g in raw_groups:
                    cats = [
                        c
                        for c in (g.get("categories", []) or [])
                        if not c.get("hidden") and not c.get("deleted")
                    ]
                    if cats:
                        category_groups.append({"name": g.get("name"), "categories": cats})
            except Exception as e:
                error_categories = str(e)

    return {
        "request": request,
        "configured": configured,
        "budgets": budgets,
        "accounts": accounts,
        "category_groups": category_groups,
        "active_budget_id": active_budget_id,
        "default_account_id": default_account_id,
        "error_budgets": error_budgets,
        "error_accounts": error_accounts,
        "error_categories": error_categories,
    }


# Save the user's selected YNAB budget id for later API calls.
def set_selected_budget_id(budget_id: str) -> None:
    conf = _load_user_conf()
    conf["ynab_budget_id"] = (budget_id or "").strip()
    _save_user_conf(conf)


# Push a single transaction to YNAB using current settings; returns result or error.
def push_transaction(
    *,
    stored_filename: str,
    date: str,
    payee: str,
    outflow: str,
    memo: str,
    account_id: str,
    category_id: str,
) -> Dict[str, Any]:
    """Push a single transaction to YNAB. Returns a JSON-serializable dict."""
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
