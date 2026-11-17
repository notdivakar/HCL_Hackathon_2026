"""
backend/api.py
Customer segmentation API (FastAPI + pandas).

Data:
  data/cleaned_telecom_data.csv  - pre-cleaned customer data (one row per customer)
  data/offer_catalouge.csv       - offer catalogue

Run:  uvicorn backend.api:app --reload
"""

from __future__ import annotations

import math
import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CUSTOMERS_CSV = DATA_DIR / "cleaned_telecom_data.csv"
OFFERS_CSV = DATA_DIR / "offer_catalouge.csv"

DORMANT_LOGIN_DAYS = 30          # no login for >= 30 days        -> Dormant
DECLINING_USAGE_PCT = -25.0      # usage dropped by >= 25%        -> Declining
LOYAL_MIN_TENURE = 12            # tenure >= 12 months            -> Loyal
LONG_CONTRACTS = {"one year", "two year"}  # committed contract  -> Loyal
PAYMENT_STATUSES = {"overdue", "due"}      # bill not yet paid   -> Payment Reminder

# --------------------------------------------------------------------------- #
# Load data (already cleaned upstream; only enforce types / sanity checks)
# --------------------------------------------------------------------------- #
CUSTOMER_REQUIRED = [
    "customerID", "tenure", "Contract", "Churn", "loyalty_tier", "plan_status",
    "days_since_login", "usage_change_pct", "status", "days_overdue",
    "amount_due", "marketing_opt_in",
]
CUSTOMER_INT = ["SeniorCitizen", "tenure", "active_services", "days_since_login",
                "days_overdue", "days_since_contact", "contacts_last_30d"]
CUSTOMER_FLOAT = ["MonthlyCharges", "TotalCharges", "usage_prev_30d", "usage_last_30d",
                  "usage_change_pct", "amount_due"]
CUSTOMER_BOOL = ["autopay", "dispute_open", "email_opt_in", "marketing_opt_in"]
CUSTOMER_DATES = ["last_login_date", "due_date", "paid_date", "last_contact_date"]


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} is missing columns: {missing}")


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"true", "yes", "1"})


def load_customers(path: Path = CUSTOMERS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require(df, CUSTOMER_REQUIRED, path.name)

    for c in CUSTOMER_INT:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in CUSTOMER_FLOAT:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in CUSTOMER_BOOL:
        if c in df:
            df[c] = _as_bool(df[c])
    for c in CUSTOMER_DATES:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date  # paid_date may be empty

    df["customerID"] = df["customerID"].astype(str).str.strip()
    df = df.drop_duplicates(subset="customerID", keep="last")
    return df.set_index("customerID", drop=False)


def load_offers(path: Path = OFFERS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require(df, ["offer_id", "eligible_segments", "eligible_tiers", "status"], path.name)

    df["min_tenure_months"] = pd.to_numeric(df.get("min_tenure_months"), errors="coerce").fillna(0)
    df["max_tenure_months"] = pd.to_numeric(df.get("max_tenure_months"), errors="coerce").fillna(math.inf)
    df["priority"] = pd.to_numeric(df.get("priority"), errors="coerce").fillna(99).astype(int)
    df["requires_marketing_consent"] = _as_bool(df.get("requires_marketing_consent", False))
    for c in ("valid_from", "valid_to"):
        df[c] = pd.to_datetime(df.get(c), errors="coerce").dt.date
    return df


# --------------------------------------------------------------------------- #
# Deterministic segmentation  (first matching rule wins)
# --------------------------------------------------------------------------- #
#   1. Payment Reminder  bill status Overdue / Due
#   2. Dormant           days_since_login >= 30
#   3. Declining         usage_change_pct <= -25
#   4. Loyal             tenure >= 12 months OR on a 1/2-year contract
#   5. Consistent        everyone else (active, paid, stable usage, short tenure)
# Note: in this dataset Churn == Yes coincides with dormancy / usage decline, so
# churn-risk customers land in Dormant / Declining and receive win-back offers.
def classify(c: pd.Series) -> dict[str, Any]:
    status = str(c["status"]).strip().lower()
    contract = str(c["Contract"]).strip().lower()
    tenure = int(c["tenure"])

    if status in PAYMENT_STATUSES:
        reasons = [f"Bill {c['bill_id']} is {c['status']}: "
                   f"{c['amount_due']:.2f} {c.get('currency', '')} due {c['due_date']}"]
        if c["days_overdue"] > 0:
            reasons.append(f"{c['days_overdue']} day(s) overdue")
        if c.get("dispute_open"):
            reasons.append("Open billing dispute")
        if not c.get("autopay"):
            reasons.append("Autopay is off")
        return {"segment": "Payment Reminder", "reasons": reasons}

    if c["days_since_login"] >= DORMANT_LOGIN_DAYS:
        return {"segment": "Dormant",
                "reasons": [f"No login for {c['days_since_login']} days "
                            f"(threshold {DORMANT_LOGIN_DAYS})"]}

    if c["usage_change_pct"] <= DECLINING_USAGE_PCT:
        return {"segment": "Declining",
                "reasons": [f"Usage changed {c['usage_change_pct']:.1f}% vs previous 30 days "
                            f"(threshold {DECLINING_USAGE_PCT:.0f}%)"]}

    if tenure >= LOYAL_MIN_TENURE or contract in LONG_CONTRACTS:
        reasons = []
        if tenure >= LOYAL_MIN_TENURE:
            reasons.append(f"Tenure {tenure} months (>= {LOYAL_MIN_TENURE})")
        if contract in LONG_CONTRACTS:
            reasons.append(f"{c['Contract']} contract")
        return {"segment": "Loyal", "reasons": reasons}

    return {"segment": "Consistent",
            "reasons": [f"Active, bills paid, stable usage, tenure {tenure} month(s)"]}


# --------------------------------------------------------------------------- #
# Offer matching
# --------------------------------------------------------------------------- #
_RULE = re.compile(r"^\s*(\w+)\s*(!=|==|=)\s*(.+?)\s*$")


def _in_list(allowed: Any, value: str) -> bool:
    """'New|Silver' style lists; 'Any' or blank matches everything."""
    if pd.isna(allowed) or str(allowed).strip().lower() in {"", "any", "all"}:
        return True
    return value.strip().lower() in {t.strip().lower() for t in str(allowed).split("|")}


def _service_ok(rule: Any, c: pd.Series) -> bool:
    """Evaluates rules like 'InternetService != No' or 'PhoneService = Yes'."""
    if pd.isna(rule) or str(rule).strip().lower() in {"", "none", "any"}:
        return True
    m = _RULE.match(str(rule))
    if not m or m.group(1) not in c.index:
        return False  # unknown rule -> fail closed
    col, op, expected = m.groups()
    equal = str(c[col]).strip().lower() == expected.strip().lower()
    return equal if op in {"=", "=="} else not equal


def eligible_offers(c: pd.Series, segment: str, offers: pd.DataFrame,
                    today: date | None = None) -> list[dict[str, Any]]:
    today = today or date.today()
    tenure = int(c["tenure"])
    matches = offers[
        (offers["status"].str.lower() == "active")
        & offers["eligible_segments"].apply(_in_list, value=segment)
        & offers["eligible_tiers"].apply(_in_list, value=str(c["loyalty_tier"]))
        & offers["required_plan_status"].apply(_in_list, value=str(c["plan_status"]))
        & (offers["min_tenure_months"] <= tenure)
        & (offers["max_tenure_months"] >= tenure)
        & offers["valid_from"].apply(lambda d: pd.isna(d) or d <= today)
        & offers["valid_to"].apply(lambda d: pd.isna(d) or d >= today)
        & (~offers["requires_marketing_consent"] | bool(c["marketing_opt_in"]))
        & offers["required_service"].apply(_service_ok, c=c)
    ]
    matches = matches.sort_values(["priority", "offer_id"])
    return [_jsonable(r) for r in matches.to_dict(orient="records")]


# --------------------------------------------------------------------------- #
# Serialization helper
# --------------------------------------------------------------------------- #
def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None or v is pd.NaT or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            out[k] = None
        elif isinstance(v, date):
            out[k] = v.isoformat()
        elif hasattr(v, "item"):  # numpy scalar
            out[k] = v.item()
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
customers_df = load_customers()
offers_df = load_offers()

app = FastAPI(title="Customer Segmentation API", version="1.1.0")


class CustomerRequest(BaseModel):
    customer_id: str


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "customers": len(customers_df), "offers": len(offers_df)}


def _process(customer_id: str) -> dict[str, Any]:
    cid = customer_id.strip()
    if cid not in customers_df.index:
        raise HTTPException(status_code=404, detail=f"Customer '{cid}' not found")
    c = customers_df.loc[cid]
    result = classify(c)
    # Payment Reminder customers get a reminder, not a promotional offer.
    offers = [] if result["segment"] == "Payment Reminder" \
        else eligible_offers(c, result["segment"], offers_df)
    return {
        "customer_id": cid,
        "segment": result["segment"],
        "reasons": result["reasons"],
        "profile": _jsonable(c.to_dict()),
        "eligible_offers": offers,
    }


@app.get("/process-customer")
def process_customer_get(customer_id: str = Query(..., description="e.g. 7590-VHVEG")):
    return _process(customer_id)


@app.post("/process-customer")
def process_customer_post(req: CustomerRequest):
    return _process(req.customer_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
