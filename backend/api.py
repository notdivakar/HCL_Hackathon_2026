"""
backend/api.py  (owner: feature-api)

Customer Loyalty & Engagement Assistant API (FastAPI + pandas).

Pipeline for one customer:
  profile -> classify (rules) -> strategy (policy) -> offer (deterministic)
  -> pre-checks -> LLM / template draft -> guardrails -> human review
  -> send gate -> MOCK outbox + audit log

Run:  uvicorn backend.api:app --reload --port 8000
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.guardrails import (  # noqa: E402
    GuardrailResult, Violation, is_blocking, precheck_customer, send_gate, validate_email_draft,
)
from src.llm_orchestrator import generate_email  # noqa: E402
from src.policy import load_policy, reminder_stage_for  # noqa: E402

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_DIR = ROOT / "data"
CUSTOMERS_CSV = DATA_DIR / "cleaned_telecom_data.csv"
RAW_CUSTOMERS_CSV = DATA_DIR / "Telco-Customer-Churn-merged.csv"
OFFERS_CSV = DATA_DIR / "offer_catalouge.csv"
OUTBOX_DIR = ROOT / "outbox"
LOG_DIR = ROOT / "logs"
AUDIT_LOG = LOG_DIR / "audit_log.jsonl"

POLICY = load_policy()
SEG = POLICY["segments"]
SENDER_EMAIL = os.getenv("SENDER_EMAIL", POLICY["company"]["sender_email"])
PROTECTED = set(POLICY.get("protected_attributes", []))
# "Today" can be pinned for a reproducible demo, e.g. APP_TODAY=2026-11-17
TODAY = date.fromisoformat(os.getenv("APP_TODAY")) if os.getenv("APP_TODAY") else date.today()

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
CUSTOMER_REQUIRED = [
    "customerID", "tenure", "Contract", "loyalty_tier", "plan_status", "days_since_login",
    "usage_change_pct", "usage_last_30d", "status", "days_overdue", "amount_due",
    "marketing_opt_in", "email_opt_in",
]
CUSTOMER_INT = ["tenure", "active_services", "days_since_login", "days_overdue",
                "days_since_contact", "contacts_last_30d"]
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
    if not path.exists():
        path = RAW_CUSTOMERS_CSV
    df = pd.read_csv(path)
    _require(df, CUSTOMER_REQUIRED, path.name)
    for c in CUSTOMER_INT:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in CUSTOMER_FLOAT:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CUSTOMER_BOOL:
        if c in df:
            df[c] = _as_bool(df[c])
    for c in CUSTOMER_DATES:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    df["customerID"] = df["customerID"].astype(str).str.strip().str.upper()
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
# Deterministic segmentation (first matching rule wins)
#  Protected attributes (gender, SeniorCitizen, Partner, Dependents) are NEVER used.
# --------------------------------------------------------------------------- #
def _clamp(x: float, lo: float = 0.5, hi: float = 0.99) -> float:
    return round(max(lo, min(hi, x)), 2)


def classify(c: pd.Series) -> dict[str, Any]:
    status = str(c["status"]).strip().lower()
    tenure = int(c["tenure"])
    login = int(c["days_since_login"])
    usage_chg = float(c["usage_change_pct"]) if pd.notna(c["usage_change_pct"]) else 0.0
    usage_last = float(c["usage_last_30d"]) if pd.notna(c["usage_last_30d"]) else 0.0
    contract = str(c["Contract"]).strip().lower()

    if status in SEG["payment_statuses"]:
        days = int(c["days_overdue"])
        stage = reminder_stage_for(days)
        codes = ["R_BILL_OVERDUE" if status == "overdue" else "R_BILL_DUE", f"R_REMINDER_STAGE_{stage['stage']}"]
        reasons = [f"Bill {c['bill_id']} is {c['status']}: ${c['amount_due']:.2f} due {c['due_date']}"]
        if days > 0:
            reasons.append(f"{days} day(s) overdue → stage {stage['stage']} ({stage['name']})")
        if bool(c.get("dispute_open")):
            codes.append("R_DISPUTE_OPEN")
            reasons.append("Open billing dispute")
        return {"segment": "Payment Reminder", "confidence": 0.99, "reason_codes": codes,
                "reasons": reasons, "reminder_stage": stage}

    if login >= SEG["dormant_login_days"] or usage_last == 0:
        codes, reasons = [], []
        if login >= SEG["dormant_login_days"]:
            codes.append("R_NO_LOGIN_60D")
            reasons.append(f"No login for {login} days (threshold {SEG['dormant_login_days']})")
        if usage_last == 0:
            codes.append("R_ZERO_USAGE_30D")
            reasons.append("Zero usage in the last 30 days")
        if len(codes) == 2:
            conf = 0.95
        elif usage_last == 0:
            conf = 0.85
        else:
            conf = _clamp(0.7 + (login - SEG["dormant_login_days"]) / 200)
        return {"segment": "Dormant", "confidence": conf, "reason_codes": codes, "reasons": reasons}

    if usage_chg <= SEG["declining_usage_pct"] or login >= SEG["declining_login_days"]:
        codes, reasons = [], []
        if usage_chg <= SEG["declining_usage_pct"]:
            codes.append("R_USAGE_DROP")
            reasons.append(f"Usage changed {usage_chg:.1f}% vs previous 30 days (threshold {SEG['declining_usage_pct']}%)")
        if login >= SEG["declining_login_days"]:
            codes.append("R_LOGIN_GAP_30D")
            reasons.append(f"No login for {login} days (threshold {SEG['declining_login_days']})")
        conf = _clamp(0.65 + abs(usage_chg - SEG["declining_usage_pct"]) / 150)
        return {"segment": "Declining", "confidence": conf, "reason_codes": codes, "reasons": reasons}

    if tenure >= SEG["loyal_min_tenure_months"] and (contract in SEG["loyal_contracts"] or bool(c.get("autopay"))):
        codes = ["R_TENURE_12M"]
        reasons = [f"Tenure {tenure} months (≥ {SEG['loyal_min_tenure_months']})", f"Loyalty tier {c['loyalty_tier']}"]
        if contract in SEG["loyal_contracts"]:
            codes.append("R_LONG_CONTRACT")
            reasons.append(f"{c['Contract']} contract")
        if bool(c.get("autopay")):
            codes.append("R_AUTOPAY")
            reasons.append("Autopay enabled")
        reasons.append("Bill paid, usage stable")
        return {"segment": "Loyal", "confidence": _clamp(0.7 + tenure / 240 + 0.05 * (len(codes) - 1)),
                "reason_codes": codes, "reasons": reasons}

    return {"segment": "Consistent", "confidence": 0.8, "reason_codes": ["R_PAID_STABLE"],
            "reasons": [f"Bill paid, usage stable ({usage_chg:+.1f}%), active in last {login} days, tenure {tenure} month(s)"]}


# --------------------------------------------------------------------------- #
# Offer eligibility (deterministic)
# --------------------------------------------------------------------------- #
_RULE = re.compile(r"^\s*(\w+)\s*(!=|==|=)\s*(.+?)\s*$")


def _in_list(allowed: Any, value: str) -> bool:
    if pd.isna(allowed) or str(allowed).strip().lower() in {"", "any", "all"}:
        return True
    return value.strip().lower() in {t.strip().lower() for t in str(allowed).split("|")}


def _service_ok(rule: Any, c: pd.Series) -> bool:
    if pd.isna(rule) or str(rule).strip().lower() in {"", "none", "any"}:
        return True
    m = _RULE.match(str(rule))
    if not m or m.group(1) not in c.index:
        return False  # unknown rule -> fail closed
    col, op, expected = m.groups()
    equal = str(c[col]).strip().lower() == expected.strip().lower()
    return equal if op in {"=", "=="} else not equal


def eligible_offers(c: pd.Series, segment: str, offers: pd.DataFrame, today: date | None = None) -> list[dict[str, Any]]:
    today = today or TODAY
    tenure = int(c["tenure"])
    m = offers[
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
    return [_jsonable(r) for r in m.sort_values(["priority", "offer_id"]).to_dict(orient="records")]


def offer_exclusions(c: pd.Series, segment: str, offers: pd.DataFrame, today: date | None = None) -> list[str]:
    """Explain why same-segment offers were NOT picked (shown to the reviewer)."""
    today = today or TODAY
    out = []
    for _, o in offers[offers["eligible_segments"].apply(_in_list, value=segment)].iterrows():
        why = []
        if str(o["status"]).lower() != "active":
            why.append(f"status {o['status']}")
        if pd.notna(o["valid_to"]) and o["valid_to"] < today:
            why.append(f"expired {o['valid_to']}")
        if not _in_list(o["eligible_tiers"], str(c["loyalty_tier"])):
            why.append("tier")
        if not (o["min_tenure_months"] <= int(c["tenure"]) <= o["max_tenure_months"]):
            why.append("tenure")
        if not _in_list(o["required_plan_status"], str(c["plan_status"])):
            why.append(f"plan {c['plan_status']}")
        if not _service_ok(o["required_service"], c):
            why.append(f"needs {o['required_service']}")
        if bool(o["requires_marketing_consent"]) and not bool(c["marketing_opt_in"]):
            why.append("no marketing consent")
        if why:
            out.append(f"{o['offer_id']}: {', '.join(why)}")
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None or v is pd.NaT or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            out[k] = None
        elif isinstance(v, (date, datetime)):
            out[k] = v.isoformat()
        elif hasattr(v, "item"):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _safe_profile(c: pd.Series) -> dict[str, Any]:
    """Profile for the reviewer: protected attributes removed (data minimisation)."""
    keep = ["customerID", "first_name", "email", "tenure", "Contract", "loyalty_tier", "plan_status",
            "InternetService", "PhoneService", "StreamingTV", "MonthlyCharges", "active_services",
            "last_login_date", "days_since_login", "usage_prev_30d", "usage_last_30d", "usage_change_pct",
            "bill_id", "amount_due", "currency", "due_date", "status", "days_overdue", "paid_date",
            "autopay", "dispute_open", "email_opt_in", "marketing_opt_in", "preferred_channel",
            "last_contact_date", "days_since_contact", "contacts_last_30d"]
    return _jsonable({k: c[k] for k in keep if k in c.index and k not in PROTECTED})


def _hash(value: Any) -> str:
    return hashlib.sha256(str(value).lower().encode()).hexdigest()[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def audit(event: str, **fields: Any) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    rec = {"ts": _now(), "event": event, "policy_version": POLICY["policy_version"], **fields}
    with open(AUDIT_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def _get_customer(customer_id: str) -> pd.Series:
    cid = str(customer_id).strip().upper()
    if cid not in customers_df.index:
        raise HTTPException(status_code=404, detail=f"Customer '{cid}' not found")
    return customers_df.loc[cid]


def _vlist(vs: list[Violation]) -> list[dict[str, Any]]:
    return [v.__dict__ for v in vs]


# --------------------------------------------------------------------------- #
# App state
# --------------------------------------------------------------------------- #
customers_df = load_customers()
offers_df = load_offers()
_SEGMENT_CACHE: dict[str, dict[str, Any]] = {cid: classify(row) for cid, row in customers_df.iterrows()}
DRAFTS: dict[str, dict[str, Any]] = {}

app = FastAPI(title="Customer Loyalty & Engagement Assistant API", version="2.0.0")


class CustomerRequest(BaseModel):
    customer_id: str


class ValidateRequest(BaseModel):
    draft_id: str
    subject: str
    body: str


class SendRequest(BaseModel):
    draft_id: str
    subject: str
    body: str
    reviewer: str
    feedback: str | None = None


class RejectRequest(BaseModel):
    draft_id: str
    reviewer: str
    reason: str


# --------------------------------------------------------------------------- #
# Core pipeline
# --------------------------------------------------------------------------- #
def _process(customer_id: str) -> dict[str, Any]:
    c = _get_customer(customer_id)
    cls = _SEGMENT_CACHE[c["customerID"]]
    strategy = dict(POLICY["strategies"][cls["segment"]])
    offers = [] if strategy["message_type"] == "reminder" else eligible_offers(c, cls["segment"], offers_df)
    return {
        "customer_id": c["customerID"],
        "segment": cls["segment"],
        "confidence": cls["confidence"],
        "reason_codes": cls["reason_codes"],
        "reasons": cls["reasons"],
        "reminder_stage": cls.get("reminder_stage"),
        "strategy": strategy,
        "profile": _safe_profile(c),
        "eligible_offers": offers,
        "excluded_offers": [] if strategy["message_type"] == "reminder" else offer_exclusions(c, cls["segment"], offers_df),
    }


def _run_checks(c: pd.Series, d: dict[str, Any], subject: str, body: str) -> GuardrailResult:
    return validate_email_draft(
        f"{body}\n\n{d['footer']}", c.to_dict(), [d["offer_id"]] if d["offer_id"] else [],
        subject=subject, message_type=d["message_type"],
        reminder_stage=d["reminder_stage"]["stage"] if d.get("reminder_stage") else None,
        today=TODAY,
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "customers": len(customers_df), "offers": len(offers_df),
            "llm_enabled": bool(os.getenv("GROQ_API_KEY")), "sender": SENDER_EMAIL,
            "today": TODAY.isoformat(), "policy_version": POLICY["policy_version"]}


@app.get("/segments")
def segments() -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in _SEGMENT_CACHE.values():
        counts[v["segment"]] = counts.get(v["segment"], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


@app.get("/customers")
def get_customers(segment: str | None = None, search: str | None = None,
                  limit: int = Query(200, ge=1, le=10_000), offset: int = Query(0, ge=0)) -> dict[str, Any]:
    ids = list(customers_df.index)
    if segment and segment != "All":
        ids = [i for i in ids if _SEGMENT_CACHE[i]["segment"] == segment]
    if search:
        s = search.strip().lower()
        ids = [i for i in ids if s in i.lower() or s in str(customers_df.at[i, "first_name"]).lower()]
    page = ids[offset:offset + limit]
    rows = []
    for cid in page:
        r = customers_df.loc[cid]
        cls = _SEGMENT_CACHE[cid]
        rows.append({
            "customer_id": cid,
            "name": r.get("first_name") or "Unknown",
            "segment": cls["segment"],
            "confidence": cls["confidence"],
            "loyalty_tier": r.get("loyalty_tier"),
            "bill_status": r.get("status"),
        })
    return {"total": len(ids), "offset": offset, "limit": limit, "customers": rows}


@app.get("/process-customer")
def process_customer_get(customer_id: str = Query(..., description="e.g. 7590-VHVEG")):
    return _process(customer_id)


@app.post("/process-customer")
def process_customer_post(req: CustomerRequest):
    return _process(req.customer_id)


@app.post("/generate-email")
def generate(req: CustomerRequest) -> dict[str, Any]:
    c = _get_customer(req.customer_id)
    info = _process(c["customerID"])
    strategy = info["strategy"]
    offer = info["eligible_offers"][0] if info["eligible_offers"] else None
    message_type = strategy["message_type"]
    if message_type == "offer" and offer is None:
        message_type = "no_offer"

    draft_id = uuid.uuid4().hex[:10]
    pre = precheck_customer(c.to_dict(), message_type)
    base = {
        "draft_id": draft_id, **info, "message_type": message_type,
        "offer": offer, "offer_id": offer["offer_id"] if offer else None,
        "from": SENDER_EMAIL, "to": c.get("email"),
        "precheck": _vlist(pre),
    }

    # REFUSE to generate if a pre-check blocks (consent, dispute, contact cap, bad data)
    if is_blocking(pre):
        base.update({"status": "REFUSED", "subject": "", "body": "", "footer": "",
                     "guardrails": {"allowed": False, "violations": [],
                                    "refusal_reason": "Not generated: " + "; ".join(
                                        v.detail for v in pre if v.severity in {"BLOCK", "ESCALATE"})},
                     "source": None, "model": None})
        DRAFTS[draft_id] = base
        audit("generation_refused", draft_id=draft_id, customer_id=c["customerID"], segment=info["segment"],
              reason_codes=info["reason_codes"], message_type=message_type, precheck=base["precheck"])
        return base

    gen = generate_email(c, segment=info["segment"], message_type=message_type,
                         objective=strategy["objective"], tone=strategy["tone"],
                         offer=offer, stage=info.get("reminder_stage"))
    base.update({k: gen[k] for k in ("subject", "body", "footer", "source", "model",
                                     "prompt_version", "template_version", "attempts", "llm_errors")})
    result = _run_checks(c, base, gen["subject"], gen["body"])
    base["guardrails"] = result.to_dict()
    base["status"] = "PENDING_REVIEW" if result.allowed else "BLOCKED"
    DRAFTS[draft_id] = base
    audit("draft_generated", draft_id=draft_id, customer_id=c["customerID"], segment=info["segment"],
          confidence=info["confidence"], reason_codes=info["reason_codes"], message_type=message_type,
          offer_id=base["offer_id"], source=gen["source"], model=gen["model"],
          prompt_version=gen["prompt_version"], template_version=gen["template_version"],
          evidence_fields=sorted(info["profile"].keys()), precheck=base["precheck"],
          guardrails=base["guardrails"], subject=gen["subject"], body=gen["body"])
    return base


@app.post("/validate-email")
def validate(req: ValidateRequest) -> dict[str, Any]:
    d = DRAFTS.get(req.draft_id)
    if not d:
        raise HTTPException(404, "Draft not found. Generate it again.")
    c = _get_customer(d["customer_id"])
    return _run_checks(c, d, req.subject, req.body).to_dict()


@app.post("/send-email")
def send(req: SendRequest) -> dict[str, Any]:
    d = DRAFTS.get(req.draft_id)
    if not d:
        raise HTTPException(404, "Draft not found. Generate it again.")
    if d["status"] in {"SENT", "REJECTED"}:
        raise HTTPException(409, f"Draft already {d['status'].lower()}.")
    c = _get_customer(d["customer_id"])

    final = _run_checks(c, d, req.subject, req.body)  # re-check the reviewer's edited text
    pre = precheck_customer(c.to_dict(), d["message_type"])
    gate = send_gate(approved_by=req.reviewer, precheck_violations=pre, final_check=final,
                     customer=c.to_dict(), message_type=d["message_type"])
    edited = (req.subject != d["subject"]) or (req.body != d["body"])

    if not gate.allowed:
        audit("send_blocked", draft_id=d["draft_id"], customer_id=d["customer_id"], reviewer=req.reviewer,
              edited=edited, gate=gate.to_dict())
        raise HTTPException(status_code=422, detail={"message": "Send blocked", **gate.to_dict()})

    OUTBOX_DIR.mkdir(exist_ok=True)
    message = {
        "message_id": f"MOCK-{d['draft_id']}",
        "from": SENDER_EMAIL,
        "to": d["to"],
        "subject": req.subject,
        "body": f"{req.body}\n\n---\n{d['footer']}",
        "sent_at": _now(),
        "delivery": "MOCK (draft-only, not actually delivered)",
    }
    (OUTBOX_DIR / f"{message['message_id']}.json").write_text(json.dumps(message, indent=2), encoding="utf-8")
    d["status"] = "SENT"
    audit("sent", draft_id=d["draft_id"], customer_id=d["customer_id"], recipient_hash=_hash(d["to"]),
          reviewer=req.reviewer, edited=edited, feedback=req.feedback, final_subject=req.subject,
          final_body=req.body, send_status="MOCK_SENT", message_id=message["message_id"])
    return {"status": "MOCK_SENT", "message_id": message["message_id"], "from": SENDER_EMAIL, "to": d["to"]}


@app.post("/reject-email")
def reject(req: RejectRequest) -> dict[str, Any]:
    d = DRAFTS.get(req.draft_id)
    if not d:
        raise HTTPException(404, "Draft not found.")
    d["status"] = "REJECTED"
    audit("rejected", draft_id=d["draft_id"], customer_id=d["customer_id"], reviewer=req.reviewer,
          reason=req.reason, send_status="NOT_SENT")
    return {"status": "REJECTED", "draft_id": d["draft_id"]}


@app.get("/audit")
def get_audit(limit: int = 50) -> list[dict[str, Any]]:
    if not AUDIT_LOG.exists():
        return []
    lines = AUDIT_LOG.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(x) for x in lines[-limit:]][::-1]


@app.get("/outbox")
def get_outbox(limit: int = 50) -> list[dict[str, Any]]:
    if not OUTBOX_DIR.exists():
        return []
    files = sorted(OUTBOX_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


@app.get("/scenarios")
def scenarios() -> list[dict[str, Any]]:
    """Ready-made demo customers: one per required scenario, plus refusals."""
    picks: list[dict[str, Any]] = []

    def first(label: str, segment: str, message_type: str, want_blocked: bool, extra=None) -> None:
        for cid, cls in _SEGMENT_CACHE.items():
            if cls["segment"] != segment:
                continue
            c = customers_df.loc[cid]
            if extra and not extra(c):
                continue
            mt = message_type
            if mt == "offer" and not eligible_offers(c, segment, offers_df):
                continue
            blocked = is_blocking(precheck_customer(c.to_dict(), mt))
            if blocked == want_blocked:
                picks.append({"label": label, "customer_id": cid, "name": c["first_name"], "segment": segment})
                return

    first("Loyal reward", "Loyal", "offer", False)
    first("Consistent thank-you", "Consistent", "offer", False)
    first("Declining re-engagement", "Declining", "offer", False)
    first("Dormant win-back", "Dormant", "offer", False)
    first("Due soon (stage 0)", "Payment Reminder", "reminder", False, lambda c: c["days_overdue"] == 0)
    first("Overdue polite (stage 1)", "Payment Reminder", "reminder", False, lambda c: 1 <= c["days_overdue"] <= 15)
    first("Overdue firm (stage 2)", "Payment Reminder", "reminder", False, lambda c: 16 <= c["days_overdue"] <= 30)
    first("Overdue final (stage 3)", "Payment Reminder", "reminder", False, lambda c: c["days_overdue"] > 30)
    first("REFUSE: dispute open", "Payment Reminder", "reminder", True, lambda c: bool(c["dispute_open"]))
    first("REFUSE: no marketing consent", "Loyal", "no_offer", True,
          lambda c: not bool(c["marketing_opt_in"]) and bool(c["email_opt_in"]))
    first("REFUSE: contacted too often", "Consistent", "no_offer", True,
          lambda c: c["contacts_last_30d"] >= 3 and bool(c["marketing_opt_in"]) and bool(c["email_opt_in"]))
    return picks


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
