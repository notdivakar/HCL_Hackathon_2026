"""
AI safety guardrails for customer emails (owner: feature-safety).

Three layers, all deterministic (no LLM involved):

1. precheck_customer()   -- BEFORE generation. Consent, contact frequency,
                            duplicate outreach, open disputes, missing or
                            contradictory billing data.
2. validate_email_draft() -- AFTER generation. Checks the draft against the
                            verified customer record and approved offers:
                            invented amounts / % / data / free periods / dates,
                            fake codes, unapproved offers, PII and internal-data
                            leaks, wrong facts, blocked language per message type
                            and reminder stage, missing dispute/support path.
3. send_gate()           -- AT SEND TIME. Blocks unless a human approved it,
                            the final (possibly edited) text passes, and no
                            pre-check is blocking.

Severities:
  BLOCK     -> must not be sent
  ESCALATE  -> must not be sent; route to a human team (e.g. billing disputes)
  WARN      -> can be sent, reviewer should look
"""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.policy import load_policy

DEFAULT_OFFERS_PATH = Path(__file__).resolve().parent.parent / "data" / "offer_catalouge.csv"
_EPS = 0.01
MARKETING_TYPES = {"offer", "no_offer"}

# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Violation:
    category: str            # e.g. "UNAPPROVED_AMOUNT", "NO_MARKETING_CONSENT"
    detail: str
    evidence: str = ""
    severity: str = "BLOCK"  # BLOCK | ESCALATE | WARN
    stage: str = "post"      # pre | post | send


@dataclass
class GuardrailResult:
    allowed: bool
    refusal_reason: str | None = None
    violations: list[Violation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "refusal_reason": self.refusal_reason,
            "violations": [asdict(v) for v in self.violations],
        }


def is_blocking(violations: Iterable[Violation]) -> bool:
    return any(v.severity in {"BLOCK", "ESCALATE"} for v in violations)


def _result(violations: list[Violation]) -> GuardrailResult:
    seen, unique = set(), []
    for v in violations:
        key = (v.category, v.detail)
        if key not in seen:
            seen.add(key)
            unique.append(v)
    blocking = [v for v in unique if v.severity in {"BLOCK", "ESCALATE"}]
    if not blocking:
        return GuardrailResult(allowed=True, violations=unique)
    reason = "Blocked by safety guardrails: " + "; ".join(v.detail for v in blocking)
    return GuardrailResult(allowed=False, refusal_reason=reason, violations=unique)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=8)
def load_offer_catalogue(path: str | Path = DEFAULT_OFFERS_PATH) -> dict[str, dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["offer_id"].strip().upper(): row for row in csv.DictReader(fh)}


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _to_float(value: Any) -> float | None:
    if value is None or str(value).strip() in {"", "nan", "None"}:
        return None
    try:
        return float(str(value).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def _to_int(value: Any, default: int = 0) -> int:
    f = _to_float(value)
    return default if f is None else int(f)


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s[:10], fmt).date()
        except ValueError:
            pass
    return None


# --------------------------------------------------------------------------- #
# Regex patterns
# --------------------------------------------------------------------------- #

_OFFER_ID_RE = re.compile(r"\bOFF-[A-Z0-9]+(?:-[A-Z0-9]+)*\b", re.IGNORECASE)
_DOLLAR_RE = re.compile(
    r"(?:\$|USD\s?|US\$)\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?"
    r"|(\d+(?:\.\d{1,2})?)\s?(?:dollars|usd|bucks)\b",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s?(?:%|percent\b|per\s?cent\b)", re.IGNORECASE)
_DATA_RE = re.compile(r"(\d+(?:\.\d+)?)\s?(GB|MB|TB|gigs?)\b", re.IGNORECASE)
_FREE_PERIOD_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|twelve)\s+(month|week|day|year)s?\s+(?:free|on us|at no (?:extra )?cost)\b",
    re.IGNORECASE,
)
_CODE_CONTEXT_RE = re.compile(
    r"\b(?:promo|promotion(?:al)?|discount|coupon|voucher|redemption|referral|gift)\s*code\s*[:=\-]?\s*[\"'“‘]?([A-Za-z0-9][A-Za-z0-9_\-]{2,})",
    re.IGNORECASE,
)
_CODE_TOKEN_RE = re.compile(r"\b(?=[A-Z0-9_-]*\d)(?=[A-Z0-9_-]*[A-Z])[A-Z0-9][A-Z0-9_-]{3,}\b")
_UNIT_TOKEN_RE = re.compile(r"^\d+(?:GB|MB|TB|AM|PM|TH|ST|ND|RD|K|X|G|MBPS|GBPS)$", re.IGNORECASE)

_ISO_DATE_RE = re.compile(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b")
_DMY_DATE_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+(20\d{2})\b", re.I)
_MDY_DATE_RE = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d{2})\b", re.I)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<![\w$.])(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?![\w.])")
_CARD_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
_CUSTOMER_ID_RE = re.compile(r"\b\d{4}-[A-Z]{5}\b", re.IGNORECASE)
_CAMPAIGN_ID_RE = re.compile(r"\bCMP-[A-Z0-9-]+\b", re.IGNORECASE)

_GREETING_RE = re.compile(r"^\s*(?:dear|hi|hello|hey|greetings)\s+([A-Z][A-Za-z'\-]+)", re.IGNORECASE | re.MULTILINE)
_TENURE_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\+?\s+(month|year)s?\s+"
    r"(?:with us|of (?:loyalty|service|being with us)|as (?:a|our) (?:valued )?customer|together)",
    re.IGNORECASE,
)
_TIER_RE = re.compile(r"\b(New|Silver|Gold|Platinum|Bronze|Diamond|VIP|Premium|Elite)\s+(?:customer|member|tier|status)\b", re.IGNORECASE)

_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12}

# Never allowed in any customer-facing copy.
_SENSITIVE_ALWAYS: list[tuple[str, re.Pattern[str]]] = [
    ("age / senior-citizen status", re.compile(r"\bsenior(?:\s+citizen)?s?\b|\byour age\b|\belderly\b", re.I)),
    ("gender", re.compile(r"\b(?:your gender|as a (?:man|woman|male|female))\b", re.I)),
    ("household / relationship status", re.compile(r"\byour (?:spouse|partner|husband|wife|dependents?|children|kids|family members?)\b|\bmarried\b", re.I)),
    ("churn / risk scoring", re.compile(r"\bchurn\w*\b|\b(?:risk|propensity|retention)\s+score\b|\bat[- ]risk\b|\blikely to (?:leave|cancel)\b", re.I)),
    ("behavioural tracking", re.compile(r"\b(?:last log(?:ged)?[- ]?in|days since (?:login|contact)|we noticed your usage|usage (?:dropped|declined|fell)|tracked)\b", re.I)),
    ("payment instrument", re.compile(r"\b(?:electronic check|mailed check|bank transfer|credit card \(automatic\)|card ending|account number|routing number|cvv|pin)\b", re.I)),
    ("credentials", re.compile(r"\b(?:password|passcode|one[- ]time code|otp|security question)\b", re.I)),
    ("collections status", re.compile(r"\b(?:delinquen\w*|debt collect\w*|in collections?)\b", re.I)),
]
# Not allowed in marketing emails (offers / thank-yous), allowed in billing reminders.
_SENSITIVE_MARKETING_ONLY: list[tuple[str, re.Pattern[str]]] = [
    ("billing / overdue status", re.compile(r"\b(?:overdue|past due|days late|outstanding balance|unpaid)\b", re.I)),
    ("account disputes", re.compile(r"\bdisputes?\b", re.I)),
]

_INTERNAL_FIELDS = {
    "customerid", "seniorcitizen", "paymentmethod", "totalcharges", "monthlycharges",
    "loyalty_tier", "plan_status", "last_login_date", "days_since_login", "usage_prev_30d",
    "usage_last_30d", "usage_change_pct", "days_overdue", "dispute_open", "email_opt_in",
    "marketing_opt_in", "preferred_channel", "last_contact_date", "days_since_contact",
    "last_channel", "last_campaign_id", "contacts_last_30d", "eligible_segments",
    "eligible_tiers", "requires_marketing_consent", "max_per_customer",
}


def _extract_dollars(text: str) -> list[tuple[float, str]]:
    out = []
    for m in _DOLLAR_RE.finditer(text):
        if m.group(1) is not None:
            out.append((float(f"{m.group(1).replace(',', '')}.{m.group(2) or '0'}"), m.group(0).strip()))
        elif m.group(3) is not None:
            out.append((float(m.group(3)), m.group(0).strip()))
    return out


def _extract_numbers(pattern: re.Pattern[str], text: str) -> list[tuple[float, str]]:
    return [(float(m.group(1)), m.group(0).strip()) for m in pattern.finditer(text)]


def _extract_data(text: str) -> list[tuple[float, str]]:
    out = []
    for m in _DATA_RE.finditer(text):
        value, unit = float(m.group(1)), m.group(2).upper()
        value = value / 1024 if unit == "MB" else value * 1024 if unit == "TB" else value
        out.append((value, m.group(0).strip()))
    return out


def _extract_dates(text: str) -> list[tuple[date, str]]:
    out = []
    for m in _ISO_DATE_RE.finditer(text):
        try:
            out.append((date(int(m.group(1)), int(m.group(2)), int(m.group(3))), m.group(0)))
        except ValueError:
            pass
    for m in _DMY_DATE_RE.finditer(text):
        try:
            out.append((date(int(m.group(3)), _MONTHS[m.group(2)[:3].lower()], int(m.group(1))), m.group(0)))
        except ValueError:
            pass
    for m in _MDY_DATE_RE.finditer(text):
        try:
            out.append((date(int(m.group(3)), _MONTHS[m.group(1)[:3].lower()], int(m.group(2))), m.group(0)))
        except ValueError:
            pass
    return out


def _word_or_int(token: str) -> int:
    token = token.lower()
    return _WORD_NUM.get(token, int(token) if token.isdigit() else -1)


def _in_allowed(value: float, allowed: Iterable[float]) -> bool:
    return any(abs(value - a) <= _EPS for a in allowed)


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _looks_like_date(s: str) -> bool:
    return bool(re.fullmatch(r"\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\s*|\s*\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\s*", s))


# --------------------------------------------------------------------------- #
# 1. PRE-CHECKS (before the LLM is called)
# --------------------------------------------------------------------------- #


def precheck_customer(
    customer: Mapping[str, Any],
    message_type: str,
    *,
    policy: Mapping[str, Any] | None = None,
) -> list[Violation]:
    """
    Decide whether we are allowed to write to this customer at all.

    message_type: "reminder" | "offer" | "no_offer"
    Returns a list of Violations (empty list = OK to generate).
    """
    policy = policy or load_policy()
    c = {str(k).strip(): v for k, v in customer.items()}
    out: list[Violation] = []

    def add(cat: str, detail: str, sev: str = "BLOCK", ev: str = "") -> None:
        out.append(Violation(cat, detail, ev, sev, "pre"))

    # Required evidence
    for fld in ("customerID", "first_name", "email"):
        if not str(c.get(fld) or "").strip() or str(c.get(fld)).lower() == "nan":
            add("MISSING_EVIDENCE", f"Required field '{fld}' is missing.", ev=fld)

    # Channel consent
    if not _truthy(c.get("email_opt_in")):
        add("NO_EMAIL_CONSENT", "Customer has not opted in to email.", ev="email_opt_in=False")
    if str(c.get("preferred_channel", "email")).strip().lower() not in {"email", ""}:
        add("CHANNEL_PREFERENCE", f"Customer prefers '{c.get('preferred_channel')}'; this draft is email.",
            sev="WARN", ev=f"preferred_channel={c.get('preferred_channel')}")

    if message_type == "reminder":
        status = str(c.get("status", "")).strip().lower()
        amount = _to_float(c.get("amount_due"))
        days = _to_int(c.get("days_overdue"))
        if status not in {"due", "overdue"}:
            add("CONTRADICTORY_DATA", f"Reminder requested but bill status is '{c.get('status')}'.", ev=f"status={c.get('status')}")
        if amount is None or amount <= 0:
            add("MISSING_EVIDENCE", "Amount due is missing or zero.", ev=f"amount_due={c.get('amount_due')}")
        if not _parse_date(c.get("due_date")):
            add("MISSING_EVIDENCE", "Due date is missing or invalid.", ev=f"due_date={c.get('due_date')}")
        if not str(c.get("bill_id") or "").strip():
            add("MISSING_EVIDENCE", "Bill ID is missing.", ev="bill_id")
        if status == "overdue" and days <= 0:
            add("CONTRADICTORY_DATA", "Status is Overdue but days_overdue is 0.", ev=f"days_overdue={days}")
        if status == "due" and days > 0:
            add("CONTRADICTORY_DATA", "Status is Due but days_overdue > 0.", ev=f"days_overdue={days}")
        if str(c.get("paid_date") or "").strip() not in {"", "nan", "None", "NaT"} and status in {"due", "overdue"}:
            add("CONTRADICTORY_DATA", "Bill has a paid date but is still marked unpaid.", ev=f"paid_date={c.get('paid_date')}")
        if _truthy(c.get("dispute_open")):
            add("DISPUTE_OPEN", "Customer has an open billing dispute; send to the billing team instead of a reminder.",
                sev="ESCALATE", ev="dispute_open=True")

    if message_type in MARKETING_TYPES:
        if not _truthy(c.get("marketing_opt_in")):
            add("NO_MARKETING_CONSENT", "Customer has not given marketing consent.", ev="marketing_opt_in=False")

    rules = policy.get("contact_rules", {})
    if message_type in rules.get("applies_to", []) or (message_type == "no_offer" and "offer" in rules.get("applies_to", [])):
        contacts = _to_int(c.get("contacts_last_30d"))
        since = _to_int(c.get("days_since_contact"), default=10_000)
        if contacts >= rules.get("max_contacts_last_30d", 3):
            add("CONTACT_CAP", f"Already contacted {contacts} times in the last 30 days (limit {rules['max_contacts_last_30d']}).",
                ev=f"contacts_last_30d={contacts}")
        if since < rules.get("min_days_since_last_contact", 7):
            add("DUPLICATE_OUTREACH", f"Last contacted {since} day(s) ago (minimum gap {rules['min_days_since_last_contact']} days).",
                ev=f"days_since_contact={since}")

    return out


# --------------------------------------------------------------------------- #
# 2. POST-CHECKS (on the generated / edited draft)
# --------------------------------------------------------------------------- #


def validate_email_draft(
    draft: str,
    customer: Mapping[str, Any],
    approved_offer_ids: Iterable[str],
    *,
    subject: str = "",
    message_type: str = "offer",
    reminder_stage: int | None = None,
    offers_path: str | Path = DEFAULT_OFFERS_PATH,
    today: date | None = None,
    channel: str = "email",
    allowed_contact_emails: Iterable[str] = (),
    allowed_phone_numbers: Iterable[str] = (),
    policy: Mapping[str, Any] | None = None,
) -> GuardrailResult:
    """
    Validate a draft (subject + body + footer). Fail-closed.

    message_type: "offer" | "no_offer" | "reminder"
    reminder_stage: 0-3 for reminders (controls which words are allowed).
    """
    policy = policy or load_policy()
    violations: list[Violation] = []
    today = today or date.today()
    text = f"{subject}\n{draft}" if subject else (draft or "")
    lowered = text.lower()

    if not (draft or "").strip():
        return _result([Violation("EMPTY_DRAFT", "The draft body is empty.")])

    try:
        catalogue = load_offer_catalogue(str(offers_path))
    except (OSError, KeyError, csv.Error) as exc:
        return _result([Violation("CATALOGUE_UNAVAILABLE", f"Offer catalogue could not be loaded: {exc}")])

    cust = {str(k).strip(): v for k, v in customer.items()}
    approved = {str(o).strip().upper() for o in approved_offer_ids if o and str(o).strip()}
    company = policy.get("company", {})
    ok_emails = {e.lower() for e in allowed_contact_emails} | {
        str(company.get("support_email", "")).lower(), str(company.get("sender_email", "")).lower()}

    if message_type == "reminder" and approved:
        violations.append(Violation("OFFER_IN_REMINDER", "Billing reminders must not contain offers."))

    # ---- Approved offers: re-verify eligibility (defence in depth) -------- #
    approved_offers: dict[str, dict[str, str]] = {}
    marketing_ok = _truthy(cust.get("marketing_opt_in", False))
    tenure = _to_float(cust.get("tenure"))
    for oid in sorted(approved):
        offer = catalogue.get(oid)
        if offer is None:
            violations.append(Violation("UNKNOWN_OFFER", f"Offer '{oid}' does not exist in the catalogue.", oid))
            continue
        if offer.get("status", "").strip().lower() != "active":
            violations.append(Violation("INACTIVE_OFFER", f"Offer '{oid}' is {offer.get('status')}, not active.", oid))
            continue
        vf, vt = _parse_date(offer.get("valid_from")), _parse_date(offer.get("valid_to"))
        if (vf and today < vf) or (vt and today > vt):
            violations.append(Violation("EXPIRED_OFFER", f"Offer '{oid}' is outside its validity window ({vf} to {vt}).", oid))
            continue
        och = offer.get("channel", "").strip().lower()
        if och and och not in {"any", "all", channel.lower()}:
            violations.append(Violation("WRONG_CHANNEL", f"Offer '{oid}' is not valid for channel '{channel}'.", oid))
            continue
        if _truthy(offer.get("requires_marketing_consent")) and not marketing_ok:
            violations.append(Violation("NO_MARKETING_CONSENT", f"Offer '{oid}' needs marketing consent.", oid))
            continue
        tier = str(cust.get("loyalty_tier", "")).strip().lower()
        tiers = {t.strip().lower() for t in offer.get("eligible_tiers", "").split("|") if t.strip()}
        if tier and tiers and not tiers & {tier, "all", "any"}:
            violations.append(Violation("INELIGIBLE_OFFER", f"Offer '{oid}' is not valid for tier '{tier}'.", oid))
            continue
        lo, hi = _to_float(offer.get("min_tenure_months")), _to_float(offer.get("max_tenure_months"))
        if tenure is not None and ((lo is not None and tenure < lo) or (hi is not None and tenure > hi)):
            violations.append(Violation("INELIGIBLE_OFFER", f"Offer '{oid}' tenure range {lo}-{hi} excludes {int(tenure)} months.", oid))
            continue
        approved_offers[oid] = offer

    # ---- Offer IDs / names mentioned in the draft ------------------------- #
    for m in _OFFER_ID_RE.finditer(text):
        oid = m.group(0).upper()
        if oid not in catalogue:
            violations.append(Violation("FAKE_OFFER_ID", f"Draft references an offer ID that does not exist: '{m.group(0)}'.", m.group(0)))
        elif oid not in approved_offers:
            violations.append(Violation("UNAPPROVED_OFFER", f"Draft references offer '{oid}', which is not approved.", m.group(0)))
    approved_names = {o["offer_name"].strip().lower() for o in approved_offers.values()}
    for oid, offer in catalogue.items():
        name = offer.get("offer_name", "").strip().lower()
        if oid not in approved_offers and name and name not in approved_names and name in lowered:
            violations.append(Violation("UNAPPROVED_OFFER", f"Draft mentions unapproved offer '{offer['offer_name']}'.", offer["offer_name"]))

    # ---- Whitelists of numbers backed by approved data -------------------- #
    approved_text = " ".join(
        f"{o.get('offer_value', '')} {o.get('description', '')} {o.get('terms', '')} {o.get('offer_name', '')}"
        for o in approved_offers.values()
    )
    allowed_dollars = {v for v, _ in _extract_dollars(approved_text)}
    allowed_percents = {v for v, _ in _extract_numbers(_PERCENT_RE, approved_text)}
    allowed_data = {v for v, _ in _extract_data(approved_text)}

    billing = [_to_float(cust.get(k)) for k in ("amount_due", "MonthlyCharges")]
    billing = [b for b in billing if b is not None]
    allowed_dollars.update(billing)
    for bill in billing:
        for credit, _ in _extract_dollars(approved_text):
            allowed_dollars.add(round(bill - credit, 2))
        for pct in allowed_percents:
            allowed_dollars.add(round(bill * pct / 100, 2))
            allowed_dollars.add(round(bill * (1 - pct / 100), 2))

    for value, raw in _extract_dollars(text):
        if not _in_allowed(value, allowed_dollars):
            violations.append(Violation("UNAPPROVED_AMOUNT", f"Amount '{raw}' is not backed by an approved offer or verified billing data.", raw))
    for value, raw in _extract_numbers(_PERCENT_RE, text):
        if not _in_allowed(value, allowed_percents):
            violations.append(Violation("UNAPPROVED_DISCOUNT", f"Percentage '{raw}' is not backed by an approved offer.", raw))
    for value, raw in _extract_data(text):
        if not _in_allowed(value, allowed_data):
            violations.append(Violation("UNAPPROVED_DATA_BONUS", f"Data allowance '{raw}' is not backed by an approved offer.", raw))

    allowed_free = {
        (_word_or_int(m.group(1)), m.group(2).lower())
        for o in approved_offers.values()
        for m in re.finditer(r"(\d+|one|two|three|four|five|six|twelve)\s+(month|week|day|year)s?\s+free",
                             o.get("offer_value", "") + " " + o.get("description", "") + " " + o.get("terms", ""), re.I)
    }
    for m in _FREE_PERIOD_RE.finditer(text):
        if (_word_or_int(m.group(1)), m.group(2).lower()) not in allowed_free:
            violations.append(Violation("UNAPPROVED_FREE_PERIOD", f"Free period '{m.group(0)}' is not backed by an approved offer.", m.group(0)))

    # ---- Dates: only due date / offer dates ------------------------------ #
    allowed_dates = {d for d, _ in _extract_dates(approved_text)}
    for o in approved_offers.values():
        for k in ("valid_to", "valid_from"):
            d = _parse_date(o.get(k))
            if d:
                allowed_dates.add(d)
    due = _parse_date(cust.get("due_date"))
    if due:
        allowed_dates.add(due)
    for d, raw in _extract_dates(text):
        if d not in allowed_dates:
            violations.append(Violation("UNAPPROVED_DATE", f"Date '{raw}' is not backed by billing or offer data.", raw))

    # ---- Fake promo codes ------------------------------------------------- #
    bill_id = str(cust.get("bill_id", "")).strip().upper()
    allowed_tokens = set(approved_offers) | ({bill_id} if bill_id else set())
    seen_codes: set[str] = set()
    for m in _CODE_CONTEXT_RE.finditer(text):
        token = m.group(1).strip(".,;:!?)\"'”’").upper()
        if token in allowed_tokens or token in seen_codes:
            continue
        seen_codes.add(token)
        violations.append(Violation("FAKE_CODE", f"Draft contains an unapproved promo code '{m.group(1)}'.", m.group(1)))
    for m in _CODE_TOKEN_RE.finditer(text):
        token = m.group(0).upper()
        if (token in allowed_tokens or token in seen_codes or token.startswith("OFF-")
                or _UNIT_TOKEN_RE.match(token) or _CUSTOMER_ID_RE.fullmatch(token)
                or _CAMPAIGN_ID_RE.fullmatch(token)):
            continue
        seen_codes.add(token)
        violations.append(Violation("FAKE_CODE", f"Draft contains a code-like token that is not approved: '{m.group(0)}'.", m.group(0)))

    # ---- Language rules by message type / stage -------------------------- #
    violations.extend(_check_language(lowered, message_type, reminder_stage, policy))

    # ---- Reminder must carry a support / dispute path -------------------- #
    if message_type == "reminder":
        support = str(company.get("support_email", "")).lower()
        if "dispute" not in lowered or (support and support not in lowered):
            violations.append(Violation("MISSING_SUPPORT_PATH", "Reminder must include the dispute / support path."))
        if bill_id and bill_id.lower() not in lowered:
            violations.append(Violation("MISSING_BILL_REFERENCE", f"Reminder should mention bill {bill_id}.", severity="WARN"))

    # ---- PII / internal data / facts -------------------------------------- #
    violations.extend(_check_pii(text, cust, ok_emails, allowed_phone_numbers, message_type))
    violations.extend(_check_customer_facts(text, cust))

    return _result(violations)


def _check_language(lowered: str, message_type: str, stage: int | None, policy: Mapping[str, Any]) -> list[Violation]:
    out: list[Violation] = []
    bl = policy.get("blocked_language", {})
    words = list(bl.get("all", []))
    if message_type == "reminder":
        words += bl.get("reminder_all", [])
        if stage is None or stage < 3:
            words += bl.get("reminder_stage_below_3", [])
    for w in words:
        pattern = re.escape(w.lower()) if not w[0].isalnum() else rf"\b{re.escape(w.lower())}\b"
        if re.search(pattern, lowered):
            out.append(Violation("BLOCKED_LANGUAGE", f"Blocked phrase for this message type/stage: '{w}'.", w))
    if re.search(r"https?://|www\.", lowered):
        out.append(Violation("UNAPPROVED_LINK", "Draft contains a link; links are not allowed."))
    if re.search(r"\[[^\]]+\]|\{[^}]+\}", lowered):
        out.append(Violation("UNFILLED_PLACEHOLDER", "Draft contains an unfilled placeholder."))
    return out


def _check_pii(text: str, cust: Mapping[str, Any], ok_emails: set[str],
               allowed_phone_numbers: Iterable[str], message_type: str) -> list[Violation]:
    out: list[Violation] = []
    own_email = str(cust.get("email", "")).strip().lower()
    for m in _EMAIL_RE.finditer(text):
        addr = m.group(0).lower()
        if addr == own_email:
            out.append(Violation("PII_LEAK", "Draft repeats the customer's email address in the body.", m.group(0), "WARN"))
        elif addr not in ok_emails:
            out.append(Violation("PII_LEAK", "Draft contains an unapproved email address.", m.group(0)))

    ok_phones = {re.sub(r"\D", "", p) for p in allowed_phone_numbers}
    for m in _PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 7 <= len(digits) <= 15 and digits not in ok_phones and not _looks_like_date(m.group(0)):
            out.append(Violation("PII_LEAK", "Draft contains an unapproved phone number.", m.group(0)))
    for m in _CARD_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            out.append(Violation("PII_LEAK", "Draft contains what looks like a card number.", "****" + digits[-4:]))
    for m in _SSN_RE.finditer(text):
        out.append(Violation("PII_LEAK", "Draft contains what looks like a national ID.", "***-**-" + m.group(0)[-4:]))
    for m in _IBAN_RE.finditer(text):
        out.append(Violation("PII_LEAK", "Draft contains what looks like a bank account number.", m.group(0)[:4] + "…"))

    cid = str(cust.get("customerID", "")).strip()
    if cid and cid.lower() in text.lower():
        out.append(Violation("PII_LEAK", "Draft exposes the internal customer ID.", cid))
    for m in _CUSTOMER_ID_RE.finditer(text):
        if m.group(0).lower() != cid.lower():
            out.append(Violation("PII_LEAK", "Draft contains another customer's identifier.", m.group(0)))

    patterns = list(_SENSITIVE_ALWAYS)
    if message_type != "reminder":
        patterns += _SENSITIVE_MARKETING_ONLY
    for label, pattern in patterns:
        m = pattern.search(text)
        if m:
            out.append(Violation("RESTRICTED_DATA", f"Draft references restricted information: {label}.", m.group(0)))

    pm = str(cust.get("PaymentMethod", "")).strip()
    if pm and pm.lower() not in {"nan", ""} and pm.lower() in text.lower():
        out.append(Violation("RESTRICTED_DATA", "Draft reveals the customer's payment method.", pm))

    lowered = text.lower()
    for fld in _INTERNAL_FIELDS:
        if re.search(rf"\b{re.escape(fld)}\b", lowered):
            out.append(Violation("INTERNAL_DATA_LEAK", f"Draft exposes internal field name '{fld}'.", fld))
    for m in _CAMPAIGN_ID_RE.finditer(text):
        out.append(Violation("INTERNAL_DATA_LEAK", "Draft exposes an internal campaign ID.", m.group(0)))
    return out


def _check_customer_facts(text: str, cust: Mapping[str, Any]) -> list[Violation]:
    out: list[Violation] = []
    first_name = str(cust.get("first_name", "")).strip()
    for m in _GREETING_RE.finditer(text):
        name = m.group(1)
        if first_name and name.lower() != first_name.lower() and name.lower() not in {"there", "customer", "valued", "friend"}:
            out.append(Violation("FACT_MISMATCH", f"Greeting uses '{name}' but the verified first name is '{first_name}'.", m.group(0).strip()))

    tenure = _to_float(cust.get("tenure"))
    if tenure is not None:
        for m in _TENURE_RE.finditer(text):
            n, unit = _word_or_int(m.group(1)), m.group(2).lower()
            ok = (unit == "month" and n <= tenure and tenure - n < 12) or (unit == "year" and n * 12 <= tenure < (n + 1) * 12)
            if not ok:
                out.append(Violation("FACT_MISMATCH", f"Tenure claim '{m.group(0)}' contradicts verified tenure of {int(tenure)} months.", m.group(0)))

    tier = str(cust.get("loyalty_tier", "")).strip()
    if tier:
        for m in _TIER_RE.finditer(text):
            if m.group(1).lower() != tier.lower():
                out.append(Violation("FACT_MISMATCH", f"Draft claims '{m.group(1)}' tier but verified tier is '{tier}'.", m.group(0)))
    return out


# --------------------------------------------------------------------------- #
# 3. SEND GATE (final check at delivery time)
# --------------------------------------------------------------------------- #


def send_gate(
    *,
    approved_by: str | None,
    precheck_violations: Iterable[Violation],
    final_check: GuardrailResult,
    customer: Mapping[str, Any],
    message_type: str,
) -> GuardrailResult:
    """Return allowed=True only if every send condition is met."""
    out: list[Violation] = []
    if not (approved_by or "").strip():
        out.append(Violation("NO_HUMAN_APPROVAL", "A named reviewer must approve before sending.", stage="send"))
    for v in precheck_violations:
        if v.severity in {"BLOCK", "ESCALATE"}:
            out.append(v)
    for v in final_check.violations:
        if v.severity in {"BLOCK", "ESCALATE"}:
            out.append(v)
    # Consent re-checked at send time (could have changed since drafting).
    if not _truthy(customer.get("email_opt_in")):
        out.append(Violation("NO_EMAIL_CONSENT", "Customer has not opted in to email.", stage="send"))
    if message_type in MARKETING_TYPES and not _truthy(customer.get("marketing_opt_in")):
        out.append(Violation("NO_MARKETING_CONSENT", "Customer has not given marketing consent.", stage="send"))
    return _result(out)


if __name__ == "__main__":
    demo = {
        "customerID": "5575-GNVDE", "first_name": "Arjun", "tenure": "34", "loyalty_tier": "Silver",
        "PaymentMethod": "Mailed check", "MonthlyCharges": "56.95", "amount_due": "56.95",
        "bill_id": "B100001", "email": "5575-gnvde@example.com", "marketing_opt_in": "True",
        "email_opt_in": "True", "due_date": "2026-11-08",
    }
    good = "Hi Arjun,\n\nAs a Silver customer, you get a $10 credit on your next bill. Thank you!"
    bad = "Hi Arjun,\n\nAs a Gold member, use promo code SAVE50NOW for 50% off and $100 cash back by 2026-12-25. We see you pay by mailed check and are at risk of churn."
    for label, d in (("GOOD", good), ("BAD", bad)):
        r = validate_email_draft(d, demo, ["OFF-LOY-SILV-01"], today=date(2026, 11, 17))
        print(label, "->", r.allowed, "|", r.refusal_reason)
