"""
AI safety guardrails for LLM-generated retention emails.

`validate_email_draft()` intercepts an LLM email draft before it is sent and
cross-references it against:
  * verified customer facts (a row from Telco-Customer-Churn-merged.csv), and
  * the approved offer IDs for this customer, as defined in the offer catalogue.

The draft is BLOCKED (allowed=False) with a refusal reason if the LLM:
  * references offer IDs / offer names that are unknown, inactive, expired or not approved,
  * invents dollar amounts, percentages or data allowances not backed by an approved offer
    or a verified billing fact,
  * includes fake discount / promo / voucher codes,
  * leaks restricted personal or internal data (PII, sensitive attributes, internal fields),
  * misstates verified customer facts (name, tenure, loyalty tier),
  * markets an offer to a customer who has not given marketing consent.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULT_OFFERS_PATH = Path(__file__).resolve().parent.parent / "data" / "offer_catalouge.csv"

# Tolerance when comparing money / numeric values.
_EPS = 0.01

# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Violation:
    category: str  # e.g. "UNAPPROVED_AMOUNT", "FAKE_CODE", "PII_LEAK"
    detail: str
    evidence: str = ""


@dataclass
class GuardrailResult:
    allowed: bool
    refusal_reason: str | None = None
    violations: list[Violation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "refusal_reason": self.refusal_reason,
            "violations": [v.__dict__ for v in self.violations],
        }


# --------------------------------------------------------------------------- #
# Offer catalogue
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=8)
def load_offer_catalogue(path: str | Path = DEFAULT_OFFERS_PATH) -> dict[str, dict[str, str]]:
    """Load the offer catalogue CSV keyed by offer_id."""
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["offer_id"].strip().upper(): row for row in csv.DictReader(fh)}


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


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
    r"\b(?:promo|promotion(?:al)?|discount|coupon|voucher|redemption|referral|offer|gift|use|apply|enter)?"
    r"\s*code\s*[:=\-]?\s*[\"'“‘]?([A-Za-z0-9][A-Za-z0-9_\-]{2,})",
    re.IGNORECASE,
)
# Uppercase alphanumeric token containing both letters and digits (looks like a code).
_CODE_TOKEN_RE = re.compile(r"\b(?=[A-Z0-9_-]*\d)(?=[A-Z0-9_-]*[A-Z])[A-Z0-9][A-Z0-9_-]{3,}\b")
_UNIT_TOKEN_RE = re.compile(r"^\d+(?:GB|MB|TB|AM|PM|TH|ST|ND|RD|K|X|G|MBPS|GBPS)$", re.IGNORECASE)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?!\w)")
_CARD_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
_CUSTOMER_ID_RE = re.compile(r"\b\d{4}-[A-Z]{5}\b")

_GREETING_RE = re.compile(r"^\s*(?:dear|hi|hello|hey|greetings)\s+([A-Z][A-Za-z'\-]+)", re.IGNORECASE | re.MULTILINE)
_TENURE_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\+?\s+(month|year)s?\s+"
    r"(?:with us|of (?:loyalty|service|being with us)|as (?:a|our) (?:valued )?customer|together)",
    re.IGNORECASE,
)
_TIER_RE = re.compile(r"\b(New|Silver|Gold|Platinum|Bronze|Diamond|VIP|Premium|Elite)\s+(?:customer|member|tier|status)\b", re.IGNORECASE)

_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
}

# Sensitive attributes that must never be surfaced in customer-facing copy.
_SENSITIVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("age / senior-citizen status", re.compile(r"\bsenior(?:\s+citizen)?s?\b|\byour age\b|\belderly\b", re.I)),
    ("gender", re.compile(r"\b(?:your gender|as a (?:man|woman|male|female))\b", re.I)),
    ("household / relationship status", re.compile(r"\byour (?:spouse|partner|husband|wife|dependents?|children|kids|family members?)\b|\b(?:single|married)\b", re.I)),
    ("churn / risk scoring", re.compile(r"\bchurn\w*\b|\b(?:risk|propensity|retention)\s+score\b|\bat[- ]risk\b|\blikely to (?:leave|cancel)\b", re.I)),
    ("account disputes", re.compile(r"\bdisputes?\b", re.I)),
    ("overdue / collections status", re.compile(r"\b(?:overdue|delinquen\w*|collections?|days late|past due)\b", re.I)),
    ("behavioural tracking", re.compile(r"\b(?:last log(?:ged)?[- ]?in|days since (?:login|contact)|we noticed your usage|usage (?:dropped|declined|fell))\b", re.I)),
    ("payment instrument", re.compile(r"\b(?:electronic check|mailed check|bank transfer|credit card \(automatic\)|card ending|account number|routing number|cvv|pin)\b", re.I)),
    ("credentials", re.compile(r"\b(?:password|passcode|one[- ]time code|otp|security question)\b", re.I)),
]

# Internal schema fields that must not leak into customer-facing copy.
_INTERNAL_FIELDS = {
    "customerid", "seniorcitizen", "partner", "dependents", "churn", "paymentmethod",
    "totalcharges", "monthlycharges", "loyalty_tier", "plan_status", "last_login_date",
    "days_since_login", "usage_prev_30d", "usage_last_30d", "usage_change_pct", "days_overdue",
    "dispute_open", "email_opt_in", "marketing_opt_in", "preferred_channel", "last_contact_date",
    "days_since_contact", "last_channel", "last_campaign_id", "contacts_last_30d",
    "eligible_segments", "eligible_tiers", "requires_marketing_consent", "max_per_customer",
}
_CAMPAIGN_ID_RE = re.compile(r"\bCMP-[A-Z0-9-]+\b", re.IGNORECASE)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def _extract_dollars(text: str) -> list[tuple[float, str]]:
    out = []
    for m in _DOLLAR_RE.finditer(text):
        if m.group(1) is not None:
            whole = m.group(1).replace(",", "")
            cents = m.group(2) or "0"
            out.append((float(f"{whole}.{cents}"), m.group(0).strip()))
        elif m.group(3) is not None:
            out.append((float(m.group(3)), m.group(0).strip()))
    return out


def _extract_numbers(pattern: re.Pattern[str], text: str) -> list[tuple[float, str]]:
    return [(float(m.group(1)), m.group(0).strip()) for m in pattern.finditer(text)]


def _extract_data(text: str) -> list[tuple[float, str]]:
    out = []
    for m in _DATA_RE.finditer(text):
        value, unit = float(m.group(1)), m.group(2).upper()
        if unit == "MB":
            value /= 1024
        elif unit == "TB":
            value *= 1024
        out.append((value, m.group(0).strip()))
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


def _normalise_customer(customer: Mapping[str, Any]) -> dict[str, Any]:
    return {str(k).strip(): v for k, v in customer.items()}


# --------------------------------------------------------------------------- #
# Main validation entry point
# --------------------------------------------------------------------------- #


def validate_email_draft(
    draft: str,
    customer: Mapping[str, Any],
    approved_offer_ids: Iterable[str],
    *,
    offers_path: str | Path = DEFAULT_OFFERS_PATH,
    today: date | None = None,
    channel: str = "email",
    allowed_contact_emails: Iterable[str] = (),
    allowed_phone_numbers: Iterable[str] = (),
    subject: str = "",
) -> GuardrailResult:
    """
    Validate an LLM-generated email draft. Fail-closed: any violation blocks the message.

    Args:
        draft: The email body produced by the LLM.
        customer: Verified customer record (dict/row from the customer dataset).
        approved_offer_ids: Offer IDs the decision engine approved for this customer.
        offers_path: Path to the offer catalogue CSV.
        today: Override for "today" (testing). Defaults to date.today().
        channel: Delivery channel; offers must be valid for it.
        allowed_contact_emails: Company support/sender addresses permitted in the body.
        allowed_phone_numbers: Company support phone numbers permitted in the body.
        subject: Optional subject line, validated together with the body.

    Returns:
        GuardrailResult(allowed, refusal_reason, violations)
    """
    violations: list[Violation] = []
    today = today or date.today()
    text = f"{subject}\n{draft}" if subject else (draft or "")

    if not text.strip():
        return _block([Violation("EMPTY_DRAFT", "The LLM returned an empty draft.")])

    try:
        catalogue = load_offer_catalogue(str(offers_path))
    except (OSError, KeyError, csv.Error) as exc:
        return _block([Violation("CATALOGUE_UNAVAILABLE", f"Offer catalogue could not be loaded: {exc}")])

    cust = _normalise_customer(customer)
    approved = {o.strip().upper() for o in approved_offer_ids if o and str(o).strip()}

    # ---- 1. Validate the approved offers themselves (defence in depth) ----- #
    approved_offers: dict[str, dict[str, str]] = {}
    marketing_ok = _truthy(cust.get("marketing_opt_in", True))
    for oid in sorted(approved):
        offer = catalogue.get(oid)
        if offer is None:
            violations.append(Violation("UNKNOWN_OFFER", f"Approved offer '{oid}' does not exist in the catalogue.", oid))
            continue
        if offer.get("status", "").strip().lower() != "active":
            violations.append(Violation("INACTIVE_OFFER", f"Offer '{oid}' is not active.", oid))
            continue
        vf, vt = _parse_date(offer.get("valid_from")), _parse_date(offer.get("valid_to"))
        if (vf and today < vf) or (vt and today > vt):
            violations.append(Violation("EXPIRED_OFFER", f"Offer '{oid}' is outside its validity window ({vf} to {vt}).", oid))
            continue
        offer_channel = offer.get("channel", "").strip().lower()
        if offer_channel and offer_channel not in {"any", "all", channel.lower()}:
            violations.append(Violation("WRONG_CHANNEL", f"Offer '{oid}' is not valid for channel '{channel}'.", oid))
            continue
        if _truthy(offer.get("requires_marketing_consent")) and not marketing_ok:
            violations.append(Violation("NO_MARKETING_CONSENT", f"Offer '{oid}' requires marketing consent the customer has not given.", oid))
            continue
        tier = str(cust.get("loyalty_tier", "")).strip()
        tiers = {t.strip().lower() for t in offer.get("eligible_tiers", "").split("|") if t.strip()}
        if tier and tiers and not tiers & {tier.lower(), "all", "any"}:
            violations.append(Violation("INELIGIBLE_OFFER", f"Offer '{oid}' is not valid for the customer's '{tier}' tier.", oid))
            continue
        approved_offers[oid] = offer

    # ---- 2. Offer IDs / names referenced in the draft ---------------------- #
    for m in _OFFER_ID_RE.finditer(text):
        oid = m.group(0).upper()
        if oid not in catalogue:
            violations.append(Violation("FAKE_OFFER_ID", f"Draft references an offer ID that does not exist: '{m.group(0)}'.", m.group(0)))
        elif oid not in approved_offers:
            violations.append(Violation("UNAPPROVED_OFFER", f"Draft references offer '{oid}', which is not approved for this customer.", m.group(0)))

    lowered = text.lower()
    approved_names = {o["offer_name"].strip().lower() for o in approved_offers.values()}
    for oid, offer in catalogue.items():
        name = offer.get("offer_name", "").strip().lower()
        if oid not in approved_offers and name and name not in approved_names and name in lowered:
            violations.append(Violation("UNAPPROVED_OFFER", f"Draft mentions unapproved offer '{offer['offer_name']}' ({oid}).", offer["offer_name"]))

    # ---- 3. Build whitelists of numbers backed by approved offers / facts -- #
    approved_text = " ".join(
        f"{o.get('offer_value', '')} {o.get('description', '')} {o.get('terms', '')} {o.get('offer_name', '')}"
        for o in approved_offers.values()
    )
    allowed_dollars = {v for v, _ in _extract_dollars(approved_text)}
    allowed_percents = {v for v, _ in _extract_numbers(_PERCENT_RE, approved_text)}
    allowed_data = {v for v, _ in _extract_data(approved_text)}

    billing_facts = [
        _to_float(cust.get(k)) for k in ("amount_due", "MonthlyCharges", "TotalCharges")
    ]
    billing_facts = [b for b in billing_facts if b is not None]
    allowed_dollars.update(billing_facts)
    # Permit deterministic derived amounts: bill minus approved credit / % discount.
    for bill in billing_facts:
        for credit in list(_extract_dollars(approved_text)):
            allowed_dollars.add(round(bill - credit[0], 2))
        for pct in allowed_percents:
            allowed_dollars.add(round(bill * pct / 100, 2))
            allowed_dollars.add(round(bill * (1 - pct / 100), 2))

    # ---- 4. Hallucinated monetary values / percentages / data -------------- #
    for value, raw in _extract_dollars(text):
        if not _in_allowed(value, allowed_dollars):
            violations.append(Violation("UNAPPROVED_AMOUNT", f"Dollar amount '{raw}' is not backed by an approved offer or verified billing data.", raw))

    for value, raw in _extract_numbers(_PERCENT_RE, text):
        if not _in_allowed(value, allowed_percents):
            violations.append(Violation("UNAPPROVED_DISCOUNT", f"Percentage '{raw}' is not backed by an approved offer.", raw))

    for value, raw in _extract_data(text):
        if not _in_allowed(value, allowed_data):
            violations.append(Violation("UNAPPROVED_DATA_BONUS", f"Data allowance '{raw}' is not backed by an approved offer.", raw))

    allowed_free = {
        (_word_or_int(m.group(1)), m.group(2).lower())
        for o in approved_offers.values()
        for m in re.finditer(r"(\d+|one|two|three|four|five|six|twelve)\s+(month|week|day|year)s?\s+free", o.get("offer_value", "") + " " + o.get("description", ""), re.I)
    }
    for m in _FREE_PERIOD_RE.finditer(text):
        if (_word_or_int(m.group(1)), m.group(2).lower()) not in allowed_free:
            violations.append(Violation("UNAPPROVED_FREE_PERIOD", f"Free period '{m.group(0)}' is not backed by an approved offer.", m.group(0)))

    # ---- 5. Fake discount / promo codes ------------------------------------ #
    bill_id = str(cust.get("bill_id", "")).strip().upper()
    allowed_tokens = set(approved_offers) | ({bill_id} if bill_id else set())
    seen_codes: set[str] = set()

    for m in _CODE_CONTEXT_RE.finditer(text):
        token = m.group(1).strip(".,;:!?)\"'”’").upper()
        if token in allowed_tokens or token in seen_codes or token.lower() in {"below", "above", "here", "at", "the", "and", "for", "when", "will", "to"}:
            continue
        seen_codes.add(token)
        violations.append(Violation("FAKE_CODE", f"Draft contains an unapproved discount/promo code '{m.group(1)}'.", m.group(1)))

    for m in _CODE_TOKEN_RE.finditer(text):
        token = m.group(0).upper()
        if (
            token in allowed_tokens
            or token in seen_codes
            or token.startswith("OFF-")  # handled by offer-ID check
            or _UNIT_TOKEN_RE.match(token)
            or _CUSTOMER_ID_RE.fullmatch(token)  # handled by PII check
            or _CAMPAIGN_ID_RE.fullmatch(token)  # handled by internal-data check
        ):
            continue
        seen_codes.add(token)
        violations.append(Violation("FAKE_CODE", f"Draft contains a code-like token that is not an approved offer: '{m.group(0)}'.", m.group(0)))

    # ---- 6. Restricted personal / internal data leakage -------------------- #
    violations.extend(_check_pii(text, cust, allowed_contact_emails, allowed_phone_numbers))

    # ---- 7. Verified customer facts ---------------------------------------- #
    violations.extend(_check_customer_facts(text, cust))

    return _block(violations) if violations else GuardrailResult(allowed=True)


# --------------------------------------------------------------------------- #
# Sub-checks
# --------------------------------------------------------------------------- #


def _check_pii(
    text: str,
    cust: Mapping[str, Any],
    allowed_contact_emails: Iterable[str],
    allowed_phone_numbers: Iterable[str],
) -> list[Violation]:
    out: list[Violation] = []

    # Email addresses: only the customer's own address or company contact addresses.
    ok_emails = {e.strip().lower() for e in allowed_contact_emails}
    own_email = str(cust.get("email", "")).strip().lower()
    for m in _EMAIL_RE.finditer(text):
        addr = m.group(0).lower()
        if addr not in ok_emails and addr != own_email:
            out.append(Violation("PII_LEAK", "Draft contains an unapproved email address.", m.group(0)))

    # Phone numbers.
    ok_phones = {re.sub(r"\D", "", p) for p in allowed_phone_numbers}
    for m in _PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 7 <= len(digits) <= 15 and digits not in ok_phones and not _looks_like_date(m.group(0)):
            out.append(Violation("PII_LEAK", "Draft contains an unapproved phone number.", m.group(0)))

    # Payment cards (Luhn-valid 13-19 digit sequences).
    for m in _CARD_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            out.append(Violation("PII_LEAK", "Draft contains what appears to be a payment card number.", "****" + digits[-4:]))

    for m in _SSN_RE.finditer(text):
        out.append(Violation("PII_LEAK", "Draft contains what appears to be a national ID / SSN.", "***-**-" + m.group(0)[-4:]))
    for m in _IBAN_RE.finditer(text):
        out.append(Violation("PII_LEAK", "Draft contains what appears to be a bank account / IBAN.", m.group(0)[:4] + "…"))

    # Internal customer identifier.
    cid = str(cust.get("customerID", "")).strip()
    if cid and cid.lower() in text.lower():
        out.append(Violation("PII_LEAK", "Draft exposes the internal customer ID.", cid))
    for m in _CUSTOMER_ID_RE.finditer(text):
        if m.group(0) != cid:
            out.append(Violation("PII_LEAK", "Draft contains another customer's identifier.", m.group(0)))

    # Sensitive attributes.
    for label, pattern in _SENSITIVE_PATTERNS:
        m = pattern.search(text)
        if m:
            out.append(Violation("RESTRICTED_DATA", f"Draft references restricted attribute: {label}.", m.group(0)))

    # Exact restricted values from the record (e.g. the customer's payment method).
    pm = str(cust.get("PaymentMethod", "")).strip()
    if pm and pm.lower() in text.lower():
        out.append(Violation("RESTRICTED_DATA", "Draft reveals the customer's payment method.", pm))

    # Internal schema field names or campaign IDs.
    lowered = text.lower()
    for fld in _INTERNAL_FIELDS:
        if re.search(rf"\b{re.escape(fld)}\b", lowered) and ("_" in fld or fld in {"customerid", "seniorcitizen", "paymentmethod", "totalcharges", "monthlycharges"}):
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
            out.append(Violation("FACT_MISMATCH", f"Greeting uses name '{name}' but the verified first name is '{first_name}'.", m.group(0).strip()))

    tenure = _to_float(cust.get("tenure"))
    if tenure is not None:
        for m in _TENURE_RE.finditer(text):
            n, unit = _word_or_int(m.group(1)), m.group(2).lower()
            claimed_ok = (
                (unit == "month" and n <= tenure and tenure - n < 12)
                or (unit == "year" and n * 12 <= tenure < (n + 1) * 12)
            )
            if not claimed_ok:
                out.append(Violation("FACT_MISMATCH", f"Tenure claim '{m.group(0)}' contradicts verified tenure of {int(tenure)} months.", m.group(0)))

    tier = str(cust.get("loyalty_tier", "")).strip()
    if tier:
        for m in _TIER_RE.finditer(text):
            if m.group(1).lower() != tier.lower():
                out.append(Violation("FACT_MISMATCH", f"Draft claims '{m.group(1)}' tier but the verified tier is '{tier}'.", m.group(0)))

    return out


def _looks_like_date(s: str) -> bool:
    return bool(re.fullmatch(r"\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\s*|\s*\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\s*", s))


def _block(violations: list[Violation]) -> GuardrailResult:
    # Deduplicate while preserving order.
    seen, unique = set(), []
    for v in violations:
        key = (v.category, v.detail)
        if key not in seen:
            seen.add(key)
            unique.append(v)
    reason = "Message blocked by safety guardrails: " + "; ".join(v.detail for v in unique)
    return GuardrailResult(allowed=False, refusal_reason=reason, violations=unique)


# --------------------------------------------------------------------------- #
# Convenience wrapper for the orchestrator
# --------------------------------------------------------------------------- #


def guard_or_refuse(draft: str, customer: Mapping[str, Any], approved_offer_ids: Iterable[str], **kwargs: Any) -> tuple[str | None, str | None]:
    """Return (safe_draft, None) if allowed, else (None, refusal_reason)."""
    result = validate_email_draft(draft, customer, approved_offer_ids, **kwargs)
    return (draft, None) if result.allowed else (None, result.refusal_reason)


if __name__ == "__main__":
    demo_customer = {
        "customerID": "5575-GNVDE", "first_name": "Arjun", "tenure": "34", "loyalty_tier": "Silver",
        "PaymentMethod": "Mailed check", "MonthlyCharges": "56.95", "amount_due": "56.95",
        "bill_id": "B100001", "email": "5575-gnvde@example.com", "marketing_opt_in": "True",
    }
    good = "Hi Arjun,\n\nAs a Silver customer, you get a $10 credit on your next bill (OFF-LOY-SILV-01). Thank you!"
    bad = "Hi Arjun,\n\nAs a Gold member, use code SAVE50NOW for 50% off and $100 cash back. We see you pay by mailed check and are at risk of churn."
    for label, d in (("GOOD", good), ("BAD", bad)):
        r = validate_email_draft(d, demo_customer, ["OFF-LOY-SILV-01"], today=date(2026, 11, 17))
        print(label, "->", r.allowed, "|", r.refusal_reason)
