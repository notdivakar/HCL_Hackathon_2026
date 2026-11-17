"""
Personalized retention/loyalty email generation using Groq via LangChain.

The LLM writes only the greeting and body copy. Facts (offer value, terms,
balances) come from the data. The disclaimer is added in code after
generation, so the model cannot leave it out or change it. A check rejects
drafts that invent numbers, use banned language, or leak internal labels.
If no draft passes, a fixed template email is returned instead.
"""

import json
import os
import re
from typing import Any, Mapping

from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
COMPANY_NAME = os.getenv("COMPANY_NAME", "Your Telco")
MAX_ATTEMPTS = 2
SUBJECT_MAX_CHARS = 70
BODY_MAX_WORDS = 160

STANDARD_DISCLAIMER = (
    "This offer is personal to you and cannot be transferred or exchanged for cash. "
    "{terms} "
    "You are receiving this email because you opted in to communications from {company}. "
    "To stop receiving these emails, reply STOP or update your preferences in your account settings."
)

BANNED_PHRASES = [
    "churn", "risk score", "at risk", "segment", "propensity", "final notice",
    "legal action", "collections", "suspend", "disconnect", "penalty",
    "act now", "last chance", "urgent", "guaranteed", "!!!",
]

# Keys we never expose to the model (internal analytics / PII not needed for copy).
INTERNAL_FIELDS = {
    "Churn", "customerID", "email", "bill_id", "last_campaign_id",
    "contacts_last_30d", "usage_change_pct", "eligible_segments",
}


class EmailDraft(BaseModel):
    subject: str = Field(description="Email subject line, max 70 characters, no emojis.")
    body: str = Field(description="Email body (greeting through sign-off), plain text, no disclaimer.")


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are a customer communications writer for {company}, a telecom provider.
You write short, warm, professional, plain-text emails.

STRICT RULES (any violation makes the email unusable):
1. Tone: friendly, appreciative, respectful, calm. Never pushy, alarming, guilt-inducing, or salesy.
2. Use ONLY the facts provided in CUSTOMER and OFFER. Do not invent amounts, dates, percentages,
   plan names, services, or benefits. Quote the offer value exactly as given: "{offer_value}".
3. Never mention internal terms: churn, risk, segment, score, propensity, analytics, usage drop.
4. Never use urgency or pressure language ("act now", "last chance", "urgent", "final notice").
5. {balance_rule}
6. Do NOT write any disclaimer, terms, legal text, or unsubscribe text. That is appended automatically.
7. Subject: max {subject_max} characters, no emojis, no ALL CAPS, no exclamation marks.
8. Body: max {body_max} words. Start with "Hi {first_name}," and end with
   "Warm regards,\\nThe {company} Team". Plain text only, no markdown, no placeholders like [Name].
9. Do not include links or URLs.

Return ONLY a JSON object: {{"subject": "...", "body": "..."}}"""

HUMAN_PROMPT = """CUSTOMER:
{customer_json}

OFFER (approved; use exactly as described):
{campaign_json}

Write the personalized email now."""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_dict(row: Any) -> dict:
    """Accept dicts, pandas Series, or namedtuples."""
    if hasattr(row, "to_dict"):
        return row.to_dict()
    if hasattr(row, "_asdict"):
        return row._asdict()
    return dict(row)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _has_overdue_balance(c: Mapping) -> bool:
    return (
        str(c.get("status", "")).strip().lower() != "paid"
        and _num(c.get("days_overdue")) > 0
        and _num(c.get("amount_due")) > 0
    )


def _customer_context(c: Mapping) -> dict:
    """Keep only the fields useful for personalization."""
    tenure = int(_num(c.get("tenure")))
    ctx = {
        "first_name": c.get("first_name") or "there",
        "tenure_months": tenure,
        "tenure_years": round(tenure / 12, 1),
        "loyalty_tier": c.get("loyalty_tier"),
        "contract": c.get("Contract"),
        "internet_service": c.get("InternetService"),
        "autopay_enabled": _truthy(c.get("autopay")),
    }
    if _has_overdue_balance(c):
        ctx.update(
            {
                "outstanding_balance": f"{c.get('currency', 'USD')} {_num(c.get('amount_due')):.2f}",
                "balance_due_date": c.get("due_date"),
            }
        )
    return {k: v for k, v in ctx.items() if k not in INTERNAL_FIELDS and v not in (None, "")}


def _campaign_context(o: Mapping) -> dict:
    keys = ["offer_name", "offer_type", "offer_value", "description", "terms", "valid_to"]
    return {k: o.get(k) for k in keys if o.get(k) not in (None, "")}


def _balance_rule(c: Mapping) -> str:
    if _has_overdue_balance(c):
        return (
            "The customer has an outstanding balance (given in CUSTOMER). Include ONE gentle, "
            "non-judgmental sentence noting the balance amount and due date as a friendly reminder, "
            "placed AFTER the offer. Do not imply the offer depends on payment. No threats or fees."
        )
    return "Do NOT mention bills, balances, payments, or overdue amounts at all."


def _build_disclaimer(o: Mapping) -> str:
    terms = str(o.get("terms", "")).strip()
    if terms and not terms.endswith("."):
        terms += "."
    return STANDARD_DISCLAIMER.format(terms=terms, company=COMPANY_NAME).strip()


def _allowed_numbers(c: Mapping, o: Mapping) -> set[str]:
    """Numbers the model may legitimately use."""
    sources = [
        str(o.get("offer_value", "")), str(o.get("description", "")),
        str(o.get("terms", "")), str(o.get("valid_to", "")),
        str(c.get("tenure", "")), str(round(_num(c.get("tenure")) / 12, 1)),
        str(int(_num(c.get("tenure")) // 12)),
    ]
    if _has_overdue_balance(c):
        sources += [f"{_num(c.get('amount_due')):.2f}", str(c.get("amount_due")), str(c.get("due_date"))]
    allowed = set()
    for s in sources:
        allowed.update(re.findall(r"\d+(?:\.\d+)?", s))
    return allowed


def _validate(draft: EmailDraft, c: Mapping, o: Mapping) -> list[str]:
    errors = []
    subject, body = draft.subject.strip(), draft.body.strip()
    text = f"{subject}\n{body}"
    lower = text.lower()

    if not subject or not body:
        errors.append("empty subject or body")
    if len(subject) > SUBJECT_MAX_CHARS:
        errors.append(f"subject exceeds {SUBJECT_MAX_CHARS} chars")
    if "!" in subject:
        errors.append("exclamation mark in subject")
    if len(body.split()) > BODY_MAX_WORDS:
        errors.append(f"body exceeds {BODY_MAX_WORDS} words")
    if re.search(r"https?://|www\.", lower):
        errors.append("contains a link")
    if re.search(r"\[[^\]]+\]|\{[^}]+\}", text):
        errors.append("contains unfilled placeholder")

    for phrase in BANNED_PHRASES:
        if phrase in lower:
            errors.append(f"banned phrase: '{phrase}'")

    offer_value = str(o.get("offer_value", "")).strip()
    if offer_value and offer_value.lower() not in lower:
        errors.append(f"offer value '{offer_value}' not quoted exactly")

    first_name = str(c.get("first_name") or "").strip()
    if first_name and first_name.lower() not in body.lower():
        errors.append("customer first name missing")

    allowed = _allowed_numbers(c, o)
    for n in re.findall(r"\d+(?:\.\d+)?", text):
        if n not in allowed:
            errors.append(f"unverified number: {n}")

    if not _has_overdue_balance(c) and re.search(r"\b(overdue|balance|past due|outstanding)\b", lower):
        errors.append("mentions balance for customer with no overdue amount")

    if re.search(r"\b(unsubscribe|reply stop|terms and conditions)\b", lower):
        errors.append("model wrote its own disclaimer")

    return errors


def _parse(raw: str) -> EmailDraft:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in model output")
    return EmailDraft(**json.loads(match.group(0)))


def _fallback_email(c: Mapping, o: Mapping) -> EmailDraft:
    """Deterministic, compliant template used if the LLM cannot pass validation."""
    name = c.get("first_name") or "there"
    tenure = int(_num(c.get("tenure")))
    tenure_line = (
        f"Thank you for being with us for {tenure} months." if tenure >= 1
        else "Thank you for choosing us."
    )
    body = (
        f"Hi {name},\n\n{tenure_line} As a thank-you, we'd like to offer you "
        f"{o.get('offer_name')}: {o.get('description')}\n"
    )
    if _has_overdue_balance(c):
        body += (
            f"\nAs a friendly reminder, your balance of {c.get('currency', 'USD')} "
            f"{_num(c.get('amount_due')):.2f} was due on {c.get('due_date')}.\n"
        )
    body += f"\nWarm regards,\nThe {COMPANY_NAME} Team"
    subject = f"A thank-you from {COMPANY_NAME}: {o.get('offer_value')}"[:SUBJECT_MAX_CHARS]
    return EmailDraft(subject=subject, body=body)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_customer_email(customer_row: Any, campaign_row: Any, llm: ChatGroq | None = None) -> dict:
    """
    Generate a personalized, compliant email for one customer and one approved offer.

    Args:
        customer_row: dict / pandas Series with customer fields
                      (first_name, tenure, loyalty_tier, amount_due, days_overdue, status, ...).
        campaign_row: dict / pandas Series with offer fields
                      (offer_name, offer_value, description, terms, status, ...).
        llm: optional pre-built ChatGroq client (reuse across calls).

    Returns:
        {"subject", "body", "disclaimer", "full_email", "source", "validation_errors"}
        source is "llm" or "fallback_template".
    """
    customer = _to_dict(customer_row)
    campaign = _to_dict(campaign_row)

    # Hard pre-checks: never generate for non-approved offers or non-consenting customers.
    if str(campaign.get("status", "active")).strip().lower() != "active":
        raise ValueError(f"Campaign {campaign.get('offer_id')} is not approved/active.")
    if _truthy(campaign.get("requires_marketing_consent", False)) and not _truthy(
        customer.get("marketing_opt_in", False)
    ):
        raise PermissionError(f"Customer {customer.get('customerID')} has not given marketing consent.")
    if not _truthy(customer.get("email_opt_in", True)):
        raise PermissionError(f"Customer {customer.get('customerID')} has opted out of email.")

    llm = llm or ChatGroq(model=GROQ_MODEL, temperature=0.3, max_tokens=600)
    chain = PROMPT | llm

    inputs = {
        "company": COMPANY_NAME,
        "offer_value": campaign.get("offer_value", ""),
        "first_name": customer.get("first_name") or "there",
        "balance_rule": _balance_rule(customer),
        "subject_max": SUBJECT_MAX_CHARS,
        "body_max": BODY_MAX_WORDS,
        "customer_json": json.dumps(_customer_context(customer), indent=2, default=str),
        "campaign_json": json.dumps(_campaign_context(campaign), indent=2, default=str),
    }

    draft, errors, source = None, [], "llm"
    for _ in range(MAX_ATTEMPTS):
        try:
            candidate = _parse(chain.invoke(inputs).content)
            errors = _validate(candidate, customer, campaign)
        except Exception as exc:  # parsing / API errors
            candidate, errors = None, [f"generation error: {exc}"]
        if candidate and not errors:
            draft = candidate
            break
        # Feed violations back for a corrective retry.
        inputs["customer_json"] += (
            "\n\nYOUR PREVIOUS DRAFT WAS REJECTED FOR: " + "; ".join(errors) + ". Fix these issues."
        )

    if draft is None:
        draft, source = _fallback_email(customer, campaign), "fallback_template"

    disclaimer = _build_disclaimer(campaign)
    body = draft.body.strip()
    return {
        "subject": draft.subject.strip(),
        "body": body,
        "disclaimer": disclaimer,
        "full_email": f"{body}\n\n---\n{disclaimer}",
        "source": source,
        "validation_errors": errors if source == "fallback_template" else [],
    }


# --------------------------------------------------------------------------- #
# Example
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    customer = {
        "customerID": "5575-GNVDE", "first_name": "Arjun", "tenure": 34,
        "loyalty_tier": "Silver", "Contract": "One year", "InternetService": "DSL",
        "amount_due": 56.95, "currency": "USD", "due_date": "2026-11-08",
        "status": "Unpaid", "days_overdue": 9, "autopay": "False",
        "email_opt_in": "True", "marketing_opt_in": "True",
    }
    campaign = {
        "offer_id": "OFF-LOY-SILV-01", "offer_name": "Silver Loyalty Credit",
        "offer_type": "reward", "offer_value": "$10 credit",
        "description": "As a Silver customer, you get a $10 credit on your next bill.",
        "terms": "Credit applied to your next bill. Redeem by 31 Jan 2027. One per customer.",
        "valid_to": "2027-01-31", "requires_marketing_consent": "True", "status": "active",
    }
    result = generate_customer_email(customer, campaign)
    print("SUBJECT:", result["subject"])
    print(result["full_email"])
    print("SOURCE:", result["source"])
