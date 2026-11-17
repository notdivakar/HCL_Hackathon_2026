"""
Email generation (owner: feature-llm).

The LLM writes ONLY the subject and the body copy. Everything factual
(offer value, terms, amount, due date, bill ID) comes from verified data,
and the footer (disclaimer / dispute path) is appended in code so the model
cannot drop it.

Message types:
  offer     -> appreciation / reward / re-engagement with ONE approved offer
  no_offer  -> thank-you or check-in when no offer is eligible
  reminder  -> billing reminder; tone set by reminder stage 0-3

If GROQ_API_KEY is not set, or the model fails validation twice, a fixed,
pre-approved template is used instead (source = "template").
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping

from pydantic import BaseModel, Field

from src.policy import load_policy

try:  # load .env if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

POLICY = load_policy()
COMPANY_NAME = os.getenv("COMPANY_NAME", POLICY["company"]["name"])
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", POLICY["company"]["support_email"])
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
PROMPT_VERSION = "prompt-v2.0"
TEMPLATE_VERSION = "template-v2.0"
MAX_ATTEMPTS = 2
SUBJECT_MAX_CHARS = 70
BODY_MAX_WORDS = 160


class EmailDraft(BaseModel):
    subject: str = Field(description="Email subject line, max 70 characters, no emojis.")
    body: str = Field(description="Email body from greeting to sign-off, plain text, no footer.")


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are a customer communications writer for {company}, a telecom provider.
You write short, plain-text emails that a human reviewer will check before sending.

GOAL OF THIS EMAIL: {objective}
TONE: {tone}

STRICT RULES (breaking any rule makes the email unusable):
1. Use ONLY the facts in FACTS. Do not invent amounts, dates, percentages, plan names,
   services, benefits, codes or links.
2. {content_rule}
3. Never mention: churn, risk, segment, score, usage drop, login history, analytics,
   age, gender, family, payment method, or any internal field names.
4. No pressure or threats: never say "act now", "last chance", "urgent", "legal action",
   "collections", "disconnect", "penalty", "late fee", "credit score".
5. Do NOT write any disclaimer, unsubscribe text, or dispute/support text. It is added automatically.
6. Subject: max {subject_max} characters, no emojis, no ALL CAPS, no exclamation marks.
7. Body: max {body_max} words. Start with "Hi {first_name}," and end with
   "Warm regards,\\nThe {company} Team". Plain text, no markdown, no placeholders.

Return ONLY a JSON object: {{"subject": "...", "body": "..."}}"""

HUMAN_PROMPT = """FACTS:
{facts_json}

Write the email now."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_dict(row: Any) -> dict:
    if hasattr(row, "to_dict"):
        return row.to_dict()
    if hasattr(row, "_asdict"):
        return row._asdict()
    return dict(row or {})


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _money(c: Mapping) -> str:
    return f"${_num(c.get('amount_due')):.2f}"


def _facts(c: Mapping, message_type: str, offer: Mapping | None, stage: Mapping | None) -> dict:
    """Minimal, non-sensitive facts the model may use (no protected attributes)."""
    tenure = int(_num(c.get("tenure")))
    facts: dict[str, Any] = {
        "first_name": c.get("first_name") or "there",
        "months_with_us": tenure,
        "loyalty_tier": c.get("loyalty_tier"),
        "contract": c.get("Contract"),
    }
    if message_type == "offer" and offer:
        facts["offer"] = {k: offer.get(k) for k in
                          ("offer_name", "offer_value", "description", "terms") if offer.get(k)}
    if message_type == "reminder":
        facts["bill"] = {
            "bill_id": c.get("bill_id"),
            "amount_due": _money(c),
            "due_date": str(c.get("due_date")),
            "days_overdue": int(_num(c.get("days_overdue"))),
            "autopay_enabled": _truthy(c.get("autopay")),
        }
        facts["reminder_stage"] = stage["name"] if stage else None
    return facts


def _content_rule(message_type: str, offer: Mapping | None, stage: Mapping | None) -> str:
    if message_type == "offer":
        return (f'Thank the customer and present the offer. Quote the offer value exactly as "{offer.get("offer_value")}". '
                "Do not mention bills, balances or payments.")
    if message_type == "no_offer":
        return "Thank the customer for being with us. Do NOT mention any offer, discount, bill or payment."
    days = int(stage["min_days_overdue"]) if stage else 0
    rule = (f"State the bill ID, the amount due and the due date exactly as given in FACTS.bill. "
            f"Ask the customer to pay or reply if they need help. Do not mention any offer.")
    if days == 0:
        rule += " The bill is not late yet; this is only a friendly heads-up."
    if stage and stage["stage"] < 3:
        rule += ' Do not use the words "final" or "final notice".'
    else:
        rule += " You may say this is a final reminder."
    return rule


def _allowed_numbers(c: Mapping, offer: Mapping | None, message_type: str) -> set[str]:
    sources = [str(int(_num(c.get("tenure")))), str(int(_num(c.get("tenure")) // 12))]
    if offer and message_type == "offer":
        sources += [str(offer.get(k, "")) for k in ("offer_value", "description", "terms", "valid_to", "offer_name")]
    if message_type == "reminder":
        sources += [_money(c), str(c.get("amount_due")), str(c.get("due_date")),
                    str(int(_num(c.get("days_overdue")))), str(c.get("bill_id"))]
    allowed: set[str] = set()
    for s in sources:
        allowed.update(re.findall(r"\d+(?:\.\d+)?", s))
    return allowed


def _quick_validate(d: EmailDraft, c: Mapping, offer: Mapping | None, message_type: str) -> list[str]:
    """Cheap checks used to trigger a retry. The full check is src.guardrails."""
    errors = []
    subject, body = d.subject.strip(), d.body.strip()
    text = f"{subject}\n{body}"
    if not subject or not body:
        errors.append("empty subject or body")
    if len(subject) > SUBJECT_MAX_CHARS:
        errors.append(f"subject longer than {SUBJECT_MAX_CHARS} characters")
    if "!" in subject:
        errors.append("exclamation mark in subject")
    if len(body.split()) > BODY_MAX_WORDS:
        errors.append(f"body longer than {BODY_MAX_WORDS} words")
    first = str(c.get("first_name") or "").strip()
    if first and first.lower() not in body.lower():
        errors.append("first name missing")
    if message_type == "offer" and offer:
        val = str(offer.get("offer_value", "")).strip()
        if val and val.lower() not in text.lower():
            errors.append(f"offer value '{val}' not quoted exactly")
    if message_type == "reminder":
        if _money(c) not in text:
            errors.append(f"amount {_money(c)} not stated exactly")
        if str(c.get("bill_id")) not in text:
            errors.append("bill ID missing")
    allowed = _allowed_numbers(c, offer, message_type)
    for n in re.findall(r"\d+(?:\.\d+)?", text):
        if n not in allowed:
            errors.append(f"unverified number {n}")
    return errors


def _parse(raw: str) -> EmailDraft:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in model output")
    return EmailDraft(**json.loads(m.group(0)))


# --------------------------------------------------------------------------- #
# Deterministic templates (used when no LLM or LLM fails)
# --------------------------------------------------------------------------- #
def _template(c: Mapping, message_type: str, segment: str, offer: Mapping | None, stage: Mapping | None) -> EmailDraft:
    name = c.get("first_name") or "there"
    tenure = int(_num(c.get("tenure")))
    sign = f"\n\nWarm regards,\nThe {COMPANY_NAME} Team"
    thanks = f"Thank you for being with us for {tenure} months." if tenure >= 1 else "Thank you for choosing us."

    if message_type == "reminder":
        s = stage["stage"] if stage else 1
        bill, amt, due = c.get("bill_id"), _money(c), c.get("due_date")
        if s == 0:
            subject = f"Your bill {bill} is coming due"
            body = (f"Hi {name},\n\nJust a friendly heads-up that your bill {bill} for {amt} is due on {due}. "
                    "If you have autopay or have already paid, there is nothing more to do.")
        elif s == 1:
            subject = f"A quick reminder about bill {bill}"
            body = (f"Hi {name},\n\nWe noticed your bill {bill} for {amt}, due on {due}, has not been paid yet. "
                    "It may simply have slipped through. Please pay when you can, or reply if you need any help.")
        elif s == 2:
            subject = f"Payment needed for bill {bill}"
            body = (f"Hi {name},\n\nYour bill {bill} for {amt} was due on {due} and is still unpaid. "
                    "Please make the payment as soon as possible. If something is stopping you from paying, "
                    "reply to this email and our team will help you find a way forward.")
        else:
            subject = f"Final reminder: bill {bill}"
            body = (f"Hi {name},\n\nThis is a final reminder that your bill {bill} for {amt}, due on {due}, "
                    "remains unpaid. Please pay it now, or contact us so we can talk through options with you.")
        return EmailDraft(subject=subject, body=body + sign)

    if message_type == "offer" and offer:
        if segment in {"Declining", "Dormant"}:
            subject = f"Something for you from {COMPANY_NAME}: {offer.get('offer_value')}"
            body = f"Hi {name},\n\n{thanks} {offer.get('description')}"
        else:
            subject = f"A thank-you from {COMPANY_NAME}: {offer.get('offer_value')}"
            body = f"Hi {name},\n\n{thanks} As a thank-you: {offer.get('description')}"
        return EmailDraft(subject=subject[:SUBJECT_MAX_CHARS], body=body + sign)

    subject = f"Thank you from {COMPANY_NAME}"
    body = (f"Hi {name},\n\n{thanks} We are glad to have you with us. "
            "If there is anything we can help with, just reply to this email.")
    return EmailDraft(subject=subject, body=body + sign)


def build_footer(message_type: str, offer: Mapping | None) -> str:
    footers = POLICY["footers"]
    if message_type == "reminder":
        return footers["reminder"].format(support_email=SUPPORT_EMAIL, company=COMPANY_NAME)
    if message_type == "offer" and offer:
        terms = str(offer.get("terms", "")).strip()
        if terms and not terms.endswith("."):
            terms += "."
        return footers["offer"].format(terms=terms, company=COMPANY_NAME)
    return footers["no_offer"].format(company=COMPANY_NAME)


_LLM_CACHE: dict[str, Any] = {}


def get_llm():
    """Return a ChatGroq client, or None if no API key is configured."""
    if not os.getenv("GROQ_API_KEY"):
        return None
    if "llm" not in _LLM_CACHE:
        from langchain_groq import ChatGroq
        _LLM_CACHE["llm"] = ChatGroq(model=GROQ_MODEL, temperature=0.3, max_tokens=600)
    return _LLM_CACHE["llm"]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_email(
    customer_row: Any,
    *,
    segment: str,
    message_type: str,
    objective: str,
    tone: str,
    offer: Mapping | None = None,
    stage: Mapping | None = None,
    llm: Any = "auto",
) -> dict:
    """
    Generate one draft. Returns:
      subject, body, footer, full_email, source ("llm" | "template"),
      model, prompt_version, template_version, attempts, llm_errors
    """
    c = _to_dict(customer_row)
    offer = _to_dict(offer) if offer is not None else None
    if message_type == "offer" and not offer:
        message_type = "no_offer"
    if llm == "auto":
        llm = get_llm()

    draft, errors, source, attempts = None, [], "template", 0
    if llm is not None:
        from langchain_core.prompts import ChatPromptTemplate
        chain = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]) | llm
        inputs = {
            "company": COMPANY_NAME,
            "objective": objective,
            "tone": stage["tone"] if (message_type == "reminder" and stage) else tone,
            "content_rule": _content_rule(message_type, offer, stage),
            "first_name": c.get("first_name") or "there",
            "subject_max": SUBJECT_MAX_CHARS,
            "body_max": BODY_MAX_WORDS,
            "facts_json": json.dumps(_facts(c, message_type, offer, stage), indent=2, default=str),
        }
        for _ in range(MAX_ATTEMPTS):
            attempts += 1
            try:
                candidate = _parse(chain.invoke(inputs).content)
                errors = _quick_validate(candidate, c, offer, message_type)
            except Exception as exc:  # API / parsing errors
                candidate, errors = None, [f"generation error: {exc}"]
            if candidate and not errors:
                draft, source = candidate, "llm"
                break
            inputs["facts_json"] += "\n\nYOUR PREVIOUS DRAFT WAS REJECTED FOR: " + "; ".join(errors) + ". Fix these."

    if draft is None:
        draft = _template(c, message_type, segment, offer, stage)

    footer = build_footer(message_type, offer)
    body = draft.body.strip()
    return {
        "subject": draft.subject.strip(),
        "body": body,
        "footer": footer,
        "full_email": f"{body}\n\n---\n{footer}",
        "message_type": message_type,
        "source": source,
        "model": GROQ_MODEL if source == "llm" else "none (template)",
        "prompt_version": PROMPT_VERSION,
        "template_version": TEMPLATE_VERSION,
        "attempts": attempts,
        "llm_errors": errors if source == "template" else [],
    }


if __name__ == "__main__":
    customer = {
        "customerID": "5575-GNVDE", "first_name": "Arjun", "tenure": 34, "loyalty_tier": "Silver",
        "Contract": "One year", "amount_due": 56.95, "due_date": "2026-11-08", "bill_id": "B100001",
        "status": "Overdue", "days_overdue": 9,
    }
    from src.policy import reminder_stage_for
    out = generate_email(customer, segment="Payment Reminder", message_type="reminder",
                         objective="Billing reminder", tone="", stage=reminder_stage_for(9))
    print(out["subject"])
    print(out["full_email"])
    print("SOURCE:", out["source"])
