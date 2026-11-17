import os
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import make_msgid

import requests


def delivery_mode() -> str:
    return os.getenv("DELIVERY_MODE", "mock").strip().lower()


def resolve_recipient(customer_email: str):
    redirect = os.getenv("REDIRECT_ALL_TO", "").strip()
    if redirect:
        return redirect, True
    return customer_email, False


def send_smtp(sender: str, to: str, subject: str, body: str) -> str:
    host = os.environ["SMTP_HOST"].strip()
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").replace(" ", "")
    use_tls = os.getenv("SMTP_STARTTLS", "true").strip().lower() == "true"

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = subject
    msg["Message-ID"] = make_msgid(domain=sender.split("@")[-1])
    msg.set_content(body)

    with smtplib.SMTP(host, port, timeout=30) as s:
        if use_tls:
            s.starttls(context=ssl.create_default_context())
        if user:
            s.login(user, password)
        s.send_message(msg)
    return msg["Message-ID"]


def send_resend(to: str, subject: str, body: str) -> str:
    key = os.getenv("RESEND_API_KEY", "").strip()
    if not key:
        raise RuntimeError("RESEND_API_KEY is missing in .env")
    r = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "from": _resend_sender(),
            "to": [to],
            "subject": subject,
            "text": body,
        },
        timeout=30,
    )
    if r.status_code >= 300:
        raise RuntimeError(f"Resend {r.status_code}: {r.text}")
    return r.json().get("id", "")

def _resend_sender() -> str:
    sender = (os.getenv("RESEND_FROM") or "").strip().strip('"').strip("'").strip()
    if "@" not in sender or " " in sender.split("<")[-1]:
        sender = "onboarding@resend.dev"
    return sender