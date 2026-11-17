# Customer Loyalty & Engagement Assistant
**Team Natural Intelligence** · HCLTech Future Ready AI Engineer Hackathon 2026 · Use Case 5

A GenAI assistant for a telecom company. It sorts customers into explainable segments, picks an approved strategy and offer, lets an LLM write a personalised email from **verified facts only**, checks every draft with deterministic guardrails, and sends it **only after a human approves it**: as a mock, through Gmail SMTP, or through the **Resend** email API.

> **AI writes the words. Code decides the facts. A human approves every send.**

| Team member | Branch | Owns |
|---|---|---|
| Sandip | `feature-llm` | Prompting, LLM calls, templates |
| Kristan | `feature-api` | Backend API, segmentation, offer rules |
| Sharan | `feature-ui` | Reviewer dashboard |
| Divakar | `feature-safety` | Guardrails, policy, delivery safety |

---

## Contents
1. [Quick start](#1-quick-start)
2. [Configuration (`.env`)](#2-configuration-env)
3. [Email delivery: mock, Gmail SMTP, Resend](#3-email-delivery-mock-gmail-smtp-resend)
4. [Architecture](#4-architecture)
5. [Project structure](#5-project-structure)
6. [Segmentation and strategy](#6-segmentation-and-strategy)
7. [LLM and prompting](#7-llm-and-prompting)
8. [Safety guardrails](#8-safety-guardrails)
9. [Human review UI](#9-human-review-ui)
10. [Audit and observability](#10-audit-and-observability)
11. [API reference](#11-api-reference)
12. [Demo scenarios](#12-demo-scenarios)
13. [Evaluation](#13-evaluation)
14. [Troubleshooting](#14-troubleshooting)
15. [Assumptions and limitations](#15-assumptions-and-limitations)

---

## 1. Quick start

**Prerequisites:** Python 3.11 or newer, and a Groq API key (optional; without one the app uses approved templates).

```powershell
# 1. Install dependencies (run inside the project folder)
python -m pip install -r requirements.txt

# 2. Create your settings file
Copy-Item .env.example .env        # macOS/Linux: cp .env.example .env
# then edit .env (see section 2)

# 3. Terminal 1: backend
python -m uvicorn backend.api:app --reload --port 8000

# 4. Terminal 2: reviewer dashboard
python -m streamlit run frontend/app.py
```

- Dashboard: **http://localhost:8501**
- Backend health check: **http://localhost:8000/health**
- Interactive API docs: **http://localhost:8000/docs**

> Always start the dashboard with `streamlit run`. The VS Code ▶ Run button runs it as plain Python, which does not work.
> The backend reads `.env` only at start-up, so restart it after any `.env` change.

---

## 2. Configuration (`.env`)

All secrets and settings live in `.env`. It is in `.gitignore` and must **never** be committed. `.env.example` is the template.

| Variable | Example | Purpose |
|---|---|---|
| `GROQ_API_KEY` | `gsk_...` | LLM access. Leave empty to run in template mode |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model name |
| `COMPANY_NAME` | `Your Telco` | Used in emails and footers |
| `SUPPORT_EMAIL` | `support@yourtelco.example` | Shown in the dispute and support line |
| `SENDER_EMAIL` | `team@gmail.com` | "From" shown in the UI; the real sender for SMTP |
| `DELIVERY_MODE` | `mock` / `smtp` / `resend` | How approved emails are delivered (section 3) |
| `REDIRECT_ALL_TO` | `you@gmail.com` | Every real email goes to this one inbox |
| `RESEND_API_KEY` | `re_...` | Resend mode only |
| `RESEND_FROM` | `onboarding@resend.dev` | Resend mode only |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_STARTTLS` | `smtp.gmail.com`, `587`, `true` | SMTP mode only |
| `SMTP_USER`, `SMTP_PASSWORD` | Gmail address, 16-character app password | SMTP mode only |
| `APP_TODAY` | `2026-11-17` | Pins "today" so offer-expiry checks are reproducible |
| `API_URL` | `http://localhost:8000` | Where the dashboard finds the backend |

**Rules:** each variable appears **once**; no quotes; no spaces around `=` or inside keys.

---

## 3. Email delivery: mock, Gmail SMTP, Resend

Delivery is controlled by `DELIVERY_MODE` and implemented in `src/mailer.py`. In every mode the email is sent **only after** a named reviewer approves it and the send gate passes (section 8).

| Mode | What happens | Best for |
|---|---|---|
| `mock` *(default)* | Nothing leaves the machine. The approved email is saved to `outbox/` and shown in the **Mock outbox** tab | Brief compliance, offline demos |
| `resend` | Real email through the **Resend HTTP API** (port 443) | Live demos; works on Wi-Fi that blocks SMTP |
| `smtp` | Real email through an SMTP server, e.g. Gmail | Live demos where port 587 is open |

### Safety in real-send modes
- The dataset's customer addresses are fake (`@example.com`), so real sends **require** `REDIRECT_ALL_TO`. Every email goes to that inbox instead.
- A redirected email starts with `[DEMO] This email was meant for <customer>; redirected to the test inbox.`
- Sending to an `@example.com` address is refused with a clear error.
- Send failures are shown in the UI and logged as `send_failed` in the audit log.

### Set up Resend (recommended for live demos)
1. Sign up at **resend.com** with the inbox you will show in the demo.
2. Go to **API Keys → Create API Key** (Sending access) and copy it with the copy icon. It is shown only once.
3. In `.env`:
   ```
   DELIVERY_MODE=resend
   RESEND_API_KEY=re_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   RESEND_FROM=onboarding@resend.dev
   REDIRECT_ALL_TO=the-email-you-signed-up-with@gmail.com
   ```
4. Restart the backend and check that `/health` shows `"delivery_mode": "resend"`.

Resend's free test sender `onboarding@resend.dev` can deliver **only to the email address that owns the Resend account**, so `REDIRECT_ALL_TO` must be that address. Sending to anyone else needs a verified domain. Sent emails also appear in the Resend dashboard under **Emails**, which is useful as delivery proof.

If `RESEND_FROM` is empty, quoted or malformed, the code falls back to `onboarding@resend.dev`.

### Set up Gmail SMTP
1. Use a spare Gmail account and turn on **2-Step Verification**.
2. Create an **app password** at myaccount.google.com/apppasswords. Your normal password will not work.
3. In `.env`:
   ```
   DELIVERY_MODE=smtp
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_STARTTLS=true
   SMTP_USER=yourteam@gmail.com
   SMTP_PASSWORD=abcdefghijklmnop
   SENDER_EMAIL=yourteam@gmail.com
   REDIRECT_ALL_TO=your.inbox@gmail.com
   ```
   `SMTP_HOST` is always `smtp.gmail.com`, not your email address. `SENDER_EMAIL` must equal `SMTP_USER`.

> Outlook.com and Hotmail no longer allow password or app-password SMTP sign-in, so they are not supported.

---

## 4. Architecture

```
                 ┌──────────────────── backend/api.py (FastAPI) ────────────────────┐
 Streamlit       │ 1  Profile        minimal fields; protected attributes removed    │
 reviewer   ───► │ 2  Classify       deterministic rules → segment, reasons, conf.   │
 dashboard       │ 3  Strategy       policy.json → objective + tone                  │
 (frontend/      │ 4  Offer          deterministic eligibility from the catalogue    │
  app.py)        │ 5  Pre-checks     consent, dispute, contact cap, bad data ────────┼─► REFUSE (no LLM call)
                 │ 6  Generate       Groq LLM (or template); footer added in code    │
                 │ 7  Guardrails     groundedness + policy + privacy checks ─────────┼─► BLOCK
                 │ 8  Human review   approve / edit (re-checked) / reject            │
                 │ 9  Send gate      approval + checks + consent re-verified ────────┼─► mock │ SMTP │ Resend
                 │ 10 Audit          logs/audit_log.jsonl + outbox/                  │
                 └───────────────────────────────────────────────────────────────────┘
```

**Design principle:** segment, offer eligibility, reminder stage and every fact in the email are **deterministic** (as the brief requires). The LLM is used **only for wording**.

**Tech stack:** Python 3.12 · FastAPI + Uvicorn · Streamlit · pandas · Pydantic · LangChain + Groq · python-dotenv · smtplib / Resend API · requests.

---

## 5. Project structure

```
HCL_Hackathon_2026/
├── backend/
│   └── api.py                 FastAPI app: segmentation, offer rules, pipeline, send, audit
├── frontend/
│   └── app.py                 Streamlit reviewer dashboard
├── src/
│   ├── policy.py              Loads config/policy.json; picks the reminder stage
│   ├── guardrails.py          Pre-checks, draft validation, send gate
│   ├── llm_orchestrator.py    Prompt, Groq call, retry, templates, footers
│   └── mailer.py              Delivery: mock / SMTP / Resend, recipient redirect
├── config/
│   └── policy.json            ALL business rules: thresholds, tones, stages, limits, blocked words
├── data/
│   ├── Telco-Customer-Churn-merged.csv   7,043 customers (raw)
│   ├── cleaned_telecom_data.csv          cleaned copy used by the API (falls back to raw)
│   └── offer_catalouge.csv               25 offers (incl. 2 expired, 1 paused)
├── scripts/
│   └── evaluate.py            Builds reports/evaluation_report.md
├── reports/
│   └── evaluation_report.md   Evaluation results
├── logs/                      audit_log.jsonl (created at runtime, git-ignored)
├── outbox/                    one JSON per sent email (created at runtime)
├── test_groq.py               Checks the Groq key and model
├── requirements.txt
├── .env.example               Settings template (no secrets)
└── README.md
```

---

## 6. Segmentation and strategy

Rules are in `backend/api.py` (`classify`) and thresholds in `config/policy.json`. **The first matching rule wins.** Protected attributes (`gender`, `SeniorCitizen`, `Partner`, `Dependents`) are never read.

| # | Segment | Rule | Strategy | Customers |
|---|---|---|---|---|
| 1 | Payment Reminder | Bill `Due` or `Overdue` | Staged billing reminder | 1,791 |
| 2 | Dormant | No login ≥ 60 days **or** zero usage in 30 days | Win-back offer | 801 |
| 3 | Declining | Usage down ≥ 25% **or** no login ≥ 30 days | Re-engagement offer | 668 |
| 4 | Loyal | Tenure ≥ 12 months **and** (1/2-year contract **or** autopay) | Reward | 2,557 |
| 5 | Consistent | Everyone else (paid, stable) | Thank-you | 1,226 |

Each result includes **reason codes** (e.g. `R_USAGE_DROP`, `R_REMINDER_STAGE_2`, `R_DISPUTE_OPEN`) and a **confidence** score.

### Reminder stages
| Stage | Days overdue | Tone |
|---|---|---|
| 0 | 0 (due) | Courteous heads-up |
| 1 | 1–15 | Polite reminder |
| 2 | 16–30 | Firm but respectful |
| 3 | 31+ | Final reminder, still courteous |

Every reminder ends with an automatic support and dispute line. Reminders never include offers.

### Offer eligibility (deterministic)
An offer is chosen only if it matches the segment, tier, tenure range, plan status and required service, is `active` and within its dates, and the customer gave marketing consent. The highest-priority match wins. The reviewer also sees **why other offers were excluded** (e.g. `OFF-LOY-EXP-01: status expired`).

---

## 7. LLM and prompting

Implemented in `src/llm_orchestrator.py`.

- **Model:** Groq `GROQ_MODEL` (default `openai/gpt-oss-120b`), temperature 0.3.
- **Input:** a minimal **FACTS** JSON: first name, months with us, tier, contract, plus **either** the approved offer **or** the bill (bill ID, amount, due date, days overdue). No protected or internal fields.
- **Prompt rules:** use only FACTS; quote the offer value exactly; no internal terms (churn, risk, segment); no pressure or threats; no disclaimers; subject ≤ 70 characters; body ≤ 160 words; return JSON `{"subject": ..., "body": ...}`.
- **Self-check and retry:** a quick check (exact offer value or amount, bill ID, name, no unverified numbers) triggers **one retry** with the errors fed back.
- **Fallback:** if the LLM is unavailable or fails twice, a **pre-approved template** for that segment and stage is used (`source = "template"`).
- **Footers are added in code**, so the model cannot drop them: offer terms and unsubscribe text, or the dispute and support line for reminders.
- **Versions** (`prompt_version`, `template_version`, model) are recorded in the audit log.

> With `gpt-oss-120b`, reasoning tokens count toward the output limit. If drafts often fall back to templates, raise `max_tokens` in `get_llm()` from `600` to `2000`.

---

## 8. Safety guardrails

Implemented in `src/guardrails.py`. All checks are deterministic. Severities: **BLOCK** (never sent), **ESCALATE** (never sent; routed to a human team), **WARN** (reviewer decides).

### Layer 1: pre-checks (before the LLM is called)
| Check | Result |
|---|---|
| Missing name, email or customer ID | BLOCK |
| No email consent | BLOCK |
| No marketing consent (offers and thank-yous) | BLOCK |
| ≥ 3 contacts in 30 days, or last contact < 7 days ago (marketing) | BLOCK |
| Open billing dispute (reminders) | **ESCALATE** to the billing team |
| Contradictory bill data (e.g. "Overdue" but 0 days late, or paid date on an unpaid bill) | BLOCK |
| Customer prefers SMS | WARN |

### Layer 2: draft checks (after generation, and again after every edit)
- **Groundedness:** every `$` amount, `%`, data allowance, free period and **date** must come from the approved offer or the customer's bill.
- **Offers:** unknown, expired, paused, wrong-tier, wrong-tenure or unapproved offers; fake promo codes.
- **Privacy:** other emails, phone or card numbers, customer IDs, internal field names, campaign IDs, payment method.
- **Fairness and sensitivity:** age, gender, family, churn or "at risk" wording, behavioural tracking language.
- **Tone:** blocked phrases per message type and stage (e.g. "final notice" before stage 3, "disconnect", "late fee", "legal action").
- **Reminders** must include the dispute and support path, and must not include offers.
- **Facts:** wrong name in the greeting, wrong tier, wrong tenure.

### Layer 3: send gate (at delivery)
- A **named reviewer** must approve.
- The final (possibly edited) text must pass Layer 2 again.
- Pre-checks and consent are re-verified.

---

## 9. Human review UI

`frontend/app.py` (Streamlit):
- **Header:** backend status, LLM on/off, sender, policy version, delivery mode and redirect inbox.
- **Sidebar:** reviewer name (required to send), demo scenarios, segment filter, search, list size.
- **Left:** segment, confidence, reason codes, strategy, reminder stage, customer evidence, and eligible vs excluded offers.
- **Right:** draft status, From/To, editable subject and body, the locked footer, pre-check and draft-check results, and **Approve & send** / **Reject** (reject needs a reason).
- **Tabs:** **Audit log** and **Mock outbox**.

Approve is disabled while any BLOCK or ESCALATE finding exists.

---

## 10. Audit and observability

Every event is appended to `logs/audit_log.jsonl`:

| Event | Key fields |
|---|---|
| `draft_generated` | segment, confidence, reason codes, offer, model, prompt/template version, evidence fields, check results, subject, body |
| `generation_refused` | pre-check reasons |
| `send_blocked` | reviewer, whether edited, gate findings |
| `send_failed` | delivery mode, error |
| `sent` | reviewer, whether edited, feedback, final text, send status, message ID, **hashed** recipient |
| `rejected` | reviewer, reason |

Every sent email is also saved to `outbox/<id>.json` with From, To, delivered-to address, subject, body and delivery method.

---

## 11. API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Status, LLM enabled, sender, policy version, delivery mode, redirect inbox |
| GET | `/segments` | Customer count per segment |
| GET | `/customers?segment=&search=&limit=&offset=` | Customer list |
| GET/POST | `/process-customer` | Segment, reasons, strategy, eligible and excluded offers |
| POST | `/generate-email` `{customer_id}` | Full pipeline; returns a draft (`PENDING_REVIEW` / `BLOCKED`) or `REFUSED` |
| POST | `/validate-email` `{draft_id, subject, body}` | Re-check edited text |
| POST | `/send-email` `{draft_id, subject, body, reviewer, feedback}` | Send gate, then delivery |
| POST | `/reject-email` `{draft_id, reviewer, reason}` | Reject and log |
| GET | `/audit`, `/outbox` | Recent audit events and sent emails |
| GET | `/scenarios` | Demo customers for every required scenario |

Full interactive docs: **http://localhost:8000/docs**.

---

## 12. Demo scenarios

Choose these from **Demo scenarios** in the sidebar:

| Scenario | Expected result |
|---|---|
| Loyal reward | Reward offer, ready for review |
| Consistent thank-you | Thank-you credit |
| Declining re-engagement | Re-engagement discount |
| Dormant win-back | Win-back discount |
| Due soon (stage 0) | Courteous heads-up |
| Overdue polite / firm / final (stages 1–3) | Progressively firmer reminders, all with a dispute path |
| **Refuse: dispute open** | Not generated; escalated to the billing team |
| **Refuse: no marketing consent** | Not generated |
| **Refuse: contacted too often** | Not generated |

**Live guardrail demo:** edit an approved draft, e.g. change `$10` to `$50` or add "promo code SAVE50", and the checks turn red and Approve is disabled.

---

## 13. Evaluation

```powershell
python scripts/evaluate.py
```
This writes `reports/evaluation_report.md`. Results (template mode, all 7,043 customers):

| Metric | Result |
|---|---|
| Classification on labelled scenarios | **12 / 12** |
| Drafts generated | 3,860 |
| Drafts passing groundedness and policy checks | **3,860 / 3,860** |
| Customers refused before generation | 3,183 (45%): no marketing consent, contact cap, no email consent, disputes, bad data |
| Red-team bad drafts blocked | **18 / 18** |
| Clean control drafts allowed | **2 / 2** |

---

## 14. Troubleshooting

| Symptom | Fix |
|---|---|
| `TypeError: 'NoneType' object is not subscriptable` in the dashboard | Start it with `python -m streamlit run frontend/app.py`, and make sure the backend is running |
| `uvicorn is not recognized` | Use `python -m uvicorn ...` |
| `File does not exist: frontend\app.py` | You are in the wrong folder; `cd` until `dir` shows `backend`, `frontend`, `src` |
| `ModuleNotFoundError: src...` | Run commands from the project root |
| `.env` changes ignored | Restart the backend. If needed: `Get-Process python \| Stop-Process -Force` |
| Header shows the wrong delivery mode | `DELIVERY_MODE` appears twice in `.env`; keep one |
| `getaddrinfo failed` (SMTP) | `SMTP_HOST` must be `smtp.gmail.com`; otherwise the network blocks it, so use a hotspot or Resend |
| `Username and Password not accepted` | Use a Gmail **app password** with 2-Step Verification on |
| `Resend 401: API key is invalid` | Create a new key and paste it with the copy icon |
| `Resend 403: only send testing emails to your own email address` | `REDIRECT_ALL_TO` must be your Resend sign-up email |
| `Resend 422: Invalid from field` | Set `RESEND_FROM=onboarding@resend.dev` |
| Groq `model_not_found` | Set `GROQ_MODEL=openai/gpt-oss-120b` |
| Drafts always show `Source: template` | Check `GROQ_API_KEY`, and raise `max_tokens` (section 7) |

---

## 15. Assumptions and limitations

- Data is synthetic. "Today" can be pinned with `APP_TODAY` for reproducible offer-expiry checks.
- All catalogue offers require marketing consent, so customers without it receive no marketing email.
- Billing reminders are service messages: they need email consent but not marketing consent.
- Segment thresholds are hand-set rules; confidence is a heuristic distance from the threshold.
- Groundedness checks are pattern-based. A misleading sentence with no number or term in it relies on human review.
- Customers who prefer SMS get an email draft with a warning; there is no SMS channel yet.
- Drafts are held in API memory; restarting the backend clears unsent drafts (the audit log persists).
- Real sending is redirected to one test inbox; unsupervised bulk sending is out of scope by design.

**Next steps:** SMS channel, business-owned thresholds and contact limits, learning tone from reviewer edits, and a verified sending domain for production email.
