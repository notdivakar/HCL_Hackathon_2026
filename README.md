# Customer Loyalty & Engagement Assistant
**Team Natural Intelligence** · HCLTech Future Ready AI Engineer Hackathon 2026

A GenAI assistant that sorts telecom customers into clear segments, picks an approved strategy and offer, drafts a grounded email, checks it with deterministic guardrails, and sends it only after a human approves it. **Delivery is mock-only.**

---

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows  (source .venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
copy .env.example .env                               # add GROQ_API_KEY (optional)

python datacleaner.py                                # builds data/cleaned_telecom_data.csv
uvicorn backend.api:app --reload --port 8000         # terminal 1
streamlit run frontend/app.py                        # terminal 2  -> http://localhost:8501
python scripts/evaluate.py                           # writes reports/evaluation_report.md
```

Without `GROQ_API_KEY` the app runs in **template mode**, using fixed, pre-approved templates and no LLM calls. That's useful offline and as a fallback.

---

## Architecture

```
            ┌──────────────── backend/api.py (FastAPI) ────────────────┐
 Streamlit  │ 1 Profile      minimal fields, protected attributes dropped│
 reviewer ─►│ 2 Classify     deterministic rules + reason codes + conf. │
 UI         │ 3 Strategy     policy.json: objective + tone per segment  │
            │ 4 Offer        deterministic eligibility from catalogue    │
            │ 5 Pre-checks   consent, dispute, contact cap, bad data ───┼─► REFUSE (no LLM call)
            │ 6 Generate     LLM (Groq) or template; footer added in code│
            │ 7 Guardrails   groundedness + policy + PII checks ────────┼─► BLOCK
            │ 8 Human review approve / edit (re-checked) / reject        │
            │ 9 Send gate    approval + checks + consent re-verified ───┼─► outbox/*.json (MOCK)
            │10 Audit        logs/audit_log.jsonl                        │
            └────────────────────────────────────────────────────────────┘
```

| File | Owner | What it does |
|---|---|---|
| `config/policy.json` | Safety | All rules in one place: thresholds, strategies, reminder stages, contact limits, blocked words, footers |
| `src/policy.py` | Safety | Loads the policy and picks the reminder stage |
| `src/guardrails.py` | Safety | `precheck_customer`, `validate_email_draft`, `send_gate` |
| `src/llm_orchestrator.py` | LLM | Prompts, structured JSON output, retry, templates, footer |
| `backend/api.py` | API | Segmentation, offer eligibility, pipeline, mock delivery, audit |
| `frontend/app.py` | UI | Reviewer dashboard, audit log, mock outbox |
| `scripts/evaluate.py` | All | Evaluation report |

---

## Data contract

**Customer** (`data/cleaned_telecom_data.csv`, one row per customer). Fields used:

| Group | Fields |
|---|---|
| Identity | `customerID`, `first_name`, `email` |
| Relationship | `tenure`, `Contract`, `loyalty_tier`, `plan_status`, `autopay` |
| Activity | `days_since_login`, `usage_last_30d`, `usage_change_pct` |
| Billing | `bill_id`, `amount_due`, `due_date`, `status` (Paid/Due/Overdue), `days_overdue`, `paid_date`, `dispute_open` |
| Consent & contact | `email_opt_in`, `marketing_opt_in`, `preferred_channel`, `contacts_last_30d`, `days_since_contact` |
| Services (offer rules) | `InternetService`, `PhoneService`, `StreamingTV` |

**Never used for any decision:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`. They are also removed from API responses.

**Offer** (`data/offer_catalouge.csv`): eligibility by segment, tier, tenure range, plan status, required service, validity dates, channel, consent, status (`active` / `expired` / `paused`).

---

## Segments (first matching rule wins)

| # | Segment | Rule | Message |
|---|---|---|---|
| 1 | Payment Reminder | bill `Due` or `Overdue` | Reminder, staged by days overdue |
| 2 | Dormant | no login ≥ 60 days **or** zero usage in last 30 days | Win-back offer |
| 3 | Declining | usage down ≥ 25% **or** no login ≥ 30 days | Re-engagement offer |
| 4 | Loyal | tenure ≥ 12 months **and** (1/2-year contract **or** autopay) | Reward |
| 5 | Consistent | everyone else (paid, stable) | Thank-you |

Each result carries **reason codes** (e.g. `R_USAGE_DROP`, `R_REMINDER_STAGE_2`) and a **confidence**.

**Reminder stages**

| Stage | Days overdue | Tone |
|---|---|---|
| 0 | 0 (due) | Courteous heads-up |
| 1 | 1–15 | Polite reminder |
| 2 | 16–30 | Firm reminder |
| 3 | 31+ | Final reminder |

Every reminder ends with a dispute and support path that code adds, so the LLM can't remove it.

---

## Prompt & structured output
- The LLM gets only a minimal **FACTS** JSON: name, tenure, tier, contract, and either the approved offer or the bill details.
- It must return `{"subject": ..., "body": ...}`.
- Rules: no invented numbers, dates, codes or links; no internal terms; no pressure; tone set by the segment or reminder stage.
- A quick check retries once with the errors fed back. If it still fails, a **pre-approved template** is used.
- The footer (offer terms and unsubscribe, or the dispute path) is appended in code.
- Model: `GROQ_MODEL` (default `llama-3.3-70b-versatile`), temperature 0.3. Prompt version `prompt-v2.0`, templates `template-v2.0`.

## Guardrails (`src/guardrails.py`)

| Layer | Checks |
|---|---|
| **Pre-check** (before LLM) | Missing evidence; no email consent; no marketing consent (offers); contact cap (≥ 3 in 30 days); duplicate outreach (< 7 days); open dispute → **ESCALATE**; contradictory bill data; SMS preference (warning) |
| **Draft check** | Amounts / % / data / free periods / **dates** must come from the approved offer or bill; fake promo codes; unknown, expired, paused, wrong-tier or wrong-tenure offers; blocked language per message type and reminder stage; links and placeholders; PII (customer ID, other emails, phone, card, IBAN); protected attributes; churn or tracking language; payment method; internal field names; wrong name, tier or tenure; reminder missing the dispute path |
| **Send gate** | Named reviewer approval; edited text re-checked; pre-checks re-run; consent re-verified |

Severities: `BLOCK` (never sent), `ESCALATE` (never sent, goes to a human team), `WARN` (reviewer decides).

## Mock delivery & observability
- **Nothing is emailed.** An approved message is written to `outbox/MOCK-<id>.json` with `From: SENDER_EMAIL`.
- `logs/audit_log.jsonl` records every event: `draft_generated`, `generation_refused`, `send_blocked`, `sent`, `rejected`.
- Each record includes the policy, prompt and template versions, model, evidence field names, segment, reason codes, offer, check results, reviewer, whether the text was edited, feedback and send status.
- Recipient addresses are stored as hashes.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Status, LLM on/off, sender, policy version |
| GET | `/segments` | Customer count per segment |
| GET | `/customers?segment=&search=&limit=` | Customer list |
| GET | `/process-customer?customer_id=` | Segment, reasons, strategy, offers |
| POST | `/generate-email` | Full pipeline; returns draft + checks (or REFUSED) |
| POST | `/validate-email` | Re-check an edited draft |
| POST | `/send-email` | Send gate + mock send |
| POST | `/reject-email` | Reject with reason |
| GET | `/audit`, `/outbox` | Logs and mock outbox |
| GET | `/scenarios` | Demo customers for every required scenario |

## Demo scenarios
Pick these from **Demo scenarios** in the UI sidebar:
- loyal reward
- consistent thank-you
- declining re-engagement
- dormant win-back
- due soon (stage 0)
- overdue stages 1, 2 and 3
- **refusals**: open dispute, no marketing consent, contacted too often

## Evaluation
`python scripts/evaluate.py` writes `reports/evaluation_report.md`. It covers:
- labelled-scenario classification
- segment distribution
- refusal reasons
- groundedness and policy pass rate on every generated draft
- an 18-case safety red-team with 2 clean controls
- known limitations

## Assumptions
- Synthetic data. "Today" can be pinned with `APP_TODAY` so offer-expiry checks are reproducible.
- All catalogue offers require marketing consent, so customers without it get no marketing email at all.
- Billing reminders are service messages: they need email consent but not marketing consent, and they never include offers.
