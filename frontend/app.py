"""
frontend/app.py  (owner: feature-ui)

Reviewer dashboard. Shows: customer evidence, segment + reasons + confidence,
selected strategy and offer, generated draft, pre-checks and guardrail results,
and approve / edit / reject actions. Sending is MOCK only (written to outbox/).

Run:  streamlit run frontend/app.py
"""

import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 60

st.set_page_config(page_title="Customer Engagement Assistant", page_icon="✉️", layout="wide")


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
def api_get(path, **params):
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"Backend error on GET {path}: {e}")
        return None


def api_post(path, payload):
    """Returns (ok, json). Never pretends success."""
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=TIMEOUT)
        try:
            body = r.json()
        except ValueError:
            body = {"detail": r.text}
        return r.ok, body
    except requests.RequestException as e:
        return False, {"detail": str(e)}


@st.cache_data(ttl=300)
def load_health():
    return api_get("/health")


@st.cache_data(ttl=300)
def load_segments():
    return api_get("/segments") or {}


@st.cache_data(ttl=300)
def load_customers(segment, search, limit):
    return api_get("/customers", segment=segment, search=search or None, limit=limit) or {"total": 0, "customers": []}


@st.cache_data(ttl=300)
def load_scenarios():
    return api_get("/scenarios") or []


SEV_ICON = {"BLOCK": "⛔", "ESCALATE": "🟠", "WARN": "⚠️"}


def show_findings(title, items):
    if not items:
        st.success(f"{title}: all checks passed")
        return
    st.markdown(f"**{title}**")
    for v in items:
        st.markdown(f"{SEV_ICON.get(v['severity'], '•')} **{v['severity']}** · `{v['category']}` — {v['detail']}"
                    + (f"  \n  _evidence: {v['evidence']}_" if v.get("evidence") else ""))


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("✉️ Customer Loyalty & Engagement Assistant")
health = load_health()
if not health:
    st.stop()
st.caption(
    f"Backend OK · {health['customers']} customers · {health['offers']} offers · "
    f"LLM: {'ON' if health['llm_enabled'] else 'OFF (template mode)'} · "
    f"Sender: {health['sender']} · Policy {health['policy_version']} · Today {health['today']} · "
    "Delivery: MOCK only"
)

tab_review, tab_audit, tab_outbox = st.tabs(["Review drafts", "Audit log", "Mock outbox"])

# --------------------------------------------------------------------------- #
# Sidebar: choose customer
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Pick a customer")
    reviewer = st.text_input("Reviewer name (required to send)", value=st.session_state.get("reviewer", ""))
    st.session_state.reviewer = reviewer

    scen = load_scenarios()
    scen_labels = ["—"] + [f"{s['label']} · {s['name']} ({s['customer_id']})" for s in scen]
    scen_pick = st.selectbox("Demo scenarios", scen_labels)

    st.divider()
    seg_counts = load_segments()
    seg_options = ["All"] + list(seg_counts.keys())
    segment = st.selectbox("Filter by segment", seg_options,
                           format_func=lambda s: s if s == "All" else f"{s} ({seg_counts.get(s, 0)})")
    search = st.text_input("Search name or customer ID")
    limit = st.slider("How many to list", 50, 2000, 200, step=50)
    data = load_customers(segment, search, limit)
    st.caption(f"Showing {len(data['customers'])} of {data['total']} matching customers")

    # Label includes the customer ID, so customers with the same name never overwrite each other.
    options = {f"{c['name']} · {c['customer_id']} ({c['segment']})": c["customer_id"] for c in data["customers"]}
    list_pick = st.selectbox("Customer", list(options.keys()) or ["(none)"])

if scen_pick != "—":
    customer_id = scen[scen_labels.index(scen_pick) - 1]["customer_id"]
else:
    customer_id = options.get(list_pick)

# --------------------------------------------------------------------------- #
# Audit + outbox tabs
# --------------------------------------------------------------------------- #
with tab_audit:
    if st.button("Refresh audit log"):
        pass
    rows = api_get("/audit", limit=100) or []
    st.caption("Every generation, refusal, block, approval and rejection is logged. Recipient emails are hashed.")
    for r in rows:
        with st.expander(f"{r['ts']} · {r['event']} · {r.get('customer_id')} · draft {r.get('draft_id')}"):
            st.json(r)

with tab_outbox:
    if st.button("Refresh outbox"):
        pass
    msgs = api_get("/outbox") or []
    if not msgs:
        st.info("No mock emails sent yet.")
    for m in msgs:
        with st.container(border=True):
            st.markdown(f"**{m['subject']}**  \nFrom `{m['from']}` → To `{m['to']}` · {m['sent_at']}")
            st.text(m["body"])
            st.caption(m["delivery"])


# --------------------------------------------------------------------------- #
# Review tab
# --------------------------------------------------------------------------- #
with tab_review:
    if not customer_id:
        st.info("Select a customer in the sidebar.")
        st.stop()

    info = api_get("/process-customer", customer_id=customer_id)
    if not info:
        st.stop()
    p = info["profile"]

    left, right = st.columns([1, 1.5], gap="large")

    with left:
        with st.container(border=True):
            st.subheader(f"{p.get('first_name')} · {info['customer_id']}")
            st.markdown(f"**Segment:** `{info['segment']}` · confidence **{info['confidence']:.0%}**")
            st.markdown("**Why:** " + " · ".join(f"`{c}`" for c in info["reason_codes"]))
            for r in info["reasons"]:
                st.markdown(f"- {r}")
            st.markdown(f"**Strategy:** {info['strategy']['objective']} · _{info['strategy']['tone']}_")
            if info.get("reminder_stage"):
                s = info["reminder_stage"]
                st.markdown(f"**Reminder stage {s['stage']}:** {s['name']} — _{s['tone']}_")

        with st.expander("Customer evidence", expanded=True):
            c1, c2 = st.columns(2)
            c1.metric("Tier", p.get("loyalty_tier"))
            c2.metric("Tenure (months)", p.get("tenure"))
            c1.metric("Days since login", p.get("days_since_login"))
            c2.metric("Usage change", f"{p.get('usage_change_pct') or 0:+.1f}%")
            c1.metric("Bill status", p.get("status"))
            c2.metric("Amount due", f"${p.get('amount_due') or 0:.2f}")
            st.markdown(
                f"- Bill `{p.get('bill_id')}` due **{p.get('due_date')}**, {p.get('days_overdue')} day(s) overdue\n"
                f"- Contract: {p.get('Contract')} · Plan: {p.get('plan_status')} · Autopay: {p.get('autopay')}\n"
                f"- Dispute open: **{p.get('dispute_open')}**\n"
                f"- Email opt-in: **{p.get('email_opt_in')}** · Marketing opt-in: **{p.get('marketing_opt_in')}** · "
                f"Prefers: {p.get('preferred_channel')}\n"
                f"- Contacts last 30 days: {p.get('contacts_last_30d')} · Last contact {p.get('days_since_contact')} days ago"
            )
            st.caption("Protected attributes (gender, age, partner, dependents) are not shown or used.")

        with st.expander("Offers"):
            if info["strategy"]["message_type"] == "reminder":
                st.write("Billing reminders never include offers.")
            elif info["eligible_offers"]:
                for o in info["eligible_offers"]:
                    st.markdown(f"✅ `{o['offer_id']}` {o['offer_name']} — {o['offer_value']} (until {o['valid_to']})")
            else:
                st.write("No eligible offer.")
            for x in info.get("excluded_offers", []):
                st.markdown(f"❌ {x}")

    with right:
        key = f"draft_{customer_id}"
        b1, b2 = st.columns([1, 1])
        if b1.button("✨ Generate draft", type="primary", use_container_width=True) or (
                key not in st.session_state and st.session_state.get("auto_gen", True)):
            with st.spinner("Generating and checking draft..."):
                ok, d = api_post("/generate-email", {"customer_id": customer_id})
            if ok:
                st.session_state[key] = d
                st.session_state[f"subj_{customer_id}"] = d["subject"]
                st.session_state[f"body_{customer_id}"] = d["body"]
            else:
                st.error(f"Generation failed: {d.get('detail')}")
        b2.toggle("Auto-generate on select", key="auto_gen", value=True)

        d = st.session_state.get(key)
        if not d:
            st.stop()

        status = d["status"]
        badge = {"PENDING_REVIEW": "🟢 Ready for review", "BLOCKED": "⛔ Blocked by guardrails",
                 "REFUSED": "⛔ Refused — not generated", "SENT": "📤 Sent (mock)", "REJECTED": "🗑️ Rejected"}
        st.subheader(badge.get(status, status))

        if status == "REFUSED":
            st.error(d["guardrails"]["refusal_reason"])
            show_findings("Pre-checks", d["precheck"])
            st.stop()

        st.markdown(f"**From:** `{d['from']}`  \n**To:** `{d['to']}`")
        st.caption(f"Type: {d['message_type']} · Offer: {d.get('offer_id') or '—'} · Source: {d['source']} · "
                   f"Model: {d['model']} · {d['prompt_version']} / {d['template_version']}")

        subj = st.text_input("Subject", key=f"subj_{customer_id}")
        body = st.text_area("Body (editable)", key=f"body_{customer_id}", height=260)
        st.text_area("Footer (added automatically, cannot be edited)", value=d["footer"], height=90, disabled=True)

        edited = subj != d["subject"] or body != d["body"]
        if edited:
            ok, check = api_post("/validate-email", {"draft_id": d["draft_id"], "subject": subj, "body": body})
            check = check if ok else {"allowed": False, "violations": [], "refusal_reason": str(check)}
            st.info("You edited the draft — checks below are re-run on your version.")
        else:
            check = d["guardrails"]

        show_findings("Pre-checks", d["precheck"])
        show_findings("Draft checks", check["violations"])

        can_send = check["allowed"] and status == "PENDING_REVIEW" and bool(reviewer.strip())
        if not reviewer.strip():
            st.warning("Enter your reviewer name in the sidebar to approve.")

        feedback = st.text_input("Feedback / reason (optional for approve, required for reject)")
        a1, a2 = st.columns(2)
        if a1.button("✅ Approve & send (mock)", disabled=not can_send, use_container_width=True):
            ok, res = api_post("/send-email", {"draft_id": d["draft_id"], "subject": subj, "body": body,
                                               "reviewer": reviewer, "feedback": feedback})
            if ok:
                d["status"] = "SENT"
                st.success(f"Mock-sent from {res['from']} to {res['to']} (id {res['message_id']}). Nothing was really emailed.")
            else:
                st.error(f"Send blocked: {res.get('detail')}")
        if a2.button("❌ Reject", disabled=status not in {"PENDING_REVIEW", "BLOCKED"}, use_container_width=True):
            if not feedback.strip() or not reviewer.strip():
                st.error("Enter your reviewer name and a reason to reject.")
            else:
                ok, res = api_post("/reject-email", {"draft_id": d["draft_id"], "reviewer": reviewer, "reason": feedback})
                if ok:
                    d["status"] = "REJECTED"
                    st.warning("Draft rejected and logged.")
                else:
                    st.error(f"Reject failed: {res.get('detail')}")
