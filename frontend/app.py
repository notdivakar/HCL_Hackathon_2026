import streamlit as st
import requests

# Page Configuration
st.set_page_config(
    page_title="AI Email Review Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Email Generation & Review Dashboard")
st.markdown("Review customer profiles, examine AI-generated drafts fetched dynamically from your FastAPI backend, and take action.")

# FastAPI backend URL (update if your port differs)
FASTAPI_URL = "http://localhost:8000"

# Function to fetch customers dynamically from FastAPI backend
@st.cache_data(ttl=60)
def fetch_customers():
    try:
        response = requests.get(f"{FASTAPI_URL}/customers")
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Failed to load customers from backend.")
            return []
    except requests.exceptions.ConnectionError:
        st.error(f"⚠️ Could not connect to FastAPI backend at `{FASTAPI_URL}`. Make sure your FastAPI server is running!")
        return []

# Fetch customer data dynamically from FastAPI
CUSTOMERS = fetch_customers()

# Layout: Two columns (Left: Customer Profiles & Segments, Right: AI Email Draft & Review Actions)
left_col, right_col = st.columns([1, 1.5], gap="large")

with left_col:
    st.subheader("👥 Customer Profiles & Segments")
    
    if CUSTOMERS:
        # Segment Filter
        segments = ["All"] + list(set(c.get("segment", "General") for c in CUSTOMERS))
        selected_segment = st.selectbox("Filter by Segment", segments)
        
        filtered_customers = (
            CUSTOMERS if selected_segment == "All" 
            else [c for c in CUSTOMERS if c.get("segment", "General") == selected_segment]
        )
        
        # Customer Selection Dropdown
        customer_options = {f"{c.get('name', 'Unknown')} ({c.get('segment', 'General')})": c for c in filtered_customers}
        
        if customer_options:
            selected_label = st.selectbox("Select Customer", list(customer_options.keys()))
            current_customer = customer_options[selected_label]
            
            # Display Customer Profile Details Card
            with st.container(border=True):
                st.markdown(f"### 📋 Profile: {current_customer.get('name', 'Unknown')}")
                st.write(f"**Email:** `{current_customer.get('email', 'N/A')}`")
                st.write(f"**Segment:** `{current_customer.get('segment', 'N/A')}`")
                st.write(f"**Last Purchase:** {current_customer.get('last_purchase', 'N/A')}")
                st.write(f"**Lifetime Value (LTV):** {current_customer.get('lifetime_value', 'N/A')}")
        else:
            current_customer = None
            st.warning("No customers found in this segment.")
    else:
        current_customer = None
        st.info("Waiting for customer data from the FastAPI backend...")

with right_col:
    st.subheader("✉️ AI-Generated Email Draft")
    
    if current_customer:
        # Fetch email draft from FastAPI backend based on selected customer ID
        email_draft = {}
        try:
            response = requests.post(f"{FASTAPI_URL}/generate-email", json={"customer_id": current_customer.get("id")})
            if response.status_code == 200:
                email_draft = response.json()
            else:
                # Fallback if backend endpoint is still under construction
                email_draft = {
                    "subject": f"Special update for {current_customer.get('name', 'Customer').split()[0]}",
                    "body": f"Hi {current_customer.get('name', 'Customer')},\n\nWe have a special tailored offer ready for your account based on your recent activity.\n\nBest regards,\nThe Team"
                }
        except requests.exceptions.RequestException:
            # Fallback mock draft if offline
            email_draft = {
                "subject": f"Exclusive offer for {current_customer.get('name', 'Customer')}",
                "body": f"Hi {current_customer.get('name', 'Customer')},\n\nThank you for being a valued part of our platform. Enjoy a special discount on your next visit!\n\nCheers,\nSupport Team"
            }

        # Initialize session state for edit mode tracking
        if "edit_mode" not in st.session_state:
            st.session_state.edit_mode = False

        # Email Editing / Display Form
        with st.form("email_review_form"):
            if st.session_state.edit_mode:
                st.info("✏️ **Edit Mode Active:** Modify the subject or body before approving.")
                edited_subject = st.text_input("Subject Line", value=email_draft.get("subject", ""))
                edited_body = st.text_area("Email Body", value=email_draft.get("body", ""), height=220)
            else:
                edited_subject = email_draft.get("subject", "")
                edited_body = email_draft.get("body", "")
                
                st.markdown(f"**Subject:** {edited_subject}")
                st.text_area("Email Body (Preview)", value=edited_body, height=220, disabled=True)

            st.markdown("---")
            
            # Action Buttons Layout
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            
            with btn_col1:
                approve_submitted = st.form_submit_button("🚀 Approve & Send", type="primary", use_container_width=True)
            with btn_col2:
                edit_submitted = st.form_submit_button("✏️ Toggle Edit", use_container_width=True)
            with btn_col3:
                reject_submitted = st.form_submit_button("❌ Reject", use_container_width=True)

        # Handle Form Button Submissions Outside the Form Context
        if edit_submitted:
            st.session_state.edit_mode = not st.session_state.edit_mode
            st.rerun()

        if approve_submitted:
            try:
                payload = {
                    "customer_id": current_customer.get("id"),
                    "recipient_email": current_customer.get("email"),
                    "subject": edited_subject,
                    "body": edited_body
                }
                res = requests.post(f"{FASTAPI_URL}/send-email", json=payload)
                if res.status_code == 200:
                    st.success(f"✅ Email successfully approved and delivered to {current_customer.get('email')}!")
                    st.balloons()
                else:
                    st.error(f"Backend reported an error: {res.text}")
            except Exception as err:
                st.error(f"Failed to communicate with FastAPI delivery endpoint: {err}")

        if reject_submitted:
            try:
                requests.post(f"{FASTAPI_URL}/reject-email", json={"customer_id": current_customer.get("id")})
                st.warning(f"🚫 Email draft for {current_customer.get('name')} was rejected and discarded.")
            except Exception as err:
                st.error(f"Failed to communicate with FastAPI rejection endpoint: {err}")
    else:
        st.info("Select a customer on the left to view their profile and AI-generated email draft.")