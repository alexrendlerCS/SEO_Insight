import streamlit as st
from utils.auth import authorize_user, save_token_from_url

st.title("Google Ads & Analytics OAuth Test")

# Get auth code from URL if present
code = st.experimental_get_query_params().get("code", [None])[0]
if code and "credentials" not in st.session_state:
    save_token_from_url(code)

# Trigger auth if needed
credentials = authorize_user()

st.success("You're authenticated with Google!")
