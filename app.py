"""
Main Streamlit application for SEO analysis dashboard.
"""

import streamlit as st
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def check_environment():
    """Check for required environment variables and log warnings if missing."""
    required_vars = ['GOOGLE_CLIENT_ID', 'GOOGLE_CLIENT_SECRET']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        st.sidebar.warning(
            "⚠️ Some environment variables are missing. "
            "Certain features may not work correctly."
        )

def main():
    """Initialize and run the Streamlit application."""
    st.set_page_config(
        page_title="SEO Analysis Dashboard",
        page_icon="🔍",
        layout="wide"
    )
    
    # Check environment variables
    check_environment()
    
    # Display welcome message
    st.title("SEO Analysis Dashboard")
    st.markdown("""
    Welcome to the SEO Analysis Dashboard! This tool helps you:
    - Analyze keyword performance
    - Identify low-performing keywords
    - Generate AI-powered suggestions
    - Export results for implementation
    """)
    
    # Navigation
    st.sidebar.title("Navigation")
    st.sidebar.info(
        "Select a page from the sidebar to begin your analysis."
    )

if __name__ == "__main__":
    main() 