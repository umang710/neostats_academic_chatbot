import os
import streamlit as st

def get_api_key(provider="gemini"):
    """Fetch API keys to abstract logic from models."""
    try:
        key_name = "GEMINI_API_KEY"
        
        api_key = os.getenv(key_name)
        if not api_key and key_name in st.secrets:
            api_key = st.secrets[key_name]
        return api_key
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None