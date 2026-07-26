import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import get_api_key

def get_gemini_model(model_name="models/gemini-2.0-flash"):
    try:
        api_key = get_api_key("gemini")
        if not api_key:
            raise RuntimeError(
                "Gemini API key not configured. Set env var or Streamlit secrets in config/config.py"
            )
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model_name,
            temperature=0.2
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini model: {str(e)}")