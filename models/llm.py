import os
import streamlit as st
from langchain_groq import ChatGroq
from config.config import get_api_key


def get_chatgroq_model(model_name):

    try:
        api_key = get_api_key("groq")

        if not api_key:
            raise RuntimeError(
                "Groq API key not configured. Set env var or Streamlit secrets in config/config.py"
            )

        return ChatGroq(
            api_key=api_key,
            model=model_name,
            temperature=0.2
        )

    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")