import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
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

def get_openai_model(model_name="gpt-4o"):
    try:
        api_key = get_api_key("openai")
        if not api_key:
            raise RuntimeError(
                "OpenAI API key not configured. Set env var or Streamlit secrets in config/config.py"
            )
        return ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=0.2
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI model: {str(e)}")