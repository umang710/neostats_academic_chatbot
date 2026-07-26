import os
import streamlit as st
from langchain_groq import ChatGroq
from config.config import get_api_key

def get_chat_model(model_name="llama3-8b-8192"):
    try:
        api_key = get_api_key("groq")
        if not api_key:
            st.error("No Groq API Key found. Please add GROQ_API_KEY to your .env or Streamlit Secrets.")
            return None
            
        return ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=0.2
        )
    except Exception as e:
        print(f"Error initializing Groq model: {e}")
        return None