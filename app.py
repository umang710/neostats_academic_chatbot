import streamlit as st

st.set_page_config(
    page_title="Academic Intelligence Assistant",
    page_icon="🎓",
    menu_items={
        'About': "An intelligent, context-aware RAG assistant built for my MSc Data Science batchmates at Christ University."
    }
)

import os
import sys
import time
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.llm import get_chat_model
from utils.document_loader import load_documents
from utils.text_splitter import split_documents
from utils.vector_store import build_vector_store
from utils.retriever import retrieve_context
from utils.web_search import search_web
from utils.response_builder import build_prompt, get_confidence_label


import re
import pandas as pd
import os

# ---------- CLEAN THEME CONFIGURED IN .streamlit/config.toml ----------

# ---------- DYNAMIC MODEL DISCOVERY ----------
@st.cache_data(ttl=3600)
def get_available_models():
    try:
        from groq import Groq
        from config.config import get_api_key
        api_key = get_api_key("groq")
        if not api_key:
            return ["llama3-8b-8192"]
        client = Groq(api_key=api_key)
        models_data = client.models.list()
        
        models = [m.id for m in models_data.data if m.active]
        
        # Guarantee 100% free-tier stability by strictly allowing known fast Groq LPU models
        safe_list = [
            "llama3-8b-8192", 
            "llama3-70b-8192", 
            "mixtral-8x7b-32768", 
            "gemma2-9b-it",
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile"
        ]
        
        free_models = [m for m in models if m in safe_list]
        # Sort to put fastest 8b/instant models first
        free_models.sort(key=lambda x: ("8b" not in x, x))
        
        return free_models if free_models else ["llama3-8b-8192"]
    except Exception:
        return ["llama3-8b-8192"]

# ---------- DIRECT TRIMESTER LOOKUP ----------
def get_trimester_direct(n):
    """Bypass FAISS and directly pull the correct trimester sheet from Excel."""
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "kb", "msc_ds_structure.xlsx")
        sheet_name = f"Trimester{n}"
        df = pd.read_excel(path, sheet_name=sheet_name)
        rows = []
        for _, row in df.iterrows():
            rows.append(
                f"{sheet_name} — {row['Course Code']} {row['Subject Name']} — "
                f"Credits {row['Credits']} — CIA {row['CIA Marks']} — ESE {row['ESE Marks']} — {row['Type']} course."
            )
        return "\n".join(rows)
    except Exception as e:
        print(f"Direct Trimester Error: {e}")
        return ""


# ---------- QUICK QUERY ----------
def quick_trimester_query(n):
    return f"What subjects are there in trimester {n}?"


# ---------- CHAT WITH MEMORY ----------
def get_chat_response(chat_model, history, system_prompt, user_query):
    try:
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        
        # 1. Provide the Strict System Bounds
        formatted = [SystemMessage(content=system_prompt)]

        # 2. Append only the last 2 interactions to avoid poisoned prompt-looping
        for m in history[-2:]:
            if m["role"] == "user" and m["content"] != user_query:
                formatted.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                formatted.append(AIMessage(content=m["content"]))

        # 3. Exactly append the final user query naturally.
        formatted.append(HumanMessage(content=user_query))

        return chat_model.invoke(formatted).content

    except Exception as e:
        return f"Model response issue. Error details: {str(e)}. Try switching model."


# ---------- INTENT ROUTER ----------
def academic_router(prompt):

    q = prompt.lower()

    syllabus_kw = [
        "trimester","subject","course","credit",
        "cia","ese","syllabus","elective","project"
    ]

    ai_kw = [
        "machine learning","deep learning","regression",
        "classification","ai","neural","model","algorithm",
        "trend","latest","technology","data science"
    ]

    blocked_kw = [
        "porn","sex","actor","president","politics",
        "disease","drug"
    ]

    if any(k in q for k in blocked_kw):
        return "blocked"

    if any(k in q for k in syllabus_kw):
        return "syllabus"

    if any(k in q for k in ai_kw):
        return "ai"

    return "general"


# ---------- BUILD VECTOR INDEX ----------
def build_index():

    progress = st.progress(0)
    status = st.empty()

    status.write("Loading academic knowledge...")
    docs = load_documents()
    progress.progress(40)

    status.write("Building semantic syllabus intelligence...")
    chunks = split_documents(docs)
    index, texts, sources = build_vector_store(chunks)
    progress.progress(100)

    status.write("System ready")

    return index, texts, sources


# ---------- TRIMESTER BUTTONS ----------
def trimester_buttons():

    st.markdown("#### 📚 Trimester Explorer")

    cols = st.columns(6)

    for i in range(6):
        if cols[i].button(f"Tri {i+1}", use_container_width=True, key=f"t{i}"):
            st.session_state.quick_query = quick_trimester_query(i+1)


# ---------- CHAT PAGE ----------
def chat_page(response_mode, selected_model):

    st.title("Academic Intelligence Assistant")
    st.caption("Context-aware knowledge system built for MSc Data Science batchmates at Christ University")

    trimester_buttons()

    chat_model = get_chat_model(selected_model)

    if "vector_ready" not in st.session_state:
        index, texts, sources = build_index()
        st.session_state.index = index
        st.session_state.texts = texts
        st.session_state.sources = sources
        st.session_state.vector_ready = True

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    st.markdown("<br>", unsafe_allow_html=True)

    prompt = st.chat_input("Ask syllabus, credits, AI concept or technology trend")

    if "quick_query" in st.session_state:
        prompt = st.session_state.quick_query
        del st.session_state.quick_query

    if prompt:

        st.session_state.messages.append({"role":"user","content":prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        intent = academic_router(prompt)

        if intent == "blocked":

            response = """
This assistant is focused on MSc Data Science academic queries.

You can ask about syllabus, subjects, credits, projects or AI concepts.
"""
            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role":"assistant","content":response})
            return

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                dynamic_top_k = 15 if "70b" in selected_model else 7

                # Direct trimester lookup: detect number in query and bypass FAISS
                trimester_match = re.search(r'trimester\s*(\d+)', prompt.lower())
                if trimester_match:
                    t_num = int(trimester_match.group(1))
                    direct_data = get_trimester_direct(t_num)
                    if direct_data:
                        rag_chunks = [direct_data]
                        rag_sources = [f"Trimester{t_num}"]
                        score = 0.5
                    else:
                        rag_chunks, rag_sources, score = retrieve_context(prompt, st.session_state.index, st.session_state.texts, st.session_state.sources, top_k=dynamic_top_k)
                else:
                    rag_chunks, rag_sources, score = retrieve_context(prompt, st.session_state.index, st.session_state.texts, st.session_state.sources, top_k=dynamic_top_k)

                web_context = ""
                used_web = False

                if intent == "ai" or score > 1.2:
                    web_context = search_web(prompt)
                    used_web = True

                user_profile = {
                    "name": st.session_state.get("user_name", ""),
                    "roll_no": st.session_state.get("roll_no", "")
                }

                system_context = build_prompt(
                    rag_chunks,
                    web_context,
                    response_mode,
                    user_profile
                )

                response = get_chat_response(
                    chat_model,
                    st.session_state.messages,
                    system_context,
                    prompt
                )

                confidence = get_confidence_label(score)

                st.markdown(response)

                st.markdown(
                    f"""
                    <div class="source-card">
                    <b>Sources:</b> {', '.join(set(rag_sources)) if rag_sources else "Academic knowledge base"} <br>
                    <b>Confidence:</b> {confidence}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if used_web:
                    st.caption("External knowledge augmentation used")

        st.session_state.messages.append({"role":"assistant","content":response})


# ---------- INSTRUCTIONS ----------
def instructions_page():

    st.title("Usage Guide")

    st.markdown("""
This assistant helps MSc Data Science students:

• explore trimester syllabus  
• check CIA / ESE / credits  
• understand course topics  
• learn AI & data science concepts  
• stay updated with latest AI trends  
""")


# ---------- MAIN ----------
def main():

    st.set_page_config(
        page_title="Academic Intelligence Assistant",
        layout="wide",
        page_icon="🎓"
    )

    # Inject Premium CSS for smooth UX
    st.markdown("""
        <style>
        div[data-testid="stButton"] button {
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.2);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            font-weight: 500;
        }
        div[data-testid="stButton"] button:hover {
            border-color: #ff4b4b;
            color: #ff4b4b;
            box-shadow: 0 4px 12px rgba(255,75,75,0.15);
            transform: translateY(-2px);
        }
        .source-card {
            background-color: rgba(128,128,128,0.05);
            border-left: 4px solid #ff4b4b;
            padding: 12px 16px;
            border-radius: 4px 8px 8px 4px;
            margin-top: 12px;
            font-size: 0.85em;
            color: gray;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:

        st.title("Controls")

        st.markdown("### Student Profile")
        st.text_input("Name", key="user_name", placeholder="Optional")
        st.text_input("Roll No.", key="roll_no", placeholder="Optional")
        
        st.markdown("---")

        page = st.radio("Navigate", ["Chat","Instructions"])

        response_mode = st.radio(
            "Response Style",
            ["Concise","Detailed"],
            index=1
        )

        available_models = get_available_models()
        
        def format_model_name(m):
            return m.replace("-", " ").title()

        selected_model = st.selectbox(
            "Model Engine",
            available_models,
            format_func=format_model_name
        )

        if page == "Chat":
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()

    if page == "Instructions":
        instructions_page()
    else:
        chat_page(response_mode, selected_model)


if __name__ == "__main__":
    main()