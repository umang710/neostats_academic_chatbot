# Academic Chatbot: MSc Data Science

**Live Demo:** [https://academicchatbotumang.streamlit.app/](https://academicchatbotumang.streamlit.app/)

An intelligent, context-aware chatbot developed as an AI engineering blueprint. This application is specifically tailored as an **MSc Data Science Academic Assistant**, designed to help students navigate their syllabus, explore course credits, and stay continuously updated with the latest trends in artificial intelligence.

---

## Use Case Objective
*"Tools don’t solve problems. People do."* 
The core objective of this project is to create an AI assistant that doesn't just regurgitate text, but actively understands context, searches smartly, and grounds its answers in reality. Instead of a generic question-answering bot, this solution addresses a real-world vertical: **University Academic Navigation**. It combines structured database extraction (syllabus parsing) with dynamic internet searching to synthesize comprehensive, accurate academic guidance.

## Agentic Architecture
This chatbot follows a **Conditional Routing Agent Workflow**, stepping beyond a simple LLM wrapper by incorporating autonomous decision-making:

1. **Intent Routing Agent:** Every user query is pre-processed by a strict intent-router (`academic_router`). It autonomously classifies the conversation into semantic buckets (e.g., specific syllabus inquiries, open-ended AI questions, or out-of-bounds/blocked topics).
2. **Local RAG Retrieval:** For safe intents, the agent queries a local `FAISS` vector database embedded with `SentenceTransformers`. It evaluates the semantic distance (confidence) of the retrieved academic chunks.
3. **Autonomous Tool Usage (Fallback):** The system exhibits true agentic behavior by evaluating its own knowledge limitations. If the RAG confidence score is too low (meaning the local syllabus doesn't contain the answer), or if the intent specifically requests trending "AI" topics, the agent automatically executes the **Live Web Search** tool (`duckduckgo_search`) to scrape real-time context from the internet before answering.
4. **Conditional Formatting:** The agent dynamically structures its final syntheses based on the user-selected `Response Mode` (Concise bullet points vs Detailed academic explanations).

## Key Features
* **Multi-Modal Document Parsing:** Directly extracts knowledge from structural `.xlsx` spreadshets, `.txt` files, and `.pdf` documents using `pandas` and `pypdf`.
* **Agentic Web Augmentation:** Seamlessly falls back to real-time DuckDuckGo web scraping when internal database context runs dry.
* **Intelligent Chunking:** Custom text-splitter logic strictly enforces word-boundaries rather than raw character counts to preserve semantic context.
* **Clean Theming Engine:** Engineered with a custom `.streamlit/config.toml` that perfectly supports native system Light/Dark modes without CSS hacking.
* **Component Modularity:** A robust `utils/` and `models/` separation architecture, complete with extensive `try/except` safeguards and `config/config.py` environment variable loading setups.

## Project Structure
```text
project/
├── .streamlit/
│   └── config.toml       # Global native theme variables
├── config/
│   └── config.py         # Centralized API key & Environment loaders
├── data/
│   └── mds_syllabus.pdf  # Fallback long-tail document data
├── kb/
│   ├── msc_ds_structure.xlsx # Structured primary syllabus knowledge
│   └── course_details/   # Secondary textual fallbacks
├── models/
│   ├── embeddings.py     # Local SentenceTransformer embedding invocation
│   └── llm.py            # Google Gemini model initialization
├── utils/
│   ├── document_loader.py # Multi-format document parsing logic
│   ├── text_splitter.py   # Word-boundary aware chunking
│   ├── vector_store.py    # FAISS indexing logic
│   ├── retriever.py       # RAG similarity search and extraction
│   ├── web_search.py      # Live DDGS internet search tool
│   └── response_builder.py# Final system prompt synthesis
├── app.py                 # Main Streamlit UI and Agent runtime
└── requirements.txt       # Project dependencies
```

## Setup & Deployment Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Academic_chatbot.git
   cd Academic_chatbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your Environment Variables:**
   Create a `.streamlit/secrets.toml` file or set your system environment variables.
   ```toml
   GEMINI_API_KEY="AIza..."
   ```

4. **Run the Application Locally:**
   ```bash
   python -m streamlit run app.py
   ```

