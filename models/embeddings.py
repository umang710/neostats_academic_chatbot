from sentence_transformers import SentenceTransformer

_embedding_model = None

def load_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model

def embed_texts(texts):
    model = load_embedding_model()
    return model.encode(texts, show_progress_bar=False)

def embed_query(query):
    model = load_embedding_model()
    return model.encode([query])[0]