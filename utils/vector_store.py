import faiss
import numpy as np
from typing import List, Dict, Tuple, Any
from models.embeddings import embed_texts

def build_vector_store(chunks: List[Dict[str, str]]) -> Tuple[Any, List[str], List[str]]:
    """
    Builds a FAISS vector index from local academic document chunks.
    
    Args:
        chunks (List[Dict[str, str]]): List of chunked text and sources.
        
    Returns:
        Tuple[faiss.IndexFlatL2, List[str], List[str]]: The FAISS index, list of chunk texts, and list of sources.
    """
    try:
        texts = [c["text"] for c in chunks]
        sources = [c["source"] for c in chunks]

        embeddings = embed_texts(texts)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        return index, texts, sources
    except Exception as e:
        print(f"Error building vector index: {e}")
        return None, [], []