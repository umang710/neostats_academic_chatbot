import numpy as np
from typing import List, Tuple, Any
from models.embeddings import embed_query

def retrieve_context(query: str, index: Any, texts: List[str], sources: List[str], top_k: int = 7) -> Tuple[List[str], List[str], float]:
    """
    Retrieves the most semantically relevant context chunks for a given user query.
    
    Args:
        query (str): The user's input query.
        index (faiss.IndexFlatL2): The FAISS vector index representation.
        texts (List[str]): List of indexed text chunks.
        sources (List[str]): List of text sources mapped to chunks.
        top_k (int): Number of top documents to retrieve.
        
    Returns:
        Tuple[List[str], List[str], float]: The retrieved chunks, their sources, and average distance score.
    """
    try:
        query_vec = embed_query(query).astype("float32")
        query_vec = np.array([query_vec])

        distances, indices = index.search(query_vec, top_k)

        retrieved_chunks = []
        retrieved_sources = []

        for idx in indices[0]:
            retrieved_chunks.append(texts[idx])
            retrieved_sources.append(sources[idx])

        avg_distance = float(np.mean(distances))
        return retrieved_chunks, retrieved_sources, avg_distance
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return [], [], 0.0