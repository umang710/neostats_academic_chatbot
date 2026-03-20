from typing import List, Dict

def split_documents(docs: List[Dict[str, str]], chunk_size: int = 900, overlap: int = 120) -> List[Dict[str, str]]:
    """
    Splits document text into logical chunks strictly utilizing word boundaries.
    
    Args:
        docs (List[Dict[str, str]]): List of documents containing 'text' and 'source'.
        chunk_size (int): Maximum character length of each chunk.
        overlap (int): Number of characters to overlap between chunks.
        
    Returns:
        List[Dict[str, str]]: A list of chunked academic documents.
    """
    try:
        chunks = []

        for doc in docs:
            text = doc["text"]
            source = doc["source"]

            start = 0
            length = len(text)

            while start < length:
                end = start + chunk_size

                # Adjust end to not cut words in half
                if end < length:
                    space_idx = text.rfind(" ", start, end)
                    if space_idx != -1 and space_idx > start + chunk_size // 2:
                        end = space_idx

                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "source": source
                    })

                start = end - overlap
                
                # Adjust start to not start middle of a word for overlap
                if start > 0 and start < length:
                    space_idx2 = text.find(" ", start, min(start + 50, length))
                    if space_idx2 != -1:
                        start = space_idx2 + 1

        return chunks
    except Exception as e:
        print(f"Error splitting text chunks: {e}")
        return []