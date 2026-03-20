from typing import List, Optional, Dict

def build_prompt(rag_chunks: List[str], web_context: str, response_mode: str, user_profile: Optional[Dict[str, str]] = None) -> str:
    """
    Constructs the final prompt context payload to send to the LLM agent.
    
    Args:
        rag_chunks (List[str]): Text segments retrieved from local syllabus FAISS index.
        web_context (str): Real-time live context retrieved via DuckDuckGo.
        response_mode (str): Requested response verbosity style.
        user_profile (Optional[Dict[str, str]]): Standard active user session information payload.
        
    Returns:
        str: The fully synthesized context injection string block.
    """
    rag_text = "\n\n".join(rag_chunks) if rag_chunks else ""
    web_text = web_context if web_context else ""

    profile_text = ""
    if user_profile and (user_profile.get("name") or user_profile.get("roll_no")):
        profile_text = f"\nSTUDENT PROFILE (Address the user appropriately):\nName: {user_profile.get('name', 'N/A')}\nRoll No: {user_profile.get('roll_no', 'N/A')}\n"

    style_block = (
        "Answer in maximum 5 lines. Be crisp. Use bullet points if listing."
        if response_mode == "Concise"
        else "Give structured academic explanation. BUT if the question asks about syllabus subjects, FIRST give a clean bullet list of subjects, THEN a short explanation."
    )

    final_prompt = f"""You are the official MSc Data Science academic assistant.
{profile_text}
BACKGROUND KNOWLEDGE:
Syllabus Database:
{rag_text}

Web Search Results:
{web_text}

INSTRUCTIONS:
1. Answer the user's question friendly and naturally.
2. Use the Syllabus Database for course questions (credits, subjects). You can aggregate and sum credits. Do NOT paste the raw syllabus text back.
3. Use the Web Search Results for AI/tech concept questions.
4. If you don't know the answer, just say you don't know. Do not invent course names.
5. Format your response clearly. Use bullet points for syllabus subjects if asked. Keep it mostly concise unless asked otherwise.

Please answer the user's question directly based on the instructions and background knowledge.
"""
    return final_prompt

def get_confidence_label(score: float) -> str:
    """
    Evaluates the context extraction spatial distance assigning a confidence score matrix label.
    
    Args:
        score (float): The average L2 distance from FAISS similarity search.
        
    Returns:
        str: The human-readable system confidence evaluation boundary label.
    """
    if score < 1.3:
        return "High (Syllabus grounded)"
    elif score < 1.9:
        return "Medium (Syllabus + Web supported)"
    else:
        return "Low (Mostly web knowledge)"