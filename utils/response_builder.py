from typing import List, Optional, Dict

def build_prompt(query: str, rag_chunks: List[str], web_context: str, response_mode: str, user_profile: Optional[Dict[str, str]] = None) -> str:
    """
    Constructs the final prompt context payload to send to the LLM agent.
    
    Args:
        query (str): The natural language user query.
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

    final_prompt = f"""
You are an MSc Data Science academic assistant.
{profile_text}
RULES:
- Be friendly and address the user by name if provided.
- Use syllabus context ONLY when the question is clearly about subjects, curriculum, trimester structure, or course content.
- If the question asks about latest trends, current developments, real-world concepts, or general technical explanations → focus on conceptual / web knowledge.
- If the user asks general out-of-scope questions (not about AI, data science, or their profile), politely decline to answer and guide them back to academic topics.
- Do NOT list trimester subjects unless the user explicitly asks about syllabus or subjects.
- Keep answers focused and relevant.

USER QUESTION:
{query}

SYLLABUS CONTEXT:
{rag_text}

WEB CONTEXT:
{web_text}

RESPONSE STYLE:
{style_block}

Now produce the best accurate answer.
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