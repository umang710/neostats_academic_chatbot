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
You are an academic assistant for an MSc Data Science program.
{profile_text}
STRICT RULES — YOU MUST FOLLOW THESE WITHOUT EXCEPTION:
1. Answer ONLY using the SYLLABUS CONTEXT provided below. Do NOT use your own training knowledge for subject names, trimester counts, or course details.
2. If the exact answer is present in the SYLLABUS CONTEXT, extract it precisely and list it accurately.
3. If the SYLLABUS CONTEXT does not contain the answer, say: "I could not find this information in the syllabus database." Do NOT guess or generalize.
4. NEVER invent subject names. NEVER say "typically" or "usually". Only state what is explicitly in the context.
5. Be friendly and address the user by name if provided in the student profile.
6. For out-of-scope topics (not related to data science, academics, or user profile), politely decline.

USER QUESTION:
{query}

SYLLABUS CONTEXT (use ONLY this to answer syllabus questions):
{rag_text}

WEB CONTEXT (use ONLY for AI/technology trend questions, not for syllabus facts):
{web_text}

RESPONSE STYLE:
{style_block}

Now answer strictly from the syllabus context above.
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