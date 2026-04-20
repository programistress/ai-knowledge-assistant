"""
Query service - handles query classification and chat responses.
"""
import os
from openai import OpenAI
from utils.text_utils import extract_query_keywords


def is_knowledge_inquiry(text: str) -> bool:
    """
    Check if the message is asking for information that would need document lookup.
    Returns True if the query should search documents, False for simple chat.
    """
    cleaned = text.lower().strip()
    
    # Question indicators
    question_starters = [
        "what", "how", "why", "when", "where", "which", "who",
        "explain", "describe", "tell me", "can you", "could you",
        "define", "show me", "give me", "list", "compare"
    ]
    
    # Check for question patterns
    if any(cleaned.startswith(q) for q in question_starters):
        return True
    if "?" in text:
        return True
    
    # Check for specific topic inquiry (more than 2 meaningful words)
    keywords = extract_query_keywords(text)
    if len(keywords) >= 2:
        return True
    
    return False


def chat_response(message: str) -> str:
    """
    Generate a simple chat response without document lookup.
    Used for greetings, small talk, and non-knowledge queries.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a friendly knowledge assistant. Keep responses brief and natural."
            },
            {"role": "user", "content": message}
        ],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()
