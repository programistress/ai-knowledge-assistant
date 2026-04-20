"""
Text processing utilities for cleaning, keyword extraction, and content analysis.
"""
import re

COMMON_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "what",
    "when", "where", "which", "who", "why", "with", "your", "you", "about"
}

BOILERPLATE_TERMS = {
    "all rights reserved", "copyright", "publisher", "published by", "isbn",
    "acknowledg", "dedication", "table of contents"
}


def extract_query_keywords(question: str) -> list[str]:
    """Extract meaningful keywords from a question, filtering out stopwords."""
    words = re.findall(r"[a-zA-Z0-9\+\#]{3,}", question.lower())
    return [word for word in words if word not in COMMON_STOPWORDS]


def is_likely_boilerplate(text: str) -> bool:
    """Check if text contains boilerplate content like copyright notices."""
    lowered = text.lower()
    return any(term in lowered for term in BOILERPLATE_TERMS)


def clean_garbled_text(text: str) -> str:
    """Remove garbled/non-printable characters from text."""
    if not text:
        return ""
    # Keep only printable ASCII and common unicode, remove control chars
    cleaned = re.sub(r'[^\x20-\x7E\u00A0-\u00FF\u0100-\u017F\u0400-\u04FF\n]', '', text)
    # Remove sequences of repeated special chars that indicate encoding issues
    cleaned = re.sub(r'[�\ufffd]{2,}', '', cleaned)
    # Collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def build_relevant_excerpt(content: str, question: str, max_chars: int = 220) -> str:
    """
    Build a relevant excerpt from content based on the question.
    Finds the most relevant sentence based on keyword matching.
    """
    if not content:
        return ""

    clean_content = clean_garbled_text(content)
    clean_content = re.sub(r"\s+", " ", clean_content).strip()
    if not clean_content:
        return ""

    keywords = extract_query_keywords(question)
    sentences = re.split(r"(?<=[.!?])\s+", clean_content)
    candidates = [s.strip() for s in sentences if s.strip()]

    if not candidates:
        return (clean_content[:max_chars - 1] + "…") if len(clean_content) > max_chars else clean_content

    best_sentence = ""
    best_score = -1

    for sentence in candidates:
        lowered = sentence.lower()
        score = 0

        if not is_likely_boilerplate(sentence):
            score += 1

        score += sum(1 for keyword in keywords if keyword in lowered)

        if len(sentence) > 30:
            score += 1

        if score > best_score:
            best_score = score
            best_sentence = sentence

    excerpt = best_sentence or candidates[0]
    return (excerpt[:max_chars - 1] + "…") if len(excerpt) > max_chars else excerpt
