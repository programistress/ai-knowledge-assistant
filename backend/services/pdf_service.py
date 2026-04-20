"""
PDF processing service - handles PDF extraction, storage paths, and text cleaning.
"""
import re
from pathlib import Path
from pypdf import PdfReader

# PDF storage directory - configured at module level
UPLOADS_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def pdf_storage_path(document_id: str) -> Path:
    """Get the file system path for storing a PDF."""
    return UPLOADS_DIR / f"{document_id}.pdf"


def pdf_public_url(document_id: str) -> str:
    """Get the public URL for accessing a stored PDF."""
    return f"/files/{document_id}.pdf"


def fix_pdf_spacing(text: str) -> str:
    """Fix common PDF extraction spacing issues."""
    # Add space before uppercase letter that follows lowercase (camelCase -> camel Case)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Add space before uppercase letter that follows punctuation without space
    text = re.sub(r'([.!?,;:])([A-Z])', r'\1 \2', text)
    # Fix multiple spaces
    text = re.sub(r' +', ' ', text)
    return text


def extract_pdf_pages(file_path: Path) -> tuple[list[tuple[int, str]], str]:
    """
    Extract text from a PDF file, page by page.
    
    Returns:
        tuple: (page_texts, full_text)
            - page_texts: List of (page_number, text) tuples
            - full_text: Combined text from all pages
    """
    reader = PdfReader(str(file_path))
    page_texts = []
    full_text_parts = []

    for page_index, page in enumerate(reader.pages):
        # Try layout mode first for better spacing
        try:
            text = (page.extract_text(extraction_mode="layout") or "").strip()
        except Exception:
            text = (page.extract_text() or "").strip()
        
        if not text:
            continue

        text = fix_pdf_spacing(text)
        page_number = page_index + 1
        page_texts.append((page_number, text))
        full_text_parts.append(text)

    full_text = "\n\n".join(full_text_parts)
    return page_texts, full_text
