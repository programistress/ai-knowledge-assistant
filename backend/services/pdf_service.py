"""
PDF processing service - handles PDF text extraction only.
"""
import re
from io import BytesIO
from pathlib import Path
from pypdf import PdfReader


def fix_pdf_spacing(text: str) -> str:
    """Fix common PDF extraction spacing issues."""
    # Add space before uppercase letter that follows lowercase (camelCase -> camel Case)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Add space before uppercase letter that follows punctuation without space
    text = re.sub(r'([.!?,;:])([A-Z])', r'\1 \2', text)
    # Fix multiple spaces
    text = re.sub(r' +', ' ', text)
    return text


def _extract_pages_from_reader(reader: PdfReader) -> tuple[list[tuple[int, str]], str]:
    """Internal helper to extract text from a PdfReader instance."""
    page_texts = []
    full_text_parts = []

    for page_index, page in enumerate(reader.pages):
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


def extract_pdf_pages(file_path: Path) -> tuple[list[tuple[int, str]], str]:
    """Extract text from a PDF file path (for dataset initialization)."""
    reader = PdfReader(str(file_path))
    return _extract_pages_from_reader(reader)


def extract_pdf_pages_from_bytes(file_bytes: bytes) -> tuple[list[tuple[int, str]], str]:
    """Extract text from PDF bytes (for user uploads - no disk I/O)."""
    reader = PdfReader(BytesIO(file_bytes))
    return _extract_pages_from_reader(reader)
