"""
Document metadata store - tracks uploaded documents separately from Pinecone vectors.
This avoids the unreliable zero-vector query hack for listing documents.
"""
import json
from pathlib import Path
from typing import Optional
from threading import Lock

STORE_PATH = Path(__file__).resolve().parent / "document_metadata.json"

_lock = Lock()


def _load_store() -> dict:
    """Load the document store from disk."""
    if not STORE_PATH.exists():
        return {}
    try:
        with open(STORE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_store(store: dict) -> None:
    """Save the document store to disk."""
    with open(STORE_PATH, 'w', encoding='utf-8') as f:
        json.dump(store, f, indent=2, ensure_ascii=False)


def register_document(
    document_id: str,
    document_name: str,
    file_type: Optional[str] = None,
    pdf_url: Optional[str] = None,
    num_chunks: int = 0
) -> None:
    """Register a new document in the metadata store."""
    with _lock:
        store = _load_store()
        store[document_id] = {
            "document_id": document_id,
            "document_name": document_name,
            "file_type": file_type,
            "pdf_url": pdf_url,
            "num_chunks": num_chunks
        }
        _save_store(store)


def unregister_document(document_id: str) -> bool:
    """Remove a document from the metadata store. Returns True if found and removed."""
    with _lock:
        store = _load_store()
        if document_id in store:
            del store[document_id]
            _save_store(store)
            return True
        return False


def get_all_documents() -> list:
    """Get all registered documents."""
    with _lock:
        store = _load_store()
        return list(store.values())


def get_document(document_id: str) -> Optional[dict]:
    """Get a single document by ID."""
    with _lock:
        store = _load_store()
        return store.get(document_id)


def document_exists(document_id: str) -> bool:
    """Check if a document exists in the store."""
    with _lock:
        store = _load_store()
        return document_id in store
