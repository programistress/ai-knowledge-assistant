"""
Shared application state - holds initialized services like Pinecone index.
"""

pinecone_index = None


def get_pinecone_index():
    """Get the Pinecone index, raising an error if not initialized."""
    if pinecone_index is None:
        raise RuntimeError("Pinecone not initialized")
    return pinecone_index


def set_pinecone_index(index):
    """Set the Pinecone index after initialization."""
    global pinecone_index
    pinecone_index = index
