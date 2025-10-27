from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os

pinecone_api_key = os.getenv("PINECONE_API_KEY")

def init_pinecone(api_key, index_name="knowledge-base"):
    pc = Pinecone(api_key=api_key)
    
    # check if index exists, create if not
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # all-MiniLM-L6-v2 produces 384-dim vectors
            metric="cosine",  # similarity for semantic search
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    
    return pc.Index(index_name)

# overlapping chunks with a fixed character count
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_text(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = model.encode(chunks, show_progress_bar=True)
    return chunk_embeddings

def store_chunks_in_pinecone(index, text, document_id, document_name=""):
    chunks = chunk_text(text)
    embeddings = embed_text(chunks)

    vectors_to_upsert = [] 

    # list of tuples (index, (chunk, embedding))
    # (0, ("The cat sat", [0.2, 0.8, ...])),
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_id = f"{document_id}_chunk_{i}"
        
        # Metadata stored alongside the vector
        metadata = {
            "document_id": document_id,
            "document_name": document_name,
            "chunk_index": i,
            "content": chunk  # store the full chunk text
        }
        
        vectors_to_upsert.append({
            "id": vector_id,
            "values": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
            "metadata": metadata
        })
    
    # upsert in batches (Pinecone recommends batch sizes of 100-200)
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch)
    
    return len(vectors_to_upsert)

def query_pinecone(index, query_text, top_k=5):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query_text])[0]
    
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    
    return results['matches']