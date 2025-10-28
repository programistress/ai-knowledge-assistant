from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
from openai import OpenAI

pinecone_api_key = os.getenv("PINECONE_API_KEY")
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model (first time only)...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model

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
    model = get_embedding_model()  # Use cached model
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
    model = get_embedding_model()
    query_embedding = model.encode([query_text])[0]
    
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    
    return results['matches']

# query with a dummy vector to get some results, then extract unique documents
# this is a workaround since Pinecone doesn't have a "list all" feature
def get_all_documents(index):
    stats = index.describe_index_stats()
    total_vectors = stats.get('total_vector_count', 0)
    
    if total_vectors == 0:
        return []

    dummy_query = [0.0] * 384
    results = index.query(
        vector=dummy_query,
        top_k=min(1000, total_vectors),  #get up to 1000 vectors
        include_metadata=True
    )
    
    # extract unique documents by document_id
    documents_dict = {}
    for match in results.get('matches', []):
        metadata = match.get('metadata', {})
        doc_id = metadata.get('document_id')
        doc_name = metadata.get('document_name', 'Unknown')
        
        if doc_id and doc_id not in documents_dict:
            documents_dict[doc_id] = {
                "document_id": doc_id,
                "document_name": doc_name,
            }
    
    return list(documents_dict.values())

def generate_response(query, context_chunks):
    chunks_text = []
    for match in context_chunks:
        content = match['metadata']['content']
        chunks_text.append(content)
    
    # combine chunks with separator
    context = "\n\n---\n\n".join(chunks_text)
    
    # limit context length (GPT-4o-mini has 128k context, but keep it reasonable)
    max_context_length = 8000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "...[truncated]"
    
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context. If the answer is not in the context, say so."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
                }
            ],
            temperature=0.2,
            max_tokens=500  # response length
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error generating response: {str(e)}"
