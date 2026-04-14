from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
import time
from openai import OpenAI

pinecone_api_key = os.getenv("PINECONE_API_KEY")
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model (first time only)...")
        try:
            # Try loading from local cache first (faster, no network)
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
            print("Loaded embedding model from local cache")
        except Exception:
            # Fall back to downloading if not cached
            print("Model not in cache, downloading from HuggingFace...")
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
def chunk_text(text, chunk_size=2000, overlap=300):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_text(chunks):
    model = get_embedding_model()  # Use cached model
    chunk_embeddings = model.encode(chunks, show_progress_bar=True, batch_size=64)
    return chunk_embeddings

def store_chunks_in_pinecone(
    index,
    text,
    document_id,
    document_name="",
    page_texts=None,
    file_type=None,
    source_url=None
):
    started_at = time.perf_counter()

    chunking_started_at = time.perf_counter()
    chunk_records = []

    if page_texts:
        for page_number, page_text in page_texts:
            if not page_text or not page_text.strip():
                continue

            page_chunks = chunk_text(page_text)
            for page_chunk in page_chunks:
                chunk_records.append({
                    "content": page_chunk,
                    "page_number": page_number
                })
    else:
        chunks = chunk_text(text)
        chunk_records = [{"content": chunk, "page_number": None} for chunk in chunks]

    chunking_duration = time.perf_counter() - chunking_started_at

    if not chunk_records:
        return 0

    embedding_started_at = time.perf_counter()
    chunk_texts = [record["content"] for record in chunk_records]
    embeddings = embed_text(chunk_texts)
    embedding_duration = time.perf_counter() - embedding_started_at

    vectors_to_upsert = [] 

    # list of tuples (index, (chunk, embedding))
    # (0, ("The cat sat", [0.2, 0.8, ...])),
    for i, (chunk_record, embedding) in enumerate(zip(chunk_records, embeddings)):
        chunk = chunk_record["content"]
        vector_id = f"{document_id}_chunk_{i}"
        
        # Metadata stored alongside the vector
        metadata = {
            "document_id": document_id,
            "document_name": document_name,
            "chunk_index": i,
            "content": chunk  # store the full chunk text
        }

        if chunk_record.get("page_number") is not None:
            metadata["page_number"] = chunk_record["page_number"]

        if file_type:
            metadata["file_type"] = file_type

        if source_url:
            metadata["source_url"] = source_url
        
        vectors_to_upsert.append({
            "id": vector_id,
            "values": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
            "metadata": metadata
        })
    
    # upsert in batches (Pinecone recommends batch sizes of 100-200)
    batch_size = 100
    upsert_started_at = time.perf_counter()
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch)
    upsert_duration = time.perf_counter() - upsert_started_at

    total_duration = time.perf_counter() - started_at
    print(
        f"[store] document_id={document_id} chunks={len(chunk_records)} "
        f"chunking={chunking_duration:.2f}s embedding={embedding_duration:.2f}s "
        f"upsert={upsert_duration:.2f}s total={total_duration:.2f}s"
    )
    
    return len(vectors_to_upsert)

def query_pinecone(index, query_text, top_k=3):
    model = get_embedding_model()
    query_embedding = model.encode([query_text], show_progress_bar=False)[0]
    
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

def delete_document(index, document_id):
    try:
        index.delete(filter={"document_id": {"$eq": document_id}})
        return True
    except Exception as e:
        print(f"Error deleting document {document_id}: {str(e)}")
        return False

def generate_response(query, context_chunks):
    chunks_text = []
    for match in context_chunks:
        metadata = match.get('metadata', {})
        content = metadata.get('content', '')
        document_name = metadata.get('document_name', 'Unknown')
        chunks_text.append(f"[Source file: {document_name}]\n{content}")
    
    # combine chunks with separator
    context = "\n\n---\n\n".join(chunks_text)
    
    # Optimized: Reduced context length for faster processing
    max_context_length = 6000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "...[truncated]"
    
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful DSA assistant. Use the provided context to answer the question.\n"
                        "Style requirements:\n"
                        "- Do not paste large chunks verbatim.\n"
                        "- Synthesize and explain in your own words.\n"
                        "- Be concise and practical.\n"
                        "- Mention source file names when helpful.\n"
                        "- If quoting, keep quotes very short (one sentence max).\n"
                        "- Use markdown formatting."
                    )
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
                }
            ],
            temperature=0.3,  # Slightly higher for faster generation
            max_tokens=800  # Increased for better code examples
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error generating response: {str(e)}"
