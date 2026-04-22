"""
emb model and pinecone init, 
chunk text semantic, embed chunks, store chunks in pinecone, 
query pinecone, 
delete document, 
generate response
"""
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
import re
import time
from openai import OpenAI

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model (first time only)...")
        try:
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
            print("Loaded embedding model from local cache")
        except Exception:
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
    # index handle for future operations
    return pc.Index(index_name)


def chunk_text_semantic(text, target_size=1500, max_size=2500, min_size=200):
    """
    1. Split text into paragraphs (double newlines) and sections (headers)
    2. Merge small paragraphs together until reaching target size
    3. Split paragraphs that exceed max size at sentence boundaries
    4. Each chunk maintains semantic coherence
    """
    if not text or not text.strip():
        return []
    
    section_pattern = re.compile(r'\n(?=#{1,6}\s|\n)')
    raw_sections = section_pattern.split(text)
    
    paragraphs = []
    for section in raw_sections:
        parts = re.split(r'\n\s*\n', section)
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                paragraphs.append(cleaned)
    
    if not paragraphs:
        return [text.strip()] if text.strip() else []
    
    def split_large_paragraph(para, max_len):
        """Split a large paragraph at sentence boundaries."""
        if len(para) <= max_len:
            return [para]
        
        sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        sentences = sentence_pattern.split(para)
        
        if len(sentences) == 1:
            chunks = []
            for i in range(0, len(para), max_len):
                chunks.append(para[i:i + max_len])
            return chunks
        
        result = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= max_len:
                current = f"{current} {sentence}".strip()
            else:
                if current:
                    result.append(current)
                current = sentence
        if current:
            result.append(current)
        return result
    
    processed = []
    for para in paragraphs:
        if len(para) > max_size:
            processed.extend(split_large_paragraph(para, max_size))
        else:
            processed.append(para)
    
    chunks = []
    current_chunk = ""
    
    for para in processed:
        potential = f"{current_chunk}\n\n{para}".strip() if current_chunk else para
        
        if len(potential) <= target_size:
            current_chunk = potential
        elif len(current_chunk) >= min_size:
            chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk = potential
    
    if current_chunk:
        if len(current_chunk) < min_size and chunks:
            chunks[-1] = f"{chunks[-1]}\n\n{current_chunk}"
        else:
            chunks.append(current_chunk)
    
    return chunks


def chunk_text(text, chunk_size=2000, overlap=300):
    return chunk_text_semantic(text, target_size=chunk_size, max_size=chunk_size + 500)

def embed_text(chunks):
    model = get_embedding_model() 
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
    """
    1. make chunks by page
    2. embed chunks
    3. create vectors w id, values, metadata
    4. upsert vectors to pinecone in batches
    """
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

    for i, (chunk_record, embedding) in enumerate(zip(chunk_records, embeddings)):
        chunk = chunk_record["content"]
        vector_id = f"{document_id}_chunk_{i}"
        
        metadata = {
            "document_id": document_id,
            "document_name": document_name,
            "chunk_index": i,
            "content": chunk  
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
    """
    convert user's question to a vector
    queries Pinecone for the top_k most similar chunks
    return matches
    """
    model = get_embedding_model()
    query_embedding = model.encode([query_text], show_progress_bar=False)[0]
    
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    
    return results.get('matches', [])


# this is a workaround since Pinecone doesn't have a "list all" feature
def get_all_documents(index):
    stats = index.describe_index_stats()
    total_vectors = stats.get('total_vector_count', 0)
    
    if total_vectors == 0:
        return []

    dummy_query = [0.0] * 384
    results = index.query(
        vector=dummy_query,
        top_k=min(1000, total_vectors),
        include_metadata=True
    )
    
    documents_dict = {}
    for match in results.get('matches', []):
        metadata = match.get('metadata', {})
        doc_id = metadata.get('document_id')
        doc_name = metadata.get('document_name', 'Unknown')
        file_type = metadata.get('file_type')
        source_url = metadata.get('source_url')
        
        if doc_id and doc_id not in documents_dict:
            documents_dict[doc_id] = {
                "document_id": doc_id,
                "document_name": doc_name,
                "file_type": file_type,
                "pdf_url": source_url if file_type == 'pdf' else None
            }
    
    return list(documents_dict.values())

def delete_document(index, document_id):
    """
    delete all vectors for a given document_id
    """
    try:
        index.delete(filter={"document_id": {"$eq": document_id}})
        return True
    except Exception as e:
        print(f"Error deleting document {document_id}: {str(e)}")
        return False

def generate_response(query, context_chunks):
    """
    format result chunks into a context string
    call openai to generate a response
    """
    chunks_text = []
    for match in context_chunks:
        metadata = match.get('metadata', {})
        content = metadata.get('content', '')
        document_name = metadata.get('document_name', 'Unknown')
        chunks_text.append(f"[Source file: {document_name}]\n{content}")
    
    # combine chunks with separator
    context = "\n\n---\n\n".join(chunks_text)
    
    # Reduced context length for faster processing
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
            temperature=0.3,  # Low creativity, more factual
            max_tokens=800  # Max tokens for response
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error generating response: {str(e)}"
