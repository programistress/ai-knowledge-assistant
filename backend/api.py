from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File, Form
from pydantic import BaseModel
import os
import time
import re
import shutil
from pathlib import Path
from pypdf import PdfReader  # type: ignore[reportMissingImports]
from dotenv import load_dotenv

load_dotenv()

from main import (
    init_pinecone,
    store_chunks_in_pinecone,
    query_pinecone,
    generate_response,
    get_all_documents,
    delete_document,
    get_embedding_model
)
from openai import OpenAI

COMMON_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "what",
    "when", "where", "which", "who", "why", "with", "your", "you", "about"
}

BOILERPLATE_TERMS = {
    "all rights reserved", "copyright", "publisher", "published by", "isbn",
    "acknowledg", "dedication", "table of contents"
}

UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _extract_query_keywords(question: str):
    words = re.findall(r"[a-zA-Z0-9\+\#]{3,}", question.lower())
    return [word for word in words if word not in COMMON_STOPWORDS]


def _is_likely_boilerplate(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in BOILERPLATE_TERMS)


def _build_relevant_excerpt(content: str, question: str, max_chars: int = 220) -> str:
    if not content:
        return ""

    clean_content = re.sub(r"\s+", " ", content).strip()
    if not clean_content:
        return ""

    keywords = _extract_query_keywords(question)
    sentences = re.split(r"(?<=[.!?])\s+", clean_content)
    candidates = [s.strip() for s in sentences if s.strip()]

    if not candidates:
        return (clean_content[: max_chars - 1] + "…") if len(clean_content) > max_chars else clean_content

    best_sentence = ""
    best_score = -1

    for sentence in candidates:
        lowered = sentence.lower()
        score = 0

        if not _is_likely_boilerplate(sentence):
            score += 1

        score += sum(1 for keyword in keywords if keyword in lowered)

        if len(sentence) > 30:
            score += 1

        if score > best_score:
            best_score = score
            best_sentence = sentence

    excerpt = best_sentence or candidates[0]
    return (excerpt[: max_chars - 1] + "…") if len(excerpt) > max_chars else excerpt


def _pdf_storage_path(document_id: str) -> Path:
    return UPLOADS_DIR / f"{document_id}.pdf"


def _public_pdf_url(document_id: str) -> str:
    return f"/files/{document_id}.pdf"


def _extract_pdf_pages(file_path: Path):
    reader = PdfReader(str(file_path))
    page_texts = []
    full_text_parts = []

    for page_index, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        page_number = page_index + 1
        page_texts.append((page_number, text))
        full_text_parts.append(text)

    full_text = "\n\n".join(full_text_parts)
    return page_texts, full_text

# pydantic models
#data the /upload endpoint expects
class DocumentUpload(BaseModel):
    document_id: str        
    document_name: str     
    content: str

# data the /query endpoint expects
class QueryRequest(BaseModel):
    question: str         
    top_k: int = 3      



# fastapi
app = FastAPI(
    title="ai knowledge assistant api",
    description="upload documents and ask questions using RAG",
    version="1.0.0"
)
app.mount("/files", StaticFiles(directory=str(UPLOADS_DIR)), name="files")

# cors middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://127.0.0.1:3000"   # Alternative localhost
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow GET, POST, etc.
    allow_headers=["*"],  # Allow all headers
)

pinecone_index = None

# startup event
@app.on_event("startup")
async def startup_event():
    global pinecone_index
    #pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    print("Initializing Pinecone connection...")
    pinecone_index = init_pinecone(api_key, index_name="knowledge-base")
    print("Pinecone ready!")
    #openai
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("openai key not found")
    else:
        print("openai API key found!")
    # Pre-load embedding model at startup (uses local cache if available)
    print("Pre-loading embedding model...")
    get_embedding_model()
    print("Embedding model ready!")

@app.get("/")
async def root():
    # http://localhost:8000/
    return {
        "message": "AI Knowledge Assistant API is running!",
        "status": "healthy",
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "pinecone_configured": pinecone_index is not None,
        "openai_configured": os.getenv("OPENAI_API_KEY") is not None
    }


#uploading documents
@app.post("/upload")
async def upload_document(doc: DocumentUpload): 
    if pinecone_index is None:
        raise HTTPException(
            status_code=500,
            detail="Pinecone not initialized"
        )
    
    try:
        request_started_at = time.perf_counter()
        print(
            f"[upload] started document_id={doc.document_id} "
            f"name={doc.document_name} chars={len(doc.content)}"
        )

        num_chunks = store_chunks_in_pinecone(
            index=pinecone_index,
            text=doc.content,
            document_id=doc.document_id,
            document_name=doc.document_name
        )

        total_duration = time.perf_counter() - request_started_at
        print(
            f"[upload] finished document_id={doc.document_id} "
            f"chunks={num_chunks} total={total_duration:.2f}s"
        )
        
        return {
            "success": True,
            "message": f"Stored {num_chunks} chunks",
            "document_id": doc.document_id,
            "num_chunks": num_chunks
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


@app.post("/upload-pdf")
async def upload_pdf_document(
    document_id: str = Form(...),
    document_name: str = Form(...),
    file: UploadFile = File(...)
):
    if pinecone_index is None:
        raise HTTPException(status_code=500, detail="Pinecone not initialized")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    request_started_at = time.perf_counter()
    stored_file_path = _pdf_storage_path(document_id)

    try:
        with stored_file_path.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file)

        page_texts, full_text = _extract_pdf_pages(stored_file_path)
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        num_chunks = store_chunks_in_pinecone(
            index=pinecone_index,
            text=full_text,
            document_id=document_id,
            document_name=document_name,
            page_texts=page_texts,
            file_type="pdf",
            source_url=_public_pdf_url(document_id)
        )

        total_duration = time.perf_counter() - request_started_at
        print(
            f"[upload-pdf] finished document_id={document_id} pages={len(page_texts)} "
            f"chunks={num_chunks} total={total_duration:.2f}s"
        )

        return {
            "success": True,
            "message": f"Stored {num_chunks} chunks",
            "document_id": document_id,
            "document_name": document_name,
            "num_chunks": num_chunks,
            "pdf_url": _public_pdf_url(document_id),
            "content": full_text
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    finally:
        file.file.close()

#query 
@app.post("/query")
async def query_documents(request: QueryRequest):
    if pinecone_index is None:
        raise HTTPException(
            status_code=500,
            detail="Pinecone not initialized"
        )
    
    try:
        matches = query_pinecone(
            index=pinecone_index,
            query_text=request.question,
            top_k=request.top_k
        )
        
        if not matches:
            return {
                "success": True,
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": []
            }
        
        answer = generate_response(
            query=request.question,
            context_chunks=matches
        )
        
        # for frontend
        sources = []
        for match in matches:
            metadata = match.get('metadata', {})
            raw_content = metadata.get('content', '')
            excerpt = _build_relevant_excerpt(raw_content, request.question)

            sources.append({
                "document_id": metadata.get('document_id', 'unknown'),
                "document_name": metadata.get('document_name', 'Unknown'),
                "chunk_index": metadata.get('chunk_index', 0),
                "score": float(match['score']),
                "content": excerpt,
                "page_number": metadata.get('page_number'),
                "pdf_url": metadata.get('source_url')
            })
        
        return {
            "success": True,
            "answer": answer,
            "sources": sources
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

# get list of documents
@app.get("/documents")
async def get_documents():
    if pinecone_index is None:
        raise HTTPException(
            status_code=500,
            detail="Pinecone not initialized"
        )
    
    try:
        documents = get_all_documents(pinecone_index)
        
        return {
            "success": True,
            "total_documents": len(documents),
            "documents": documents,
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving documents: {str(e)}"
        )

# delete a document
@app.delete("/documents/{document_id}")
async def delete_document_endpoint(document_id: str):
    if pinecone_index is None:
        raise HTTPException(
            status_code=500,
            detail="Pinecone not initialized"
        )
    
    try:
        success = delete_document(pinecone_index, document_id)
        pdf_path = _pdf_storage_path(document_id)
        if pdf_path.exists():
            pdf_path.unlink()
        
        if success:
            return {
                "success": True,
                "message": f"Document {document_id} deleted successfully"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to delete document"
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )

@app.post("/initialize-dataset")
async def initialize_dataset():
    if pinecone_index is None:
        raise HTTPException(
            status_code=500,
            detail="Pinecone not initialized"
        )
    
    try:
        import os
        import glob
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Get dataset directory (one level up from backend folder)
        dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
        dataset_dir = os.path.abspath(dataset_dir)
        
        if not os.path.exists(dataset_dir):
            raise HTTPException(
                status_code=404,
                detail=f"Dataset directory not found: {dataset_dir}"
            )
        
        # Find all markdown files except README
        md_files = glob.glob(os.path.join(dataset_dir, '*.md'))
        md_files = [f for f in md_files if not f.endswith('README.md')]
        md_files.sort()  # Sort for consistent ordering
        
        def process_file(filepath):
            """Process a single file and return its info"""
            # Read file content
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract title from first line (assuming it starts with #)
            lines = content.split('\n')
            title_line = next((line for line in lines if line.strip().startswith('#')), None)
            if title_line:
                title = title_line.replace('#', '').strip()
            else:
                title = os.path.basename(filepath).replace('.md', '')
            
            # Generate document ID from filename
            doc_id = os.path.basename(filepath).replace('.md', '')
            
            # Upload to Pinecone
            num_chunks = store_chunks_in_pinecone(
                index=pinecone_index,
                text=content,
                document_id=doc_id,
                document_name=title
            )
            
            return {
                "id": doc_id,
                "title": title,
                "chunks": num_chunks
            }
        
        # Optimized: Process files in parallel using ThreadPoolExecutor
        uploaded_docs = []
        total_chunks = 0
        
        # Use 4 workers for parallel processing (adjust based on your CPU)
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {executor.submit(process_file, filepath): filepath for filepath in md_files}
            
            for future in as_completed(future_to_file):
                try:
                    result = future.result()
                    uploaded_docs.append(result)
                    total_chunks += result["chunks"]
                except Exception as e:
                    filepath = future_to_file[future]
                    print(f"Error processing {filepath}: {str(e)}")
        
        # Sort results by id for consistent ordering
        uploaded_docs.sort(key=lambda x: x["id"])
        
        return {
            "success": True,
            "message": "Dataset initialized successfully",
            "documents_uploaded": len(uploaded_docs),
            "total_chunks": total_chunks,
            "documents": uploaded_docs
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error initializing dataset: {str(e)}"
        )

# New endpoint to generate suggested questions based on document content
@app.post("/generate-questions")
async def generate_suggested_questions(doc: DocumentUpload):
    """
    Generate 3 suggested questions based on the content of the uploaded document.
    This endpoint takes the same document data and uses OpenAI to create relevant questions.
    """
    try:
        def normalize_question(question: str, max_chars: int = 65) -> str:
            cleaned = question.strip().lstrip("-*0123456789. ").strip()
            if not cleaned:
                return ""
            if len(cleaned) <= max_chars:
                return cleaned
            return cleaned[: max_chars - 1].rstrip() + "…"

        request_started_at = time.perf_counter()
        print(
            f"[generate-questions] started document_id={doc.document_id} "
            f"name={doc.document_name} chars={len(doc.content)}"
        )

        # Get OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Truncate content if too long (OpenAI has token limits)
        max_content_length = 3000
        content = doc.content
        if len(content) > max_content_length:
            content = content[:max_content_length] + "...[truncated]"
        
        # Create a prompt for short, lightweight suggested questions
        prompt = f"""Generate exactly 3 short suggested questions for this document.

Requirements:
- Keep each question simple and easy to read.
- Maximum 10 words per question.
- Each question must be clearly about this document.
- No numbering or bullet points.
- One question per line.

Document Title: {doc.document_name}
Document Content: {content}
"""

        openai_started_at = time.perf_counter()
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using the same model as in generate_response
            messages=[
                {
                    "role": "system", 
                    "content": "You create short, friendly question prompts for a chat UI."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,  # Slightly higher temperature for more creative questions
            max_tokens=200    # Limit response length
        )
        openai_duration = time.perf_counter() - openai_started_at
        
        # Parse the response to extract questions
        questions_text = response.choices[0].message.content.strip()
        questions = [normalize_question(q) for q in questions_text.split('\n') if q.strip()]
        questions = [q for q in questions if q]
        
        # Limit to maximum 3 questions, but allow fewer
        if len(questions) > 3:
            questions = questions[:3]

        total_duration = time.perf_counter() - request_started_at
        print(
            f"[generate-questions] finished document_id={doc.document_id} "
            f"questions={len(questions)} openai={openai_duration:.2f}s total={total_duration:.2f}s"
        )
        
        return {
            "success": True,
            "document_id": doc.document_id,
            "document_name": doc.document_name,
            "suggested_questions": questions
        }
    
    except Exception as e:
        # Log error and return empty questions
        total_duration = time.perf_counter() - request_started_at
        print(
            f"[generate-questions] failed document_id={doc.document_id} "
            f"after={total_duration:.2f}s error={str(e)}"
        )
        
        return {
            "success": False,
            "document_id": doc.document_id,
            "document_name": doc.document_name,
            "suggested_questions": [],
            "error": str(e)
        }

