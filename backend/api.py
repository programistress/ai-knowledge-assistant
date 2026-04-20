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
    delete_document as delete_from_pinecone,
    get_embedding_model
)
from document_store import (
    register_document,
    unregister_document,
    get_all_documents as get_all_documents_from_store
)
from openai import OpenAI

# for pdf access functionality !!!! need to change for deployment
UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# fastapi initialization
app = FastAPI(
    title="ai knowledge assistant api",
    description="upload documents and ask questions using RAG",
    version="1.0.0"
)
# static file server for pdfs !!!! need to change for deployment
app.mount("/files", StaticFiles(directory=str(UPLOADS_DIR)), name="files")

# cors middleware to communicate with frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://127.0.0.1:3000"   # Alternative localhost
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  
)

# deferred initialization pattern
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


COMMON_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "what",
    "when", "where", "which", "who", "why", "with", "your", "you", "about"
}

BOILERPLATE_TERMS = {
    "all rights reserved", "copyright", "publisher", "published by", "isbn",
    "acknowledg", "dedication", "table of contents"
}

def _is_knowledge_inquiry(text: str) -> bool:
    """Check if the message is asking for information that would need document lookup."""
    cleaned = text.lower().strip()
    
    # Question indicators
    question_starters = ["what", "how", "why", "when", "where", "which", "who", 
                         "explain", "describe", "tell me", "can you", "could you",
                         "define", "show me", "give me", "list", "compare"]
    
    # Check for question patterns
    if any(cleaned.startswith(q) for q in question_starters):
        return True
    if "?" in text:
        return True
    
    # Check for specific topic inquiry (more than 3 meaningful words)
    keywords = _extract_query_keywords(text)
    if len(keywords) >= 2:
        return True
    
    return False

def _chat_response(message: str) -> str:
    """Generate a simple chat response without document lookup."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a friendly knowledge assistant. Keep responses brief and natural."},
            {"role": "user", "content": message}
        ],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()


def _extract_query_keywords(question: str):
    words = re.findall(r"[a-zA-Z0-9\+\#]{3,}", question.lower())
    return [word for word in words if word not in COMMON_STOPWORDS]


def _is_likely_boilerplate(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in BOILERPLATE_TERMS)


def _clean_garbled_text(text: str) -> str:
    """Remove garbled/non-printable characters from text."""
    if not text:
        return ""
    # Keep only printable ASCII and common unicode, remove control chars and garbled sequences
    cleaned = re.sub(r'[^\x20-\x7E\u00A0-\u00FF\u0100-\u017F\u0400-\u04FF\n]', '', text)
    # Remove sequences of repeated special chars that indicate encoding issues
    cleaned = re.sub(r'[�\ufffd]{2,}', '', cleaned)
    # Collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def _build_relevant_excerpt(content: str, question: str, max_chars: int = 220) -> str:
    if not content:
        return ""

    # Clean garbled text first
    clean_content = _clean_garbled_text(content)
    clean_content = re.sub(r"\s+", " ", clean_content).strip()
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


def _fix_pdf_spacing(text: str) -> str:
    """Fix common PDF extraction spacing issues."""
    import re
    # Add space before uppercase letter that follows lowercase (camelCase -> camel Case)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Add space before uppercase letter that follows punctuation without space
    text = re.sub(r'([.!?,;:])([A-Z])', r'\1 \2', text)
    # Fix multiple spaces
    text = re.sub(r' +', ' ', text)
    return text


def _extract_pdf_pages(file_path: Path):
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

        # Fix spacing issues
        text = _fix_pdf_spacing(text)

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

        register_document(
            document_id=doc.document_id,
            document_name=doc.document_name,
            file_type="note",
            num_chunks=num_chunks
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
    print(f"[upload-pdf] received: document_id={document_id}, document_name={document_name}, filename={file.filename}, content_type={file.content_type}")
    
    if pinecone_index is None:
        raise HTTPException(status_code=500, detail="Pinecone not initialized")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        print(f"[upload-pdf] rejected: filename={file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    request_started_at = time.perf_counter()
    stored_file_path = _pdf_storage_path(document_id)

    try:
        with stored_file_path.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file)

        page_texts, full_text = _extract_pdf_pages(stored_file_path)
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="Sorry, we can't read text from your PDF :( It might be a scanned document or image-based. Try a different PDF with selectable text.")

        num_chunks = store_chunks_in_pinecone(
            index=pinecone_index,
            text=full_text,
            document_id=document_id,
            document_name=document_name,
            page_texts=page_texts,
            file_type="pdf",
            source_url=_public_pdf_url(document_id)
        )

        register_document(
            document_id=document_id,
            document_name=document_name,
            file_type="pdf",
            pdf_url=_public_pdf_url(document_id),
            num_chunks=num_chunks
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
    
    # If not a knowledge inquiry, just chat normally
    if not _is_knowledge_inquiry(request.question):
        try:
            answer = _chat_response(request.question)
            return {
                "success": True,
                "answer": answer,
                "sources": []
            }
        except Exception as e:
            print(f"[query] chat response failed: {e}")
    
    try:
        matches = query_pinecone(
            index=pinecone_index,
            query_text=request.question,
            top_k=request.top_k
        )
        
        if not matches:
            # Fall back to normal chat if no docs found
            try:
                answer = _chat_response(request.question)
            except:
                answer = "I couldn't find any relevant information in your documents."
            return {
                "success": True,
                "answer": answer,
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
        import traceback
        print(f"[query] ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

# get list of documents
@app.get("/documents")
async def get_documents():
    try:
        documents = get_all_documents_from_store()
        
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
        pinecone_success = delete_from_pinecone(pinecone_index, document_id)
        unregister_document(document_id)
        
        pdf_path = _pdf_storage_path(document_id)
        if pdf_path.exists():
            pdf_path.unlink()
        
        if pinecone_success:
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


@app.delete("/documents")
async def clear_all_documents():
    """Delete all documents from the knowledge base."""
    if pinecone_index is None:
        raise HTTPException(status_code=500, detail="Pinecone not initialized")
    
    try:
        documents = get_all_documents_from_store()
        deleted_count = 0
        
        for doc in documents:
            doc_id = doc.get("document_id")
            if doc_id:
                delete_from_pinecone(pinecone_index, doc_id)
                unregister_document(doc_id)
                
                pdf_path = _pdf_storage_path(doc_id)
                if pdf_path.exists():
                    pdf_path.unlink()
                
                deleted_count += 1
        
        return {
            "success": True,
            "message": f"Deleted {deleted_count} documents",
            "deleted_count": deleted_count
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing documents: {str(e)}"
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
        md_files.sort()
        
        # Find all PDF files
        pdf_files = glob.glob(os.path.join(dataset_dir, '*.pdf'))
        pdf_files.sort()
        
        def process_md_file(filepath):
            """Process a markdown file and return its info"""
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            title_line = next((line for line in lines if line.strip().startswith('#')), None)
            if title_line:
                title = title_line.replace('#', '').strip()
            else:
                title = os.path.basename(filepath).replace('.md', '')
            
            doc_id = os.path.basename(filepath).replace('.md', '')
            
            num_chunks = store_chunks_in_pinecone(
                index=pinecone_index,
                text=content,
                document_id=doc_id,
                document_name=title
            )
            
            register_document(
                document_id=doc_id,
                document_name=title,
                file_type="md",
                num_chunks=num_chunks
            )
            
            return {
                "id": doc_id,
                "title": title,
                "chunks": num_chunks,
                "type": "md"
            }
        
        def process_pdf_file(filepath):
            """Process a PDF file and return its info"""
            filename = os.path.basename(filepath)
            doc_id = f"demo_{filename.replace('.pdf', '').replace(' ', '_').lower()}"
            
            # Copy PDF to uploads folder so it can be served
            dest_path = UPLOADS_DIR / f"{doc_id}.pdf"
            shutil.copy2(filepath, dest_path)
            
            # Extract text from PDF
            page_texts, full_text = _extract_pdf_pages(Path(filepath))
            
            if not full_text.strip():
                print(f"[initialize-dataset] Skipping {filename} - no extractable text")
                return None
            
            num_chunks = store_chunks_in_pinecone(
                index=pinecone_index,
                text=full_text,
                document_id=doc_id,
                document_name=filename,
                page_texts=page_texts,
                file_type="pdf",
                source_url=_public_pdf_url(doc_id)
            )
            
            register_document(
                document_id=doc_id,
                document_name=filename,
                file_type="pdf",
                pdf_url=_public_pdf_url(doc_id),
                num_chunks=num_chunks
            )
            
            return {
                "id": doc_id,
                "title": filename,
                "chunks": num_chunks,
                "type": "pdf",
                "pdf_url": _public_pdf_url(doc_id)
            }
        
        uploaded_docs = []
        total_chunks = 0
        
        # Process all files in parallel
        all_tasks = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit markdown files
            for filepath in md_files:
                all_tasks.append((executor.submit(process_md_file, filepath), filepath))
            
            # Submit PDF files
            for filepath in pdf_files:
                all_tasks.append((executor.submit(process_pdf_file, filepath), filepath))
            
            for future, filepath in all_tasks:
                try:
                    result = future.result()
                    if result:  # PDF processing can return None
                        uploaded_docs.append(result)
                        total_chunks += result["chunks"]
                        print(f"[initialize-dataset] Processed {result['title']}: {result['chunks']} chunks")
                except Exception as e:
                    print(f"[initialize-dataset] Error processing {filepath}: {str(e)}")
        
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
        def normalize_question(question: str, max_chars: int = 40) -> str:
            cleaned = question.strip().lstrip("-*0123456789. ").strip()
            if not cleaned:
                return ""
            if len(cleaned) <= max_chars:
                return cleaned
            # If too long, just skip it rather than truncate with ...
            return ""

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
        
        # Create a prompt for short, simple questions
        prompt = f"""Generate 3 SHORT questions about this document.

CRITICAL: Keep questions under 6 words. Be simple and direct.

Good examples:
- What is a pointer?
- How do arrays work?
- What does malloc do?

Bad examples (TOO LONG):
- How can datasets be loaded using torch.utils.data.DataLoader?
- What is the difference between stack and heap memory allocation?

Rules:
- MAX 6 words per question
- Simple vocabulary
- No numbering
- One per line

Document: {doc.document_name}
Content: {content}
"""

        openai_started_at = time.perf_counter()
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using the same model as in generate_response
            messages=[
                {
                    "role": "system", 
                    "content": "You generate very short, simple questions. Maximum 6 words. Never exceed this limit."
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

