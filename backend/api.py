from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

from main import (
    init_pinecone,
    store_chunks_in_pinecone,
    query_pinecone,
    generate_response,
    get_all_documents
)

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
        num_chunks = store_chunks_in_pinecone(
            index=pinecone_index,
            text=doc.content,
            document_id=doc.document_id,
            document_name=doc.document_name
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
            sources.append({
                "document_id": match['metadata'].get('document_id', 'unknown'),
                "document_name": match['metadata'].get('document_name', 'Unknown'),
                "chunk_index": match['metadata'].get('chunk_index', 0),
                "score": float(match['score']),
                "content": match['metadata'].get('content', '')[:200] + "..."  
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

