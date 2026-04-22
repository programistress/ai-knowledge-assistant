"""
startup, health checks
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv

load_dotenv()

from main import init_pinecone, get_embedding_model
from services.pdf_service import UPLOADS_DIR
import state

# Import routers
from routes.documents import router as documents_router
from routes.query import router as query_router
from routes.dataset import router as dataset_router

# FastAPI initialization
app = FastAPI(
    title="AI Knowledge Assistant API",
    description="Upload documents and ask questions using RAG",
    version="1.0.0"
)

# Static file server for PDFs
app.mount("/files", StaticFiles(directory=str(UPLOADS_DIR)), name="files")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# register all the routes
app.include_router(documents_router)
app.include_router(query_router)
app.include_router(dataset_router)


@app.on_event("startup")
async def startup_event():
    """Initialize Pinecone and pre-load the embedding model."""
    api_key = os.getenv("PINECONE_API_KEY")
    print("Initializing Pinecone connection...")
    index = init_pinecone(api_key, index_name="knowledge-base")
    state.set_pinecone_index(index)
    print("Pinecone ready!")

    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OpenAI key not found")
    else:
        print("OpenAI API key found!")

    print("Pre-loading embedding model...")
    get_embedding_model()
    print("Embedding model ready!")


# is server running check
@app.get("/")
async def root():
    return {"message": "AI Knowledge Assistant API is running!", "status": "healthy"}

# status check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "pinecone_configured": state.pinecone_index is not None,
        "openai_configured": os.getenv("OPENAI_API_KEY") is not None
    }
