"""
startup, health checks
"""
print("=== Loading API module ===", flush=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()
print("Environment loaded", flush=True)

from main import init_pinecone, get_embedding_model
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

# CORS middleware - allow frontend origins from env or defaults
FRONTEND_ORIGINS = os.getenv("FRONTEND_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
origins = [origin.strip() for origin in FRONTEND_ORIGINS.split(",")]
print(f"CORS origins configured: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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
    print("=== Starting AI Knowledge Assistant API ===", flush=True)
    
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print("ERROR: PINECONE_API_KEY not found!", flush=True)
        else:
            print("Initializing Pinecone connection...", flush=True)
            index = init_pinecone(api_key, index_name="knowledge-base")
            state.set_pinecone_index(index)
            print("Pinecone ready!", flush=True)
    except Exception as e:
        print(f"ERROR initializing Pinecone: {e}", flush=True)

    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OpenAI key not found", flush=True)
    else:
        print("OpenAI API key found!", flush=True)

    try:
        print("Pre-loading embedding model (this may take a minute)...", flush=True)
        get_embedding_model()
        print("Embedding model ready!", flush=True)
    except Exception as e:
        print(f"ERROR loading embedding model: {e}", flush=True)
    
    print("=== Startup complete ===", flush=True)


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
        "openai_configured": os.getenv("OPENAI_API_KEY") is not None,
        "cors_origins": origins
    }
