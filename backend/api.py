"""
startup, health checks
"""
print("=== Loading API module ===", flush=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()
print("Environment loaded", flush=True)

from main import init_pinecone, get_embedding_model
import state

embedding_model_ready = False

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


async def load_embedding_model_background():
    """Load embedding model in background thread to not block startup."""
    global embedding_model_ready
    try:
        print("Pre-loading embedding model in background...", flush=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, get_embedding_model)
        embedding_model_ready = True
        print("Embedding model ready!", flush=True)
    except Exception as e:
        print(f"ERROR loading embedding model: {e}", flush=True)


@app.on_event("startup")
async def startup_event():
    """Initialize Pinecone and start loading embedding model in background."""
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

    asyncio.create_task(load_embedding_model_background())
    
    print("=== Startup complete (model loading in background) ===", flush=True)


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
        "embedding_model_ready": embedding_model_ready,
        "cors_origins": origins
    }

# simple test endpoint for CORS verification
@app.get("/ping")
async def ping():
    return {"pong": True}
