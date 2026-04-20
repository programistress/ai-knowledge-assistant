"""
Query routes - handle RAG queries against the knowledge base.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from state import get_pinecone_index
from main import query_pinecone, generate_response
from services.query_service import is_knowledge_inquiry, chat_response
from utils.text_utils import build_relevant_excerpt

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


@router.post("/query")
async def query_documents(request: QueryRequest):
    index = get_pinecone_index()

    # If not a knowledge inquiry, just chat normally
    if not is_knowledge_inquiry(request.question):
        try:
            answer = chat_response(request.question)
            return {"success": True, "answer": answer, "sources": []}
        except Exception as e:
            print(f"[query] chat response failed: {e}")

    try:
        matches = query_pinecone(
            index=index,
            query_text=request.question,
            top_k=request.top_k
        )

        if not matches:
            try:
                answer = chat_response(request.question)
            except:
                answer = "I couldn't find any relevant information in your documents."
            return {"success": True, "answer": answer, "sources": []}

        answer = generate_response(query=request.question, context_chunks=matches)

        # Build sources for frontend
        sources = []
        for match in matches:
            metadata = match.get('metadata', {})
            raw_content = metadata.get('content', '')
            excerpt = build_relevant_excerpt(raw_content, request.question)

            sources.append({
                "document_id": metadata.get('document_id', 'unknown'),
                "document_name": metadata.get('document_name', 'Unknown'),
                "chunk_index": metadata.get('chunk_index', 0),
                "score": float(match['score']),
                "content": excerpt,
                "page_number": metadata.get('page_number'),
                "pdf_url": metadata.get('source_url')
            })

        return {"success": True, "answer": answer, "sources": sources}

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        import traceback
        print(f"[query] ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
