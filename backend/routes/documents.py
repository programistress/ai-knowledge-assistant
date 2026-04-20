"""
Document management routes - upload, list, delete documents.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import time
import shutil

from state import get_pinecone_index
from main import store_chunks_in_pinecone, delete_document as delete_from_pinecone
from document_store import register_document, unregister_document, get_all_documents as get_all_documents_from_store
from services.pdf_service import pdf_storage_path, pdf_public_url, extract_pdf_pages

router = APIRouter(tags=["documents"])


class DocumentUpload(BaseModel):
    document_id: str
    document_name: str
    content: str


@router.post("/upload")
async def upload_document(doc: DocumentUpload):
    index = get_pinecone_index()

    try:
        request_started_at = time.perf_counter()
        print(f"[upload] started document_id={doc.document_id} name={doc.document_name} chars={len(doc.content)}")

        num_chunks = store_chunks_in_pinecone(
            index=index,
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
        print(f"[upload] finished document_id={doc.document_id} chunks={num_chunks} total={total_duration:.2f}s")

        return {
            "success": True,
            "message": f"Stored {num_chunks} chunks",
            "document_id": doc.document_id,
            "num_chunks": num_chunks
        }
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.post("/upload-pdf")
async def upload_pdf_document(
    document_id: str = Form(...),
    document_name: str = Form(...),
    file: UploadFile = File(...)
):
    print(f"[upload-pdf] received: document_id={document_id}, document_name={document_name}, filename={file.filename}")
    index = get_pinecone_index()

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    request_started_at = time.perf_counter()
    stored_file_path = pdf_storage_path(document_id)

    try:
        with stored_file_path.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file)

        page_texts, full_text = extract_pdf_pages(stored_file_path)
        if not full_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Sorry, we can't read text from your PDF :( It might be a scanned document or image-based."
            )

        num_chunks = store_chunks_in_pinecone(
            index=index,
            text=full_text,
            document_id=document_id,
            document_name=document_name,
            page_texts=page_texts,
            file_type="pdf",
            source_url=pdf_public_url(document_id)
        )

        register_document(
            document_id=document_id,
            document_name=document_name,
            file_type="pdf",
            pdf_url=pdf_public_url(document_id),
            num_chunks=num_chunks
        )

        total_duration = time.perf_counter() - request_started_at
        print(f"[upload-pdf] finished document_id={document_id} pages={len(page_texts)} chunks={num_chunks} total={total_duration:.2f}s")

        return {
            "success": True,
            "message": f"Stored {num_chunks} chunks",
            "document_id": document_id,
            "document_name": document_name,
            "num_chunks": num_chunks,
            "pdf_url": pdf_public_url(document_id),
            "content": full_text
        }
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    finally:
        file.file.close()


@router.get("/documents")
async def get_documents():
    try:
        documents = get_all_documents_from_store()
        return {
            "success": True,
            "total_documents": len(documents),
            "documents": documents,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")


@router.delete("/documents/{document_id}")
async def delete_document_endpoint(document_id: str):
    index = get_pinecone_index()

    try:
        pinecone_success = delete_from_pinecone(index, document_id)
        unregister_document(document_id)

        pdf_path = pdf_storage_path(document_id)
        if pdf_path.exists():
            pdf_path.unlink()

        if pinecone_success:
            return {"success": True, "message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete document")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@router.delete("/documents")
async def clear_all_documents():
    """Delete all documents from the knowledge base."""
    index = get_pinecone_index()

    try:
        documents = get_all_documents_from_store()
        deleted_count = 0

        for doc in documents:
            doc_id = doc.get("document_id")
            if doc_id:
                delete_from_pinecone(index, doc_id)
                unregister_document(doc_id)

                pdf_path = pdf_storage_path(doc_id)
                if pdf_path.exists():
                    pdf_path.unlink()

                deleted_count += 1

        return {
            "success": True,
            "message": f"Deleted {deleted_count} documents",
            "deleted_count": deleted_count
        }
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")
