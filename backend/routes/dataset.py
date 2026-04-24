"""
initialize dataset from files in dataset directory
question generation for uploaded document
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import glob
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

from state import get_pinecone_index
from main import store_chunks_in_pinecone
from document_store import register_document, document_exists
from services.pdf_service import extract_pdf_pages
from services.r2_service import upload_pdf, pdf_exists as r2_pdf_exists, get_pdf_url

router = APIRouter(tags=["dataset"])

# validation document upload for question generation
class DocumentUpload(BaseModel):
    document_id: str
    document_name: str
    content: str


@router.post("/initialize-dataset")
async def initialize_dataset():
    """
    initialize the dataset by uploading the files to pinecone
    1. find files
    2. process files md and pdf (get title, store in pinecone)
    3. use 4 worker threads to process files concurrently
    """
    index = get_pinecone_index()

    try:
        dataset_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset')
        dataset_dir = os.path.abspath(dataset_dir)

        if not os.path.exists(dataset_dir):
            raise HTTPException(status_code=404, detail=f"Dataset directory not found: {dataset_dir}")

        md_files = [f for f in glob.glob(os.path.join(dataset_dir, '*.md')) if not f.endswith('README.md')]
        md_files.sort()

        pdf_files = glob.glob(os.path.join(dataset_dir, '*.pdf'))
        pdf_files.sort()

        def process_md_file(filepath):
            doc_id = os.path.basename(filepath).replace('.md', '')

            # Skip if already processed
            if document_exists(doc_id):
                print(f"[initialize-dataset] Skipping {doc_id}.md - already exists")
                return {"id": doc_id, "title": doc_id, "chunks": 0, "type": "md", "skipped": True}

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')
            title_line = next((line for line in lines if line.strip().startswith('#')), None)
            title = title_line.replace('#', '').strip() if title_line else doc_id

            num_chunks = store_chunks_in_pinecone(index=index, text=content, document_id=doc_id, document_name=title)
            register_document(document_id=doc_id, document_name=title, file_type="md", num_chunks=num_chunks)

            return {"id": doc_id, "title": title, "chunks": num_chunks, "type": "md"}

        def process_pdf_file(filepath):
            filename = os.path.basename(filepath)
            doc_id = f"demo_{filename.replace('.pdf', '').replace(' ', '_').lower()}"

            # Skip if already processed (exists in both document store and R2)
            if document_exists(doc_id) and r2_pdf_exists(doc_id):
                r2_url = get_pdf_url(doc_id)
                print(f"[initialize-dataset] Skipping {filename} - already exists in R2")
                return {"id": doc_id, "title": filename, "chunks": 0, "type": "pdf", "pdf_url": r2_url, "skipped": True}

            page_texts, full_text = extract_pdf_pages(Path(filepath))

            if not full_text.strip():
                print(f"[initialize-dataset] Skipping {filename} - no extractable text")
                return None

            # Upload to R2
            r2_url = upload_pdf(doc_id, Path(filepath))
            print(f"[initialize-dataset] uploaded to R2: {r2_url}")

            num_chunks = store_chunks_in_pinecone(
                index=index,
                text=full_text,
                document_id=doc_id,
                document_name=filename,
                page_texts=page_texts,
                file_type="pdf",
                source_url=r2_url
            )

            register_document(
                document_id=doc_id,
                document_name=filename,
                file_type="pdf",
                pdf_url=r2_url,
                num_chunks=num_chunks
            )

            return {"id": doc_id, "title": filename, "chunks": num_chunks, "type": "pdf", "pdf_url": r2_url}

        processed_docs = []
        total_chunks = 0
        skipped_count = 0

        with ThreadPoolExecutor(max_workers=4) as executor:
            all_tasks = [(executor.submit(process_md_file, f), f) for f in md_files]
            all_tasks += [(executor.submit(process_pdf_file, f), f) for f in pdf_files]

            for future, filepath in all_tasks:
                try:
                    result = future.result()
                    if result:
                        processed_docs.append(result)
                        if result.get("skipped"):
                            skipped_count += 1
                        else:
                            total_chunks += result["chunks"]
                            print(f"[initialize-dataset] Processed {result['title']}: {result['chunks']} chunks")
                except Exception as e:
                    print(f"[initialize-dataset] Error processing {filepath}: {str(e)}")

        processed_docs.sort(key=lambda x: x["id"])
        new_uploads = len(processed_docs) - skipped_count

        return {
            "success": True,
            "message": f"Dataset initialized: {new_uploads} new, {skipped_count} skipped (already exist)",
            "documents_uploaded": new_uploads,
            "documents_skipped": skipped_count,
            "total_chunks": total_chunks,
            "documents": processed_docs
        }
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing dataset: {str(e)}")


def sample_document_content(text: str, total_chars: int = 3000) -> str:
    """
    Sample content from beginning, middle, and end of document.
    This gives broader coverage than just taking the first N characters.
    """
    if len(text) <= total_chars:
        return text
    
    chunk_size = total_chars // 3  # ~1000 chars each
    
    beginning = text[:chunk_size]
    
    middle_start = (len(text) - chunk_size) // 2
    middle = text[middle_start:middle_start + chunk_size]
    
    end = text[-chunk_size:]
    
    return f"{beginning}\n\n[...middle section...]\n\n{middle}\n\n[...end section...]\n\n{end}"


@router.post("/generate-questions")
async def generate_suggested_questions(doc: DocumentUpload):
    """
    generate 3 suggested questions based on the uploaded document content
    1. sample content from beginning, middle, and end (3000 chars total)
    2. prompt openai to generate questions
    3. normalize questions
    4. return questions
    """
    request_started_at = time.perf_counter()

    try:
        print(f"[generate-questions] started document_id={doc.document_id} name={doc.document_name} chars={len(doc.content)}")

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        content = sample_document_content(doc.content, total_chars=3000)

        prompt = f"""Generate 3 SHORT questions about this document.

CRITICAL: Keep questions under 6 words. Be simple and direct.
Cover different topics from the document (not just the beginning).

Good examples:
- What is a pointer?
- How do arrays work?
- What does malloc do?

Rules:
- MAX 6 words per question
- Simple vocabulary
- No numbering
- One per line
- Questions should cover different parts/topics

Document: {doc.document_name}
Content (sampled from beginning, middle, end):
{content}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You generate very short, simple questions. Maximum 6 words."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )

        def normalize(q: str, max_chars: int = 40) -> str:
            cleaned = q.strip().lstrip("-*0123456789. ").strip()
            return cleaned if cleaned and len(cleaned) <= max_chars else ""

        questions = [normalize(q) for q in response.choices[0].message.content.strip().split('\n') if q.strip()]
        questions = [q for q in questions if q][:3]

        print(f"[generate-questions] finished document_id={doc.document_id} questions={len(questions)} total={time.perf_counter() - request_started_at:.2f}s")

        return {
            "success": True,
            "document_id": doc.document_id,
            "document_name": doc.document_name,
            "suggested_questions": questions
        }

    except Exception as e:
        print(f"[generate-questions] failed document_id={doc.document_id} after={time.perf_counter() - request_started_at:.2f}s error={str(e)}")
        return {
            "success": False,
            "document_id": doc.document_id,
            "document_name": doc.document_name,
            "suggested_questions": [],
            "error": str(e)
        }
