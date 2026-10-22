from app.rag_engine import answer_question
from pydantic import BaseModel, Field

from uuid import uuid4
from app.vector_store import VectorStore

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.chunking import chunk_document

import pymupdf



app = FastAPI(
    title="AI Document Intelligence Assistant",
    description="A RAG-powered API for asking questions about PDF documents.",
    version="1.0.0",
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)
    document_id: str | None = None



class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question about an uploaded PDF",
    )
    document_id: str | None = Field(
        default=None,
        max_length=100,
    )
    top_k: int = Field(default=3, ge=1, le=20)




@app.get("/")
def home():
    return {
        "message": "Welcome to the AI Document Intelligence Assistant",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF content type.",
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds the 10 MB size limit.",
        )

    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid PDF.",
        )

    
    try:
        pages = []

        with pymupdf.open(stream=content, filetype="pdf") as document:
            if document.needs_pass:
                raise ValueError("Password-protected PDFs are not supported.")

            for page_index, page in enumerate(document):
                text = page.get_text("text").strip()

                if text:
                    pages.append({
                        "page_number": page_index + 1,
                        "text": text,
                    })

    except (pymupdf.FileDataError, pymupdf.EmptyFileError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to process PDF: {exc}",
        ) from exc

    finally:
        await file.close()


   
    chunks = chunk_document(
        pages,
        chunk_size=1000,
        chunk_overlap=200,
    )


    document_id = str(uuid4())

    stored_chunks = 0

    if chunks:
        try:
            vector_store = VectorStore()
            stored_chunks = vector_store.add_chunks(
                chunks,
                document_id=document_id,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Failed to index the uploaded document.",
            ) from exc


    
    return {
        "document_id": document_id,
        "filename": file.filename,
        "pages_extracted": len(pages),
        "total_characters": sum(len(page["text"]) for page in pages),
        "total_chunks": len(chunks),
        "stored_chunks": stored_chunks,
        "pages": pages,
        "chunks": chunks,
    }


@app.post("/search")
def search_documents(request: SearchRequest):
    """Search uploaded PDF documents using semantic similarity."""

    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty.",
        )

    try:
        vector_store = VectorStore()

       
        results = vector_store.search(
            query=request.query,
            top_k=request.top_k,
            document_id=request.document_id,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Document search failed.",
        ) from exc

    return {
        "query": request.query,
        "total_results": len(results),
        "results": results,
    }


@app.post("/ask")
def ask_document(request: AskRequest):
    """Answer questions using uploaded PDF documents."""

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        result = answer_question(
            question=request.question,
            document_id=request.document_id,
            top_k=request.top_k,
        )

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Unable to generate an AI answer right now.",
        )

