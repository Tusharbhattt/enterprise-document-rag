
from pathlib import Path
import sys
import shutil
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# Make backend/app modules importable when running
# the application through Uvicorn from the project root.
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


from parser import extract_pdf_pages
from chunker import create_semantic_chunks, generate_embeddings
from vector_store import index_chunks_to_db, query_vector_db
from llm import generate_rag_response


app = FastAPI(
    title="DocuMind - Enterprise Document Intelligence API",
    description=(
        "API for PDF processing, vector indexing, "
        "and document-based question answering."
    ),
    version="1.0.0",
)


# Allow the local React frontend to connect later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)


@app.get("/")
def root():
    return {
        "status": "online",
        "message": "DocuMind API is running!",
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload, process, embed, and index a PDF."""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    if Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    upload_dir = PROJECT_ROOT / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Use a generated filename instead of trusting the
    # filename supplied by the user.
    safe_filename = f"{uuid.uuid4().hex}.pdf"
    temp_file_path = upload_dir / safe_filename

    try:
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # PDF processing pipeline
        pages = extract_pdf_pages(str(temp_file_path))

        if not pages:
            raise HTTPException(
                status_code=400,
                detail="No readable pages found in the PDF.",
            )

        chunks = create_semantic_chunks(pages)
        embedded_chunks = generate_embeddings(chunks)

        index_chunks_to_db(embedded_chunks)

        return {
            "filename": file.filename,
            "status": "success",
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "message": "PDF processed and indexed successfully.",
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}",
        )

    finally:
        # Remove the temporary uploaded PDF after processing.
        if temp_file_path.exists():
            temp_file_path.unlink()

        await file.close()


@app.post("/api/query")
def query_document(request: QueryRequest):
    """Retrieve document chunks and return a mock RAG response."""

    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    try:
        retrieved_chunks = query_vector_db(
            request.query,
            n_results=request.top_k,
        )

        response_data = generate_rag_response(
            request.query,
            retrieved_chunks,
        )

        return {
            "query": request.query,
            **response_data,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query execution failed: {str(e)}",
        )
