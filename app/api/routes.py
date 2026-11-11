import logging
import secrets
from pathlib import Path

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Header,
    Request
)
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.rate_limit import chat_limiter, ingest_limiter
from app.rag.workflow import ask
from app.rag.vectorstore import add_documents
from app.services.ingestion import (
    load_file,
    chunk_documents,
    SUPPORTED
)
from app.services.audit import write_audit

logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE = 15 * 1024 * 1024  # 15 MB
STREAM_CHUNK_SIZE = 1024 * 1024      # 1 MB

router = APIRouter(prefix="/api")
settings = get_settings()

class ChatRequest(BaseModel):
    question: str = Field(
        min_length=2,
        max_length=3000
    )

@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.app_name
    }

@router.post("/chat")
def chat(payload: ChatRequest, request: Request):
    chat_limiter.check(request)
    try:
        result = ask(payload.question)

        write_audit(
            payload.question,
            result["source_used"],
            result.get("trace", [])
        )

        return {
            "answer": result["answer"],
            "source_used": result["source_used"],
            "trace": result.get("trace", []),
            "citations": result.get("citations", []),
            "rewritten_query": result.get(
                "current_query",
                payload.question
            )
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to process chat query: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request. Please try again or contact IT support."
        ) from exc

@router.post("/ingest")
async def ingest(
    request: Request,
    file: UploadFile = File(...),
    x_admin_key: str = Header(default="")
):
    ingest_limiter.check(request)

    if not x_admin_key or not secrets.compare_digest(x_admin_key, settings.admin_api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid admin key"
        )

    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Supported: {', '.join(sorted(SUPPORTED))}"
        )

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    dest = upload_dir / Path(file.filename).name

    try:
        size_read = 0
        with open(dest, "wb") as f_out:
            while True:
                chunk = await file.read(STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                size_read += len(chunk)
                if size_read > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum upload size of {MAX_UPLOAD_SIZE // (1024 * 1024)}MB."
                    )
                f_out.write(chunk)

        docs = load_file(dest)
        chunks = chunk_documents(docs)
        if not chunks:
            raise ValueError("Document contains no readable text content.")
        ids = add_documents(chunks)
    except HTTPException:
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise
    except ValueError as ve:
        if dest.exists():
            dest.unlink(missing_ok=True)
        logger.warning("Ingestion validation error for '%s': %s", file.filename, ve)
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        ) from ve
    except Exception as exc:
        if dest.exists():
            dest.unlink(missing_ok=True)
        logger.exception("Unexpected error during document ingestion for '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to index document due to an internal server error."
        ) from exc

    return {
        "message": "Document indexed",
        "file": dest.name,
        "chunks": len(chunks),
        "ids_created": len(ids)
    }
