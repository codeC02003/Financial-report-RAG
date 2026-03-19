"""FastAPI backend for Multimodal Document QA."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import io
import yaml
import base64
import logging
import asyncio
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.qa_engine.engine import DocumentQAEngine

logger = logging.getLogger(__name__)

engine: DocumentQAEngine | None = None


def load_config() -> dict:
    config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    config = load_config()
    engine = DocumentQAEngine(config)
    logger.info("QA Engine initialized.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Multimodal Document QA",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5
    history: list[ChatMessage] = []


class QAResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    is_unanswerable: bool
    source_pages: list[int]
    evidence: list[dict]
    used_vision: bool


class DocumentInfo(BaseModel):
    file_path: str
    total_pages: int
    pages_processed: int
    total_chunks: int
    pages_with_tables: int
    pages_with_charts: int


@app.post("/api/upload", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a PDF document."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    max_size = load_config().get("api", {}).get("max_file_size_mb", 50) * 1024 * 1024
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(status_code=400, detail="File too large.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Run CPU-heavy processing in thread pool so it doesn't block
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, engine.load_document, tmp_path)
        return DocumentInfo(**info)
    except Exception as e:
        logger.error(f"Failed to process document: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/api/ask", response_model=QAResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about the uploaded document."""
    if not engine or not engine.document:
        raise HTTPException(status_code=400, detail="No document loaded. Upload a PDF first.")

    try:
        loop = asyncio.get_event_loop()
        history = [{"role": m.role, "content": m.content} for m in request.history]
        result = await loop.run_in_executor(
            None, partial(engine.ask, request.question, top_k=request.top_k, history=history)
        )
        return QAResponse(
            question=result.question,
            answer=result.answer,
            confidence=result.confidence,
            is_unanswerable=result.is_unanswerable,
            source_pages=result.source_pages,
            evidence=result.evidence_chunks,
            used_vision=result.used_vision,
        )
    except Exception as e:
        logger.error(f"QA failed: {e}")
        raise HTTPException(status_code=500, detail=f"QA failed: {str(e)}")


@app.get("/api/page/{page_number}")
async def get_page_image(page_number: int):
    """Get a rendered page image as base64 (on-demand rendering)."""
    if not engine or not engine.document:
        raise HTTPException(status_code=400, detail="No document loaded.")

    loop = asyncio.get_event_loop()
    image = await loop.run_in_executor(None, engine.get_page_image, page_number)
    if image is None:
        raise HTTPException(status_code=404, detail=f"Page {page_number} not found.")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"page_number": page_number, "image": img_base64}


@app.get("/api/document")
async def get_document_info():
    if not engine or not engine.document:
        raise HTTPException(status_code=400, detail="No document loaded.")
    return engine.get_document_summary()


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "model_loaded": engine is not None and engine.vision_model.model is not None}
