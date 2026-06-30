"""
FastAPI application entry point — Phase 1 + Phase 2.

Phase 1: Document ingestion and parsing.
Phase 2: Semantic chunking, embedding, and vector search.

Routers registered:
  POST /api/v1/upload  — Phase 1 document ingestion
  POST /api/v1/index   — Phase 2 document indexing
  POST /api/v1/search  — Phase 2 semantic search
  GET  /health         — System health (includes Qdrant status)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.upload import router as upload_router
from api.index import router as index_router
from api.search import router as search_router
from utils.file_utils import ensure_directories
from utils.logger import get_logger
from vector_db.qdrant_client import ensure_collection, qdrant_health_check

logger = get_logger(__name__)

REQUIRED_DIRS = [
    Path("uploads"),
    Path("output/json"),
    Path("output/images"),
    Path("logs"),
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle handler."""
    logger.info("=" * 60)
    logger.info("  Multimodal RAG — Phase 1 + Phase 2")
    logger.info("  Document Ingestion + Semantic Search")
    logger.info("=" * 60)

    ensure_directories(*REQUIRED_DIRS)

    # Initialise Qdrant collection on startup
    try:
        ensure_collection()
        logger.info("Qdrant collection ready.")
    except Exception as exc:
        logger.warning("Qdrant not available at startup: %s", exc)
        logger.warning("Indexing and search will fail until Qdrant is running.")

    logger.info("Application ready.")
    yield
    logger.info("Application shutting down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Multimodal RAG — Research Assistant API",
        description=(
            "Phase 1: Document ingestion pipeline (PDF, images, DOCX, PPTX).\n"
            "Phase 2: Semantic chunking, BGE-M3 embeddings, Qdrant vector search."
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    prefix = "/api/v1"
    app.include_router(upload_router, prefix=prefix, tags=["Phase 1 — Ingestion"])
    app.include_router(index_router, prefix=prefix, tags=["Phase 2 — Indexing"])
    app.include_router(search_router, prefix=prefix, tags=["Phase 2 — Search"])

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    # ── Enhanced health check ─────────────────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health_check() -> dict:
        """System health including Qdrant connectivity."""
        qdrant_status = qdrant_health_check()
        return {
            "status": "healthy",
            "phase": 2,
            "service": "multimodal-rag",
            "version": "2.0.0",
            "qdrant": qdrant_status,
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)