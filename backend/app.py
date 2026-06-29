"""
FastAPI application entry point for the Multimodal RAG ingestion pipeline.

Phase 1: Document ingestion and parsing.

The application is structured for easy extension in later phases:
  - Phase 2: Embeddings and vector database ingestion
  - Phase 3: Multimodal retrieval
  - Phase 4: LLM integration and RAG query API
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.upload import router as upload_router
from utils.file_utils import ensure_directories
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Directories created on startup ────────────────────────────────────────────

REQUIRED_DIRS = [
    Path("uploads"),
    Path("output/json"),
    Path("output/images"),
    Path("logs"),
]


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Runs startup/shutdown logic using FastAPI's modern lifespan interface.
    All required output directories are created here so the API is
    immediately ready to handle uploads after startup.
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Multimodal RAG — Phase 1: Document Ingestion Pipeline")
    logger.info("=" * 60)

    ensure_directories(*REQUIRED_DIRS)
    logger.info("Required directories verified.")
    logger.info("Application ready. Listening for document uploads.")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Application shutting down. Goodbye.")


# ── Application factory ───────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance.
    """
    app = FastAPI(
        title="Multimodal RAG — Document Ingestion API",
        description=(
            "Phase 1 of the Multimodal Retrieval-Augmented Generation system. "
            "Accepts research documents (PDF, images, DOCX, PPTX) and extracts "
            "structured content including sections, tables, and images."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(upload_router, prefix="/api/v1", tags=["Document Ingestion"])

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc),
                "path": str(request.url),
            },
        )

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/health", tags=["System"], summary="Health check")
    async def health_check() -> dict:
        """Return service health status."""
        return {
            "status": "healthy",
            "phase": 1,
            "service": "document-ingestion",
            "version": "1.0.0",
        }

    return app


# ── Module-level app instance (for uvicorn) ───────────────────────────────────

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )