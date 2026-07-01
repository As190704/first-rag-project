"""
FastAPI application — Phase 1 + Phase 2 + Phase 3.

Routers:
  POST /api/v1/upload              — Phase 1: document ingestion
  POST /api/v1/index               — Phase 2: text indexing
  POST /api/v1/search              — Phase 2: text search
  POST /api/v1/multimodal/index    — Phase 3: multimodal indexing
  POST /api/v1/multimodal/search   — Phase 3: hybrid search
  GET  /health                     — system health
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
from api.multimodal import router as multimodal_router
from utils.file_utils import ensure_directories
from utils.logger import get_logger
from vector_db.qdrant_client import ensure_collection, qdrant_health_check
from vector_db.multimodal_indexer import ensure_multimodal_collection

logger = get_logger(__name__)

REQUIRED_DIRS = [
    Path("uploads"),
    Path("output/json"),
    Path("output/images"),
    Path("logs"),
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  Multimodal RAG — Phase 1 + 2 + 3")
    logger.info("  Ingestion + Text Search + Multimodal Search")
    logger.info("=" * 60)

    ensure_directories(*REQUIRED_DIRS)

    try:
        ensure_collection()
        logger.info("Phase 2 Qdrant collection ready.")
    except Exception as exc:
        logger.warning("Phase 2 Qdrant collection unavailable: %s", exc)

    try:
        ensure_multimodal_collection()
        logger.info("Phase 3 multimodal Qdrant collection ready.")
    except Exception as exc:
        logger.warning("Phase 3 Qdrant collection unavailable: %s", exc)

    logger.info("Application ready.")
    yield
    logger.info("Application shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Multimodal RAG — Research Assistant API",
        description=(
            "Phase 1: Document ingestion (PDF, images, DOCX, PPTX).\n"
            "Phase 2: Text chunking, BGE-M3 embeddings, semantic search.\n"
            "Phase 3: Figure/chart/table/equation indexing, hybrid retrieval."
        ),
        version="3.0.0",
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

    prefix = "/api/v1"
    app.include_router(upload_router, prefix=prefix, tags=["Phase 1 — Ingestion"])
    app.include_router(index_router, prefix=prefix, tags=["Phase 2 — Text Indexing"])
    app.include_router(search_router, prefix=prefix, tags=["Phase 2 — Text Search"])
    app.include_router(
        multimodal_router,
        prefix=f"{prefix}/multimodal",
        tags=["Phase 3 — Multimodal"],
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    @app.get("/health", tags=["System"])
    async def health_check() -> dict:
        qdrant = qdrant_health_check()
        return {
            "status": "healthy",
            "phase": 3,
            "service": "multimodal-rag",
            "version": "3.0.0",
            "qdrant": qdrant,
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)