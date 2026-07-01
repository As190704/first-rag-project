"""
Multimodal API router — Phase 3 endpoints.

Endpoints:
  POST /multimodal/index  — index visual elements of a parsed document
  POST /multimodal/search — hybrid text + visual search
  GET  /multimodal/health — collection health check
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from schemas.multimodal_chunk import (
    MultimodalIndexRequest,
    MultimodalIndexResponse,
    MultimodalSearchRequest,
    MultimodalSearchResponse,
)
from services.multimodal_service import MultimodalService
from vector_db.multimodal_indexer import ensure_multimodal_collection
from vector_db.qdrant_client import get_qdrant_client
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()
_multimodal_service = MultimodalService()


# ── Index endpoint ────────────────────────────────────────────────────────────


@router.post(
    "/index",
    response_model=MultimodalIndexResponse,
    status_code=status.HTTP_200_OK,
    summary="Index visual elements of a parsed document",
    description=(
        "Processes all images, tables, charts, and equations extracted "
        "during Phase 1 parsing. Generates Qwen2-VL descriptions, "
        "ColPali visual embeddings, and BGE-M3 text embeddings. "
        "Stores everything in the 'research_multimodal' Qdrant collection."
    ),
)
async def index_multimodal(
    request: MultimodalIndexRequest,
) -> MultimodalIndexResponse:
    """
    Multimodal indexing endpoint.

    Requires Phase 1 parsing to have completed for the given document_id.
    Phase 2 text indexing does NOT need to run first.

    Args:
        request: MultimodalIndexRequest with document_id and feature flags.

    Returns:
        MultimodalIndexResponse with processing statistics.
    """
    logger.info(
        "POST /multimodal/index | document_id=%s force=%s",
        request.document_id,
        request.force_reindex,
    )

    try:
        stats = _multimodal_service.index_document(request)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("[Multimodal API] Unexpected error during indexing: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {exc}",
        )

    return MultimodalIndexResponse(
        document_id=stats["document_id"],
        figures_processed=stats["figures_processed"],
        charts_processed=stats["charts_processed"],
        tables_processed=stats["tables_processed"],
        equations_detected=stats["equations_detected"],
        vectors_stored=stats["vectors_stored"],
        collection_name=stats["collection_name"],
        duration_seconds=stats["duration_seconds"],
        message=stats["message"],
    )


# ── Search endpoint ───────────────────────────────────────────────────────────


@router.post(
    "/search",
    response_model=MultimodalSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Hybrid multimodal semantic search",
    description=(
        "Search over indexed visual elements using natural language. "
        "Supports text-only, visual-only, or hybrid (default) retrieval. "
        "Returns figures, charts, tables, and equations sorted by relevance.\n\n"
        "Example queries:\n"
        "- 'Show all confusion matrices'\n"
        "- 'Find the CNN architecture diagram'\n"
        "- 'Locate tables comparing accuracy metrics'\n"
        "- 'Find charts showing training loss curves'\n"
        "- 'Find equations containing cross entropy'"
    ),
)
async def search_multimodal(
    request: MultimodalSearchRequest,
) -> MultimodalSearchResponse:
    """
    Hybrid multimodal search endpoint.

    Args:
        request: MultimodalSearchRequest with query, filters, and search mode.

    Returns:
        MultimodalSearchResponse with ranked multimodal results.
    """
    logger.info(
        "POST /multimodal/search | query='%s' mode=%s top_k=%d",
        request.query[:80],
        request.search_mode,
        request.top_k,
    )

    if request.search_mode not in ("text", "visual", "hybrid"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid search_mode '{request.search_mode}'. "
                   "Use 'text', 'visual', or 'hybrid'.",
        )

    try:
        response = _multimodal_service.search(request)
    except RuntimeError as exc:
        error_str = str(exc).lower()
        if "qdrant" in error_str or "connect" in error_str:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Search service unavailable: {exc}",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("[Multimodal API] Unexpected search error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {exc}",
        )

    return response


# ── Health check ──────────────────────────────────────────────────────────────


@router.get(
    "/health",
    tags=["System"],
    summary="Multimodal collection health check",
)
async def multimodal_health() -> dict:
    """Check Qdrant multimodal collection status."""
    try:
        client = get_qdrant_client()
        collection = client.get_collection("research_multimodal")
        return {
            "status": "healthy",
            "collection": "research_multimodal",
            "vectors_count": collection.vectors_count,
            "points_count": collection.points_count,
        }
    except Exception as exc:
        return {
            "status": "degraded",
            "collection": "research_multimodal",
            "error": str(exc),
        }