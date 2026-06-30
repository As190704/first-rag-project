"""
Index API router — POST /index endpoint.

Triggers the full indexing pipeline for a previously parsed document.
The document must have been parsed through Phase 1 first.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from schemas.chunk import IndexRequest, IndexResponse
from services.indexing_service import IndexingService
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/index",
    response_model=IndexResponse,
    status_code=status.HTTP_200_OK,
    summary="Index a parsed document into Qdrant",
    description=(
        "Takes a document_id from Phase 1 parsing, loads the output JSON, "
        "creates semantic chunks, generates BGE-M3 embeddings, and stores "
        "all vectors in the Qdrant research_documents collection."
    ),
)
async def index_document(request: IndexRequest) -> IndexResponse:
    """
    Index endpoint: chunk → embed → store.

    Args:
        request: IndexRequest with document_id and optional force_reindex flag.

    Returns:
        IndexResponse with indexing statistics.

    Raises:
        HTTPException 404: If document JSON not found.
        HTTPException 409: If already indexed (without force_reindex).
        HTTPException 500: On processing failures.
    """
    logger.info(
        "POST /index | document_id=%s force_reindex=%s",
        request.document_id,
        request.force_reindex,
    )

    service = IndexingService()

    try:
        stats = service.index_document(
            document_id=request.document_id,
            force_reindex=request.force_reindex,
        )
    except FileNotFoundError as exc:
        logger.warning("[Index API] Document not found: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except ValueError as exc:
        logger.error("[Index API] Invalid document: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except RuntimeError as exc:
        logger.error("[Index API] Indexing failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    return IndexResponse(
        document_id=stats["document_id"],
        chunks_created=stats["chunks_created"],
        embeddings_generated=stats["embeddings_generated"],
        vectors_stored=stats["vectors_stored"],
        collection_name=stats["collection_name"],
        duration_seconds=stats["duration_seconds"],
        message=stats["message"],
    )