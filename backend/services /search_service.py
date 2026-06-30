"""
Search service — orchestrates query embedding and Qdrant retrieval.

Provides a clean interface for the API layer, decoupling the endpoint
logic from the embedding and search internals.

Phase 3 extension:
  - Add multimodal_search() that embeds an image query and searches
    the "image" named vector space in Qdrant.
  - Add hybrid_search() combining dense vector + BM25 sparse scores.
"""

from __future__ import annotations

from embeddings.embedder import embedding_engine
from schemas.chunk import ChunkType, SearchRequest, SearchResponse, SearchResult
from vector_db.search import SemanticSearcher
from utils.logger import get_logger

logger = get_logger(__name__)


class SearchService:
    """
    Orchestrates the semantic search pipeline.

    Flow:
      query string → embed → Qdrant search → SearchResponse

    Usage::

        service = SearchService()
        response = service.search(request)
    """

    def __init__(self) -> None:
        self.searcher = SemanticSearcher()

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, request: SearchRequest) -> SearchResponse:
        """
        Execute a semantic search from a SearchRequest.

        Args:
            request: Validated SearchRequest from the API layer.

        Returns:
            SearchResponse with ranked results and latency info.

        Raises:
            RuntimeError: On embedding or search failure.
        """
        logger.info(
            "[SearchService] Query='%s' top_k=%d filters: doc=%s page=%s type=%s",
            request.query[:80],
            request.top_k,
            request.document_id or "all",
            str(request.page) if request.page else "all",
            request.chunk_type.value if request.chunk_type else "all",
        )

        # ── Step 1: Embed query ───────────────────────────────────────────────
        query_vector = self._embed_query(request.query)

        # ── Step 2: Search Qdrant ─────────────────────────────────────────────
        results, latency_ms = self.searcher.search(
            query_vector=query_vector,
            top_k=request.top_k,
            document_id=request.document_id,
            heading=request.heading,
            page=request.page,
            chunk_type=request.chunk_type,
        )

        logger.info(
            "[SearchService] Found %d results in %.2fms",
            len(results),
            latency_ms,
        )

        return SearchResponse(
            query=request.query,
            total_results=len(results),
            results=results,
            latency_ms=round(latency_ms, 2),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding vector for the search query.

        Args:
            query: Raw query string from the user.

        Returns:
            Normalised embedding vector.

        Raises:
            RuntimeError: If embedding fails.
        """
        try:
            return embedding_engine.embed_query(query)
        except Exception as exc:
            logger.error("[SearchService] Query embedding failed: %s", exc)
            raise RuntimeError(f"Failed to embed search query: {exc}") from exc