"""
Semantic search module for Qdrant vector retrieval.

Supports:
  - Pure vector search (cosine similarity)
  - Filtered search (document_id, heading, page, chunk_type)
  - Top-K retrieval with configurable result count
  - Score threshold filtering
  - Search latency measurement

Phase 3 extensions:
  - Hybrid search: combine dense text vector with sparse BM25 scores
  - Multimodal search: search across text + image vector spaces
  - Re-ranking with a cross-encoder model
"""

from __future__ import annotations

import time

from qdrant_client.http import models as qmodels

from schemas.chunk import ChunkType, SearchResult
from vector_db.qdrant_client import COLLECTION_NAME, get_qdrant_client
from utils.logger import get_logger

logger = get_logger(__name__)

# Minimum cosine similarity score to include in results
DEFAULT_SCORE_THRESHOLD: float = 0.0


class SemanticSearcher:
    """
    Performs semantic similarity search over the Qdrant collection.

    Usage::

        searcher = SemanticSearcher()
        results = searcher.search(query_vector, top_k=5)
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> None:
        """
        Args:
            collection_name:  Qdrant collection to search.
            score_threshold:  Minimum similarity score for results.
        """
        self.collection_name = collection_name
        self.score_threshold = score_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        document_id: str | None = None,
        heading: str | None = None,
        page: int | None = None,
        chunk_type: ChunkType | None = None,
    ) -> tuple[list[SearchResult], float]:
        """
        Execute a cosine similarity search with optional payload filters.

        Args:
            query_vector: Normalised embedding of the search query.
            top_k:        Maximum number of results to return.
            document_id:  If set, restrict results to this document.
            heading:      If set, restrict results to this heading.
            page:         If set, restrict results to this page number.
            chunk_type:   If set, restrict results to this chunk type.

        Returns:
            Tuple of (list of SearchResult, latency in milliseconds).
        """
        search_filter = self._build_filter(
            document_id=document_id,
            heading=heading,
            page=page,
            chunk_type=chunk_type,
        )

        client = get_qdrant_client()
        t_start = time.perf_counter()

        try:
            raw_results = client.search(
                collection_name=self.collection_name,
                query_vector=("text", query_vector),
                query_filter=search_filter,
                limit=top_k,
                score_threshold=self.score_threshold,
                with_payload=True,
                with_vectors=False,  # Don't return vectors — saves bandwidth
            )
        except Exception as exc:
            logger.error("[Search] Qdrant search failed: %s", exc)
            raise RuntimeError(f"Search failed: {exc}") from exc

        latency_ms = (time.perf_counter() - t_start) * 1000

        results = [self._point_to_result(hit) for hit in raw_results]

        logger.info(
            "[Search] Returned %d results in %.2fms | filters: doc=%s page=%s type=%s",
            len(results),
            latency_ms,
            document_id or "all",
            str(page) if page else "all",
            chunk_type.value if chunk_type else "all",
        )

        return results, latency_ms

    def search_by_document(
        self,
        query_vector: list[float],
        document_id: str,
        top_k: int = 10,
    ) -> tuple[list[SearchResult], float]:
        """
        Convenience method: search within a single document only.

        Args:
            query_vector: Query embedding.
            document_id:  Target document ID.
            top_k:        Result limit.

        Returns:
            Tuple of (results, latency_ms).
        """
        return self.search(query_vector, top_k=top_k, document_id=document_id)

    # ── Filter builder ────────────────────────────────────────────────────────

    @staticmethod
    def _build_filter(
        document_id: str | None,
        heading: str | None,
        page: int | None,
        chunk_type: ChunkType | None,
    ) -> qmodels.Filter | None:
        """
        Build a Qdrant filter from optional search parameters.

        All provided filters are combined with AND logic (must conditions).

        Args:
            document_id: Exact match on document_id payload field.
            heading:     Exact match on heading payload field.
            page:        Exact match on page payload field.
            chunk_type:  Exact match on chunk_type payload field.

        Returns:
            A Qdrant Filter object, or None if no filters are active.
        """
        conditions: list[qmodels.Condition] = []

        if document_id:
            conditions.append(
                qmodels.FieldCondition(
                    key="document_id",
                    match=qmodels.MatchValue(value=document_id),
                )
            )

        if heading:
            conditions.append(
                qmodels.FieldCondition(
                    key="heading",
                    match=qmodels.MatchValue(value=heading),
                )
            )

        if page is not None:
            conditions.append(
                qmodels.FieldCondition(
                    key="page",
                    match=qmodels.MatchValue(value=page),
                )
            )

        if chunk_type is not None:
            conditions.append(
                qmodels.FieldCondition(
                    key="chunk_type",
                    match=qmodels.MatchValue(value=chunk_type.value),
                )
            )

        if not conditions:
            return None

        return qmodels.Filter(must=conditions)

    # ── Result conversion ─────────────────────────────────────────────────────

    @staticmethod
    def _point_to_result(hit) -> SearchResult:
        """
        Convert a raw Qdrant ScoredPoint into a SearchResult model.

        Args:
            hit: A qdrant_client ScoredPoint from a search response.

        Returns:
            Populated SearchResult instance.
        """
        payload = hit.payload or {}

        return SearchResult(
            score=round(float(hit.score), 6),
            chunk_id=payload.get("chunk_id", ""),
            document_id=payload.get("document_id", ""),
            text=payload.get("text", ""),
            page=int(payload.get("page", 1)),
            heading=payload.get("heading", ""),
            chunk_type=payload.get("chunk_type", "unknown"),
            source_file=payload.get("source_file", ""),
            document_title=payload.get("source_file", ""),
        )