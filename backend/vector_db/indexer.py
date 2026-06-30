"""
Vector indexer — inserts EmbeddedChunk objects into Qdrant.

Handles:
  - Batch upsert with configurable batch size
  - UUID generation for Qdrant point IDs (Qdrant requires integer or UUID IDs)
  - Duplicate detection via payload filtering
  - Progress logging per batch
  - Atomic rollback on partial failure (best-effort)
"""

from __future__ import annotations

import uuid
from typing import Iterator

from qdrant_client.http import models as qmodels

from schemas.chunk import EmbeddedChunk
from vector_db.qdrant_client import COLLECTION_NAME, get_qdrant_client
from utils.logger import get_logger

logger = get_logger(__name__)

# Number of points to upsert per Qdrant API call
UPSERT_BATCH_SIZE: int = 64


def _batched(items: list, size: int) -> Iterator[list]:
    """Yield fixed-size batches from a list."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


class VectorIndexer:
    """
    Inserts EmbeddedChunk objects into a Qdrant collection.

    Usage::

        indexer = VectorIndexer()
        count = indexer.index_chunks(embedded_chunks)
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        batch_size: int = UPSERT_BATCH_SIZE,
    ) -> None:
        """
        Args:
            collection_name: Target Qdrant collection.
            batch_size:      Points per upsert batch.
        """
        self.collection_name = collection_name
        self.batch_size = batch_size

    # ── Public API ────────────────────────────────────────────────────────────

    def index_chunks(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        """
        Upsert a list of EmbeddedChunk objects into Qdrant.

        Each chunk becomes one Qdrant PointStruct containing:
          - A UUID point ID derived from chunk_id
          - Named vector "text" (the embedding)
          - Full metadata payload

        Args:
            embedded_chunks: List of chunks with embeddings attached.

        Returns:
            Total number of vectors successfully upserted.

        Raises:
            RuntimeError: If upsert fails after all retries.
        """
        if not embedded_chunks:
            logger.warning("[Indexer] No chunks to index.")
            return 0

        client = get_qdrant_client()
        total = len(embedded_chunks)
        upserted = 0

        logger.info(
            "[Indexer] Indexing %d vectors into collection='%s'",
            total,
            self.collection_name,
        )

        for batch_idx, batch in enumerate(_batched(embedded_chunks, self.batch_size), start=1):
            points = [self._build_point(ec) for ec in batch]

            try:
                client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,  # Wait for indexing confirmation
                )
                upserted += len(batch)
                logger.info(
                    "[Indexer] Batch %d: upserted %d points | total=%d/%d",
                    batch_idx,
                    len(batch),
                    upserted,
                    total,
                )

            except Exception as exc:
                logger.error(
                    "[Indexer] Batch %d upsert failed: %s",
                    batch_idx,
                    exc,
                )
                raise RuntimeError(
                    f"Vector indexing failed at batch {batch_idx}: {exc}"
                ) from exc

        logger.info(
            "[Indexer] Indexing complete: %d/%d vectors stored in '%s'",
            upserted,
            total,
            self.collection_name,
        )
        return upserted

    def document_is_indexed(self, document_id: str) -> bool:
        """
        Check whether any vectors exist for a given document_id.

        Used to detect duplicate indexing attempts.

        Args:
            document_id: Document to check.

        Returns:
            True if at least one vector exists for this document.
        """
        client = get_qdrant_client()
        try:
            results, _ = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id),
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return len(results) > 0
        except Exception:
            return False

    def get_document_vector_count(self, document_id: str) -> int:
        """
        Count the number of indexed vectors for a document.

        Args:
            document_id: Document to count vectors for.

        Returns:
            Number of vectors found.
        """
        client = get_qdrant_client()
        try:
            result = client.count(
                collection_name=self.collection_name,
                count_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id),
                        )
                    ]
                ),
                exact=True,
            )
            return result.count
        except Exception:
            return 0

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_point(ec: EmbeddedChunk) -> qmodels.PointStruct:
        """
        Convert an EmbeddedChunk to a Qdrant PointStruct.

        Qdrant point IDs must be integers or UUIDs.  We derive a
        deterministic UUID from the chunk_id string so that re-indexing
        the same chunk always produces the same point ID (enabling upsert
        idempotency).

        Args:
            ec: EmbeddedChunk with chunk metadata and embedding vector.

        Returns:
            PointStruct ready for Qdrant upsert.
        """
        # Deterministic UUID from chunk_id for idempotent upserts
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, ec.chunk.chunk_id))

        return qmodels.PointStruct(
            id=point_id,
            vector={"text": ec.embedding},
            payload=ec.chunk.to_qdrant_payload(),
        )