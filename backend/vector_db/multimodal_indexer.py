"""
Multimodal vector indexer for Qdrant.

Manages the 'research_multimodal' collection which stores both
visual (ColPali) and text (BGE-M3) named vectors for every
visual element extracted from research documents.

Collection schema:
  - Named vector "visual": 128-dim ColPali (visual search)
  - Named vector "text":   1024-dim BGE-M3 (text search)

This dual-vector design enables:
  - Pure text search:   "find tables comparing accuracy"
  - Pure visual search: match query to visual layout
  - Hybrid search:      weighted combination of both scores

Phase 4 extension:
  - Add "sparse" vector for BM25 hybrid retrieval.
  - Add cross-encoder reranking step after initial retrieval.
"""

from __future__ import annotations

import time
import uuid
from typing import Iterator

from qdrant_client.http import models as qmodels

from multimodal.image_embedder import COLPALI_EMBEDDING_DIM, TEXT_EMBEDDING_DIM
from schemas.multimodal_chunk import EmbeddedMultimodalChunk
from vector_db.qdrant_client import get_qdrant_client
from utils.logger import get_logger

logger = get_logger(__name__)

MULTIMODAL_COLLECTION: str = "research_multimodal"
UPSERT_BATCH_SIZE: int = 32


def _batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ── Collection management ─────────────────────────────────────────────────────


def ensure_multimodal_collection() -> None:
    """
    Create the multimodal Qdrant collection if it does not exist.

    Uses named vectors to store both visual and text embeddings
    independently, enabling per-modality search.
    """
    client = get_qdrant_client()

    try:
        existing = client.get_collection(MULTIMODAL_COLLECTION)
        logger.info(
            "[MultimodalIndexer] Collection '%s' exists (%d vectors).",
            MULTIMODAL_COLLECTION,
            existing.vectors_count,
        )
        return
    except Exception:
        pass

    logger.info(
        "[MultimodalIndexer] Creating collection '%s'...",
        MULTIMODAL_COLLECTION,
    )

    client.create_collection(
        collection_name=MULTIMODAL_COLLECTION,
        vectors_config={
            "visual": qmodels.VectorParams(
                size=COLPALI_EMBEDDING_DIM,
                distance=qmodels.Distance.COSINE,
            ),
            "text": qmodels.VectorParams(
                size=TEXT_EMBEDDING_DIM,
                distance=qmodels.Distance.COSINE,
            ),
        },
        hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=100),
    )

    # Create payload indices for fast filtered search
    for field_name in ("document_id", "chunk_type", "page", "source_file"):
        client.create_payload_index(
            collection_name=MULTIMODAL_COLLECTION,
            field_name=field_name,
            field_schema=qmodels.PayloadSchemaType.KEYWORD
            if field_name != "page"
            else qmodels.PayloadSchemaType.INTEGER,
        )

    logger.info(
        "[MultimodalIndexer] Collection '%s' created with visual+text vectors.",
        MULTIMODAL_COLLECTION,
    )


def delete_document_multimodal_vectors(document_id: str) -> None:
    """
    Remove all multimodal vectors for a given document.

    Args:
        document_id: Document whose vectors should be deleted.
    """
    client = get_qdrant_client()
    client.delete(
        collection_name=MULTIMODAL_COLLECTION,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchValue(value=document_id),
                    )
                ]
            )
        ),
    )
    logger.info(
        "[MultimodalIndexer] Deleted multimodal vectors for document_id=%s",
        document_id,
    )


# ── Indexer class ─────────────────────────────────────────────────────────────


class MultimodalIndexer:
    """
    Inserts EmbeddedMultimodalChunk objects into the multimodal collection.

    Each chunk is stored as one Qdrant point with:
      - A "visual" named vector (ColPali)
      - A "text" named vector (BGE-M3)
      - Full metadata payload

    Usage::

        indexer = MultimodalIndexer()
        count = indexer.index(embedded_chunks)
    """

    def __init__(
        self,
        collection_name: str = MULTIMODAL_COLLECTION,
        batch_size: int = UPSERT_BATCH_SIZE,
    ) -> None:
        self.collection_name = collection_name
        self.batch_size = batch_size

    # ── Public API ────────────────────────────────────────────────────────────

    def index(self, embedded_chunks: list[EmbeddedMultimodalChunk]) -> int:
        """
        Upsert a list of EmbeddedMultimodalChunk objects into Qdrant.

        Chunks without a visual embedding get a zero vector (still
        searchable via text vector). Chunks without text embeddings
        similarly get a zero text vector.

        Args:
            embedded_chunks: List of chunks with embeddings.

        Returns:
            Total number of vectors successfully upserted.
        """
        if not embedded_chunks:
            logger.warning("[MultimodalIndexer] Nothing to index.")
            return 0

        client = get_qdrant_client()
        total = len(embedded_chunks)
        upserted = 0

        logger.info(
            "[MultimodalIndexer] Indexing %d multimodal chunks into '%s'",
            total,
            self.collection_name,
        )

        t_start = time.perf_counter()

        for batch_idx, batch in enumerate(
            _batched(embedded_chunks, self.batch_size), start=1
        ):
            points = [self._build_point(ec) for ec in batch]
            try:
                client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )
                upserted += len(batch)
                logger.info(
                    "[MultimodalIndexer] Batch %d: upserted %d | total=%d/%d",
                    batch_idx,
                    len(batch),
                    upserted,
                    total,
                )
            except Exception as exc:
                logger.error(
                    "[MultimodalIndexer] Batch %d failed: %s",
                    batch_idx,
                    exc,
                )
                raise RuntimeError(
                    f"Multimodal indexing failed at batch {batch_idx}: {exc}"
                ) from exc

        elapsed = time.perf_counter() - t_start
        logger.info(
            "[MultimodalIndexer] Done: %d vectors in %.2fs",
            upserted,
            elapsed,
        )
        return upserted

    def document_is_indexed(self, document_id: str) -> bool:
        """Check if any multimodal vectors exist for a document."""
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

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_point(ec: EmbeddedMultimodalChunk) -> qmodels.PointStruct:
        """
        Convert an EmbeddedMultimodalChunk to a Qdrant PointStruct.

        Both visual and text vectors are stored as named vectors.
        Missing embeddings are replaced with zero vectors.

        Args:
            ec: Embedded multimodal chunk.

        Returns:
            Qdrant PointStruct ready for upsert.
        """
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, ec.chunk.chunk_id))

        visual_vec = (
            ec.visual_embedding
            if ec.has_visual_embedding
            else [0.0] * COLPALI_EMBEDDING_DIM
        )
        text_vec = (
            ec.text_embedding
            if ec.has_text_embedding
            else [0.0] * TEXT_EMBEDDING_DIM
        )

        return qmodels.PointStruct(
            id=point_id,
            vector={
                "visual": visual_vec,
                "text": text_vec,
            },
            payload=ec.chunk.to_qdrant_payload(),
        )