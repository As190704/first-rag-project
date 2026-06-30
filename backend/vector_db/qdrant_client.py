"""
Qdrant client factory and collection management.

Provides:
  - A singleton QdrantClient configured from environment variables
  - Collection creation with cosine similarity and named vectors
  - Collection existence checks and deletion utilities
  - Health check for startup validation

Environment variables:
  QDRANT_HOST  (default: localhost)
  QDRANT_PORT  (default: 6333)
  QDRANT_API_KEY (default: None — not required for local Docker)

Phase 3 extensions:
  - Add a second collection for image embeddings (1024-dim CLIP vectors)
  - Add a sparse vector field when BGE-M3 sparse vectors are needed
"""

from __future__ import annotations

import os
import time

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from embeddings.embedder import EMBEDDING_DIMENSION
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY", None)

COLLECTION_NAME: str = "research_documents"

# Number of connection retries on startup
MAX_RETRIES: int = 5
RETRY_DELAY_SECONDS: float = 2.0


# ── Singleton client ──────────────────────────────────────────────────────────

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """
    Return the singleton QdrantClient, creating it on first call.

    Implements retry logic for Docker startup race conditions where
    Qdrant may not be ready immediately when FastAPI starts.

    Returns:
        Configured QdrantClient instance.

    Raises:
        RuntimeError: If Qdrant is unreachable after MAX_RETRIES attempts.
    """
    global _client

    if _client is not None:
        return _client

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "[Qdrant] Connecting to %s:%d (attempt %d/%d)",
                QDRANT_HOST,
                QDRANT_PORT,
                attempt,
                MAX_RETRIES,
            )
            client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                api_key=QDRANT_API_KEY,
                timeout=30,
            )
            # Verify connection with a lightweight API call
            client.get_collections()
            _client = client
            logger.info("[Qdrant] Connected successfully to %s:%d", QDRANT_HOST, QDRANT_PORT)
            return _client

        except Exception as exc:
            last_error = exc
            logger.warning(
                "[Qdrant] Connection attempt %d failed: %s. Retrying in %.1fs...",
                attempt,
                exc,
                RETRY_DELAY_SECONDS,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"Cannot connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT} "
        f"after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


# ── Collection management ─────────────────────────────────────────────────────


def ensure_collection(
    collection_name: str = COLLECTION_NAME,
    vector_size: int = EMBEDDING_DIMENSION,
) -> None:
    """
    Create the Qdrant collection if it does not already exist.

    Uses cosine distance because BGE-M3 vectors are L2-normalised,
    making cosine similarity equivalent to dot product (faster).

    The collection schema is designed for Phase 3 extensibility:
    using named vectors ("text") so an "image" vector can be added
    without migrating existing data.

    Args:
        collection_name: Name of the Qdrant collection.
        vector_size:     Dimensionality of the embedding vectors.
    """
    client = get_qdrant_client()

    try:
        existing = client.get_collection(collection_name)
        logger.info(
            "[Qdrant] Collection '%s' already exists (%d vectors).",
            collection_name,
            existing.vectors_count,
        )
        return
    except Exception:
        pass  # Collection does not exist — create it

    logger.info(
        "[Qdrant] Creating collection '%s' (dim=%d, metric=cosine)...",
        collection_name,
        vector_size,
    )

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            # Named vector "text" — Phase 3 can add "image" alongside it
            "text": qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
                on_disk=False,  # Keep in RAM for fast retrieval
            )
        },
        # HNSW index configuration for high-recall approximate search
        hnsw_config=qmodels.HnswConfigDiff(
            m=16,              # Connections per node
            ef_construct=100,  # Build-time accuracy vs. speed tradeoff
        ),
        # Optimiser settings for write-heavy indexing workloads
        optimizers_config=qmodels.OptimizersConfigDiff(
            indexing_threshold=20_000,  # Build HNSW after 20k vectors
        ),
    )

    logger.info("[Qdrant] Collection '%s' created successfully.", collection_name)


def collection_exists(collection_name: str = COLLECTION_NAME) -> bool:
    """
    Check whether a Qdrant collection exists.

    Args:
        collection_name: Collection to check.

    Returns:
        True if the collection exists.
    """
    client = get_qdrant_client()
    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False


def delete_collection(collection_name: str = COLLECTION_NAME) -> None:
    """
    Permanently delete a Qdrant collection.

    Used by force_reindex to clear existing vectors before re-ingestion.

    Args:
        collection_name: Collection to delete.
    """
    client = get_qdrant_client()
    client.delete_collection(collection_name)
    logger.warning("[Qdrant] Collection '%s' deleted.", collection_name)


def delete_document_vectors(
    document_id: str,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """
    Delete all vectors belonging to a specific document.

    More targeted than delete_collection — only removes vectors
    matching the document_id payload filter.

    Args:
        document_id:     The document whose vectors should be deleted.
        collection_name: Collection to operate on.

    Returns:
        Number of vectors deleted (approximate from Qdrant).
    """
    client = get_qdrant_client()

    result = client.delete(
        collection_name=collection_name,
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
        "[Qdrant] Deleted vectors for document_id=%s | status=%s",
        document_id,
        result.status,
    )
    return 0  # Qdrant delete doesn't return count; return 0 as sentinel


def qdrant_health_check() -> dict[str, str]:
    """
    Perform a lightweight health check against Qdrant.

    Returns:
        Dictionary with 'status' and 'collections' count.
    """
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        return {
            "status": "healthy",
            "collections": str(len(collections.collections)),
        }
    except Exception as exc:
        return {"status": "unhealthy", "error": str(exc)}