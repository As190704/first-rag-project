"""
Indexing service — orchestrates the full chunk → embed → store pipeline.

This service is the single entry point for indexing a document.
It coordinates the chunker, embedding engine, and vector indexer,
providing a clean interface for the API layer.

Design for Phase 3:
  - Add index_images() to embed and store image vectors in a separate
    "image" named vector field without changing this service's interface.
  - Add index_tables() similarly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from embeddings.chunker import SemanticChunker
from embeddings.embedder import embedding_engine
from schemas.chunk import Chunk, EmbeddedChunk
from vector_db.indexer import VectorIndexer
from vector_db.qdrant_client import (
    COLLECTION_NAME,
    delete_document_vectors,
    ensure_collection,
)
from utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_JSON_DIR = Path("output/json")


class IndexingService:
    """
    Orchestrates the document indexing pipeline.

    Flow:
      load JSON → chunk → embed → store in Qdrant → return statistics

    Usage::

        service = IndexingService()
        stats = service.index_document("abc123", force_reindex=False)
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        """
        Args:
            collection_name: Target Qdrant collection name.
        """
        self.collection_name = collection_name
        self.chunker = SemanticChunker()
        self.indexer = VectorIndexer(collection_name=collection_name)

    # ── Public API ────────────────────────────────────────────────────────────

    def index_document(
        self,
        document_id: str,
        force_reindex: bool = False,
    ) -> dict:
        """
        Index a single document by its document_id.

        Loads the Phase 1 JSON output, chunks it, generates embeddings,
        and stores all vectors in Qdrant.

        Args:
            document_id:    ID of the document to index.
            force_reindex:  If True, remove existing vectors and re-index.

        Returns:
            Statistics dictionary with chunk/embedding/vector counts
            and elapsed time.

        Raises:
            FileNotFoundError: If the Phase 1 JSON is not found.
            RuntimeError:      On embedding or storage failures.
        """
        t_start = time.perf_counter()

        logger.info(
            "[IndexingService] Starting indexing for document_id=%s force=%s",
            document_id,
            force_reindex,
        )

        # ── Step 1: Load Phase 1 JSON ─────────────────────────────────────────
        doc = self._load_document_json(document_id)
        logger.info(
            "[IndexingService] Loaded JSON: title='%s'",
            doc.get("title", "")[:60],
        )

        # ── Step 2: Ensure collection exists ──────────────────────────────────
        ensure_collection(self.collection_name)

        # ── Step 3: Handle duplicate indexing ────────────────────────────────
        if not force_reindex and self.indexer.document_is_indexed(document_id):
            existing_count = self.indexer.get_document_vector_count(document_id)
            logger.warning(
                "[IndexingService] Document %s already indexed (%d vectors). "
                "Use force_reindex=True to re-index.",
                document_id,
                existing_count,
            )
            return {
                "document_id": document_id,
                "chunks_created": existing_count,
                "embeddings_generated": 0,
                "vectors_stored": 0,
                "collection_name": self.collection_name,
                "duration_seconds": 0.0,
                "message": (
                    f"Document already indexed with {existing_count} vectors. "
                    "Use force_reindex=True to overwrite."
                ),
                "already_indexed": True,
            }

        if force_reindex:
            logger.info(
                "[IndexingService] force_reindex=True — removing existing vectors for %s",
                document_id,
            )
            delete_document_vectors(document_id, self.collection_name)

        # ── Step 4: Chunk document ────────────────────────────────────────────
        chunks = self._chunk_document(doc)

        if not chunks:
            logger.warning("[IndexingService] No chunks produced for %s", document_id)
            return self._stats(
                document_id, 0, 0, 0, t_start, "No chunks could be extracted."
            )

        # ── Step 5: Generate embeddings ───────────────────────────────────────
        embedded_chunks = self._embed_chunks(chunks)

        # ── Step 6: Store in Qdrant ───────────────────────────────────────────
        vectors_stored = self.indexer.index_chunks(embedded_chunks)

        elapsed = time.perf_counter() - t_start
        stats = self._stats(
            document_id=document_id,
            chunks_created=len(chunks),
            embeddings_generated=len(embedded_chunks),
            vectors_stored=vectors_stored,
            t_start=t_start,
            message=(
                f"Successfully indexed {vectors_stored} vectors "
                f"in {elapsed:.2f}s."
            ),
        )

        logger.info(
            "[IndexingService] Done: %d chunks | %d embeddings | %d vectors | %.2fs",
            stats["chunks_created"],
            stats["embeddings_generated"],
            stats["vectors_stored"],
            elapsed,
        )
        return stats

    # ── Private pipeline steps ────────────────────────────────────────────────

    def _load_document_json(self, document_id: str) -> dict:
        """
        Load the Phase 1 parsed document JSON from disk.

        Args:
            document_id: Document ID used as the JSON filename stem.

        Returns:
            Parsed document dictionary.

        Raises:
            FileNotFoundError: If JSON file does not exist.
            ValueError:        If JSON is malformed.
        """
        json_path = OUTPUT_JSON_DIR / f"{document_id}.json"

        if not json_path.exists():
            raise FileNotFoundError(
                f"Parsed document JSON not found: {json_path}. "
                "Ensure Phase 1 parsing completed successfully before indexing."
            )

        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
            logger.debug("[IndexingService] Loaded %s (%.1f KB)", json_path.name, json_path.stat().st_size / 1024)
            return doc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in {json_path}: {exc}") from exc

    def _chunk_document(self, doc: dict) -> list[Chunk]:
        """
        Apply semantic chunking to the parsed document.

        Args:
            doc: Parsed document dict from Phase 1.

        Returns:
            List of Chunk objects.
        """
        logger.info("[IndexingService] Chunking document...")
        try:
            chunks = self.chunker.chunk_document(doc)
            logger.info("[IndexingService] Created %d chunks", len(chunks))
            return chunks
        except Exception as exc:
            logger.error("[IndexingService] Chunking failed: %s", exc)
            raise RuntimeError(f"Chunking failed: {exc}") from exc

    def _embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """
        Generate embeddings for all chunks.

        Extracts text from each chunk, calls the embedding engine in
        batch mode, then pairs each vector back with its chunk.

        Args:
            chunks: List of Chunk objects to embed.

        Returns:
            List of EmbeddedChunk objects with vectors attached.
        """
        logger.info(
            "[IndexingService] Generating embeddings for %d chunks...", len(chunks)
        )

        texts = [chunk.text for chunk in chunks]

        try:
            vectors = embedding_engine.embed_texts(texts)
        except Exception as exc:
            logger.error("[IndexingService] Embedding failed: %s", exc)
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc

        embedded = [
            EmbeddedChunk(
                chunk=chunk,
                embedding=vector,
                embedding_model=embedding_engine.model_name,
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        logger.info("[IndexingService] Embeddings generated: %d", len(embedded))
        return embedded

    @staticmethod
    def _stats(
        document_id: str,
        chunks_created: int,
        embeddings_generated: int,
        vectors_stored: int,
        t_start: float,
        message: str,
    ) -> dict:
        """Build the statistics response dictionary."""
        return {
            "document_id": document_id,
            "chunks_created": chunks_created,
            "embeddings_generated": embeddings_generated,
            "vectors_stored": vectors_stored,
            "collection_name": COLLECTION_NAME,
            "duration_seconds": round(time.perf_counter() - t_start, 3),
            "message": message,
        }