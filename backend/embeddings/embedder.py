"""
BGE-M3 embedding engine.

Wraps the sentence-transformers library to provide:
  - Lazy model initialisation (loads once on first use)
  - Automatic GPU / CPU device selection
  - Batch embedding with configurable batch size
  - L2 normalisation for cosine similarity compatibility
  - Progress logging for long embedding jobs

BGE-M3 produces 1024-dimensional dense vectors.
It supports 100+ languages and long contexts (up to 8192 tokens).

Phase 3 extension points:
  - Add encode_image() using CLIP or similar vision encoder
  - Add encode_table() with a specialised tabular encoder
"""

from __future__ import annotations

import time
from typing import Iterator

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Model configuration ───────────────────────────────────────────────────────

MODEL_NAME: str = "BAAI/bge-m3"
EMBEDDING_DIMENSION: int = 1024
DEFAULT_BATCH_SIZE: int = 32
MAX_SEQ_LENGTH: int = 8192


def _batched(items: list, size: int) -> Iterator[list]:
    """
    Yield successive fixed-size batches from a list.

    Args:
        items: Source list.
        size:  Maximum batch size.

    Yields:
        Sub-lists of length <= size.
    """
    for i in range(0, len(items), size):
        yield items[i : i + size]


class EmbeddingEngine:
    """
    Singleton embedding engine wrapping BAAI/bge-m3.

    The model is loaded lazily on first call to avoid slow startup
    when the server starts. All public methods are thread-safe after
    the initial load because the model is read-only during inference.

    Usage::

        engine = EmbeddingEngine()
        vectors = engine.embed_texts(["What is attention?", "Transformers..."])
    """

    _instance: "EmbeddingEngine | None" = None
    _model = None
    _device: str = "cpu"

    def __new__(cls) -> "EmbeddingEngine":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ── Initialisation ────────────────────────────────────────────────────────

    def _ensure_model_loaded(self) -> None:
        """
        Lazily load the BGE-M3 model on first use.

        Automatically selects CUDA if available, otherwise CPU.

        Raises:
            RuntimeError: If sentence-transformers is not installed or
                          model download fails.
        """
        if self._model is not None:
            return

        try:
            import torch
            from sentence_transformers import SentenceTransformer

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                "Loading embedding model '%s' on device='%s'...",
                MODEL_NAME,
                self._device,
            )

            start = time.perf_counter()
            self._model = SentenceTransformer(
                MODEL_NAME,
                device=self._device,
            )
            self._model.max_seq_length = MAX_SEQ_LENGTH

            elapsed = time.perf_counter() - start
            logger.info(
                "Model loaded in %.2fs | dimension=%d | device=%s",
                elapsed,
                EMBEDDING_DIMENSION,
                self._device,
            )

        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{MODEL_NAME}': {exc}"
            ) from exc

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Generate L2-normalised embeddings for a list of text strings.

        Processes in batches to control memory usage. Each embedding is
        L2-normalised so that dot product == cosine similarity, which is
        required for Qdrant's cosine distance metric.

        Args:
            texts:         List of text strings to embed.
            batch_size:    Number of texts to process per forward pass.
            show_progress: Log progress for long jobs.

        Returns:
            List of embedding vectors, one per input text.
            Each vector has EMBEDDING_DIMENSION (1024) floats.

        Raises:
            ValueError: If texts list is empty.
            RuntimeError: If embedding fails.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

        self._ensure_model_loaded()

        total = len(texts)
        logger.info(
            "[Embedder] Embedding %d texts in batches of %d on %s",
            total,
            batch_size,
            self._device,
        )

        all_embeddings: list[np.ndarray] = []
        processed = 0
        t_start = time.perf_counter()

        for batch in _batched(texts, batch_size):
            try:
                batch_embeddings = self._model.encode(
                    batch,
                    batch_size=batch_size,
                    normalize_embeddings=True,   # L2 normalise for cosine sim
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                all_embeddings.append(batch_embeddings)
                processed += len(batch)

                if show_progress:
                    pct = processed / total * 100
                    logger.info(
                        "[Embedder] Progress: %d/%d (%.1f%%)",
                        processed,
                        total,
                        pct,
                    )

            except Exception as exc:
                logger.error(
                    "[Embedder] Batch embedding failed at index %d: %s",
                    processed,
                    exc,
                )
                raise RuntimeError(f"Embedding generation failed: {exc}") from exc

        # Stack all batches into one array, then convert to Python lists
        full_matrix = np.vstack(all_embeddings)
        elapsed = time.perf_counter() - t_start

        logger.info(
            "[Embedder] Complete: %d embeddings in %.2fs (%.1f emb/s) | dim=%d",
            total,
            elapsed,
            total / elapsed if elapsed > 0 else 0,
            full_matrix.shape[1],
        )

        return full_matrix.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.

        BGE-M3 benefits from the "Represent this sentence:" prefix
        for retrieval tasks (asymmetric retrieval pattern).

        Args:
            query: Natural language search query.

        Returns:
            Single normalised embedding vector of length EMBEDDING_DIMENSION.
        """
        if not query.strip():
            raise ValueError("Query text must not be empty.")

        # BGE retrieval instruction prefix
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        logger.debug("[Embedder] Embedding query: '%s'", query[:80])

        vectors = self.embed_texts([prefixed], batch_size=1, show_progress=False)
        return vectors[0]

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        return EMBEDDING_DIMENSION

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return MODEL_NAME


# ── Module-level singleton ────────────────────────────────────────────────────

embedding_engine = EmbeddingEngine()