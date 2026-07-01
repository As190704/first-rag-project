"""
ColPali-based visual embedding pipeline.

ColPali (https://github.com/illuin-tech/colpali) uses a PaliGemma
backbone to generate patch-level embeddings for document page images.
It is specifically designed for document retrieval tasks.

For multimodal search, we store two embeddings per chunk:
  1. Visual embedding (ColPali) — captures visual structure and layout
  2. Text embedding (BGE-M3)   — captures description semantics

Both are stored as named vectors in Qdrant so search can use either
or both simultaneously (hybrid retrieval).

Phase 4 extension:
  - Enable late interaction (MaxSim) scoring for ColPali patch vectors.
  - Integrate CLIP for zero-shot cross-modal queries.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image as PILImage

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Model configuration ───────────────────────────────────────────────────────

COLPALI_MODEL_NAME: str = "vidore/colpali-v1.2"
COLPALI_EMBEDDING_DIM: int = 128   # ColPali produces 128-dim patch-averaged vectors
TEXT_EMBEDDING_DIM: int = 1024     # BGE-M3 dimension (from Phase 2)

DEFAULT_BATCH_SIZE: int = 4        # Small batch for memory-constrained systems


def _batched(items: list, size: int) -> Iterator[list]:
    """Yield fixed-size batches from a list."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


class ColPaliEmbedder:
    """
    Generates visual embeddings using ColPali.

    Singleton pattern — loads the model once and reuses across requests.
    Gracefully degrades to zero vectors if model cannot be loaded.

    Usage::

        embedder = ColPaliEmbedder()
        vectors = embedder.embed_images([pil_img_1, pil_img_2])
    """

    _instance: "ColPaliEmbedder | None" = None
    _model = None
    _processor = None
    _device: str = "cpu"
    _model_loaded: bool = False
    _load_failed: bool = False

    def __new__(cls) -> "ColPaliEmbedder":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def visual_dim(self) -> int:
        """Return the visual embedding dimension."""
        return COLPALI_EMBEDDING_DIM

    @property
    def text_dim(self) -> int:
        """Return the text embedding dimension (BGE-M3)."""
        return TEXT_EMBEDDING_DIM

    def embed_images(
        self,
        images: list[PILImage.Image],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[list[float]]:
        """
        Generate ColPali visual embeddings for a list of PIL images.

        Each image is encoded into a 128-dim vector (mean-pooled over
        all patch tokens). This representation captures visual layout,
        colour patterns, and spatial structure.

        Args:
            images:     List of PIL images to embed.
            batch_size: Images to process per forward pass.

        Returns:
            List of 128-dim embedding vectors (one per image).
            Returns zero vectors if model is unavailable.
        """
        if not images:
            return []

        if not self._model_loaded and not self._load_failed:
            self._load_model()

        if self._load_failed or self._model is None:
            logger.warning(
                "[ColPali] Model unavailable, returning zero vectors for %d images.",
                len(images),
            )
            return [[0.0] * COLPALI_EMBEDDING_DIM for _ in images]

        all_embeddings: list[list[float]] = []
        total = len(images)
        processed = 0
        t_start = time.perf_counter()

        for batch in _batched(images, batch_size):
            try:
                batch_embeddings = self._encode_batch(batch)
                all_embeddings.extend(batch_embeddings)
                processed += len(batch)
                logger.info(
                    "[ColPali] Embedded %d/%d images",
                    processed,
                    total,
                )
            except Exception as exc:
                logger.error(
                    "[ColPali] Batch embedding failed at %d: %s",
                    processed,
                    exc,
                )
                # Return zero vectors for failed batch
                all_embeddings.extend(
                    [[0.0] * COLPALI_EMBEDDING_DIM for _ in batch]
                )
                processed += len(batch)

        elapsed = time.perf_counter() - t_start
        logger.info(
            "[ColPali] Complete: %d images in %.2fs | dim=%d",
            total,
            elapsed,
            COLPALI_EMBEDDING_DIM,
        )
        return all_embeddings

    def embed_query_text(self, query: str) -> list[float]:
        """
        Embed a text query for visual search using ColPali's text tower.

        ColPali supports text queries against visual embeddings by encoding
        the query through its language model component.

        Args:
            query: Natural language search query.

        Returns:
            128-dim query embedding for visual search.
        """
        if not self._model_loaded and not self._load_failed:
            self._load_model()

        if self._load_failed or self._model is None:
            return [0.0] * COLPALI_EMBEDDING_DIM

        try:
            import torch

            inputs = self._processor.process_queries([query])
            if self._device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                query_embeddings = self._model(**inputs)

            # Mean pool over sequence tokens
            vector = query_embeddings.mean(dim=1).squeeze(0)
            vector = vector / (vector.norm() + 1e-8)
            return vector.cpu().numpy().tolist()

        except Exception as exc:
            logger.warning("[ColPali] Query embedding failed: %s", exc)
            return [0.0] * COLPALI_EMBEDDING_DIM

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Lazily load the ColPali model and processor.

        ColPali requires the colpali-engine package and a PaliGemma checkpoint.
        """
        try:
            import torch
            from colpali_engine.models import ColPali, ColPaliProcessor  # type: ignore

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                "[ColPali] Loading %s on %s...",
                COLPALI_MODEL_NAME,
                self._device,
            )

            t_start = time.perf_counter()
            dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

            self._model = ColPali.from_pretrained(
                COLPALI_MODEL_NAME,
                torch_dtype=dtype,
                device_map="auto" if self._device == "cuda" else None,
            )
            self._processor = ColPaliProcessor.from_pretrained(COLPALI_MODEL_NAME)

            if self._device == "cpu":
                self._model = self._model.to("cpu")

            self._model_loaded = True
            logger.info(
                "[ColPali] Loaded in %.1fs on %s.",
                time.perf_counter() - t_start,
                self._device,
            )

        except ImportError as exc:
            logger.error(
                "[ColPali] colpali-engine not installed: %s. "
                "Install with: pip install colpali-engine",
                exc,
            )
            self._load_failed = True
        except Exception as exc:
            logger.error("[ColPali] Failed to load ColPali: %s", exc)
            self._load_failed = True

    def _encode_batch(self, images: list[PILImage.Image]) -> list[list[float]]:
        """
        Encode a batch of images through ColPali.

        Args:
            images: List of PIL images (same batch).

        Returns:
            List of 128-dim normalised embedding vectors.
        """
        import torch

        # Prepare inputs
        inputs = self._processor.process_images(images)
        if self._device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            image_embeddings = self._model(**inputs)

        # Mean pool over patch tokens → one vector per image
        # Shape: [batch, num_patches, 128] → [batch, 128]
        pooled = image_embeddings.mean(dim=1)

        # L2 normalise
        norms = pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normalised = pooled / norms

        return normalised.cpu().numpy().tolist()


# ── Module-level singleton ────────────────────────────────────────────────────

colpali_embedder = ColPaliEmbedder()
