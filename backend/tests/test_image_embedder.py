"""Tests for ColPaliEmbedder — visual embedding generation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage

from multimodal.image_embedder import (
    ColPaliEmbedder,
    COLPALI_EMBEDDING_DIM,
    _batched,
)


@pytest.fixture(autouse=True)
def reset_embedder_singleton():
    ColPaliEmbedder._instance = None
    ColPaliEmbedder._model = None
    ColPaliEmbedder._processor = None
    ColPaliEmbedder._model_loaded = False
    ColPaliEmbedder._load_failed = False
    yield
    ColPaliEmbedder._instance = None
    ColPaliEmbedder._model = None
    ColPaliEmbedder._processor = None
    ColPaliEmbedder._model_loaded = False
    ColPaliEmbedder._load_failed = False


def make_mock_embedder() -> ColPaliEmbedder:
    """Create an embedder with a mocked ColPali model."""
    embedder = ColPaliEmbedder()
    embedder._model_loaded = True
    embedder._device = "cpu"

    mock_model = MagicMock()
    mock_processor = MagicMock()

    # Simulate patch-level output: [batch, patches, 128]
    def fake_model(**kwargs):
        batch_size = kwargs["input_ids"].shape[0] if "input_ids" in kwargs else 1
        output = MagicMock()
        output.mean.return_value = MagicMock(
            norm=lambda **kw: MagicMock(clamp=lambda **kw2: MagicMock()),
            __truediv__=lambda self, other: MagicMock(
                cpu=lambda: MagicMock(
                    numpy=lambda: np.random.rand(batch_size, COLPALI_EMBEDDING_DIM).astype(np.float32)
                )
            )
        )
        return output

    embedder._model = mock_model
    embedder._processor = mock_processor
    return embedder


class TestColPaliEmbedderFallback:
    """Tests using load_failed=True so no model is needed."""

    def test_empty_list_returns_empty(self):
        embedder = ColPaliEmbedder()
        embedder._load_failed = True
        result = embedder.embed_images([])
        assert result == []

    def test_load_failed_returns_zero_vectors(self):
        embedder = ColPaliEmbedder()
        embedder._load_failed = True
        images = [PILImage.new("RGB", (100, 100)) for _ in range(3)]
        result = embedder.embed_images(images)
        assert len(result) == 3
        assert all(len(v) == COLPALI_EMBEDDING_DIM for v in result)
        assert all(all(x == 0.0 for x in v) for v in result)

    def test_query_fallback_returns_zero_vector(self):
        embedder = ColPaliEmbedder()
        embedder._load_failed = True
        result = embedder.embed_query_text("attention mechanism")
        assert len(result) == COLPALI_EMBEDDING_DIM
        assert all(x == 0.0 for x in result)

    def test_visual_dim_property(self):
        embedder = ColPaliEmbedder()
        assert embedder.visual_dim == COLPALI_EMBEDDING_DIM

    def test_text_dim_property(self):
        embedder = ColPaliEmbedder()
        assert embedder.text_dim == 1024


class TestBatched:
    def test_exact_batches(self):
        result = list(_batched([1, 2, 3, 4], 2))
        assert result == [[1, 2], [3, 4]]

    def test_remainder_batch(self):
        result = list(_batched([1, 2, 3], 2))
        assert result == [[1, 2], [3]]

    def test_empty(self):
        result = list(_batched([], 4))
        assert result == []