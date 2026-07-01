"""
Unit tests for the BGE-M3 embedding engine.

Uses mocking to avoid loading the actual 2GB model in CI/CD environments.
Tests validate the interface contract, batch logic, and error handling.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embeddings.embedder import EmbeddingEngine, _batched


# ── Tests: _batched helper ────────────────────────────────────────────────────


class TestBatched:
    def test_even_split(self):
        result = list(_batched([1, 2, 3, 4], 2))
        assert result == [[1, 2], [3, 4]]

    def test_uneven_split(self):
        result = list(_batched([1, 2, 3, 4, 5], 2))
        assert result == [[1, 2], [3, 4], [5]]

    def test_batch_larger_than_list(self):
        result = list(_batched([1, 2], 10))
        assert result == [[1, 2]]

    def test_empty_list(self):
        result = list(_batched([], 5))
        assert result == []


# ── Tests: EmbeddingEngine ────────────────────────────────────────────────────


class TestEmbeddingEngine:
    """
    Tests for EmbeddingEngine using a mocked SentenceTransformer.

    We patch at the sentence_transformers import level so that the
    real BGE-M3 model is never downloaded during testing.
    """

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the singleton state before each test."""
        EmbeddingEngine._instance = None
        EmbeddingEngine._model = None
        yield
        EmbeddingEngine._instance = None
        EmbeddingEngine._model = None

    @pytest.fixture
    def mock_model(self):
        """Return a mock SentenceTransformer that produces fake 1024-dim vectors."""
        model = MagicMock()
        model.max_seq_length = 8192

        def fake_encode(texts, **kwargs):
            n = len(texts)
            vecs = np.random.rand(n, 1024).astype(np.float32)
            # L2 normalise
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / norms

        model.encode.side_effect = fake_encode
        return model

    def _get_engine_with_mock(self, mock_model) -> EmbeddingEngine:
        engine = EmbeddingEngine()
        engine._model = mock_model
        engine._device = "cpu"
        return engine

    def test_embed_texts_returns_correct_count(self, mock_model):
        engine = self._get_engine_with_mock(mock_model)
        texts = ["Hello world", "Transformers are great", "Attention mechanism"]
        result = engine.embed_texts(texts)
        assert len(result) == 3

    def test_embed_texts_correct_dimension(self, mock_model):
        engine = self._get_engine_with_mock(mock_model)
        result = engine.embed_texts(["Test sentence"])
        assert len(result[0]) == 1024

    def test_embed_texts_returns_lists(self, mock_model):
        engine = self._get_engine_with_mock(mock_model)
        result = engine.embed_texts(["Hello"])
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], float)

    def test_embed_texts_empty_raises(self, mock_model):
        engine = self._get_engine_with_mock(mock_model)
        with pytest.raises(ValueError, match="empty"):
            engine.embed_texts([])

    def test_embed_query_returns_single_vector(self, mock_model):
        engine = self._get_engine_with_mock(mock_model)
        result = engine.embed_query("What is attention?")
        assert isinstance(result, list)
        assert len(result) == 1024

    def test_embed_query_empty_raises(self, mock_model):
        engine = self._get_engine_with_mock(mock_model)
        with pytest.raises(ValueError, match="empty"):
            engine.embed_query("   ")

    def test_batch_processing(self, mock_model):
        engine = self._get_engine_with_mock(mock_model)
        texts = [f"Sentence number {i}" for i in range(10)]
        result = engine.embed_texts(texts, batch_size=3)
        assert len(result) == 10
        # Verify encode was called multiple times (batched)
        assert mock_model.encode.call_count == 4  # ceil(10/3) = 4

    def test_dimension_property(self, mock_model):
        engine = self._get_engine_with_mock(mock_model)
        assert engine.dimension == 1024

    def test_model_name_property(self, mock_model):
        engine = self._get_engine_with_mock(mock_model)
        assert "bge-m3" in engine.model_name.lower()