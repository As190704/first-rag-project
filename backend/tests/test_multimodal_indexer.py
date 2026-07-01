"""Tests for MultimodalIndexer — Qdrant multimodal vector storage."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from schemas.multimodal_chunk import (
    EmbeddedMultimodalChunk,
    MultimodalChunk,
    VisualChunkType,
)
from vector_db.multimodal_indexer import (
    COLPALI_EMBEDDING_DIM,
    MultimodalIndexer,
)


def make_embedded_chunk(
    chunk_id: str = "mm_abc123",
    document_id: str = "doc1",
    chunk_type: VisualChunkType = VisualChunkType.FIGURE,
    has_visual: bool = True,
    has_text: bool = True,
) -> EmbeddedMultimodalChunk:
    chunk = MultimodalChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        page=1,
        chunk_type=chunk_type,
        image_path="output/images/doc1/img_001.png",
        caption="Test figure",
        description="A detailed description of the figure.",
        source_file="test.pdf",
    )
    return EmbeddedMultimodalChunk(
        chunk=chunk,
        visual_embedding=[0.1] * COLPALI_EMBEDDING_DIM if has_visual else None,
        text_embedding=[0.2] * 1024 if has_text else None,
    )


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.upsert.return_value = MagicMock(status="completed")
    client.scroll.return_value = ([], None)
    return client


@pytest.fixture
def indexer(mock_client):
    with patch("vector_db.multimodal_indexer.get_qdrant_client", return_value=mock_client):
        yield MultimodalIndexer(collection_name="test_multimodal", batch_size=2)


class TestMultimodalIndexer:
    def test_index_single_chunk(self, indexer, mock_client):
        chunks = [make_embedded_chunk()]
        with patch(
            "vector_db.multimodal_indexer.get_qdrant_client", return_value=mock_client
        ):
            count = indexer.index(chunks)
        assert count == 1

    def test_index_empty_returns_zero(self, indexer, mock_client):
        with patch(
            "vector_db.multimodal_indexer.get_qdrant_client", return_value=mock_client
        ):
            count = indexer.index([])
        assert count == 0

    def test_batching(self, mock_client):
        """With batch_size=2 and 5 chunks, upsert called 3 times."""
        with patch(
            "vector_db.multimodal_indexer.get_qdrant_client", return_value=mock_client
        ):
            idx = MultimodalIndexer(collection_name="test", batch_size=2)
            chunks = [make_embedded_chunk(f"mm_{i:03d}") for i in range(5)]
            idx.index(chunks)

        assert mock_client.upsert.call_count == 3

    def test_point_has_both_named_vectors(self, mock_client):
        with patch(
            "vector_db.multimodal_indexer.get_qdrant_client", return_value=mock_client
        ):
            idx = MultimodalIndexer(collection_name="test")
            chunks = [make_embedded_chunk()]
            idx.index(chunks)

        call_kwargs = mock_client.upsert.call_args.kwargs
        point = call_kwargs["points"][0]
        assert "visual" in point.vector
        assert "text" in point.vector
        assert len(point.vector["visual"]) == COLPALI_EMBEDDING_DIM
        assert len(point.vector["text"]) == 1024

    def test_missing_visual_uses_zeros(self, mock_client):
        chunk = make_embedded_chunk(has_visual=False)
        with patch(
            "vector_db.multimodal_indexer.get_qdrant_client", return_value=mock_client
        ):
            idx = MultimodalIndexer(collection_name="test")
            idx.index([chunk])

        point = mock_client.upsert.call_args.kwargs["points"][0]
        assert all(x == 0.0 for x in point.vector["visual"])

    def test_upsert_failure_raises(self, mock_client):
        mock_client.upsert.side_effect = Exception("Connection timeout")
        with patch(
            "vector_db.multimodal_indexer.get_qdrant_client", return_value=mock_client
        ):
            idx = MultimodalIndexer(collection_name="test")
            with pytest.raises(RuntimeError, match="Multimodal indexing failed"):
                idx.index([make_embedded_chunk()])

    def test_document_is_indexed_true(self, mock_client):
        mock_client.scroll.return_value = ([MagicMock()], None)
        with patch(
            "vector_db.multimodal_indexer.get_qdrant_client", return_value=mock_client
        ):
            idx = MultimodalIndexer()
            assert idx.document_is_indexed("doc1") is True

    def test_document_is_indexed_false(self, mock_client):
        mock_client.scroll.return_value = ([], None)
        with patch(
            "vector_db.multimodal_indexer.get_qdrant_client", return_value=mock_client
        ):
            idx = MultimodalIndexer()
            assert idx.document_is_indexed("unknown") is False