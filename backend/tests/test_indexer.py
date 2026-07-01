"""
Unit tests for the VectorIndexer.

Uses mocked Qdrant client to test indexing logic without
requiring a live Qdrant instance.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from schemas.chunk import Chunk, ChunkType, EmbeddedChunk
from vector_db.indexer import VectorIndexer


# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_embedded_chunk(
    chunk_id: str = "chunk_00001_doc1",
    document_id: str = "doc1",
    text: str = "Sample text for indexing.",
    page: int = 1,
) -> EmbeddedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        page=page,
        heading="Introduction",
        chunk_type=ChunkType.PARAGRAPH,
        text=text,
        source_file="paper.pdf",
        token_count=5,
    )
    return EmbeddedChunk(
        chunk=chunk,
        embedding=[0.1] * 1024,
        embedding_model="BAAI/bge-m3",
    )


@pytest.fixture
def mock_qdrant_client():
    client = MagicMock()
    client.upsert.return_value = MagicMock(status="completed")
    client.scroll.return_value = ([], None)
    client.count.return_value = MagicMock(count=0)
    return client


@pytest.fixture
def indexer(mock_qdrant_client):
    with patch(
        "vector_db.indexer.get_qdrant_client",
        return_value=mock_qdrant_client,
    ):
        yield VectorIndexer(collection_name="test_collection", batch_size=2)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestVectorIndexer:
    def test_index_single_chunk(self, indexer, mock_qdrant_client):
        chunks = [make_embedded_chunk()]
        with patch("vector_db.indexer.get_qdrant_client", return_value=mock_qdrant_client):
            count = indexer.index_chunks(chunks)
        assert count == 1

    def test_index_multiple_chunks(self, indexer, mock_qdrant_client):
        chunks = [make_embedded_chunk(f"chunk_{i:05d}_doc1", text=f"Text {i}") for i in range(5)]
        with patch("vector_db.indexer.get_qdrant_client", return_value=mock_qdrant_client):
            count = indexer.index_chunks(chunks)
        assert count == 5

    def test_batching_calls_upsert_multiple_times(self, mock_qdrant_client):
        """With batch_size=2 and 5 chunks, upsert should be called 3 times."""
        with patch("vector_db.indexer.get_qdrant_client", return_value=mock_qdrant_client):
            indexer = VectorIndexer(collection_name="test", batch_size=2)
            chunks = [make_embedded_chunk(f"chunk_{i:05d}_doc", text=f"Text {i}") for i in range(5)]
            indexer.index_chunks(chunks)

        assert mock_qdrant_client.upsert.call_count == 3  # ceil(5/2)

    def test_empty_chunks_returns_zero(self, indexer, mock_qdrant_client):
        with patch("vector_db.indexer.get_qdrant_client", return_value=mock_qdrant_client):
            count = indexer.index_chunks([])
        assert count == 0

    def test_upsert_failure_raises_runtime_error(self, mock_qdrant_client):
        mock_qdrant_client.upsert.side_effect = Exception("Connection refused")
        with patch("vector_db.indexer.get_qdrant_client", return_value=mock_qdrant_client):
            indexer = VectorIndexer(collection_name="test")
            chunks = [make_embedded_chunk()]
            with pytest.raises(RuntimeError, match="Vector indexing failed"):
                indexer.index_chunks(chunks)

    def test_document_is_indexed_true(self, mock_qdrant_client):
        mock_qdrant_client.scroll.return_value = (
            [MagicMock()],  # Non-empty result
            None,
        )
        with patch("vector_db.indexer.get_qdrant_client", return_value=mock_qdrant_client):
            indexer = VectorIndexer()
            assert indexer.document_is_indexed("doc1") is True

    def test_document_is_indexed_false(self, mock_qdrant_client):
        mock_qdrant_client.scroll.return_value = ([], None)
        with patch("vector_db.indexer.get_qdrant_client", return_value=mock_qdrant_client):
            indexer = VectorIndexer()
            assert indexer.document_is_indexed("doc_unknown") is False

    def test_point_has_named_vector(self, mock_qdrant_client):
        """Each upserted point must use the named 'text' vector key."""
        chunks = [make_embedded_chunk()]
        with patch("vector_db.indexer.get_qdrant_client", return_value=mock_qdrant_client):
            indexer = VectorIndexer()
            indexer.index_chunks(chunks)

        call_args = mock_qdrant_client.upsert.call_args
        points = call_args.kwargs["points"]
        assert len(points) == 1
        assert "text" in points[0].vector
        assert len(points[0].vector["text"]) == 1024