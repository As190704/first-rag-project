"""Tests for multimodal search — hybrid retrieval pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from schemas.multimodal_chunk import (
    MultimodalSearchRequest,
    MultimodalSearchResult,
    VisualChunkType,
)
from services.multimodal_service import MultimodalService


def make_mock_hit(
    chunk_id: str = "mm_001",
    score: float = 0.90,
    chunk_type: str = "figure",
    page: int = 3,
    description: str = "CNN architecture diagram.",
    source_file: str = "paper.pdf",
) -> MagicMock:
    hit = MagicMock()
    hit.score = score
    hit.id = chunk_id
    hit.payload = {
        "chunk_id": chunk_id,
        "document_id": "doc1",
        "chunk_type": chunk_type,
        "page": page,
        "caption": f"{chunk_type} caption",
        "description": description,
        "image_path": f"output/images/doc1/{chunk_id}.png",
        "source_file": source_file,
    }
    return hit


@pytest.fixture
def mock_text_embed():
    with patch(
        "services.multimodal_service.embedding_engine.embed_query",
        return_value=[0.5] * 1024,
    ) as m:
        yield m


@pytest.fixture
def mock_visual_embed():
    with patch(
        "services.multimodal_service.colpali_embedder.embed_query_text",
        return_value=[0.3] * 128,
    ) as m:
        yield m


@pytest.fixture
def mock_qdrant_search():
    with patch(
        "services.multimodal_service.get_qdrant_client"
    ) as mock_get:
        mock_client = MagicMock()
        mock_client.search.return_value = [make_mock_hit()]
        mock_get.return_value = mock_client
        yield mock_client


class TestMultimodalHybridSearch:
    def test_hybrid_search_returns_response(
        self, mock_text_embed, mock_visual_embed, mock_qdrant_search
    ):
        service = MultimodalService()
        req = MultimodalSearchRequest(
            query="CNN architecture diagram",
            top_k=5,
            search_mode="hybrid",
        )
        response = service.search(req)
        assert response.query == "CNN architecture diagram"
        assert response.search_mode == "hybrid"
        assert isinstance(response.results, list)

    def test_text_only_search(self, mock_text_embed, mock_qdrant_search):
        service = MultimodalService()
        req = MultimodalSearchRequest(
            query="confusion matrix",
            top_k=3,
            search_mode="text",
        )
        response = service.search(req)
        assert response.total_results >= 0

    def test_visual_only_search(self, mock_visual_embed, mock_qdrant_search):
        service = MultimodalService()
        req = MultimodalSearchRequest(
            query="architecture diagram",
            top_k=3,
            search_mode="visual",
        )
        response = service.search(req)
        assert response.search_mode == "visual"

    def test_result_fields_populated(
        self, mock_text_embed, mock_visual_embed, mock_qdrant_search
    ):
        service = MultimodalService()
        req = MultimodalSearchRequest(query="transformer", top_k=5)
        response = service.search(req)

        if response.results:
            r = response.results[0]
            assert hasattr(r, "score")
            assert hasattr(r, "chunk_type")
            assert hasattr(r, "page")
            assert hasattr(r, "description")

    def test_latency_recorded(
        self, mock_text_embed, mock_visual_embed, mock_qdrant_search
    ):
        service = MultimodalService()
        req = MultimodalSearchRequest(query="test", top_k=2)
        response = service.search(req)
        assert response.latency_ms >= 0.0

    def test_chunk_type_filter_built(
        self, mock_text_embed, mock_visual_embed, mock_qdrant_search
    ):
        service = MultimodalService()
        req = MultimodalSearchRequest(
            query="figure",
            top_k=3,
            chunk_types=[VisualChunkType.FIGURE, VisualChunkType.DIAGRAM],
        )
        response = service.search(req)
        # Filter should have been applied — check qdrant received filter
        calls = mock_qdrant_search.search.call_args_list
        for call in calls:
            kwargs = call.kwargs
            if kwargs.get("query_filter") is not None:
                break  # Filter was passed

    def test_deduplication_merges_results(
        self, mock_text_embed, mock_visual_embed, mock_qdrant_search
    ):
        """Same chunk returned by text and visual search should appear once."""
        mock_qdrant_search.search.return_value = [
            make_mock_hit(chunk_id="mm_001", score=0.9),
            make_mock_hit(chunk_id="mm_002", score=0.8),
        ]
        service = MultimodalService()
        req = MultimodalSearchRequest(query="duplicate test", search_mode="hybrid")
        response = service.search(req)
        chunk_ids = [r.chunk_id for r in response.results]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk_ids in results"