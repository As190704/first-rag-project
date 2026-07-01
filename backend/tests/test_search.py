"""
Unit tests for semantic search.

Uses mocked Qdrant client and embedding engine to validate
the search pipeline without external dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from schemas.chunk import ChunkType, SearchRequest
from services.search_service import SearchService
from vector_db.search import SemanticSearcher


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_qdrant_hit(
    score: float = 0.92,
    chunk_id: str = "chunk_00001",
    document_id: str = "doc1",
    text: str = "Transformers use attention.",
    page: int = 3,
    heading: str = "Architecture",
    chunk_type: str = "paragraph",
    source_file: str = "paper.pdf",
) -> MagicMock:
    hit = MagicMock()
    hit.score = score
    hit.payload = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "text": text,
        "page": page,
        "heading": heading,
        "chunk_type": chunk_type,
        "source_file": source_file,
    }
    return hit


# ── Tests: SemanticSearcher ───────────────────────────────────────────────────


class TestSemanticSearcher:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.search.return_value = [make_qdrant_hit()]
        return client

    @pytest.fixture
    def searcher(self, mock_client):
        with patch("vector_db.search.get_qdrant_client", return_value=mock_client):
            yield SemanticSearcher()

    def test_search_returns_results(self, searcher, mock_client):
        query_vector = [0.1] * 1024
        with patch("vector_db.search.get_qdrant_client", return_value=mock_client):
            results, latency = searcher.search(query_vector, top_k=5)
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.92, abs=1e-3)

    def test_search_result_fields_populated(self, searcher, mock_client):
        query_vector = [0.1] * 1024
        with patch("vector_db.search.get_qdrant_client", return_value=mock_client):
            results, _ = searcher.search(query_vector)
        r = results[0]
        assert r.chunk_id == "chunk_00001"
        assert r.document_id == "doc1"
        assert r.text == "Transformers use attention."
        assert r.page == 3
        assert r.heading == "Architecture"

    def test_latency_is_positive(self, searcher, mock_client):
        with patch("vector_db.search.get_qdrant_client", return_value=mock_client):
            _, latency = searcher.search([0.1] * 1024)
        assert latency >= 0

    def test_filter_built_for_document_id(self, searcher, mock_client):
        with patch("vector_db.search.get_qdrant_client", return_value=mock_client):
            searcher.search([0.1] * 1024, document_id="doc1")
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["query_filter"] is not None

    def test_no_filter_when_no_args(self, searcher, mock_client):
        with patch("vector_db.search.get_qdrant_client", return_value=mock_client):
            searcher.search([0.1] * 1024)
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["query_filter"] is None

    def test_search_failure_raises_runtime_error(self, mock_client):
        mock_client.search.side_effect = Exception("Qdrant down")
        with patch("vector_db.search.get_qdrant_client", return_value=mock_client):
            searcher = SemanticSearcher()
            with pytest.raises(RuntimeError, match="Search failed"):
                searcher.search([0.1] * 1024)

    def test_empty_results(self, mock_client):
        mock_client.search.return_value = []
        with patch("vector_db.search.get_qdrant_client", return_value=mock_client):
            searcher = SemanticSearcher()
            results, _ = searcher.search([0.1] * 1024)
        assert results == []


# ── Tests: SearchService ──────────────────────────────────────────────────────


class TestSearchService:
    @pytest.fixture
    def mock_embed(self):
        with patch(
            "services.search_service.embedding_engine.embed_query",
            return_value=[0.5] * 1024,
        ) as m:
            yield m

    @pytest.fixture
    def mock_searcher_search(self):
        from schemas.chunk import SearchResult
        fake_result = SearchResult(
            score=0.88,
            chunk_id="chunk_00002",
            document_id="doc2",
            text="Positional encoding adds order information.",
            page=4,
            heading="Transformer Architecture",
            chunk_type="paragraph",
            source_file="attention.pdf",
        )
        with patch(
            "services.search_service.SemanticSearcher.search",
            return_value=([fake_result], 12.5),
        ) as m:
            yield m

    def test_search_returns_response(self, mock_embed, mock_searcher_search):
        service = SearchService()
        req = SearchRequest(query="What is positional encoding?", top_k=5)
        response = service.search(req)
        assert response.query == "What is positional encoding?"
        assert response.total_results == 1
        assert len(response.results) == 1
        assert response.latency_ms == pytest.approx(12.5)

    def test_search_result_score(self, mock_embed, mock_searcher_search):
        service = SearchService()
        req = SearchRequest(query="attention mechanism")
        response = service.search(req)
        assert response.results[0].score == pytest.approx(0.88)

    def test_embed_failure_raises_runtime_error(self):
        with patch(
            "services.search_service.embedding_engine.embed_query",
            side_effect=Exception("Model error"),
        ):
            service = SearchService()
            req = SearchRequest(query="test query")
            with pytest.raises(RuntimeError, match="Failed to embed"):
                service.search(req)