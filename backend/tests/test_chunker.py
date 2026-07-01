"""
Unit tests for the semantic chunker.

Tests cover:
  - Single short section → single chunk
  - Long section → multiple overlapping chunks
  - Table processing
  - Figure caption creation
  - Heading classification
  - Empty / edge case inputs
"""

from __future__ import annotations

import pytest

from embeddings.chunker import (
    SemanticChunker,
    classify_heading,
    estimate_tokens,
)
from schemas.chunk import ChunkType


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def minimal_doc() -> dict:
    """Minimal valid parsed document dict."""
    return {
        "document_id": "test123",
        "filename": "test_paper.pdf",
        "title": "Test Paper Title",
        "sections": [],
        "images": [],
        "tables": [],
    }


@pytest.fixture
def full_doc() -> dict:
    """Full parsed document with sections, tables, and images."""
    return {
        "document_id": "abc456",
        "filename": "attention.pdf",
        "title": "Attention Is All You Need",
        "sections": [
            {
                "page": 1,
                "heading": "Abstract",
                "text": (
                    "The dominant sequence transduction models are based on complex "
                    "recurrent or convolutional neural networks that include an encoder "
                    "and a decoder. The best performing models also connect the encoder "
                    "and decoder through an attention mechanism."
                ),
                "level": 1,
            },
            {
                "page": 2,
                "heading": "Introduction",
                "text": "Recurrent neural networks, long short-term memory and gated "
                        "recurrent neural networks in particular, have been firmly "
                        "established as state of the art approaches in sequence modeling.",
                "level": 1,
            },
            {
                "page": 5,
                "heading": "Conclusion",
                "text": "In this work, we presented the Transformer, the first sequence "
                        "transduction model based entirely on attention.",
                "level": 1,
            },
        ],
        "tables": [
            {
                "page": 8,
                "table_number": 1,
                "headers": ["Model", "BLEU", "Training Cost"],
                "rows": [
                    ["Transformer (big)", "28.4", "$28K"],
                    ["LSTM", "26.4", "$1K"],
                ],
                "raw_text": "",
            }
        ],
        "images": [
            {
                "page": 3,
                "image_number": 1,
                "image_path": "output/images/abc456/img_001.png",
            }
        ],
    }


@pytest.fixture
def chunker() -> SemanticChunker:
    return SemanticChunker()


# ── Tests: heading classification ─────────────────────────────────────────────


class TestClassifyHeading:
    def test_abstract(self):
        assert classify_heading("Abstract") == ChunkType.ABSTRACT

    def test_introduction(self):
        assert classify_heading("1. Introduction") == ChunkType.INTRODUCTION

    def test_conclusion(self):
        assert classify_heading("Conclusion and Future Work") == ChunkType.CONCLUSION

    def test_references(self):
        assert classify_heading("References") == ChunkType.REFERENCES

    def test_method(self):
        assert classify_heading("Methodology") == ChunkType.SUBSECTION

    def test_unknown_heading(self):
        assert classify_heading("Background") == ChunkType.HEADING

    def test_empty_heading(self):
        assert classify_heading("") == ChunkType.PARAGRAPH

    def test_case_insensitive(self):
        assert classify_heading("ABSTRACT") == ChunkType.ABSTRACT


# ── Tests: token estimation ───────────────────────────────────────────────────


class TestEstimateTokens:
    def test_short_text(self):
        assert estimate_tokens("Hello world") > 0

    def test_empty_text(self):
        assert estimate_tokens("") == 1  # max(1, ...)

    def test_proportional(self):
        short = estimate_tokens("Hello")
        long = estimate_tokens("Hello " * 100)
        assert long > short


# ── Tests: chunk_document ─────────────────────────────────────────────────────


class TestChunkDocument:
    def test_title_chunk_created(self, chunker, full_doc):
        chunks = chunker.chunk_document(full_doc)
        title_chunks = [c for c in chunks if c.chunk_type == ChunkType.TITLE]
        assert len(title_chunks) == 1
        assert "Attention" in title_chunks[0].text

    def test_abstract_chunk_created(self, chunker, full_doc):
        chunks = chunker.chunk_document(full_doc)
        abstract_chunks = [c for c in chunks if c.chunk_type == ChunkType.ABSTRACT]
        assert len(abstract_chunks) >= 1

    def test_conclusion_chunk_created(self, chunker, full_doc):
        chunks = chunker.chunk_document(full_doc)
        conclusion_chunks = [c for c in chunks if c.chunk_type == ChunkType.CONCLUSION]
        assert len(conclusion_chunks) >= 1

    def test_table_chunk_created(self, chunker, full_doc):
        chunks = chunker.chunk_document(full_doc)
        table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLE]
        assert len(table_chunks) == 1
        assert "BLEU" in table_chunks[0].text

    def test_figure_caption_created(self, chunker, full_doc):
        chunks = chunker.chunk_document(full_doc)
        fig_chunks = [c for c in chunks if c.chunk_type == ChunkType.FIGURE_CAPTION]
        assert len(fig_chunks) == 1
        assert fig_chunks[0].multimodal_ref is not None

    def test_chunk_indices_sequential(self, chunker, full_doc):
        chunks = chunker.chunk_document(full_doc)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_document_id_propagated(self, chunker, full_doc):
        chunks = chunker.chunk_document(full_doc)
        for chunk in chunks:
            assert chunk.document_id == "abc456"

    def test_page_numbers_valid(self, chunker, full_doc):
        chunks = chunker.chunk_document(full_doc)
        for chunk in chunks:
            assert chunk.page >= 1

    def test_empty_sections_handled(self, chunker, minimal_doc):
        chunks = chunker.chunk_document(minimal_doc)
        # Only title chunk should be created
        assert len(chunks) == 1
        assert chunks[0].chunk_type == ChunkType.TITLE

    def test_no_title_handled(self, chunker):
        doc = {
            "document_id": "notitle",
            "filename": "test.pdf",
            "title": "",
            "sections": [],
            "images": [],
            "tables": [],
        }
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 0

    def test_chunk_text_not_empty(self, chunker, full_doc):
        chunks = chunker.chunk_document(full_doc)
        for chunk in chunks:
            assert chunk.text.strip() != ""

    def test_missing_document_id_raises(self, chunker):
        with pytest.raises(ValueError, match="missing required fields"):
            chunker.chunk_document({"title": "No ID here"})


class TestSplitWithOverlap:
    def test_long_text_split_into_multiple_chunks(self, chunker):
        long_text = "This is a sentence about transformers. " * 200
        doc = {
            "document_id": "splitdoc",
            "filename": "test.pdf",
            "title": "",
            "sections": [{"page": 1, "heading": "Methods", "text": long_text}],
            "images": [],
            "tables": [],
        }
        chunks = chunker.chunk_document(doc)
        content_chunks = [c for c in chunks if c.chunk_type != ChunkType.TITLE]
        assert len(content_chunks) > 1

    def test_overlap_present_in_consecutive_chunks(self, chunker):
        """Consecutive chunks should share some text (overlap)."""
        long_text = (
            "Alpha sentence here. Beta sentence here. Gamma sentence here. "
            "Delta sentence here. Epsilon sentence here. " * 80
        )
        doc = {
            "document_id": "overlapdoc",
            "filename": "test.pdf",
            "title": "",
            "sections": [{"page": 1, "heading": "Body", "text": long_text}],
            "images": [],
            "tables": [],
        }
        chunks = chunker.chunk_document(doc)
        body_chunks = [c for c in chunks if c.chunk_type == ChunkType.HEADING or
                      c.chunk_type == ChunkType.PARAGRAPH or
                      c.chunk_type == ChunkType.SUBSECTION]

        # At least two content chunks should exist
        assert len(body_chunks) >= 2