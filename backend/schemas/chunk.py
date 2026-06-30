"""
Pydantic schemas for document chunks.

Chunks are the atomic unit of the indexing pipeline. Every chunk
carries enough metadata to reconstruct its origin and to support
filtered semantic search without requiring a database join.

Design notes for Phase 3:
  - chunk_type is an enum so image/table/equation types can be added cleanly.
  - embedding is kept as an optional field so the same model can represent
    a pre-embedding chunk and a post-embedding chunk without duplication.
  - multimodal_ref is reserved for Phase 3 image/figure cross-references.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ── Chunk type taxonomy ───────────────────────────────────────────────────────


class ChunkType(str, Enum):
    """
    Semantic classification of a chunk's content role.

    Extending for Phase 3:
      Add IMAGE = "image", EQUATION = "equation", CAPTION = "caption"
      without touching existing code paths.
    """

    TITLE = "title"
    ABSTRACT = "abstract"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    SUBSECTION = "subsection"
    TABLE = "table"
    FIGURE_CAPTION = "figure_caption"
    CONCLUSION = "conclusion"
    INTRODUCTION = "introduction"
    REFERENCES = "references"
    UNKNOWN = "unknown"


# ── Core chunk model ──────────────────────────────────────────────────────────


class Chunk(BaseModel):
    """
    A single semantic chunk extracted from a parsed document.

    This is the canonical unit passed between chunker → embedder → indexer.
    """

    chunk_id: str = Field(
        default_factory=lambda: f"chunk_{uuid4().hex[:12]}",
        description="Unique chunk identifier",
    )
    document_id: str = Field(..., description="Parent document identifier")
    page: int = Field(default=1, ge=1, description="1-based source page number")
    heading: str = Field(default="", description="Nearest section heading")
    chunk_type: ChunkType = Field(
        default=ChunkType.PARAGRAPH,
        description="Semantic role of this chunk",
    )
    text: str = Field(..., min_length=1, description="Raw chunk text content")
    source_file: str = Field(default="", description="Original filename")
    token_count: int = Field(default=0, ge=0, description="Approximate token count")
    chunk_index: int = Field(
        default=0,
        ge=0,
        description="Sequential index within document (for ordering)",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of chunk creation",
    )

    # Reserved for Phase 3 multimodal cross-references
    multimodal_ref: str | None = Field(
        default=None,
        description="[Phase 3] Path to associated image or figure",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk text must contain non-whitespace content.")
        return v.strip()

    def to_qdrant_payload(self) -> dict[str, Any]:
        """
        Serialise chunk metadata into a Qdrant point payload.

        Only stores metadata (no embedding vector — that goes in the
        point's vector field). Designed for efficient filtered search.

        Returns:
            Dictionary safe for Qdrant payload storage.
        """
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "page": self.page,
            "heading": self.heading,
            "chunk_type": self.chunk_type.value,
            "text": self.text,
            "source_file": self.source_file,
            "token_count": self.token_count,
            "chunk_index": self.chunk_index,
            "created_at": self.created_at.isoformat(),
            "multimodal_ref": self.multimodal_ref,
        }


# ── Embedding-enriched chunk ──────────────────────────────────────────────────


class EmbeddedChunk(BaseModel):
    """
    A Chunk paired with its dense embedding vector.

    Kept as a separate model to enforce that embedding is always
    present before insertion into Qdrant.
    """

    chunk: Chunk
    embedding: list[float] = Field(..., description="Dense embedding vector")
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="Model used to generate this embedding",
    )

    @property
    def vector_dimension(self) -> int:
        """Return the dimensionality of the embedding vector."""
        return len(self.embedding)


# ── Search result model ───────────────────────────────────────────────────────


class SearchResult(BaseModel):
    """A single result returned by semantic search."""

    score: float = Field(..., description="Cosine similarity score (0–1)")
    chunk_id: str
    document_id: str
    text: str
    page: int
    heading: str
    chunk_type: str
    source_file: str
    document_title: str = Field(default="", description="Human-readable document title")


# ── API request / response models ─────────────────────────────────────────────


class IndexRequest(BaseModel):
    """Request body for POST /index."""

    document_id: str = Field(..., description="Document ID from Phase 1 parse output")
    force_reindex: bool = Field(
        default=False,
        description="If True, delete existing vectors and re-index from scratch",
    )


class IndexResponse(BaseModel):
    """Response body for POST /index."""

    document_id: str
    chunks_created: int
    embeddings_generated: int
    vectors_stored: int
    collection_name: str
    duration_seconds: float
    message: str


class SearchRequest(BaseModel):
    """Request body for POST /search."""

    query: str = Field(..., min_length=1, description="Natural language search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")

    # Optional filters
    document_id: str | None = Field(default=None, description="Filter to a single document")
    heading: str | None = Field(default=None, description="Filter by section heading")
    page: int | None = Field(default=None, ge=1, description="Filter to a specific page")
    chunk_type: ChunkType | None = Field(default=None, description="Filter by chunk type")


class SearchResponse(BaseModel):
    """Response body for POST /search."""

    query: str
    total_results: int
    results: list[SearchResult]
    latency_ms: float