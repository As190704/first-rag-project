"""
Pydantic schemas for multimodal document chunks.

Every visual element extracted from a research document (figure,
chart, table, equation, diagram) becomes a MultimodalChunk that
can be independently embedded, indexed, and retrieved.

Design for Phase 4:
  - MultimodalChunk.rerank_score is reserved for cross-encoder reranking.
  - MultimodalChunk.citation_context will carry surrounding text for
    citation-aware RAG answer generation.
  - The HybridSearchRequest already supports score fusion weights so
    Phase 4 can tune retrieval without schema changes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ── Visual element taxonomy ───────────────────────────────────────────────────


class VisualChunkType(str, Enum):
    """
    Semantic classification of a visual element's role.

    Mirrors Phase 2 ChunkType but is scoped to multimodal content.
    Phase 4 can add VIDEO_FRAME, AUDIO_TRANSCRIPT, etc.
    """

    FIGURE = "figure"
    DIAGRAM = "diagram"
    FLOWCHART = "flowchart"
    ARCHITECTURE = "architecture"
    SCREENSHOT = "screenshot"
    CHART = "chart"
    TABLE = "table"
    EQUATION = "equation"
    UNKNOWN = "unknown"


class ImageClassification(str, Enum):
    """
    Fine-grained visual classification produced by the image classifier.
    Maps 1-to-1 with VisualChunkType but separated for clarity.
    """

    FIGURE = "figure"
    DIAGRAM = "diagram"
    FLOWCHART = "flowchart"
    ARCHITECTURE = "architecture"
    SCREENSHOT = "screenshot"
    CHART = "chart"
    TABLE = "table"
    EQUATION = "equation"
    UNKNOWN = "unknown"


# ── Bounding box ──────────────────────────────────────────────────────────────


class BoundingBox(BaseModel):
    """Pixel-space bounding box within a page."""

    x0: float = Field(..., description="Left coordinate")
    y0: float = Field(..., description="Top coordinate")
    x1: float = Field(..., description="Right coordinate")
    y1: float = Field(..., description="Bottom coordinate")


# ── Chart structured data ─────────────────────────────────────────────────────


class ChartData(BaseModel):
    """Structured representation of an extracted chart."""

    chart_type: str = Field(default="unknown", description="bar|line|pie|scatter|unknown")
    title: str = Field(default="", description="Chart title if detected")
    x_axis: str = Field(default="", description="X-axis label")
    y_axis: str = Field(default="", description="Y-axis label")
    legend: list[str] = Field(default_factory=list, description="Legend entries")
    values: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted data points [{label: value}]",
    )
    description: str = Field(default="", description="Natural language description")


# ── Table structured data ─────────────────────────────────────────────────────


class TableData(BaseModel):
    """Structured representation of an extracted table."""

    columns: list[str] = Field(default_factory=list, description="Column headers")
    rows: list[list[str]] = Field(default_factory=list, description="Data rows")
    description: str = Field(default="", description="Natural language description")
    extraction_method: str = Field(
        default="unknown",
        description="camelot|docling|fallback",
    )


# ── Equation metadata ─────────────────────────────────────────────────────────


class EquationData(BaseModel):
    """Metadata for a detected mathematical equation."""

    raw_ocr_text: str = Field(default="", description="Raw OCR text if available")
    latex: str = Field(default="", description="LaTeX representation if detectable")
    equation_type: str = Field(
        default="unknown",
        description="inline|display|numbered",
    )
    description: str = Field(default="", description="Natural language description")


# ── Core multimodal chunk ─────────────────────────────────────────────────────


class MultimodalChunk(BaseModel):
    """
    A single multimodal element extracted from a research document.

    This is the canonical unit for visual content: one chunk per figure,
    chart, table, or equation. Each chunk carries both its structured
    metadata and a natural language description suitable for embedding.
    """

    chunk_id: str = Field(
        default_factory=lambda: f"mm_{uuid4().hex[:12]}",
        description="Unique multimodal chunk identifier",
    )
    document_id: str = Field(..., description="Parent document identifier")
    page: int = Field(default=1, ge=1, description="1-based source page number")
    chunk_type: VisualChunkType = Field(
        default=VisualChunkType.UNKNOWN,
        description="Semantic type of this visual element",
    )
    image_number: int = Field(default=0, description="Sequential image index in document")
    image_path: str = Field(default="", description="Relative path to the image file")
    caption: str = Field(default="", description="Short caption or label")
    description: str = Field(
        default="",
        description="Detailed Qwen2-VL generated description of visual content",
    )
    bounding_box: BoundingBox | None = Field(
        default=None,
        description="Spatial location on page",
    )
    source_file: str = Field(default="", description="Original document filename")

    # Type-specific structured data
    chart_data: ChartData | None = Field(
        default=None,
        description="Structured chart data (charts only)",
    )
    table_data: TableData | None = Field(
        default=None,
        description="Structured table data (tables only)",
    )
    equation_data: EquationData | None = Field(
        default=None,
        description="Equation metadata (equations only)",
    )

    # Embedding metadata
    embedding_model: str = Field(
        default="colpali",
        description="Model used for visual embedding",
    )
    embedding_text: str = Field(
        default="",
        description="Text used as input for text-side embedding (description + caption)",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Phase 4 reserved fields
    rerank_score: float | None = Field(
        default=None,
        description="[Phase 4] Cross-encoder reranking score",
    )
    citation_context: str = Field(
        default="",
        description="[Phase 4] Surrounding text context for citation-aware RAG",
    )

    @field_validator("description")
    @classmethod
    def description_stripped(cls, v: str) -> str:
        return v.strip()

    def get_embedding_text(self) -> str:
        """
        Build the text string used for generating the text-side embedding.

        Combines caption and description so that text queries like
        "CNN architecture diagram" can match figure descriptions.

        Returns:
            Combined searchable text representation.
        """
        parts = []
        if self.caption:
            parts.append(f"Caption: {self.caption}")
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.chunk_type == VisualChunkType.TABLE and self.table_data:
            parts.append(f"Table: {self.table_data.description}")
        if self.chunk_type == VisualChunkType.CHART and self.chart_data:
            parts.append(f"Chart: {self.chart_data.description}")
        if self.chunk_type == VisualChunkType.EQUATION and self.equation_data:
            parts.append(f"Equation: {self.equation_data.description}")
        return " ".join(parts) if parts else f"{self.chunk_type.value} on page {self.page}"

    def to_qdrant_payload(self) -> dict[str, Any]:
        """
        Serialise chunk metadata for Qdrant point payload storage.

        Returns:
            Dictionary suitable for Qdrant payload field.
        """
        payload: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "page": self.page,
            "chunk_type": self.chunk_type.value,
            "image_number": self.image_number,
            "image_path": self.image_path,
            "caption": self.caption,
            "description": self.description,
            "source_file": self.source_file,
            "embedding_model": self.embedding_model,
            "embedding_text": self.get_embedding_text(),
            "created_at": self.created_at.isoformat(),
        }
        if self.bounding_box:
            payload["bounding_box"] = {
                "x0": self.bounding_box.x0,
                "y0": self.bounding_box.y0,
                "x1": self.bounding_box.x1,
                "y1": self.bounding_box.y1,
            }
        if self.chart_data:
            payload["chart_json"] = self.chart_data.model_dump()
        if self.table_data:
            payload["table_json"] = self.table_data.model_dump()
        if self.equation_data:
            payload["equation_json"] = self.equation_data.model_dump()
        return payload


# ── Embedded multimodal chunk ─────────────────────────────────────────────────


class EmbeddedMultimodalChunk(BaseModel):
    """
    A MultimodalChunk paired with its visual and text embeddings.

    Both vectors are stored so that hybrid search can query either
    the visual space (ColPali) or the text space (BGE-M3) independently.
    """

    chunk: MultimodalChunk
    visual_embedding: list[float] | None = Field(
        default=None,
        description="ColPali visual embedding (128-dim patch-averaged)",
    )
    text_embedding: list[float] | None = Field(
        default=None,
        description="BGE-M3 text embedding of description+caption (1024-dim)",
    )

    @property
    def has_visual_embedding(self) -> bool:
        return self.visual_embedding is not None and len(self.visual_embedding) > 0

    @property
    def has_text_embedding(self) -> bool:
        return self.text_embedding is not None and len(self.text_embedding) > 0


# ── API models ────────────────────────────────────────────────────────────────


class MultimodalIndexRequest(BaseModel):
    """Request body for POST /multimodal/index."""

    document_id: str = Field(..., description="Document ID from Phase 1")
    force_reindex: bool = Field(default=False)
    run_captioning: bool = Field(
        default=True,
        description="Set False to skip Qwen2-VL (faster, lower quality)",
    )
    run_chart_ocr: bool = Field(default=True)
    run_table_extraction: bool = Field(default=True)
    run_equation_detection: bool = Field(default=True)


class MultimodalIndexResponse(BaseModel):
    """Response body for POST /multimodal/index."""

    document_id: str
    figures_processed: int
    charts_processed: int
    tables_processed: int
    equations_detected: int
    vectors_stored: int
    collection_name: str
    duration_seconds: float
    message: str


class MultimodalSearchRequest(BaseModel):
    """Request body for POST /multimodal/search."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: str | None = None
    chunk_types: list[VisualChunkType] | None = Field(
        default=None,
        description="Filter to specific visual element types",
    )
    search_mode: str = Field(
        default="hybrid",
        description="text|visual|hybrid",
    )
    # Phase 4: fusion weight between text and visual scores
    text_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    visual_weight: float = Field(default=0.5, ge=0.0, le=1.0)


class MultimodalSearchResult(BaseModel):
    """A single result from multimodal search."""

    score: float
    chunk_type: str
    page: int
    caption: str
    description: str
    image_path: str
    document_id: str
    source_file: str
    chunk_id: str
    chart_data: dict | None = None
    table_data: dict | None = None
    equation_data: dict | None = None
    # Phase 4 reserved
    rerank_score: float | None = None


class MultimodalSearchResponse(BaseModel):
    """Response body for POST /multimodal/search."""

    query: str
    total_results: int
    results: list[MultimodalSearchResult]
    latency_ms: float
    search_mode: str