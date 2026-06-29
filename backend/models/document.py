"""
Pydantic data models for the document ingestion pipeline.

These models define the canonical schema for parsed documents,
sections, images, tables, and API responses throughout Phase 1.
"""

from typing import Any
from pydantic import BaseModel, Field


# ── Sub-models ────────────────────────────────────────────────────────────────


class DocumentMetadata(BaseModel):
    """Bibliographic and structural metadata extracted from a document."""

    authors: list[str] = Field(default_factory=list, description="List of author names")
    year: str = Field(default="", description="Publication year")
    pages: int = Field(default=0, description="Total page count")
    doi: str = Field(default="", description="Digital Object Identifier if present")
    keywords: list[str] = Field(default_factory=list, description="Extracted keywords")
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata fields",
    )


class Section(BaseModel):
    """A single logical section (heading + body text) from a document."""

    page: int = Field(..., description="1-based page number where section starts")
    heading: str = Field(default="", description="Section heading / title")
    text: str = Field(default="", description="Full body text of the section")
    level: int = Field(
        default=1,
        description="Heading hierarchy level (1=top, 2=sub, etc.)",
    )


class BoundingBox(BaseModel):
    """Pixel-space bounding box for a region within a page."""

    x0: float = Field(..., description="Left coordinate")
    y0: float = Field(..., description="Top coordinate")
    x1: float = Field(..., description="Right coordinate")
    y1: float = Field(..., description="Bottom coordinate")


class ExtractedImage(BaseModel):
    """Metadata for a single image extracted from a document."""

    page: int = Field(..., description="1-based page number containing the image")
    image_number: int = Field(..., description="Sequential image index on the page")
    image_path: str = Field(..., description="Relative path to the saved image file")
    bounding_box: BoundingBox | None = Field(
        default=None,
        description="Bounding box within the page (if available)",
    )
    width: int = Field(default=0, description="Image width in pixels")
    height: int = Field(default=0, description="Image height in pixels")


class TableCell(BaseModel):
    """A single cell within an extracted table."""

    row: int
    col: int
    text: str


class ExtractedTable(BaseModel):
    """A table extracted from a document page."""

    page: int = Field(..., description="1-based page number containing the table")
    table_number: int = Field(..., description="Sequential table index in document")
    headers: list[str] = Field(default_factory=list, description="Column headers")
    rows: list[list[str]] = Field(default_factory=list, description="Data rows")
    raw_text: str = Field(default="", description="Raw text fallback for the table")


class ParsedDocument(BaseModel):
    """
    Top-level model representing a fully parsed document.

    This is the canonical output schema written to JSON and returned
    by the API. All downstream phases (embeddings, retrieval) consume
    this model.
    """

    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original uploaded filename")
    title: str = Field(default="", description="Document title")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    sections: list[Section] = Field(default_factory=list)
    images: list[ExtractedImage] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
    ocr_applied: bool = Field(
        default=False,
        description="True if OCR was used on any page",
    )
    parse_errors: list[str] = Field(
        default_factory=list,
        description="Non-fatal errors encountered during parsing",
    )


# ── API Response model ────────────────────────────────────────────────────────


class UploadResponse(BaseModel):
    """Response schema for the POST /upload endpoint."""

    document_id: str
    filename: str
    pages: int
    sections: list[Section]
    images: list[ExtractedImage]
    tables: list[ExtractedTable]
    metadata: DocumentMetadata
    ocr_applied: bool
    output_json_path: str
    message: str = "Document parsed successfully"