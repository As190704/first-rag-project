"""
PDF parsing module — the core of Phase 1.

Strategy:
  1. Attempt structured extraction via Docling (layout, headings, tables).
  2. For each page, check whether it is scanned (low text density).
  3. If scanned, fall back to PaddleOCR on a PyMuPDF-rendered raster.
  4. Extract all embedded images via ImageExtractor.

The module returns a fully populated ParsedDocument model.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import fitz  # PyMuPDF

from models.document import (
    DocumentMetadata,
    ExtractedTable,
    ParsedDocument,
    Section,
)
from parser.image_extractor import ImageExtractor
from parser.ocr import ocr_engine
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Tuneable constants ────────────────────────────────────────────────────────

# Chars-per-page threshold below which a page is treated as scanned
SCANNED_TEXT_THRESHOLD: int = 50

# DPI used when rasterising pages for OCR
OCR_RENDER_DPI: int = 200


class PDFParser:
    """
    Parse a PDF document into a structured ParsedDocument.

    Combines Docling (structured extraction) with PyMuPDF (rendering/images)
    and PaddleOCR (scanned-page fallback) in a single cohesive pipeline.

    Usage::

        parser = PDFParser(pdf_path, output_image_dir, document_id)
        result = parser.parse()
    """

    def __init__(
        self,
        pdf_path: Path,
        output_image_dir: Path,
        document_id: str,
    ) -> None:
        """
        Args:
            pdf_path:         Path to the PDF file.
            output_image_dir: Directory to save extracted images.
            document_id:      Unique ID for this document.
        """
        self.pdf_path = pdf_path
        self.output_image_dir = output_image_dir
        self.document_id = document_id

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self) -> ParsedDocument:
        """
        Execute the full parsing pipeline.

        Returns:
            A populated ParsedDocument model.

        Raises:
            ValueError: For encrypted or password-protected PDFs.
            RuntimeError: For unrecoverable parse failures.
        """
        logger.info("[PDF] Starting parse for document_id=%s", self.document_id)

        self._validate_pdf()

        # ── Step 1: Docling structured extraction ─────────────────────────────
        docling_result = self._extract_with_docling()

        # ── Step 2: PyMuPDF page analysis + OCR fallback ──────────────────────
        sections, ocr_applied, parse_errors = self._process_pages_with_mupdf(
            docling_sections=docling_result.get("sections", []),
        )

        # ── Step 3: Image extraction ──────────────────────────────────────────
        images = self._extract_images()

        # ── Step 4: Assemble ParsedDocument ──────────────────────────────────
        document = ParsedDocument(
            document_id=self.document_id,
            filename=self.pdf_path.name,
            title=docling_result.get("title", ""),
            metadata=DocumentMetadata(
                authors=docling_result.get("authors", []),
                year=docling_result.get("year", ""),
                pages=docling_result.get("page_count", 0),
                keywords=docling_result.get("keywords", []),
            ),
            sections=sections,
            images=images,
            tables=docling_result.get("tables", []),
            ocr_applied=ocr_applied,
            parse_errors=parse_errors,
        )

        logger.info(
            "[PDF] Parse complete — sections=%d, images=%d, tables=%d, ocr=%s",
            len(document.sections),
            len(document.images),
            len(document.tables),
            document.ocr_applied,
        )
        return document

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_pdf(self) -> None:
        """
        Validate the PDF is readable and not encrypted.

        Raises:
            ValueError: If PDF is encrypted/password-protected.
            RuntimeError: If PDF is corrupted or unreadable.
        """
        try:
            doc = fitz.open(str(self.pdf_path))
        except fitz.FileDataError as exc:
            raise RuntimeError(f"Corrupted PDF file: {self.pdf_path.name}") from exc
        except Exception as exc:
            raise RuntimeError(f"Cannot open PDF: {exc}") from exc

        try:
            if doc.is_encrypted:
                raise ValueError(
                    f"PDF is encrypted/password-protected: {self.pdf_path.name}. "
                    "Please provide a decrypted copy."
                )
        finally:
            doc.close()

        logger.debug("PDF validation passed: %s", self.pdf_path.name)

    # ── Docling extraction ────────────────────────────────────────────────────

    def _extract_with_docling(self) -> dict:
        """
        Use Docling to extract structured content from the PDF.

        Docling understands document layout and can identify headings,
        paragraphs, and tables from a PDF's logical structure.

        Returns:
            Dictionary with keys: title, authors, year, keywords,
            page_count, sections, tables.
        """
        logger.info("[Docling] Starting extraction: %s", self.pdf_path.name)

        try:
            from docling.document_converter import DocumentConverter  # type: ignore

            converter = DocumentConverter()
            result = converter.convert(str(self.pdf_path))
            doc = result.document

            # ── Title ─────────────────────────────────────────────────────────
            title = self._extract_docling_title(doc)

            # ── Metadata ──────────────────────────────────────────────────────
            authors, year, keywords = self._extract_docling_metadata(doc)

            # ── Page count ────────────────────────────────────────────────────
            page_count = self._get_page_count_mupdf()

            # ── Sections ──────────────────────────────────────────────────────
            sections = self._extract_docling_sections(doc)

            # ── Tables ────────────────────────────────────────────────────────
            tables = self._extract_docling_tables(doc)

            logger.info(
                "[Docling] Extracted title='%s', %d sections, %d tables",
                title,
                len(sections),
                len(tables),
            )

            return {
                "title": title,
                "authors": authors,
                "year": year,
                "keywords": keywords,
                "page_count": page_count,
                "sections": sections,
                "tables": tables,
            }

        except ImportError:
            logger.warning("Docling not available; falling back to PyMuPDF-only extraction.")
            return self._fallback_mupdf_extraction()
        except Exception as exc:
            logger.warning("[Docling] Extraction failed (%s); using fallback.", exc)
            return self._fallback_mupdf_extraction()

    def _extract_docling_title(self, doc) -> str:
        """Extract document title from Docling document object."""
        try:
            # Docling exposes a list of body items with typed labels
            for item, _ in doc.iterate_items():
                label = str(getattr(item, "label", "")).lower()
                if "title" in label:
                    text = getattr(item, "text", "")
                    if text:
                        return text.strip()
        except Exception as exc:
            logger.debug("Title extraction failed: %s", exc)
        return ""

    def _extract_docling_metadata(self, doc) -> tuple[list[str], str, list[str]]:
        """
        Extract authors, year, and keywords from Docling metadata.

        Returns:
            Tuple of (authors list, year string, keywords list).
        """
        authors: list[str] = []
        year: str = ""
        keywords: list[str] = []

        try:
            meta = getattr(doc, "metadata", None)
            if meta:
                authors = getattr(meta, "authors", []) or []
                pub_info = getattr(meta, "publication", None)
                if pub_info:
                    year = str(getattr(pub_info, "year", "") or "")
                keywords = getattr(meta, "keywords", []) or []
        except Exception as exc:
            logger.debug("Metadata extraction error: %s", exc)

        return authors, year, keywords

    def _extract_docling_sections(self, doc) -> list[Section]:
        """
        Walk Docling document items and build Section objects.

        Docling items have a `label` attribute (e.g., 'section_header',
        'text', 'list_item') and a `text` attribute.

        Returns:
            List of Section objects.
        """
        sections: list[Section] = []
        current_heading = ""
        current_text_parts: list[str] = []
        current_page = 1
        current_level = 1

        def flush_section() -> None:
            """Commit accumulated text into a Section."""
            if current_heading or current_text_parts:
                sections.append(
                    Section(
                        page=current_page,
                        heading=current_heading,
                        text="\n".join(current_text_parts).strip(),
                        level=current_level,
                    )
                )

        try:
            for item, _ in doc.iterate_items():
                label = str(getattr(item, "label", "")).lower()
                text = str(getattr(item, "text", "") or "").strip()
                page = self._get_item_page(item)

                if not text:
                    continue

                if "section_header" in label or "heading" in label or "title" in label:
                    flush_section()
                    current_heading = text
                    current_text_parts = []
                    current_page = page
                    # Estimate heading level from font size or label suffix
                    current_level = self._infer_heading_level(item, label)

                elif label in ("text", "paragraph", "list_item", "body"):
                    current_text_parts.append(text)

        except Exception as exc:
            logger.warning("Section extraction partial failure: %s", exc)

        flush_section()
        return sections

    @staticmethod
    def _get_item_page(item) -> int:
        """Safely retrieve page number from a Docling item."""
        try:
            prov = getattr(item, "prov", None)
            if prov and len(prov) > 0:
                return int(prov[0].page_no)
        except Exception:
            pass
        return 1

    @staticmethod
    def _infer_heading_level(item, label: str) -> int:
        """Infer heading hierarchy level from label or font metadata."""
        if "h1" in label or "title" in label:
            return 1
        if "h2" in label:
            return 2
        if "h3" in label:
            return 3
        return 2

    def _extract_docling_tables(self, doc) -> list[ExtractedTable]:
        """
        Extract tables from Docling document.

        Returns:
            List of ExtractedTable objects.
        """
        tables: list[ExtractedTable] = []

        try:
            table_counter = 0
            for item, _ in doc.iterate_items():
                label = str(getattr(item, "label", "")).lower()
                if "table" not in label:
                    continue

                table_counter += 1
                page = self._get_item_page(item)

                # Try structured table data
                rows: list[list[str]] = []
                headers: list[str] = []

                table_data = getattr(item, "data", None)
                if table_data:
                    grid = getattr(table_data, "grid", None)
                    if grid:
                        for row_idx, row in enumerate(grid):
                            row_texts = [
                                str(getattr(cell, "text", "") or "") for cell in row
                            ]
                            if row_idx == 0:
                                headers = row_texts
                            else:
                                rows.append(row_texts)

                raw_text = str(getattr(item, "text", "") or "")

                tables.append(
                    ExtractedTable(
                        page=page,
                        table_number=table_counter,
                        headers=headers,
                        rows=rows,
                        raw_text=raw_text,
                    )
                )

        except Exception as exc:
            logger.warning("Table extraction error: %s", exc)

        return tables

    # ── PyMuPDF fallback extraction ───────────────────────────────────────────

    def _fallback_mupdf_extraction(self) -> dict:
        """
        Minimal extraction using only PyMuPDF when Docling is unavailable.

        Returns:
            Dictionary matching the format expected by _extract_with_docling().
        """
        logger.info("[PyMuPDF] Running fallback extraction.")

        sections: list[Section] = []
        page_count = 0

        try:
            doc = fitz.open(str(self.pdf_path))
            page_count = len(doc)

            for page_number in range(page_count):
                page = doc[page_number]
                text = page.get_text("text").strip()
                if text:
                    sections.append(
                        Section(
                            page=page_number + 1,
                            heading="",
                            text=text,
                            level=1,
                        )
                    )
            doc.close()
        except Exception as exc:
            logger.error("PyMuPDF fallback failed: %s", exc)

        return {
            "title": sections[0].text[:80] if sections else "",
            "authors": [],
            "year": "",
            "keywords": [],
            "page_count": page_count,
            "sections": sections,
            "tables": [],
        }

    # ── Page-level OCR integration ────────────────────────────────────────────

    def _process_pages_with_mupdf(
        self,
        docling_sections: list[Section],
    ) -> tuple[list[Section], bool, list[str]]:
        """
        Scan each PDF page for scanned content and apply OCR where needed.

        For pages where Docling extracted little or no text (scanned pages),
        we render the page as a raster image and run PaddleOCR.

        Args:
            docling_sections: Sections already extracted by Docling.

        Returns:
            Tuple of:
              - Final sections list (Docling + OCR sections merged)
              - Boolean: True if any OCR was applied
              - List of non-fatal error strings
        """
        logger.info("[OCR-Check] Analysing pages for scanned content.")

        ocr_applied = False
        parse_errors: list[str] = []
        ocr_sections: list[Section] = []

        # Build a fast lookup: page_number → Docling section text length
        docling_text_by_page: dict[int, int] = {}
        for sec in docling_sections:
            prev = docling_text_by_page.get(sec.page, 0)
            docling_text_by_page[sec.page] = prev + len(sec.text)

        try:
            doc = fitz.open(str(self.pdf_path))
        except Exception as exc:
            parse_errors.append(f"Cannot re-open PDF for OCR check: {exc}")
            return docling_sections, False, parse_errors

        try:
            for page_number in range(len(doc)):
                page_1based = page_number + 1
                native_text = doc[page_number].get_text("text").strip()
                docling_chars = docling_text_by_page.get(page_1based, 0)

                is_scanned = (
                    len(native_text) < SCANNED_TEXT_THRESHOLD
                    and docling_chars < SCANNED_TEXT_THRESHOLD
                )

                if is_scanned:
                    logger.info(
                        "[OCR] Page %d appears scanned (text=%d chars). Running OCR.",
                        page_1based,
                        len(native_text),
                    )
                    ocr_section = self._run_ocr_on_page(doc, page_number, page_1based)
                    if ocr_section:
                        ocr_sections.append(ocr_section)
                        ocr_applied = True
                    else:
                        parse_errors.append(
                            f"OCR produced no text for page {page_1based}."
                        )
        finally:
            doc.close()

        # Merge OCR sections into Docling sections
        # Strategy: replace any Docling section for an OCR'd page, or append
        ocr_page_numbers = {s.page for s in ocr_sections}
        final_sections = [
            s for s in docling_sections if s.page not in ocr_page_numbers
        ]
        final_sections.extend(ocr_sections)
        final_sections.sort(key=lambda s: s.page)

        return final_sections, ocr_applied, parse_errors

    def _run_ocr_on_page(
        self,
        doc: fitz.Document,
        page_index: int,
        page_1based: int,
    ) -> Section | None:
        """
        Render a PDF page and run OCR on the resulting image.

        Args:
            doc:        Open fitz.Document.
            page_index: 0-based page index.
            page_1based: 1-based page number for Section metadata.

        Returns:
            A Section containing OCR text, or None on failure.
        """
        try:
            page = doc[page_index]
            mat = fitz.Matrix(OCR_RENDER_DPI / 72, OCR_RENDER_DPI / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

            # Convert pixmap to numpy array for PaddleOCR
            import numpy as np
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )

            blocks = ocr_engine.run_on_numpy_array(img_array)
            if not blocks:
                return None

            text = "\n".join(b.text for b in blocks)
            return Section(
                page=page_1based,
                heading="",
                text=text,
                level=1,
            )

        except Exception as exc:
            logger.error("[OCR] Failed on page %d: %s", page_1based, exc)
            return None

    # ── Image extraction ──────────────────────────────────────────────────────

    def _extract_images(self):
        """
        Delegate image extraction to ImageExtractor.

        Returns:
            List of ExtractedImage models.
        """
        try:
            extractor = ImageExtractor(self.pdf_path, self.output_image_dir)
            return extractor.extract_all()
        except Exception as exc:
            logger.warning("[Images] Extraction failed: %s", exc)
            return []

    # ── Utility ───────────────────────────────────────────────────────────────

    def _get_page_count_mupdf(self) -> int:
        """Open the PDF with PyMuPDF to reliably count pages."""
        try:
            doc = fitz.open(str(self.pdf_path))
            count = len(doc)
            doc.close()
            return count
        except Exception:
            return 0