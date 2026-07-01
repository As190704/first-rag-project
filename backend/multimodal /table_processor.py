"""
Table extraction pipeline.

Strategy:
  1. Attempt Camelot lattice detection (best for bordered tables).
  2. Try Camelot stream detection (borderless tables).
  3. Fall back to Docling table data already in Phase 1 JSON.
  4. Emergency fallback: OCR the table image region.

Every successfully extracted table is converted to a structured
TableData model and a natural language description for embedding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image as PILImage

from schemas.multimodal_chunk import TableData
from utils.logger import get_logger

logger = get_logger(__name__)


class TableProcessor:
    """
    Extracts structured table data from PDFs and table images.

    Can operate in two modes:
      1. PDF-level: extract tables from a page using Camelot
      2. Image-level: extract table from a pre-cropped image using OCR

    Usage::

        processor = TableProcessor()
        tables = processor.extract_from_pdf_page(pdf_path, page_number)
        # or
        table_data = processor.extract_from_image(image_path, pil_image)
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_from_pdf_page(
        self,
        pdf_path: Path,
        page_number: int,
    ) -> list[TableData]:
        """
        Extract all tables from a single PDF page.

        Tries Camelot lattice → Camelot stream → returns empty list.

        Args:
            pdf_path:    Path to the PDF file.
            page_number: 1-based page number.

        Returns:
            List of TableData objects found on the page.
        """
        # Camelot uses 1-based page strings
        page_str = str(page_number)

        # ── Try Camelot lattice (bordered tables) ─────────────────────────────
        tables = self._camelot_extract(pdf_path, page_str, flavor="lattice")

        # ── Try Camelot stream (borderless tables) ────────────────────────────
        if not tables:
            tables = self._camelot_extract(pdf_path, page_str, flavor="stream")

        if tables:
            logger.info(
                "[TableProcessor] Extracted %d table(s) from page %d via Camelot",
                len(tables),
                page_number,
            )
        else:
            logger.debug(
                "[TableProcessor] No tables found on page %d via Camelot",
                page_number,
            )

        return tables

    def extract_from_image(
        self,
        image_path: Path,
        pil_image: PILImage.Image,
        vlm_description: str = "",
    ) -> TableData:
        """
        Extract table data from a table image using OCR.

        This is used when an image has been classified as a TABLE
        but is not directly extractable from the PDF (e.g., embedded
        as a raster image in the PDF).

        Args:
            image_path:      Path to the table image.
            pil_image:       PIL image.
            vlm_description: Optional Qwen2-VL description for context.

        Returns:
            TableData with OCR-extracted content.
        """
        logger.info("[TableProcessor] Extracting table from image: %s", image_path.name)

        raw_text = self._ocr_image(image_path)
        table_data = self._parse_ocr_table(raw_text)
        table_data.extraction_method = "ocr"

        if vlm_description:
            table_data.description = vlm_description
        elif not table_data.description:
            table_data.description = self._build_description(table_data)

        return table_data

    def from_phase1_table(self, table_dict: dict) -> TableData:
        """
        Convert a Phase 1 parsed table dict into a TableData model.

        This is the primary integration point with Phase 1 output,
        using the tables already extracted by Docling.

        Args:
            table_dict: Table dict from Phase 1 JSON with keys:
                        headers, rows, raw_text.

        Returns:
            Populated TableData model.
        """
        columns = table_dict.get("headers", [])
        rows = [
            [str(cell) for cell in row]
            for row in table_dict.get("rows", [])
        ]
        raw_text = table_dict.get("raw_text", "")

        table_data = TableData(
            columns=columns,
            rows=rows,
            description="",
            extraction_method="docling",
        )
        table_data.description = self._build_description(table_data)
        return table_data

    # ── Camelot extraction ────────────────────────────────────────────────────

    def _camelot_extract(
        self,
        pdf_path: Path,
        page_str: str,
        flavor: str = "lattice",
    ) -> list[TableData]:
        """
        Use Camelot to extract tables from a PDF page.

        Args:
            pdf_path:  Path to the PDF.
            page_str:  Camelot page string (e.g., "3").
            flavor:    'lattice' (bordered) or 'stream' (borderless).

        Returns:
            List of TableData objects.
        """
        try:
            import camelot  # type: ignore

            tables = camelot.read_pdf(
                str(pdf_path),
                pages=page_str,
                flavor=flavor,
                suppress_stdout=True,
            )

            result = []
            for table in tables:
                df = table.df
                if df.empty:
                    continue

                # First row as header if it looks like a header
                headers = list(df.iloc[0].astype(str))
                rows = [
                    list(row.astype(str))
                    for _, row in df.iloc[1:].iterrows()
                ]

                table_data = TableData(
                    columns=headers,
                    rows=rows,
                    description="",
                    extraction_method=f"camelot_{flavor}",
                )
                table_data.description = self._build_description(table_data)
                result.append(table_data)

            return result

        except ImportError:
            logger.debug("[TableProcessor] Camelot not installed.")
            return []
        except Exception as exc:
            logger.debug("[TableProcessor] Camelot %s failed: %s", flavor, exc)
            return []

    # ── OCR fallback ──────────────────────────────────────────────────────────

    def _ocr_image(self, image_path: Path) -> str:
        """Extract text from a table image using PaddleOCR."""
        try:
            from parser.ocr import ocr_engine
            return ocr_engine.extract_text(image_path)
        except Exception as exc:
            logger.warning("[TableProcessor] OCR failed for %s: %s", image_path.name, exc)
            return ""

    def _parse_ocr_table(self, ocr_text: str) -> TableData:
        """
        Parse OCR text into a TableData structure.

        Heuristic: lines that look like rows (contain | or tabs or
        consistent spacing) are parsed as table rows.

        Args:
            ocr_text: Raw OCR output from the table image.

        Returns:
            Best-effort TableData.
        """
        lines = [ln.strip() for ln in ocr_text.split("\n") if ln.strip()]
        if not lines:
            return TableData()

        # Detect delimiter
        delimiter = "|" if any("|" in ln for ln in lines) else None

        if delimiter:
            parsed_rows = [
                [cell.strip() for cell in ln.split(delimiter) if cell.strip()]
                for ln in lines
            ]
            parsed_rows = [row for row in parsed_rows if row]
            columns = parsed_rows[0] if parsed_rows else []
            rows = parsed_rows[1:] if len(parsed_rows) > 1 else []
        else:
            # Treat each line as a row with space-separated cells
            columns = lines[0].split() if lines else []
            rows = [ln.split() for ln in lines[1:]]

        return TableData(columns=columns, rows=rows)

    # ── Description builder ───────────────────────────────────────────────────

    @staticmethod
    def _build_description(table_data: TableData) -> str:
        """
        Build a natural language description of a table.

        Args:
            table_data: Structured table data.

        Returns:
            Description string suitable for embedding.
        """
        parts = ["Table"]

        if table_data.columns:
            col_str = ", ".join(str(c) for c in table_data.columns[:6])
            if len(table_data.columns) > 6:
                col_str += f" and {len(table_data.columns) - 6} more columns"
            parts.append(f"with columns: {col_str}")

        if table_data.rows:
            parts.append(f"containing {len(table_data.rows)} rows of data")

        return " ".join(parts) + "."