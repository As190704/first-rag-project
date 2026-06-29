"""
Export service — persists ParsedDocument as a formatted JSON file.

Writes to output/json/<document_id>.json and returns the file path.
The JSON format is the canonical interchange format for downstream phases.
"""

from __future__ import annotations

import json
from pathlib import Path

from models.document import ParsedDocument
from utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_JSON_DIR = Path("output/json")


class ExportService:
    """
    Handles serialisation and persistence of parsed documents.

    Usage::

        service = ExportService()
        path = service.save(parsed_document)
    """

    def __init__(self, output_dir: Path = OUTPUT_JSON_DIR) -> None:
        """
        Args:
            output_dir: Root directory for JSON output files.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def save(self, document: ParsedDocument) -> Path:
        """
        Serialise a ParsedDocument to a JSON file.

        Args:
            document: The fully parsed document model.

        Returns:
            Path to the written JSON file.

        Raises:
            IOError: If the file cannot be written.
        """
        output_path = self.output_dir / f"{document.document_id}.json"

        payload = self._build_payload(document)

        try:
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
            logger.info(
                "Document JSON saved: %s (%d bytes)",
                output_path,
                output_path.stat().st_size,
            )
        except IOError as exc:
            logger.error("Failed to write JSON for %s: %s", document.document_id, exc)
            raise

        return output_path

    def load(self, document_id: str) -> dict:
        """
        Load a previously saved document JSON by ID.

        Args:
            document_id: The document ID (filename without extension).

        Returns:
            Parsed JSON dictionary.

        Raises:
            FileNotFoundError: If no JSON file exists for the given ID.
        """
        path = self.output_dir / f"{document_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"No saved document found for ID: {document_id}")

        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_payload(document: ParsedDocument) -> dict:
        """
        Build the canonical JSON payload dictionary from a ParsedDocument.

        Follows the output schema defined in the project spec.

        Args:
            document: Fully parsed document.

        Returns:
            Dictionary ready for JSON serialisation.
        """
        return {
            "document_id": document.document_id,
            "filename": document.filename,
            "title": document.title,
            "metadata": {
                "authors": document.metadata.authors,
                "year": document.metadata.year,
                "pages": document.metadata.pages,
                "doi": document.metadata.doi,
                "keywords": document.metadata.keywords,
                **document.metadata.extra,
            },
            "sections": [
                {
                    "page": s.page,
                    "heading": s.heading,
                    "text": s.text,
                    "level": s.level,
                }
                for s in document.sections
            ],
            "images": [
                {
                    "page": img.page,
                    "image_number": img.image_number,
                    "image_path": img.image_path,
                    "width": img.width,
                    "height": img.height,
                    "bounding_box": (
                        {
                            "x0": img.bounding_box.x0,
                            "y0": img.bounding_box.y0,
                            "x1": img.bounding_box.x1,
                            "y1": img.bounding_box.y1,
                        }
                        if img.bounding_box
                        else None
                    ),
                }
                for img in document.images
            ],
            "tables": [
                {
                    "page": t.page,
                    "table_number": t.table_number,
                    "headers": t.headers,
                    "rows": t.rows,
                    "raw_text": t.raw_text,
                }
                for t in document.tables
            ],
            "ocr_applied": document.ocr_applied,
            "parse_errors": document.parse_errors,
        }