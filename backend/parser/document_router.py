"""
Document router — detects file type and dispatches to the correct parser.

Acts as the single entry point for the parsing pipeline.  DOCX and PPTX
have placeholder implementations ready for Phase 2 extension.
"""

from __future__ import annotations

from pathlib import Path

from models.document import ParsedDocument, Section
from parser.pdf_parser import PDFParser
from utils.file_utils import detect_file_type
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentRouter:
    """
    Routes a document to its appropriate parser based on detected file type.

    Usage::

        router = DocumentRouter(
            file_path=path,
            document_id="abc123",
            output_image_dir=Path("output/images/abc123"),
        )
        result = router.route()
    """

    def __init__(
        self,
        file_path: Path,
        document_id: str,
        output_image_dir: Path,
    ) -> None:
        """
        Args:
            file_path:        Path to the uploaded file on disk.
            document_id:      Unique identifier for this document.
            output_image_dir: Directory where extracted images are stored.
        """
        self.file_path = file_path
        self.document_id = document_id
        self.output_image_dir = output_image_dir

    # ── Public API ────────────────────────────────────────────────────────────

    def route(self) -> ParsedDocument:
        """
        Detect file type and invoke the corresponding parser.

        Returns:
            A fully populated ParsedDocument.

        Raises:
            ValueError: For unsupported file types.
            RuntimeError: For parser-level failures.
        """
        file_type = detect_file_type(self.file_path)
        logger.info(
            "Routing document_id=%s | file=%s | type=%s",
            self.document_id,
            self.file_path.name,
            file_type,
        )

        dispatch = {
            "pdf": self._parse_pdf,
            "image": self._parse_image,
            "docx": self._parse_docx,
            "pptx": self._parse_pptx,
        }

        handler = dispatch.get(file_type)
        if handler is None:
            raise ValueError(f"No parser registered for file type: '{file_type}'")

        return handler()

    # ── Parsers ───────────────────────────────────────────────────────────────

    def _parse_pdf(self) -> ParsedDocument:
        """Invoke the full PDF parsing pipeline."""
        parser = PDFParser(
            pdf_path=self.file_path,
            output_image_dir=self.output_image_dir,
            document_id=self.document_id,
        )
        return parser.parse()

    def _parse_image(self) -> ParsedDocument:
        """
        Parse a standalone image file using PaddleOCR.

        Extracts text via OCR and returns a single-section document.
        The image itself is copied to the output directory.
        """
        logger.info("[Image] Parsing image file: %s", self.file_path.name)

        from parser.ocr import ocr_engine
        from models.document import DocumentMetadata, ExtractedImage
        import shutil

        parse_errors: list[str] = []
        sections: list[Section] = []
        ocr_applied = False

        # ── Run OCR ────────────────────────────────────────────────────────────
        try:
            text = ocr_engine.extract_text(self.file_path)
            if text.strip():
                sections.append(
                    Section(page=1, heading="", text=text, level=1)
                )
            ocr_applied = True
        except Exception as exc:
            logger.error("[Image] OCR failed: %s", exc)
            parse_errors.append(f"OCR failed: {exc}")

        # ── Copy image to output dir ───────────────────────────────────────────
        self.output_image_dir.mkdir(parents=True, exist_ok=True)
        dest = self.output_image_dir / self.file_path.name
        try:
            shutil.copy2(self.file_path, dest)
        except Exception as exc:
            logger.warning("Could not copy image to output: %s", exc)

        images = [
            ExtractedImage(
                page=1,
                image_number=1,
                image_path=str(dest),
            )
        ]

        return ParsedDocument(
            document_id=self.document_id,
            filename=self.file_path.name,
            title="",
            metadata=DocumentMetadata(pages=1),
            sections=sections,
            images=images,
            tables=[],
            ocr_applied=ocr_applied,
            parse_errors=parse_errors,
        )

    def _parse_docx(self) -> ParsedDocument:
        """
        DOCX parser — placeholder for Phase 2.

        Phase 2 implementation will use python-docx to extract:
        - Headings, paragraphs, and lists
        - Embedded images
        - Tables with structured cell data
        """
        logger.warning("[DOCX] Parser not yet implemented (Phase 2 placeholder).")
        return self._placeholder_document(
            note="DOCX parsing will be implemented in Phase 2 using python-docx."
        )

    def _parse_pptx(self) -> ParsedDocument:
        """
        PPTX parser — placeholder for Phase 2.

        Phase 2 implementation will use python-pptx to extract:
        - Slide titles and body text
        - Speaker notes
        - Embedded images and charts
        """
        logger.warning("[PPTX] Parser not yet implemented (Phase 2 placeholder).")
        return self._placeholder_document(
            note="PPTX parsing will be implemented in Phase 2 using python-pptx."
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _placeholder_document(self, note: str) -> ParsedDocument:
        """
        Return a ParsedDocument shell for unsupported-but-known file types.

        Args:
            note: Human-readable explanation stored in parse_errors.

        Returns:
            A minimally populated ParsedDocument.
        """
        from models.document import DocumentMetadata

        return ParsedDocument(
            document_id=self.document_id,
            filename=self.file_path.name,
            title="",
            metadata=DocumentMetadata(),
            sections=[],
            images=[],
            tables=[],
            ocr_applied=False,
            parse_errors=[note],
        )