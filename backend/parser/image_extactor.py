"""
Image extraction module for PDF documents.

Uses PyMuPDF (fitz) to locate and export every embedded image from a PDF,
saving them under output/images/<document_id>/.  Metadata including page
number, image index, and bounding box is returned for each image.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image as PILImage

from models.document import BoundingBox, ExtractedImage
from utils.logger import get_logger

logger = get_logger(__name__)


class ImageExtractor:
    """
    Extract and persist all images embedded in a PDF document.

    Usage::

        extractor = ImageExtractor(pdf_path, output_dir)
        images = extractor.extract_all()
    """

    # Minimum pixel dimensions — ignore tiny icons / decorative elements
    MIN_WIDTH: int = 50
    MIN_HEIGHT: int = 50

    def __init__(self, pdf_path: Path, output_dir: Path) -> None:
        """
        Args:
            pdf_path:   Path to the source PDF file.
            output_dir: Directory where extracted images will be saved.
                        Created automatically if it does not exist.
        """
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_all(self) -> list[ExtractedImage]:
        """
        Extract every image from the PDF and save to output_dir.

        Returns:
            List of ExtractedImage models, one per extracted image.

        Raises:
            RuntimeError: If the PDF cannot be opened.
        """
        logger.info("Starting image extraction from: %s", self.pdf_path.name)

        try:
            doc = fitz.open(str(self.pdf_path))
        except Exception as exc:
            raise RuntimeError(f"Cannot open PDF for image extraction: {exc}") from exc

        extracted: list[ExtractedImage] = []
        global_img_counter = 0

        try:
            for page_number in range(len(doc)):
                page = doc[page_number]
                page_images = self._extract_page_images(
                    doc=doc,
                    page=page,
                    page_number=page_number + 1,  # 1-based
                    global_counter=global_img_counter,
                )
                global_img_counter += len(page_images)
                extracted.extend(page_images)
        finally:
            doc.close()

        logger.info(
            "Image extraction complete: %d images saved to %s",
            len(extracted),
            self.output_dir,
        )
        return extracted

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_page_images(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        page_number: int,
        global_counter: int,
    ) -> list[ExtractedImage]:
        """
        Extract all images from a single PDF page.

        Args:
            doc:            The open fitz.Document.
            page:           The current fitz.Page.
            page_number:    1-based page index.
            global_counter: Running total of images already extracted.

        Returns:
            List of ExtractedImage models for this page.
        """
        results: list[ExtractedImage] = []
        image_list = page.get_images(full=True)

        if not image_list:
            logger.debug("Page %d: no embedded images found.", page_number)
            return results

        logger.debug("Page %d: found %d image(s).", page_number, len(image_list))

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                extracted = self._extract_single_image(
                    doc=doc,
                    page=page,
                    xref=xref,
                    page_number=page_number,
                    img_index=img_index + 1,  # 1-based per page
                    global_index=global_counter + img_index + 1,
                )
                if extracted:
                    results.append(extracted)
            except Exception as exc:
                logger.warning(
                    "Skipping image xref=%d on page %d: %s",
                    xref,
                    page_number,
                    exc,
                )

        return results

    def _extract_single_image(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        xref: int,
        page_number: int,
        img_index: int,
        global_index: int,
    ) -> ExtractedImage | None:
        """
        Extract, validate, and save a single image by its XREF.

        Args:
            doc:          The open fitz.Document.
            page:         The page the image appears on.
            xref:         Image XREF within the PDF.
            page_number:  1-based page number.
            img_index:    Image index within the page (1-based).
            global_index: Global sequential image index (1-based).

        Returns:
            An ExtractedImage model, or None if image is below size threshold.
        """
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image.get("ext", "png").lower()

        # Normalise extension
        if image_ext == "jpeg":
            image_ext = "jpg"

        # Verify and resize-check via Pillow
        try:
            pil_img = PILImage.open(__import__("io").BytesIO(image_bytes))
            width, height = pil_img.size
        except Exception as exc:
            raise ValueError(f"Cannot decode image bytes: {exc}") from exc

        if width < self.MIN_WIDTH or height < self.MIN_HEIGHT:
            logger.debug(
                "Skipping tiny image %dx%d on page %d.", width, height, page_number
            )
            return None

        # Build output filename
        filename = f"img_{global_index:03d}_p{page_number}.{image_ext}"
        save_path = self.output_dir / filename

        # Save image
        pil_img.save(str(save_path))
        logger.debug("Saved image: %s (%dx%d)", filename, width, height)

        # Attempt bounding box retrieval
        bbox = self._get_image_bbox(page, xref)

        return ExtractedImage(
            page=page_number,
            image_number=img_index,
            image_path=str(save_path),
            bounding_box=bbox,
            width=width,
            height=height,
        )

    @staticmethod
    def _get_image_bbox(page: fitz.Page, xref: int) -> BoundingBox | None:
        """
        Retrieve the bounding box of an image on a page using its XREF.

        Args:
            page: The fitz.Page the image belongs to.
            xref: The image XREF to look up.

        Returns:
            A BoundingBox if found, otherwise None.
        """
        try:
            for img_item in page.get_image_info(xrefs=True):
                if img_item.get("xref") == xref:
                    rect = img_item.get("bbox")
                    if rect and len(rect) == 4:
                        return BoundingBox(
                            x0=rect[0], y0=rect[1],
                            x1=rect[2], y1=rect[3],
                        )
        except Exception as exc:
            logger.debug("Could not retrieve bbox for xref=%d: %s", xref, exc)
        return None