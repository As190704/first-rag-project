"""
OCR module using PaddleOCR.

Provides page-level and image-level OCR with structured text extraction.
Automatically handles initialization, error recovery, and confidence
filtering so callers receive clean text blocks.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

from utils.logger import get_logger

logger = get_logger(__name__)


class OCRBlock(NamedTuple):
    """A single recognized text block from OCR."""

    text: str
    confidence: float
    bbox: list[list[float]]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]


class OCREngine:
    """
    Singleton-style wrapper around PaddleOCR.

    Lazy-initializes the PaddleOCR model on first use to avoid loading
    the heavy model at import time.  All public methods are safe to call
    even if PaddleOCR is not installed — they will raise a clear error.
    """

    _instance: "OCREngine | None" = None
    _ocr = None  # PaddleOCR instance

    def __new__(cls) -> "OCREngine":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ── Initialisation ────────────────────────────────────────────────────────

    def _ensure_initialized(self) -> None:
        """Lazily initialize PaddleOCR on first call."""
        if self._ocr is not None:
            return

        try:
            from paddleocr import PaddleOCR  # type: ignore

            logger.info("Initializing PaddleOCR engine (first use)...")
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False,
                use_gpu=False,  # Set True if CUDA is available
            )
            logger.info("PaddleOCR engine initialized successfully.")
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. "
                "Install it with: pip install paddleocr paddlepaddle"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize PaddleOCR: {exc}") from exc

    # ── Public API ────────────────────────────────────────────────────────────

    def run_on_image_path(
        self,
        image_path: Path,
        confidence_threshold: float = 0.5,
    ) -> list[OCRBlock]:
        """
        Run OCR on an image file.

        Args:
            image_path: Path to a PNG/JPG image.
            confidence_threshold: Minimum confidence score to include a block.

        Returns:
            List of OCRBlock objects sorted top-to-bottom.

        Raises:
            FileNotFoundError: If image_path does not exist.
            RuntimeError: If OCR processing fails.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for OCR: {image_path}")

        self._ensure_initialized()
        logger.info("Running OCR on: %s", image_path.name)

        try:
            result = self._ocr.ocr(str(image_path), cls=True)
            return self._parse_result(result, confidence_threshold)
        except Exception as exc:
            logger.error("OCR failed on %s: %s", image_path.name, exc)
            raise RuntimeError(f"OCR processing failed for {image_path.name}: {exc}") from exc

    def run_on_numpy_array(
        self,
        image_array: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> list[OCRBlock]:
        """
        Run OCR directly on a NumPy image array (e.g., from PyMuPDF pixmap).

        Args:
            image_array: RGB numpy array of shape (H, W, 3).
            confidence_threshold: Minimum confidence score to include a block.

        Returns:
            List of OCRBlock objects sorted top-to-bottom.
        """
        self._ensure_initialized()
        logger.debug("Running OCR on in-memory image array shape=%s", image_array.shape)

        try:
            result = self._ocr.ocr(image_array, cls=True)
            return self._parse_result(result, confidence_threshold)
        except Exception as exc:
            logger.error("OCR failed on array: %s", exc)
            raise RuntimeError(f"OCR processing failed on image array: {exc}") from exc

    def extract_text(
        self,
        image_path: Path,
        confidence_threshold: float = 0.5,
    ) -> str:
        """
        Convenience method: return all OCR text as a single joined string.

        Args:
            image_path: Path to a PNG/JPG image.
            confidence_threshold: Minimum confidence to include a line.

        Returns:
            Plain text string with lines separated by newlines.
        """
        blocks = self.run_on_image_path(image_path, confidence_threshold)
        text = "\n".join(block.text for block in blocks)
        logger.debug("OCR extracted %d blocks, total chars=%d", len(blocks), len(text))
        return text

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_result(
        raw_result: list | None,
        confidence_threshold: float,
    ) -> list[OCRBlock]:
        """
        Parse PaddleOCR raw output into structured OCRBlock objects.

        PaddleOCR returns:  [ [ [bbox, (text, confidence)], ... ], ... ]
        where the outer list corresponds to pages (always 1 for single image).

        Args:
            raw_result: Raw PaddleOCR output.
            confidence_threshold: Blocks below this confidence are discarded.

        Returns:
            Sorted list of OCRBlock objects.
        """
        blocks: list[OCRBlock] = []

        if not raw_result:
            return blocks

        for page_result in raw_result:
            if not page_result:
                continue
            for line in page_result:
                if not line or len(line) < 2:
                    continue
                bbox, (text, confidence) = line[0], line[1]
                if confidence >= confidence_threshold and text.strip():
                    blocks.append(OCRBlock(text=text.strip(), confidence=confidence, bbox=bbox))

        # Sort blocks top-to-bottom by the y-coordinate of the first bbox point
        blocks.sort(key=lambda b: b.bbox[0][1])
        return blocks


# ── Module-level singleton ────────────────────────────────────────────────────

ocr_engine = OCREngine()