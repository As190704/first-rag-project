"""
Image preprocessing and classification pipeline.

Responsibilities:
  1. Load and validate every image extracted by Phase 1.
  2. Classify each image into a semantic category (figure, chart,
     flowchart, architecture, table, equation, screenshot, unknown).
  3. Preprocess images into a consistent format for downstream VLM
     and embedding models.

Classification strategy:
  - Uses lightweight OpenCV heuristics (edge density, colour histogram,
    aspect ratio, text-to-image ratio) to classify without loading a
    heavy model for this step.
  - This keeps startup fast and avoids GPU contention with Qwen2-VL.
  - Phase 4 can replace/augment this with a fine-tuned CLIP classifier.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image as PILImage

from schemas.multimodal_chunk import BoundingBox, ImageClassification
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_SHORT_SIDE: int = 1024   # Resize long images for VLM input
MAX_IMAGE_SIZE: int = 2048       # Hard cap for any dimension
MIN_IMAGE_SIZE: int = 32         # Below this we skip the image


class ProcessedImage(NamedTuple):
    """Container for a loaded and classified image."""

    path: Path
    pil_image: PILImage.Image
    classification: ImageClassification
    width: int
    height: int
    confidence: float
    page: int
    image_number: int
    bounding_box: BoundingBox | None


class ImageProcessor:
    """
    Loads, validates, classifies, and preprocesses document images.

    Usage::

        processor = ImageProcessor()
        results = processor.process_image_list(image_metadata_list)
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def process_image_list(
        self,
        image_records: list[dict],
    ) -> list[ProcessedImage]:
        """
        Process a list of image metadata records from Phase 1.

        Args:
            image_records: List of image dicts from parsed document JSON.
                           Each dict has keys: page, image_number, image_path,
                           width, height, bounding_box.

        Returns:
            List of ProcessedImage objects ready for captioning/embedding.
        """
        results: list[ProcessedImage] = []

        for record in image_records:
            try:
                processed = self._process_single(record)
                if processed:
                    results.append(processed)
            except Exception as exc:
                logger.warning(
                    "[ImageProcessor] Skipping image on page %d: %s",
                    record.get("page", 0),
                    exc,
                )

        logger.info(
            "[ImageProcessor] Processed %d/%d images successfully.",
            len(results),
            len(image_records),
        )
        return results

    def preprocess_for_vlm(self, image: PILImage.Image) -> PILImage.Image:
        """
        Resize and normalise an image for VLM input.

        Maintains aspect ratio while ensuring the shorter side is at
        most TARGET_SHORT_SIDE pixels, which balances quality with
        Qwen2-VL context window constraints.

        Args:
            image: Source PIL image.

        Returns:
            Resized RGB PIL image.
        """
        image = image.convert("RGB")
        w, h = image.size

        if max(w, h) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(w, h)
            image = image.resize(
                (int(w * scale), int(h * scale)),
                PILImage.LANCZOS,
            )
            w, h = image.size

        if min(w, h) > TARGET_SHORT_SIDE:
            scale = TARGET_SHORT_SIDE / min(w, h)
            image = image.resize(
                (int(w * scale), int(h * scale)),
                PILImage.LANCZOS,
            )

        return image

    def image_to_bytes(self, image: PILImage.Image, fmt: str = "PNG") -> bytes:
        """
        Convert a PIL image to raw bytes.

        Args:
            image: PIL image to convert.
            fmt:   Output format ('PNG' or 'JPEG').

        Returns:
            Raw image bytes.
        """
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        return buf.getvalue()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _process_single(self, record: dict) -> ProcessedImage | None:
        """
        Process a single image metadata record.

        Args:
            record: Image metadata dict from Phase 1 JSON.

        Returns:
            ProcessedImage or None if image is invalid/too small.
        """
        image_path = Path(record.get("image_path", ""))

        if not image_path.exists():
            logger.warning("[ImageProcessor] Image not found: %s", image_path)
            return None

        try:
            pil_img = PILImage.open(image_path).convert("RGB")
        except Exception as exc:
            logger.warning("[ImageProcessor] Cannot open %s: %s", image_path.name, exc)
            return None

        w, h = pil_img.size

        if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
            logger.debug(
                "[ImageProcessor] Image too small (%dx%d), skipping: %s",
                w, h, image_path.name,
            )
            return None

        classification, confidence = self._classify_image(pil_img)

        bbox = self._parse_bbox(record.get("bounding_box"))

        logger.debug(
            "[ImageProcessor] %s → %s (conf=%.2f) [%dx%d]",
            image_path.name,
            classification.value,
            confidence,
            w,
            h,
        )

        return ProcessedImage(
            path=image_path,
            pil_image=pil_img,
            classification=classification,
            width=w,
            height=h,
            confidence=confidence,
            page=record.get("page", 1),
            image_number=record.get("image_number", 0),
            bounding_box=bbox,
        )

    def _classify_image(
        self,
        image: PILImage.Image,
    ) -> tuple[ImageClassification, float]:
        """
        Classify an image using lightweight OpenCV heuristics.

        Heuristic pipeline:
          1. Equation: very high edge density + mostly white background
          2. Chart: structured colours + rectangular regions
          3. Flowchart: many rectangular bounding boxes with arrows
          4. Architecture: dense network of nodes and edges
          5. Table: high horizontal edge density + grid structure
          6. Figure: default for complex images with diverse content
          7. Screenshot: very wide aspect ratio + monospace font patterns

        Args:
            image: RGB PIL image.

        Returns:
            Tuple of (ImageClassification, confidence_score 0–1).
        """
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # ── Feature extraction ────────────────────────────────────────────────

        # Edge density (Canny)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / (w * h)

        # White pixel ratio (for equation / whitespace detection)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        white_ratio = np.count_nonzero(thresh) / (w * h)

        # Aspect ratio
        aspect = w / max(h, 1)

        # Colour diversity (std of hue channel)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        hue_std = float(np.std(hsv[:, :, 0]))

        # Contour count
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour_count = len(contours)

        # Horizontal line density (table heuristic)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        h_line_density = np.count_nonzero(horizontal_lines) / (w * h)

        # ── Classification rules ──────────────────────────────────────────────

        # Equation: high white background + moderate edges + low colour
        if white_ratio > 0.75 and edge_density < 0.12 and hue_std < 15:
            return ImageClassification.EQUATION, 0.72

        # Table: strong horizontal line structure
        if h_line_density > 0.015 and contour_count > 10:
            return ImageClassification.TABLE, 0.68

        # Screenshot: very wide or very tall + low hue diversity
        if (aspect > 2.0 or aspect < 0.4) and hue_std < 20:
            return ImageClassification.SCREENSHOT, 0.60

        # Chart: moderate colour diversity + structured contours
        if hue_std > 30 and contour_count < 50 and edge_density < 0.15:
            return ImageClassification.CHART, 0.65

        # Flowchart: many medium-size rectangular contours
        rect_count = self._count_rect_contours(contours, min_area=500)
        if rect_count > 5 and edge_density > 0.05:
            return ImageClassification.FLOWCHART, 0.62

        # Architecture diagram: dense + many edges + moderate colour
        if edge_density > 0.10 and contour_count > 30 and hue_std < 40:
            return ImageClassification.ARCHITECTURE, 0.58

        # Diagram: catch-all for structured but not chart/flowchart
        if contour_count > 15 and edge_density > 0.05:
            return ImageClassification.DIAGRAM, 0.55

        return ImageClassification.FIGURE, 0.50

    @staticmethod
    def _count_rect_contours(
        contours: list,
        min_area: int = 500,
    ) -> int:
        """
        Count contours that approximate rectangles.

        Args:
            contours: OpenCV contour list.
            min_area: Minimum contour area to consider.

        Returns:
            Count of rectangular contours.
        """
        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
            if len(approx) == 4:
                count += 1
        return count

    @staticmethod
    def _parse_bbox(bbox_dict: dict | None) -> BoundingBox | None:
        """
        Parse a bounding box dict from Phase 1 into a BoundingBox model.

        Args:
            bbox_dict: Dict with keys x0, y0, x1, y1 or None.

        Returns:
            BoundingBox or None.
        """
        if not bbox_dict:
            return None
        try:
            return BoundingBox(
                x0=float(bbox_dict.get("x0", 0)),
                y0=float(bbox_dict.get("y0", 0)),
                x1=float(bbox_dict.get("x1", 0)),
                y1=float(bbox_dict.get("y1", 0)),
            )
        except (TypeError, ValueError):
            return None