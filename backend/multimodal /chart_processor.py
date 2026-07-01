"""
Chart understanding pipeline.

Strategy:
  1. Attempt ChartOCR for structured data extraction (bar, line, pie, scatter).
  2. Fall back to pure OCR (PaddleOCR) if ChartOCR fails.
  3. Use Qwen2-VL description as the final fallback source.

All extracted chart data is stored as ChartData + a natural language
description that can be embedded for text-based retrieval.
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image as PILImage

from schemas.multimodal_chunk import ChartData
from utils.logger import get_logger

logger = get_logger(__name__)


class ChartProcessor:
    """
    Extracts structured data and descriptions from chart images.

    Usage::

        processor = ChartProcessor()
        chart_data = processor.process(image_path, pil_image)
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def process(
        self,
        image_path: Path,
        pil_image: PILImage.Image,
        vlm_description: str = "",
    ) -> ChartData:
        """
        Extract structured data from a chart image.

        Tries ChartOCR first, then OCR fallback, then VLM description parsing.

        Args:
            image_path:      Path to the chart image on disk.
            pil_image:       PIL image of the chart.
            vlm_description: Qwen2-VL description for description fallback.

        Returns:
            Populated ChartData model.
        """
        logger.info("[ChartProcessor] Processing chart: %s", image_path.name)

        # ── Try ChartOCR ──────────────────────────────────────────────────────
        chart_data = self._try_chartocr(image_path, pil_image)

        if chart_data:
            logger.info(
                "[ChartProcessor] ChartOCR success: type=%s values=%d",
                chart_data.chart_type,
                len(chart_data.values),
            )
        else:
            # ── Fallback: OCR-based extraction ────────────────────────────────
            logger.info("[ChartProcessor] ChartOCR failed, using OCR fallback.")
            chart_data = self._ocr_fallback(image_path, pil_image)

        # ── Enhance description from VLM if available ─────────────────────────
        if vlm_description and not chart_data.description:
            chart_data.description = vlm_description
        elif not chart_data.description:
            chart_data.description = self._build_description(chart_data)

        return chart_data

    # ── ChartOCR extraction ───────────────────────────────────────────────────

    def _try_chartocr(
        self,
        image_path: Path,
        pil_image: PILImage.Image,
    ) -> ChartData | None:
        """
        Attempt structured chart extraction via ChartOCR.

        ChartOCR (https://github.com/soap117/DeepRule) expects image paths
        and returns structured output. We wrap it defensively.

        Args:
            image_path: Path to chart image.
            pil_image:  PIL image (fallback).

        Returns:
            ChartData if successful, None on failure.
        """
        try:
            from chartocr.src.run import ChartOCRPipeline  # type: ignore

            pipeline = ChartOCRPipeline()
            raw = pipeline.run(str(image_path))

            if not raw:
                return None

            return ChartData(
                chart_type=self._normalise_chart_type(raw.get("type", "unknown")),
                title=raw.get("title", ""),
                x_axis=raw.get("x_axis", ""),
                y_axis=raw.get("y_axis", ""),
                legend=raw.get("legend", []),
                values=self._parse_chartocr_values(raw.get("data", [])),
                description="",
            )
        except ImportError:
            logger.debug("[ChartProcessor] ChartOCR not installed.")
            return None
        except Exception as exc:
            logger.warning("[ChartProcessor] ChartOCR failed: %s", exc)
            return None

    # ── OCR fallback ──────────────────────────────────────────────────────────

    def _ocr_fallback(
        self,
        image_path: Path,
        pil_image: PILImage.Image,
    ) -> ChartData:
        """
        Extract chart information using PaddleOCR text extraction.

        This is a best-effort fallback that extracts all visible text
        and attempts to infer axes, title, and chart type from it.

        Args:
            image_path: Path to chart image.
            pil_image:  PIL image.

        Returns:
            Partially populated ChartData.
        """
        ocr_text = ""

        try:
            from parser.ocr import ocr_engine
            ocr_text = ocr_engine.extract_text(image_path)
        except Exception as exc:
            logger.warning("[ChartProcessor] OCR fallback failed: %s", exc)

        chart_type = self._infer_chart_type_from_image(pil_image)
        title, x_axis, y_axis = self._parse_axes_from_text(ocr_text)

        return ChartData(
            chart_type=chart_type,
            title=title,
            x_axis=x_axis,
            y_axis=y_axis,
            legend=[],
            values=[],
            description="",
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_chart_type(raw_type: str) -> str:
        """Map ChartOCR type strings to normalised names."""
        mapping = {
            "bar": "bar",
            "grouped_bar": "bar",
            "stacked_bar": "bar",
            "line": "line",
            "scatter": "scatter",
            "pie": "pie",
            "area": "line",
            "histogram": "bar",
        }
        return mapping.get(raw_type.lower(), "unknown")

    @staticmethod
    def _parse_chartocr_values(raw_data: list) -> list[dict]:
        """
        Convert ChartOCR raw data into a list of {label: value} dicts.

        Args:
            raw_data: ChartOCR data output.

        Returns:
            List of value dicts.
        """
        values = []
        for item in raw_data:
            if isinstance(item, dict):
                values.append(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                values.append({"x": item[0], "y": item[1]})
        return values[:100]  # Cap to avoid oversized payloads

    @staticmethod
    def _infer_chart_type_from_image(image: PILImage.Image) -> str:
        """
        Use simple colour and shape heuristics to guess chart type.

        Args:
            image: PIL chart image.

        Returns:
            Chart type string: 'bar'|'pie'|'scatter'|'line'|'unknown'.
        """
        import numpy as np
        import cv2

        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Detect circles (pie chart indicator)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=min(gray.shape) // 2,
        )
        if circles is not None and len(circles[0]) >= 1:
            return "pie"

        # Detect horizontal lines (bar chart) vs continuous curves (line)
        edges = cv2.Canny(gray, 50, 150)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)

        h_count = np.count_nonzero(h_lines)
        v_count = np.count_nonzero(v_lines)

        if v_count > h_count * 2:
            return "bar"

        return "line"

    @staticmethod
    def _parse_axes_from_text(ocr_text: str) -> tuple[str, str, str]:
        """
        Attempt to extract title, x-axis, and y-axis from OCR text.

        Uses simple heuristics: first line is likely the title,
        short repeated terms may be axis labels.

        Args:
            ocr_text: Raw OCR text from the chart.

        Returns:
            Tuple of (title, x_axis, y_axis).
        """
        lines = [ln.strip() for ln in ocr_text.split("\n") if ln.strip()]
        title = lines[0] if lines else ""
        x_axis = ""
        y_axis = ""

        # Look for common axis label patterns
        axis_pattern = re.compile(
            r"(accuracy|loss|epoch|iteration|f1|precision|recall|"
            r"score|value|count|percentage|time|error)",
            re.I,
        )
        for line in lines[1:]:
            if axis_pattern.search(line) and not x_axis:
                x_axis = line
            elif axis_pattern.search(line) and not y_axis:
                y_axis = line

        return title, x_axis, y_axis

    @staticmethod
    def _build_description(chart_data: ChartData) -> str:
        """
        Build a natural language description from structured chart data.

        Args:
            chart_data: Partially or fully populated ChartData.

        Returns:
            Natural language description string.
        """
        parts = []

        if chart_data.chart_type and chart_data.chart_type != "unknown":
            parts.append(f"A {chart_data.chart_type} chart")
        else:
            parts.append("A chart")

        if chart_data.title:
            parts.append(f"titled '{chart_data.title}'")

        if chart_data.x_axis and chart_data.y_axis:
            parts.append(
                f"showing {chart_data.y_axis} versus {chart_data.x_axis}"
            )
        elif chart_data.x_axis:
            parts.append(f"with x-axis: {chart_data.x_axis}")

        if chart_data.legend:
            parts.append(f"with legend entries: {', '.join(chart_data.legend[:5])}")

        if chart_data.values:
            parts.append(f"containing {len(chart_data.values)} data points")

        return " ".join(parts) + "."