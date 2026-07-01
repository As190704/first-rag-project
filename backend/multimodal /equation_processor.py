"""
Equation detection and metadata extraction.

Detects mathematical equations in document images using a combination
of image heuristics and OCR. Does NOT solve equations.

Detection approach:
  1. Images pre-classified as EQUATION by ImageProcessor.
  2. OCR extracts visible mathematical symbols.
  3. LaTeX pattern matching attempts to identify equation structure.
  4. All detected equations are stored with page, bbox, and OCR text.

Phase 4 extension:
  - Integrate LaTeX-OCR (pix2tex) for high-quality LaTeX transcription.
  - Index equations in a dedicated 'equation' vector namespace.
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image as PILImage

from schemas.multimodal_chunk import EquationData
from utils.logger import get_logger

logger = get_logger(__name__)

# ── LaTeX symbol patterns ─────────────────────────────────────────────────────

MATH_SYMBOL_PATTERN = re.compile(
    r"[\∑∏∫∂∇±×÷≤≥≠∞√α-ωΑ-Ω]|"
    r"\b(lim|sum|prod|int|frac|sqrt|log|exp|sin|cos|tan|max|min|arg)\b",
    re.UNICODE,
)

EQUATION_KEYWORDS = re.compile(
    r"(cross.?entropy|loss|accuracy|gradient|derivative|"
    r"probability|expectation|variance|covariance|KL.?div)",
    re.I,
)


class EquationProcessor:
    """
    Detects and extracts metadata from mathematical equations.

    Processes images that have been classified as EQUATION by the
    ImageProcessor. Extracts OCR text, infers equation type, and
    builds a searchable description.

    Usage::

        processor = EquationProcessor()
        data = processor.process(image_path, pil_image)
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def process(
        self,
        image_path: Path,
        pil_image: PILImage.Image,
        vlm_description: str = "",
    ) -> EquationData:
        """
        Extract equation metadata from an equation image.

        Args:
            image_path:      Path to the equation image.
            pil_image:       PIL image.
            vlm_description: Optional Qwen2-VL description.

        Returns:
            EquationData with OCR text, inferred type, and description.
        """
        logger.info("[EquationProcessor] Processing equation: %s", image_path.name)

        # ── OCR extraction ────────────────────────────────────────────────────
        raw_ocr = self._run_ocr(image_path)

        # ── LaTeX-OCR attempt (optional) ──────────────────────────────────────
        latex = self._try_latex_ocr(image_path) or self._infer_latex_from_ocr(raw_ocr)

        # ── Equation type ─────────────────────────────────────────────────────
        eq_type = self._classify_equation_type(pil_image, raw_ocr)

        # ── Description ───────────────────────────────────────────────────────
        description = (
            vlm_description
            or self._build_description(raw_ocr, latex, eq_type)
        )

        eq_data = EquationData(
            raw_ocr_text=raw_ocr,
            latex=latex,
            equation_type=eq_type,
            description=description,
        )

        logger.debug(
            "[EquationProcessor] Equation type=%s latex=%s",
            eq_type,
            bool(latex),
        )
        return eq_data

    def is_equation_image(self, image_path: Path) -> bool:
        """
        Quick heuristic check whether an image is likely an equation.

        Used as a secondary filter after ImageProcessor classification.

        Args:
            image_path: Path to image to check.

        Returns:
            True if the image is likely to contain an equation.
        """
        try:
            ocr_text = self._run_ocr(image_path)
            return bool(MATH_SYMBOL_PATTERN.search(ocr_text))
        except Exception:
            return False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_ocr(self, image_path: Path) -> str:
        """Extract text from equation image via PaddleOCR."""
        try:
            from parser.ocr import ocr_engine
            return ocr_engine.extract_text(image_path, confidence_threshold=0.3)
        except Exception as exc:
            logger.debug("[EquationProcessor] OCR failed: %s", exc)
            return ""

    @staticmethod
    def _try_latex_ocr(image_path: Path) -> str:
        """
        Attempt LaTeX transcription using pix2tex if available.

        pix2tex (https://github.com/lukas-blecher/LaTeX-OCR) specialises
        in converting equation images to LaTeX.

        Args:
            image_path: Path to equation image.

        Returns:
            LaTeX string or empty string if unavailable.
        """
        try:
            from pix2tex.cli import LatexOCR  # type: ignore
            from PIL import Image

            model = LatexOCR()
            img = Image.open(image_path)
            return model(img)
        except ImportError:
            return ""
        except Exception as exc:
            logger.debug("[EquationProcessor] LaTeX-OCR failed: %s", exc)
            return ""

    @staticmethod
    def _infer_latex_from_ocr(ocr_text: str) -> str:
        """
        Attempt to clean OCR text into a LaTeX-like representation.

        This is a best-effort transformation for simple equations.

        Args:
            ocr_text: Raw OCR output.

        Returns:
            Cleaned equation string (not full LaTeX but readable).
        """
        if not ocr_text.strip():
            return ""

        # Basic symbol normalisation
        cleaned = ocr_text
        substitutions = [
            (r"\bsigma\b", "σ"),
            (r"\balpha\b", "α"),
            (r"\bbeta\b", "β"),
            (r"\blambda\b", "λ"),
            (r"\bmu\b", "μ"),
            (r"(\w+)\^(\w+)", r"\1^\2"),   # Exponents
            (r"(\w+)_(\w+)", r"\1_\2"),   # Subscripts
        ]
        for pattern, replacement in substitutions:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.I)

        return cleaned.strip()

    @staticmethod
    def _classify_equation_type(
        image: PILImage.Image,
        ocr_text: str,
    ) -> str:
        """
        Classify whether an equation is inline, display, or numbered.

        Heuristic:
          - Numbered: OCR contains a number in parentheses like (1) or (2.3)
          - Display: image is wide relative to height (typically centered)
          - Inline: narrow height, appears mid-text

        Args:
            image:    PIL equation image.
            ocr_text: OCR text from the equation.

        Returns:
            'inline'|'display'|'numbered'
        """
        # Check for equation number pattern like (1) or [eq. 3]
        numbered = re.search(r"\(\s*\d+\s*\)|\[eq\.?\s*\d+\s*\]", ocr_text, re.I)
        if numbered:
            return "numbered"

        w, h = image.size
        aspect = w / max(h, 1)

        if aspect > 3.0:
            return "display"

        return "inline"

    @staticmethod
    def _build_description(
        ocr_text: str,
        latex: str,
        eq_type: str,
    ) -> str:
        """
        Build a natural language description of an equation.

        Args:
            ocr_text: Raw OCR output.
            latex:    LaTeX or cleaned equation string.
            eq_type:  'inline'|'display'|'numbered'.

        Returns:
            Description string for embedding.
        """
        parts = [f"Mathematical {eq_type} equation."]

        display = latex or ocr_text
        if display:
            parts.append(f"Expression: {display[:200]}")

        # Check for known equation types
        if EQUATION_KEYWORDS.search(ocr_text):
            match = EQUATION_KEYWORDS.search(ocr_text)
            if match:
                parts.append(f"Related to: {match.group(0)}.")

        return " ".join(parts)