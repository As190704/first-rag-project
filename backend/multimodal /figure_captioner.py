"""
Figure captioning pipeline using Qwen2-VL-7B-Instruct.

Generates detailed, factual descriptions of visual elements.
The VLM is prompted to describe exactly what is visible —
not to summarise or interpret — so that descriptions are
maximally useful for retrieval.

Singleton pattern prevents loading the 7B model multiple times.

Phase 4 extension:
  - Pass surrounding text as context to generate citation-aware
    descriptions that mention the figure's referenced content.
  - Add structured output mode to extract specific fields.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image as PILImage

from schemas.multimodal_chunk import ImageClassification, VisualChunkType
from utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# ── Prompt templates per visual type ─────────────────────────────────────────

DESCRIPTION_PROMPTS: dict[str, str] = {
    ImageClassification.FIGURE: (
        "Describe this scientific figure in detail. "
        "State exactly what is shown: the type of visualisation, "
        "axes labels if present, data trends, colour coding, annotations, "
        "and any text visible in the image. Be specific and factual."
    ),
    ImageClassification.DIAGRAM: (
        "Describe this diagram precisely. "
        "Identify all components, their relationships, directional flows, "
        "labels, and any hierarchical structure. Do not interpret — describe."
    ),
    ImageClassification.FLOWCHART: (
        "Describe this flowchart. List every node, decision point, "
        "and edge in the flow. Include all visible labels and the "
        "overall process being represented."
    ),
    ImageClassification.ARCHITECTURE: (
        "Describe this system or neural network architecture diagram. "
        "Identify all layers, modules, connections, data flow directions, "
        "and any labels, dimensions, or annotations shown."
    ),
    ImageClassification.CHART: (
        "Describe this chart in detail. "
        "State the chart type, title, axis labels, units, legend entries, "
        "approximate values for key data points, and any visible trends."
    ),
    ImageClassification.TABLE: (
        "Describe this table. "
        "State the column headers, the number of rows, and summarise "
        "the key values or comparisons presented."
    ),
    ImageClassification.EQUATION: (
        "Describe this mathematical expression. "
        "State the symbols, operators, and what equation or formula "
        "is shown. If LaTeX notation is recognisable, state it."
    ),
    ImageClassification.SCREENSHOT: (
        "Describe this screenshot. "
        "State what application or interface is shown, visible text, "
        "UI elements, and any data or content displayed."
    ),
    ImageClassification.UNKNOWN: (
        "Describe this image in detail. "
        "State everything that is visible including objects, text, "
        "colours, shapes, and spatial relationships."
    ),
}

DEFAULT_PROMPT = DESCRIPTION_PROMPTS[ImageClassification.UNKNOWN]


class FigureCaptioner:
    """
    Generates detailed descriptions of visual elements using Qwen2-VL.

    The model is loaded lazily on first use to avoid blocking API startup.
    Gracefully degrades to a template-based fallback if the model
    cannot be loaded (e.g., insufficient VRAM).

    Usage::

        captioner = FigureCaptioner()
        description = captioner.describe(pil_image, classification)
    """

    _instance: "FigureCaptioner | None" = None
    _model = None
    _processor = None
    _device: str = "cpu"
    _model_loaded: bool = False
    _load_failed: bool = False

    def __new__(cls) -> "FigureCaptioner":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ── Public API ────────────────────────────────────────────────────────────

    def describe(
        self,
        image: PILImage.Image,
        classification: ImageClassification = ImageClassification.UNKNOWN,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Generate a detailed textual description of a visual element.

        Args:
            image:          PIL image of the visual element.
            classification: Pre-classified image type (sets the prompt).
            max_new_tokens: Maximum tokens in the generated description.

        Returns:
            Detailed description string. Falls back to a template
            description if the VLM cannot be loaded.
        """
        if not self._model_loaded and not self._load_failed:
            self._load_model()

        if self._load_failed or self._model is None:
            return self._fallback_description(classification)

        prompt = DESCRIPTION_PROMPTS.get(classification, DEFAULT_PROMPT)

        try:
            t_start = time.perf_counter()
            description = self._run_inference(image, prompt, max_new_tokens)
            elapsed = (time.perf_counter() - t_start) * 1000
            logger.debug(
                "[Captioner] Generated description in %.0fms (%d chars)",
                elapsed,
                len(description),
            )
            return description.strip()
        except Exception as exc:
            logger.warning("[Captioner] Inference failed: %s. Using fallback.", exc)
            return self._fallback_description(classification)

    def describe_batch(
        self,
        images_and_types: list[tuple[PILImage.Image, ImageClassification]],
        max_new_tokens: int = 512,
    ) -> list[str]:
        """
        Generate descriptions for a batch of images.

        Processes sequentially (Qwen2-VL-7B does not support native batching
        for variable-size images). Progress is logged per image.

        Args:
            images_and_types: List of (PIL image, classification) tuples.
            max_new_tokens:   Token budget per description.

        Returns:
            List of description strings in input order.
        """
        descriptions = []
        total = len(images_and_types)

        for idx, (image, classification) in enumerate(images_and_types, start=1):
            logger.info(
                "[Captioner] Describing image %d/%d (type=%s)",
                idx,
                total,
                classification.value,
            )
            desc = self.describe(image, classification, max_new_tokens)
            descriptions.append(desc)

        return descriptions

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Lazily load Qwen2-VL-7B-Instruct from HuggingFace.

        Sets _load_failed = True if loading fails, triggering
        fallback mode for all subsequent calls.
        """
        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                "[Captioner] Loading Qwen2-VL-7B on %s (this may take a while)...",
                self._device,
            )

            t_start = time.perf_counter()

            # Use bfloat16 on GPU for memory efficiency
            dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                torch_dtype=dtype,
                device_map="auto" if self._device == "cuda" else None,
            )
            self._processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct"
            )

            if self._device == "cpu":
                self._model = self._model.to("cpu")

            self._model_loaded = True
            logger.info(
                "[Captioner] Qwen2-VL loaded in %.1fs on %s.",
                time.perf_counter() - t_start,
                self._device,
            )

        except ImportError as exc:
            logger.error(
                "[Captioner] transformers not installed: %s. "
                "Install with: pip install transformers>=4.45.0",
                exc,
            )
            self._load_failed = True
        except Exception as exc:
            logger.error(
                "[Captioner] Failed to load Qwen2-VL: %s. "
                "Falling back to template descriptions.",
                exc,
            )
            self._load_failed = True

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run_inference(
        self,
        image: PILImage.Image,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        """
        Run Qwen2-VL inference on a single image + prompt.

        Args:
            image:          RGB PIL image.
            prompt:         Task-specific instruction prompt.
            max_new_tokens: Generation budget.

        Returns:
            Generated description string.
        """
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply Qwen2-VL chat template
        text_input = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            text=[text_input],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        if self._device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Decode only the generated tokens (exclude input tokens)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        description = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return description

    # ── Fallback ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_description(classification: ImageClassification) -> str:
        """
        Return a template description when VLM is unavailable.

        Args:
            classification: Visual element type.

        Returns:
            Simple placeholder description.
        """
        templates = {
            ImageClassification.FIGURE: "Scientific figure extracted from research document.",
            ImageClassification.DIAGRAM: "Technical diagram showing system components and relationships.",
            ImageClassification.FLOWCHART: "Flowchart depicting a process or algorithm.",
            ImageClassification.ARCHITECTURE: "Neural network or system architecture diagram.",
            ImageClassification.CHART: "Data chart or graph visualising experimental results.",
            ImageClassification.TABLE: "Data table with rows and columns.",
            ImageClassification.EQUATION: "Mathematical equation or formula.",
            ImageClassification.SCREENSHOT: "Screenshot of an interface or application.",
            ImageClassification.UNKNOWN: "Visual element extracted from research document.",
        }
        return templates.get(classification, templates[ImageClassification.UNKNOWN])


# ── Module-level singleton ────────────────────────────────────────────────────

figure_captioner = FigureCaptioner()