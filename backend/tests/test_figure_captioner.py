"""Tests for FigureCaptioner — VLM description generation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from multimodal.figure_captioner import FigureCaptioner, DESCRIPTION_PROMPTS
from schemas.multimodal_chunk import ImageClassification


@pytest.fixture(autouse=True)
def reset_captioner_singleton():
    """Reset singleton state between tests."""
    FigureCaptioner._instance = None
    FigureCaptioner._model = None
    FigureCaptioner._processor = None
    FigureCaptioner._model_loaded = False
    FigureCaptioner._load_failed = False
    yield
    FigureCaptioner._instance = None
    FigureCaptioner._model = None
    FigureCaptioner._processor = None
    FigureCaptioner._model_loaded = False
    FigureCaptioner._load_failed = False


def make_captioner_with_mock() -> tuple[FigureCaptioner, MagicMock]:
    """Return a captioner with a mocked model."""
    mock_model = MagicMock()
    mock_processor = MagicMock()

    mock_processor.apply_chat_template.return_value = "formatted_prompt"
    mock_processor.return_value = {"input_ids": MagicMock(shape=(1, 10))}
    mock_model.generate.return_value = MagicMock()
    mock_processor.batch_decode.return_value = ["This is a CNN architecture diagram."]

    captioner = FigureCaptioner()
    captioner._model = mock_model
    captioner._processor = mock_processor
    captioner._model_loaded = True
    captioner._device = "cpu"

    return captioner, mock_model


class TestFallbackDescription:
    def test_fallback_for_each_type(self):
        captioner = FigureCaptioner()
        captioner._load_failed = True

        for classification in ImageClassification:
            result = captioner.describe(
                PILImage.new("RGB", (100, 100)),
                classification,
            )
            assert isinstance(result, str)
            assert len(result) > 0

    def test_fallback_does_not_raise(self):
        captioner = FigureCaptioner()
        captioner._load_failed = True

        img = PILImage.new("RGB", (200, 200))
        result = captioner.describe(img, ImageClassification.FIGURE)
        assert "figure" in result.lower()


class TestPrompts:
    def test_all_types_have_prompts(self):
        for cls in ImageClassification:
            assert cls in DESCRIPTION_PROMPTS or cls == ImageClassification.UNKNOWN


class TestDescribeBatch:
    def test_batch_returns_correct_count(self):
        captioner = FigureCaptioner()
        captioner._load_failed = True

        images_and_types = [
            (PILImage.new("RGB", (100, 100)), ImageClassification.FIGURE),
            (PILImage.new("RGB", (100, 100)), ImageClassification.CHART),
            (PILImage.new("RGB", (100, 100)), ImageClassification.TABLE),
        ]
        results = captioner.describe_batch(images_and_types)
        assert len(results) == 3

    def test_batch_all_strings(self):
        captioner = FigureCaptioner()
        captioner._load_failed = True

        images_and_types = [
            (PILImage.new("RGB", (50, 50)), ImageClassification.EQUATION)
        ]
        results = captioner.describe_batch(images_and_types)
        assert all(isinstance(r, str) for r in results)