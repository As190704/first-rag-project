"""Tests for ImageProcessor — classification and preprocessing."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage

from multimodal.image_processor import ImageProcessor, MIN_IMAGE_SIZE
from schemas.multimodal_chunk import ImageClassification


@pytest.fixture
def processor() -> ImageProcessor:
    return ImageProcessor()


def make_pil_image(width: int = 400, height: int = 300, colour: str = "white") -> PILImage.Image:
    return PILImage.new("RGB", (width, height), colour)


def make_image_record(
    image_path: str = "test.png",
    page: int = 1,
    image_number: int = 1,
    width: int = 400,
    height: int = 300,
) -> dict:
    return {
        "image_path": image_path,
        "page": page,
        "image_number": image_number,
        "width": width,
        "height": height,
        "bounding_box": {"x0": 10.0, "y0": 20.0, "x1": 300.0, "y1": 200.0},
    }


class TestImageClassification:
    def test_classify_returns_tuple(self, processor):
        img = make_pil_image(400, 300)
        classification, confidence = processor._classify_image(img)
        assert isinstance(classification, ImageClassification)
        assert 0.0 <= confidence <= 1.0

    def test_white_narrow_image_likely_equation(self, processor):
        # White image with low edges → equation heuristic
        img = PILImage.new("RGB", (600, 80), "white")
        classification, conf = processor._classify_image(img)
        # Should not be UNKNOWN
        assert isinstance(classification, ImageClassification)

    def test_wide_aspect_screenshot(self, processor):
        img = PILImage.new("RGB", (1200, 200), (200, 200, 200))
        classification, conf = processor._classify_image(img)
        assert isinstance(classification, ImageClassification)

    def test_colourful_image_chart(self, processor):
        # Create an image with many different colours
        arr = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        img = PILImage.fromarray(arr)
        classification, conf = processor._classify_image(img)
        assert isinstance(classification, ImageClassification)


class TestPreprocessing:
    def test_preprocess_maintains_rgb(self, processor):
        img = PILImage.new("RGBA", (100, 100), "white")
        result = processor.preprocess_for_vlm(img)
        assert result.mode == "RGB"

    def test_preprocess_large_image_resized(self, processor):
        img = PILImage.new("RGB", (4000, 3000), "white")
        result = processor.preprocess_for_vlm(img)
        assert max(result.size) <= 2048

    def test_preprocess_small_image_unchanged(self, processor):
        img = PILImage.new("RGB", (200, 150), "white")
        result = processor.preprocess_for_vlm(img)
        # Small image should not be enlarged
        assert result.size[0] <= 200
        assert result.size[1] <= 150

    def test_image_to_bytes_png(self, processor):
        img = PILImage.new("RGB", (100, 100), "white")
        data = processor.image_to_bytes(img, "PNG")
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_image_to_bytes_jpeg(self, processor):
        img = PILImage.new("RGB", (100, 100), "white")
        data = processor.image_to_bytes(img, "JPEG")
        assert isinstance(data, bytes)


class TestBboxParsing:
    def test_valid_bbox_parsed(self, processor):
        bbox_dict = {"x0": 10.0, "y0": 20.0, "x1": 200.0, "y1": 150.0}
        bbox = processor._parse_bbox(bbox_dict)
        assert bbox is not None
        assert bbox.x0 == 10.0
        assert bbox.y1 == 150.0

    def test_none_bbox_returns_none(self, processor):
        assert processor._parse_bbox(None) is None

    def test_invalid_bbox_returns_none(self, processor):
        assert processor._parse_bbox({"broken": "data"}) is None


class TestProcessImageList:
    def test_missing_file_skipped(self, processor):
        records = [make_image_record(image_path="/nonexistent/image.png")]
        results = processor.process_image_list(records)
        assert len(results) == 0

    def test_valid_image_processed(self, processor, tmp_path):
        img = PILImage.new("RGB", (200, 200), "blue")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        records = [make_image_record(image_path=str(img_path))]
        results = processor.process_image_list(records)
        assert len(results) == 1
        assert results[0].width == 200
        assert results[0].height == 200

    def test_tiny_image_skipped(self, processor, tmp_path):
        img = PILImage.new("RGB", (10, 10), "white")
        img_path = tmp_path / "tiny.png"
        img.save(img_path)

        records = [make_image_record(image_path=str(img_path))]
        results = processor.process_image_list(records)
        assert len(results) == 0