"""Tests for ChartProcessor — chart data extraction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from multimodal.chart_processor import ChartProcessor
from schemas.multimodal_chunk import ChartData


@pytest.fixture
def processor() -> ChartProcessor:
    return ChartProcessor()


@pytest.fixture
def chart_image() -> PILImage.Image:
    return PILImage.new("RGB", (640, 480), "white")


class TestChartTypeNormalisation:
    def test_known_types_normalised(self, processor):
        assert processor._normalise_chart_type("bar") == "bar"
        assert processor._normalise_chart_type("grouped_bar") == "bar"
        assert processor._normalise_chart_type("line") == "line"
        assert processor._normalise_chart_type("scatter") == "scatter"
        assert processor._normalise_chart_type("pie") == "pie"

    def test_unknown_type_returns_unknown(self, processor):
        assert processor._normalise_chart_type("xyz") == "unknown"


class TestDescriptionBuilder:
    def test_full_chart_description(self, processor):
        chart = ChartData(
            chart_type="bar",
            title="Model Comparison",
            x_axis="Model",
            y_axis="Accuracy",
            legend=["ResNet", "EfficientNet"],
            values=[],
        )
        desc = processor._build_description(chart)
        assert "bar" in desc
        assert "Model Comparison" in desc
        assert "Accuracy" in desc

    def test_minimal_chart_description(self, processor):
        chart = ChartData()
        desc = processor._build_description(chart)
        assert "chart" in desc.lower()

    def test_legend_included(self, processor):
        chart = ChartData(legend=["alpha", "beta", "gamma"])
        desc = processor._build_description(chart)
        assert "alpha" in desc


class TestOCRFallback:
    def test_fallback_returns_chart_data(self, processor, chart_image, tmp_path):
        img_path = tmp_path / "chart.png"
        chart_image.save(img_path)

        with patch.object(
            processor,
            "_ocr_fallback",
            return_value=ChartData(chart_type="line", description="A line chart."),
        ):
            result = processor.process(img_path, chart_image)

        assert isinstance(result, ChartData)

    def test_process_returns_chart_data(self, processor, chart_image, tmp_path):
        img_path = tmp_path / "chart.png"
        chart_image.save(img_path)

        # Mock ChartOCR to fail, triggering fallback
        with patch.object(processor, "_try_chartocr", return_value=None):
            result = processor.process(img_path, chart_image)

        assert isinstance(result, ChartData)

    def test_vlm_description_used_when_available(self, processor, chart_image, tmp_path):
        img_path = tmp_path / "chart.png"
        chart_image.save(img_path)

        with patch.object(processor, "_try_chartocr", return_value=None):
            result = processor.process(
                img_path, chart_image, vlm_description="A bar chart showing accuracy."
            )

        assert result.description == "A bar chart showing accuracy."


class TestAxesExtraction:
    def test_parses_title_from_text(self, processor):
        text = "Training Loss\nEpoch\nLoss"
        title, x, y = processor._parse_axes_from_text(text)
        assert title == "Training Loss"

    def test_detects_axis_keywords(self, processor):
        text = "Chart Title\nAccuracy vs Epoch\nLoss value"
        title, x, y = processor._parse_axes_from_text(text)
        assert "accuracy" in x.lower() or "epoch" in x.lower()

    def test_empty_text_handled(self, processor):
        title, x, y = processor._parse_axes_from_text("")
        assert title == ""
        assert x == ""
        assert y == ""