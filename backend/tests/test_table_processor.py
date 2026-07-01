"""Tests for TableProcessor — table data extraction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from multimodal.table_processor import TableProcessor
from schemas.multimodal_chunk import TableData


@pytest.fixture
def processor() -> TableProcessor:
    return TableProcessor()


@pytest.fixture
def table_image() -> PILImage.Image:
    return PILImage.new("RGB", (400, 300), "white")


class TestFromPhase1Table:
    def test_basic_table_parsed(self, processor):
        table_dict = {
            "headers": ["Model", "Accuracy", "F1"],
            "rows": [
                ["ResNet", "92.3", "91.8"],
                ["EfficientNet", "94.1", "93.7"],
            ],
            "raw_text": "",
        }
        result = processor.from_phase1_table(table_dict)
        assert result.columns == ["Model", "Accuracy", "F1"]
        assert len(result.rows) == 2
        assert result.extraction_method == "docling"

    def test_empty_table_handled(self, processor):
        result = processor.from_phase1_table({})
        assert isinstance(result, TableData)
        assert result.columns == []

    def test_description_generated(self, processor):
        table_dict = {
            "headers": ["A", "B"],
            "rows": [["1", "2"]],
            "raw_text": "",
        }
        result = processor.from_phase1_table(table_dict)
        assert len(result.description) > 0
        assert "Table" in result.description


class TestDescriptionBuilder:
    def test_description_includes_columns(self, processor):
        table = TableData(
            columns=["Name", "Score", "Rank"],
            rows=[["Alice", "95", "1"]],
        )
        desc = processor._build_description(table)
        assert "Name" in desc
        assert "Score" in desc

    def test_description_includes_row_count(self, processor):
        table = TableData(
            columns=["Col"],
            rows=[["a"], ["b"], ["c"]],
        )
        desc = processor._build_description(table)
        assert "3" in desc

    def test_many_columns_truncated(self, processor):
        columns = [f"Col{i}" for i in range(10)]
        table = TableData(columns=columns, rows=[])
        desc = processor._build_description(table)
        assert "more columns" in desc


class TestOCRTableParsing:
    def test_pipe_delimited_parsed(self, processor):
        text = "Model | Accuracy | F1\nResNet | 92 | 91\nViT | 94 | 93"
        result = processor._parse_ocr_table(text)
        assert "Model" in result.columns or len(result.columns) > 0

    def test_empty_text_returns_empty_table(self, processor):
        result = processor._parse_ocr_table("")
        assert isinstance(result, TableData)

    def test_space_delimited_fallback(self, processor):
        text = "A B C\n1 2 3\n4 5 6"
        result = processor._parse_ocr_table(text)
        assert len(result.columns) > 0


class TestExtractFromImage:
    def test_returns_table_data(self, processor, table_image, tmp_path):
        img_path = tmp_path / "table.png"
        table_image.save(img_path)

        with patch.object(processor, "_ocr_image", return_value="A | B\n1 | 2"):
            result = processor.extract_from_image(img_path, table_image)

        assert isinstance(result, TableData)
        assert result.extraction_method == "ocr"

    def test_vlm_description_used(self, processor, table_image, tmp_path):
        img_path = tmp_path / "table.png"
        table_image.save(img_path)

        with patch.object(processor, "_ocr_image", return_value=""):
            result = processor.extract_from_image(
                img_path, table_image, vlm_description="A comparison table."
            )

        assert result.description == "A comparison table."