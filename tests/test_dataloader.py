"""app.dataloader 單元測試。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.dataloader import LoadedData, LoaderConfig, LocalFileLoader


class TestLocalFileLoader:
    def test_load_json(self, tmp_data_dir: Path):
        loader = LocalFileLoader(LoaderConfig(base_path=str(tmp_data_dir)))
        result = loader.load("sample.json")
        assert isinstance(result, LoadedData)
        assert result.content_type == "json"
        assert result.content["key"] == "value"

    def test_load_csv(self, tmp_data_dir: Path):
        loader = LocalFileLoader(LoaderConfig(base_path=str(tmp_data_dir)))
        result = loader.load("sample.csv")
        assert result.content_type == "csv"
        assert len(result.content) == 2
        assert result.content[0]["name"] == "Alice"

    def test_load_txt(self, tmp_data_dir: Path):
        loader = LocalFileLoader(LoaderConfig(base_path=str(tmp_data_dir)))
        result = loader.load("sample.txt")
        assert result.content_type == "text"
        assert "Hello World" in result.content

    def test_load_markdown(self, tmp_data_dir: Path):
        loader = LocalFileLoader(LoaderConfig(base_path=str(tmp_data_dir)))
        result = loader.load("sample.md")
        assert result.content_type == "markdown"
        assert "# Title" in result.content

    def test_load_not_found(self, tmp_data_dir: Path):
        loader = LocalFileLoader(LoaderConfig(base_path=str(tmp_data_dir)))
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent.json")

    def test_list_sources(self, tmp_data_dir: Path):
        loader = LocalFileLoader(LoaderConfig(base_path=str(tmp_data_dir)))
        sources = loader.list_sources()
        assert len(sources) >= 4
        assert "sample.json" in sources

    def test_load_many(self, tmp_data_dir: Path):
        loader = LocalFileLoader(LoaderConfig(base_path=str(tmp_data_dir)))
        results = loader.load_many(["sample.json", "sample.txt"])
        assert len(results) == 2
