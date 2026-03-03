"""app.logger (MLflow) + app.prompts 單元測試。"""

from __future__ import annotations

import os
from pathlib import Path

from app.logger import init_mlflow, is_mlflow_available
from app.prompts import PromptManager
from app.utils.config import init_config


class TestInitMlflow:
    def test_init_disabled(self, tmp_path: Path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("mlflow:\n  enabled: false\n", encoding="utf-8")
        cfg = init_config(str(cfg_file))
        init_mlflow(cfg)

    def test_init_none(self):
        init_mlflow(None)
        assert is_mlflow_available() is False

    def test_is_mlflow_available_returns_bool(self):
        result = is_mlflow_available()
        assert isinstance(result, bool)


class TestPromptManager:
    def test_register_and_load_local(self, tmp_path: Path):
        pm = PromptManager()
        pm._export_dir = tmp_path / "prompts"
        os.makedirs(pm._export_dir, exist_ok=True)
        pm.register("test_prompt", "Hello {{ name }}!")
        loaded = pm.load("test_prompt")
        assert "Hello" in str(loaded)

    def test_load_and_format(self, tmp_path: Path):
        pm = PromptManager()
        pm._export_dir = tmp_path / "prompts"
        os.makedirs(pm._export_dir, exist_ok=True)
        pm.register("greet", "Hi {{ user }}, welcome!")
        rendered = pm.load_and_format("greet", user="Alice")
        assert "Alice" in rendered

    def test_load_not_found_raises(self):
        pm = PromptManager()
        import pytest
        with pytest.raises(FileNotFoundError):
            pm.load("nonexistent_prompt_xyz")
