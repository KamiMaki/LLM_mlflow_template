"""app.tracking 單元測試。"""

from __future__ import annotations

import os
from pathlib import Path

from app.tracking import PromptManager, init_mlflow, is_mlflow_available
from app.tracking.tracer import span, trace_llm_call, trace_node
from app.utils.config import init_config


class TestInitMlflow:
    def test_init_disabled(self, tmp_path: Path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("mlflow:\n  enabled: false\n", encoding="utf-8")
        cfg = init_config(str(cfg_file))
        init_mlflow(cfg)

    def test_is_mlflow_available_returns_bool(self):
        result = is_mlflow_available()
        assert isinstance(result, bool)


class TestTracerNoOp:
    """MLflow 未啟用時，tracer decorators 應為 no-op。"""

    def test_trace_llm_call_passthrough(self):
        @trace_llm_call
        def my_func(x):
            return x * 2

        assert my_func(5) == 10

    def test_trace_node_passthrough(self):
        @trace_node("test_node")
        def my_node(state):
            return {"result": "ok"}

        assert my_node({})["result"] == "ok"

    def test_span_context_manager(self):
        with span("test_span") as s:
            result = 1 + 1
        assert result == 2


class TestPromptManager:
    def test_register_and_load_local(self, tmp_path: Path):
        pm = PromptManager()
        pm._export_dir = tmp_path / "prompts"
        os.makedirs(pm._export_dir, exist_ok=True)
        pm.register("test_prompt", "Hello {{ name }}!")
        loaded = pm.load("test_prompt")
        assert "Hello" in loaded

    def test_render(self, tmp_path: Path):
        pm = PromptManager()
        pm._export_dir = tmp_path / "prompts"
        os.makedirs(pm._export_dir, exist_ok=True)
        pm.register("greet", "Hi {{ user }}, welcome!")
        rendered = pm.render("greet", user="Alice")
        assert "Alice" in rendered

    def test_load_not_found_raises(self):
        pm = PromptManager()
        import pytest
        with pytest.raises(FileNotFoundError):
            pm.load("nonexistent_prompt_xyz")
