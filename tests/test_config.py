"""app.utils.config 單元測試。"""

from __future__ import annotations

import pytest

from app.utils.config import AppConfig, get_config, init_config, reset_config


class TestInitConfig:
    def test_returns_app_config(self):
        cfg = init_config()
        assert isinstance(cfg, AppConfig)

    def test_has_project_name(self):
        cfg = init_config()
        assert cfg.project_name is not None

    def test_dot_notation_access(self):
        cfg = init_config()
        assert cfg.api.port == 8000

    def test_dict_access(self):
        cfg = init_config()
        assert cfg["api"]["port"] == 8000

    def test_nested_sections(self):
        cfg = init_config()
        assert cfg.logging.level == "INFO"
        assert cfg.mlflow.enabled is True
        assert cfg.dataloader.base_path == "./data"

    def test_custom_config_path(self, tmp_path):
        config_file = tmp_path / "custom.yaml"
        config_file.write_text("foo: bar\nnested:\n  key: 42\n", encoding="utf-8")
        cfg = init_config(str(config_file))
        assert cfg.foo == "bar"
        assert cfg.nested.key == 42

    def test_env_var_resolution(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_TEST_VAR", "hello")
        config_file = tmp_path / "env.yaml"
        config_file.write_text("val: ${MY_TEST_VAR}\n", encoding="utf-8")
        cfg = init_config(str(config_file))
        assert cfg.val == "hello"

    def test_env_var_default(self, tmp_path):
        config_file = tmp_path / "env.yaml"
        config_file.write_text("val: ${NONEXISTENT_VAR_XYZ:fallback}\n", encoding="utf-8")
        cfg = init_config(str(config_file))
        assert cfg.val == "fallback"


class TestGetConfig:
    def test_raises_before_init(self):
        with pytest.raises(RuntimeError):
            get_config()

    def test_returns_after_init(self):
        init_config()
        cfg = get_config()
        assert isinstance(cfg, AppConfig)


class TestResetConfig:
    def test_reset_clears_state(self):
        init_config()
        reset_config()
        with pytest.raises(RuntimeError):
            get_config()
