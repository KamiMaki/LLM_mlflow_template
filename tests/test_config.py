"""app.utils.config 單元測試。"""

from __future__ import annotations

import pytest
from omegaconf import DictConfig

from app.utils.config import get_config, get_llm_config, init_config, reset_config


class TestInitConfig:
    def test_returns_dictconfig(self):
        cfg = init_config()
        assert isinstance(cfg, DictConfig)

    def test_has_project_name(self):
        cfg = init_config()
        assert cfg.project_name is not None

    def test_env_override(self):
        cfg = init_config(overrides=["env=prod"])
        assert cfg.logging.level == "INFO"

    def test_test_env_llm_override(self):
        cfg = init_config(overrides=["env=test"])
        assert cfg.llm.temperature == 0.0
        assert cfg.llm.max_retries == 2


class TestGetConfig:
    def test_raises_before_init(self):
        with pytest.raises(RuntimeError):
            get_config()

    def test_returns_after_init(self):
        init_config()
        cfg = get_config()
        assert isinstance(cfg, DictConfig)


class TestGetLlmConfig:
    def test_returns_dict(self):
        init_config()
        llm_cfg = get_llm_config()
        assert isinstance(llm_cfg, dict)
        assert "default_model" in llm_cfg


class TestResetConfig:
    def test_reset_clears_state(self):
        init_config()
        reset_config()
        with pytest.raises(RuntimeError):
            get_config()
