"""llm_service 單元測試。"""

from __future__ import annotations

import warnings
from unittest.mock import patch, MagicMock

from llm_service.config import LLMConfig


class TestLLMConfig:
    def test_default_values(self):
        """LLMConfig 應有合理的預設值。"""
        cfg = LLMConfig()
        assert cfg.api_base == "http://localhost:11434/v1"
        assert cfg.model == "gpt-4o"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096

    def test_custom_values(self):
        """可自訂所有欄位。"""
        cfg = LLMConfig(
            api_base="https://custom.api/v1",
            api_key="test-key",
            model="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=2048,
            extra_headers={"X-Team": "test"},
        )
        assert cfg.api_base == "https://custom.api/v1"
        assert cfg.api_key == "test-key"
        assert cfg.extra_headers["X-Team"] == "test"

    def test_from_yaml(self, tmp_path):
        """from_yaml 應正確讀取 YAML。"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(
            'api_base: "https://test.api/v1"\n'
            'api_key: "yaml-key"\n'
            'model: "test-model"\n'
        )
        cfg = LLMConfig.from_yaml(str(yaml_file))
        assert cfg.api_base == "https://test.api/v1"
        assert cfg.api_key == "yaml-key"
        assert cfg.model == "test-model"

    def test_env_override(self, tmp_path, monkeypatch):
        """環境變數應覆寫 YAML 值。"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text('model: "yaml-model"\n')

        monkeypatch.setenv("LLM_MODEL", "env-model")
        monkeypatch.setenv("LLM_API_KEY", "env-key")

        cfg = LLMConfig.from_yaml(str(yaml_file))
        assert cfg.model == "env-model"
        assert cfg.api_key == "env-key"

    def test_from_yaml_missing_file(self):
        """YAML 檔案不存在時應使用預設值。"""
        cfg = LLMConfig.from_yaml("nonexistent.yaml")
        assert cfg.model == "gpt-4o"


class TestFactory:
    def test_get_litellm_kwargs(self):
        """get_litellm_kwargs 應回傳正確的 dict。"""
        from llm_service.factory import get_litellm_kwargs

        cfg = LLMConfig(
            api_base="https://test.api/v1",
            api_key="test-key",
            model="gpt-4o",
        )
        kwargs = get_litellm_kwargs(cfg)
        assert kwargs["model"] == "openai/gpt-4o"
        assert kwargs["api_base"] == "https://test.api/v1"
        assert kwargs["api_key"] == "test-key"

    def test_get_litellm_kwargs_preserves_provider_prefix(self):
        """已有 provider/ 前綴時不應重複加入。"""
        from llm_service.factory import get_litellm_kwargs

        cfg = LLMConfig(model="anthropic/claude-3-opus")
        kwargs = get_litellm_kwargs(cfg)
        assert kwargs["model"] == "anthropic/claude-3-opus"

    def test_get_litellm_kwargs_overrides(self):
        """overrides 應覆寫 config 值。"""
        from llm_service.factory import get_litellm_kwargs

        cfg = LLMConfig(model="gpt-4o", temperature=0.7)
        kwargs = get_litellm_kwargs(cfg, model="gpt-3.5-turbo", temperature=0.1)
        assert kwargs["model"] == "openai/gpt-3.5-turbo"
        assert kwargs["temperature"] == 0.1

    @patch("langchain_litellm.ChatLiteLLM")
    def test_get_langchain_llm(self, mock_cls):
        """get_langchain_llm 應建立 ChatLiteLLM。"""
        from llm_service.factory import get_langchain_llm

        cfg = LLMConfig(model="gpt-4o", api_base="https://test/v1", api_key="k")
        get_langchain_llm(cfg)

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["api_base"] == "https://test/v1"

    @patch("openai.OpenAI")
    def test_get_openai_client(self, mock_cls):
        """get_openai_client 應建立 OpenAI client。"""
        from llm_service.factory import get_openai_client

        cfg = LLMConfig(api_base="https://test/v1", api_key="k")
        get_openai_client(cfg)

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["base_url"] == "https://test/v1"
        assert call_kwargs["api_key"] == "k"


class TestLLMClientDeprecated:
    def test_deprecation_warning(self):
        """LLMClient 應發出 DeprecationWarning。"""
        from llm_service.client import LLMClient

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LLMClient(config=LLMConfig())
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
