"""llm_service 單元測試。"""

from __future__ import annotations

import warnings
from unittest.mock import patch, MagicMock

import pytest

from llm_service.config import (
    AuthConfig,
    LLMConfig,
    ModelConfig,
    ResolvedModelConfig,
    SharedConfig,
    VALID_ZONES,
)


# === 新版多模型配置測試 ===

class TestSharedConfig:
    def test_default_zone(self):
        sc = SharedConfig(auth_urls={"DEV": "https://auth/token"})
        assert sc.default_zone == "DEV"

    def test_zone_normalization(self):
        sc = SharedConfig(default_zone="dev", auth_urls={"DEV": "https://auth/token"})
        assert sc.default_zone == "DEV"

    def test_invalid_zone(self):
        with pytest.raises(ValueError, match="Invalid zone"):
            SharedConfig(default_zone="INVALID")

    def test_get_auth_url(self):
        sc = SharedConfig(auth_urls={
            "DEV": "https://auth-dev/token",
            "PROD": "https://auth-prod/token",
        })
        assert sc.get_auth_url("DEV") == "https://auth-dev/token"
        assert sc.get_auth_url("PROD") == "https://auth-prod/token"

    def test_get_auth_url_missing_zone(self):
        sc = SharedConfig(auth_urls={"DEV": "https://auth/token"})
        with pytest.raises(ValueError, match="No auth_url configured"):
            sc.get_auth_url("PROD")


class TestModelConfig:
    def test_basic_model_config(self):
        mc = ModelConfig(
            model_name="qwen3",
            j1_token="test-token",
            api_endpoints={"DEV": "https://llm-dev/v1"},
            temperature=0.5,
        )
        assert mc.model_name == "qwen3"
        assert mc.temperature == 0.5

    def test_get_api_endpoint(self):
        mc = ModelConfig(
            model_name="qwen3",
            api_endpoints={"DEV": "https://dev/v1", "PROD": "https://prod/v1"},
        )
        assert mc.get_api_endpoint("DEV") == "https://dev/v1"
        assert mc.get_api_endpoint("PROD") == "https://prod/v1"

    def test_get_api_endpoint_missing_zone(self):
        mc = ModelConfig(model_name="qwen3", api_endpoints={"DEV": "https://dev/v1"})
        with pytest.raises(ValueError, match="No api_endpoint configured"):
            mc.get_api_endpoint("PROD")

    def test_get_hyperparams(self):
        mc = ModelConfig(
            model_name="qwen3",
            temperature=0.5,
            max_tokens=2048,
            top_p=0.9,
        )
        params = mc.get_hyperparams()
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 2048
        assert params["top_p"] == 0.9
        assert "top_k" not in params  # None values excluded

    def test_hyperparams_exclude_none(self):
        mc = ModelConfig(model_name="qwen3")
        params = mc.get_hyperparams()
        assert "top_p" not in params
        assert "stop" not in params


class TestLLMConfigMultiModel:
    def _make_config(self, **overrides):
        defaults = {
            "shared_config": SharedConfig(
                default_zone="DEV",
                auth_urls={"DEV": "https://auth-dev/token", "PROD": "https://auth-prod/token"},
            ),
            "model_configs": {
                "QWEN3": ModelConfig(
                    j1_token="j1-qwen3",
                    model_name="qwen3",
                    api_endpoints={"DEV": "https://llm-dev/v1", "PROD": "https://llm-prod/v1"},
                ),
                "QWEN3VL": ModelConfig(
                    j1_token="j1-vl",
                    model_name="qwen3-vl",
                    api_endpoints={"DEV": "https://vl-dev/v1", "PROD": "https://vl-prod/v1"},
                    temperature=0.3,
                ),
            },
        }
        defaults.update(overrides)
        return LLMConfig(**defaults)

    def test_list_models(self):
        cfg = self._make_config()
        assert cfg.list_models() == ["QWEN3", "QWEN3VL"]

    def test_get_model_config(self):
        cfg = self._make_config()
        mc = cfg.get_model_config("QWEN3")
        assert mc.model_name == "qwen3"

    def test_get_model_config_not_found(self):
        cfg = self._make_config()
        with pytest.raises(KeyError, match="not found"):
            cfg.get_model_config("NONEXISTENT")

    def test_resolve(self):
        cfg = self._make_config()
        with patch("llm_service.auth.TokenExchanger") as mock_cls:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "j2-token"
            mock_cls.return_value = mock_exchanger

            resolved = cfg.resolve("QWEN3")
            assert isinstance(resolved, ResolvedModelConfig)
            assert resolved.model_name == "qwen3"
            assert resolved.api_base == "https://llm-dev/v1"
            assert resolved.api_key == "j2-token"

    def test_resolve_with_zone(self):
        cfg = self._make_config()
        with patch("llm_service.auth.TokenExchanger") as mock_cls:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "j2-prod"
            mock_cls.return_value = mock_exchanger

            resolved = cfg.resolve("QWEN3", zone="PROD")
            assert resolved.api_base == "https://llm-prod/v1"

    def test_resolve_missing_j1_token(self):
        cfg = LLMConfig(
            shared_config=SharedConfig(
                auth_urls={"DEV": "https://auth/token"},
            ),
            model_configs={
                "QWEN3": ModelConfig(
                    j1_token="",
                    model_name="qwen3",
                    api_endpoints={"DEV": "https://dev/v1"},
                ),
            },
        )
        with pytest.raises(ValueError, match="No J1 token"):
            cfg.resolve("QWEN3")

    def test_validation_requires_auth_urls(self):
        with pytest.raises(ValueError, match="auth_urls is required"):
            LLMConfig(
                shared_config=SharedConfig(auth_urls={}),
                model_configs={
                    "QWEN3": ModelConfig(
                        model_name="qwen3",
                        api_endpoints={"DEV": "https://dev/v1"},
                    ),
                },
            )

    def test_validation_requires_default_zone_endpoint(self):
        with pytest.raises(ValueError, match="missing api_endpoint"):
            LLMConfig(
                shared_config=SharedConfig(
                    default_zone="DEV",
                    auth_urls={"DEV": "https://auth/token"},
                ),
                model_configs={
                    "QWEN3": ModelConfig(
                        model_name="qwen3",
                        api_endpoints={"PROD": "https://prod/v1"},  # missing DEV
                    ),
                },
            )

    def test_clear_token_cache_specific(self):
        cfg = self._make_config()
        with patch("llm_service.auth.TokenExchanger") as mock_cls:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "j2"
            mock_cls.return_value = mock_exchanger

            cfg.resolve("QWEN3")
            cfg.clear_token_cache("QWEN3")
            mock_exchanger.clear_cache.assert_called_once()

    def test_clear_token_cache_all(self):
        cfg = self._make_config()
        with patch("llm_service.auth.TokenExchanger") as mock_cls:
            mock_e1 = MagicMock()
            mock_e1.get_token.return_value = "j2"
            mock_e2 = MagicMock()
            mock_e2.get_token.return_value = "j2"
            mock_cls.side_effect = [mock_e1, mock_e2]

            cfg.resolve("QWEN3")
            cfg.resolve("QWEN3VL")
            cfg.clear_token_cache()
            mock_e1.clear_cache.assert_called_once()
            mock_e2.clear_cache.assert_called_once()

    def test_from_yaml_multimodel(self, tmp_path):
        yaml_file = tmp_path / "llm_config.yaml"
        yaml_file.write_text(
            'shared_config:\n'
            '  default_zone: "DEV"\n'
            '  auth_urls:\n'
            '    DEV: "https://auth-dev/token"\n'
            'model_configs:\n'
            '  QWEN3:\n'
            '    j1_token: "yaml-j1"\n'
            '    model_name: "qwen3"\n'
            '    api_endpoints:\n'
            '      DEV: "https://llm-dev/v1"\n'
            '    temperature: 0.5\n'
        )
        cfg = LLMConfig.from_yaml(str(yaml_file))
        assert "QWEN3" in cfg.model_configs
        assert cfg.model_configs["QWEN3"].model_name == "qwen3"
        assert cfg.model_configs["QWEN3"].temperature == 0.5

    def test_from_yaml_zone_env_override(self, tmp_path, monkeypatch):
        yaml_file = tmp_path / "llm_config.yaml"
        yaml_file.write_text(
            'shared_config:\n'
            '  default_zone: "DEV"\n'
            '  auth_urls:\n'
            '    DEV: "https://auth-dev/token"\n'
            '    PROD: "https://auth-prod/token"\n'
            'model_configs:\n'
            '  QWEN3:\n'
            '    model_name: "qwen3"\n'
            '    api_endpoints:\n'
            '      DEV: "https://dev/v1"\n'
            '      PROD: "https://prod/v1"\n'
        )
        monkeypatch.setenv("LLM_ZONE", "PROD")
        cfg = LLMConfig.from_yaml(str(yaml_file))
        assert cfg.shared_config.default_zone == "PROD"

    def test_from_yaml_model_token_env_override(self, tmp_path, monkeypatch):
        yaml_file = tmp_path / "llm_config.yaml"
        yaml_file.write_text(
            'shared_config:\n'
            '  auth_urls:\n'
            '    DEV: "https://auth/token"\n'
            'model_configs:\n'
            '  QWEN3:\n'
            '    j1_token: "yaml-j1"\n'
            '    model_name: "qwen3"\n'
            '    api_endpoints:\n'
            '      DEV: "https://dev/v1"\n'
        )
        monkeypatch.setenv("LLM_AUTH_TOKEN_QWEN3", "env-j1")
        cfg = LLMConfig.from_yaml(str(yaml_file))
        assert cfg.model_configs["QWEN3"].j1_token == "env-j1"


# === 向後相容測試（舊版單一模型）===

class TestLLMConfigLegacy:
    def test_default_values(self):
        cfg = LLMConfig()
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096

    def test_custom_values(self):
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

    def test_from_yaml_legacy(self, tmp_path):
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
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text('model: "yaml-model"\n')
        monkeypatch.setenv("LLM_MODEL", "env-model")
        monkeypatch.setenv("LLM_API_KEY", "env-key")
        cfg = LLMConfig.from_yaml(str(yaml_file))
        assert cfg.model == "env-model"
        assert cfg.api_key == "env-key"

    def test_from_yaml_missing_file(self):
        cfg = LLMConfig.from_yaml("nonexistent.yaml")
        assert cfg.model == ""


class TestFactory:
    def test_get_litellm_kwargs_legacy(self):
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
        from llm_service.factory import get_litellm_kwargs

        cfg = LLMConfig(model="anthropic/claude-3-opus")
        kwargs = get_litellm_kwargs(cfg)
        assert kwargs["model"] == "anthropic/claude-3-opus"

    def test_get_litellm_kwargs_overrides(self):
        from llm_service.factory import get_litellm_kwargs

        cfg = LLMConfig(model="gpt-4o", temperature=0.7)
        kwargs = get_litellm_kwargs(cfg, model="gpt-3.5-turbo", temperature=0.1)
        assert kwargs["model"] == "openai/gpt-3.5-turbo"
        assert kwargs["temperature"] == 0.1

    def test_get_litellm_kwargs_multimodel(self):
        from llm_service.factory import get_litellm_kwargs

        cfg = LLMConfig(
            shared_config=SharedConfig(
                auth_urls={"DEV": "https://auth/token"},
            ),
            model_configs={
                "QWEN3": ModelConfig(
                    j1_token="j1",
                    model_name="qwen3",
                    api_endpoints={"DEV": "https://dev/v1"},
                    temperature=0.5,
                ),
            },
        )
        with patch("llm_service.auth.TokenExchanger") as mock_cls:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "j2"
            mock_cls.return_value = mock_exchanger

            kwargs = get_litellm_kwargs(cfg, model_alias="QWEN3")
            assert kwargs["model"] == "openai/qwen3"
            assert kwargs["api_base"] == "https://dev/v1"
            assert kwargs["api_key"] == "j2"
            assert kwargs["temperature"] == 0.5

    @patch("langchain_litellm.ChatLiteLLM")
    def test_get_langchain_llm_legacy(self, mock_cls):
        from llm_service.factory import get_langchain_llm

        cfg = LLMConfig(model="gpt-4o", api_base="https://test/v1", api_key="k")
        get_langchain_llm(cfg)

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["api_base"] == "https://test/v1"

    @patch("openai.OpenAI")
    def test_get_openai_client_legacy(self, mock_cls):
        from llm_service.factory import get_openai_client

        cfg = LLMConfig(api_base="https://test/v1", api_key="k")
        get_openai_client(cfg)

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["base_url"] == "https://test/v1"
        assert call_kwargs["api_key"] == "k"


class TestBuildMultimodalMessages:
    def test_text_only(self):
        from llm_service.factory import build_multimodal_messages

        messages = build_multimodal_messages(user_text="Hello")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_with_system_prompt(self):
        from llm_service.factory import build_multimodal_messages

        messages = build_multimodal_messages(user_text="Hi", system_prompt="Be helpful")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_with_single_image(self):
        from llm_service.factory import build_multimodal_messages

        messages = build_multimodal_messages(
            user_text="Describe",
            image_base64="abc123",
        )
        user_msg = messages[0]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0]["type"] == "text"
        assert user_msg["content"][1]["type"] == "image_url"
        assert user_msg["content"][1]["image_url"]["url"] == "data:image/png;base64,abc123"

    def test_with_data_uri_prefix(self):
        from llm_service.factory import build_multimodal_messages

        messages = build_multimodal_messages(
            user_text="Describe",
            image_base64="data:image/jpeg;base64,xyz",
        )
        url = messages[0]["content"][1]["image_url"]["url"]
        assert url == "data:image/jpeg;base64,xyz"

    def test_with_multiple_images(self):
        from llm_service.factory import build_multimodal_messages

        messages = build_multimodal_messages(
            user_text="Compare",
            image_base64=["img1", "img2"],
        )
        content = messages[0]["content"]
        assert len(content) == 3  # text + 2 images


class TestTokenExchange:
    def test_resolve_api_key_without_auth(self):
        cfg = LLMConfig(api_key="direct-key")
        assert cfg.resolve_api_key() == "direct-key"

    def test_resolve_api_key_with_auth(self):
        cfg = LLMConfig(
            auth=AuthConfig(
                auth_url="https://auth.test/token",
                auth_token="j1-token",
            ),
        )
        with patch("llm_service.auth.TokenExchanger") as mock_cls:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "j2-token"
            mock_cls.return_value = mock_exchanger

            result = cfg.resolve_api_key()
            assert result == "j2-token"

    def test_resolve_api_key_caches_exchanger(self):
        cfg = LLMConfig(
            auth=AuthConfig(
                auth_url="https://auth.test/token",
                auth_token="j1-token",
            ),
        )
        with patch("llm_service.auth.TokenExchanger") as mock_cls:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "j2"
            mock_cls.return_value = mock_exchanger

            cfg.resolve_api_key()
            cfg.resolve_api_key()
            mock_cls.assert_called_once()
            assert mock_exchanger.get_token.call_count == 2

    def test_from_yaml_with_auth(self, tmp_path):
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(
            'api_base: "https://llm.test/v1"\n'
            'model: "gpt-4o"\n'
            'auth:\n'
            '  auth_url: "https://auth.test/token"\n'
            '  auth_token: "j1-from-yaml"\n'
            '  token_field: "token"\n'
        )
        cfg = LLMConfig.from_yaml(str(yaml_file))
        assert cfg.auth is not None
        assert cfg.auth.auth_url == "https://auth.test/token"
        assert cfg.auth.auth_token == "j1-from-yaml"
        assert cfg.auth.token_field == "token"

    def test_from_yaml_auth_token_env_override(self, tmp_path, monkeypatch):
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(
            'auth:\n'
            '  auth_url: "https://auth.test/token"\n'
            '  auth_token: "yaml-j1"\n'
        )
        monkeypatch.setenv("LLM_AUTH_TOKEN", "env-j1")
        cfg = LLMConfig.from_yaml(str(yaml_file))
        assert cfg.auth.auth_token == "env-j1"


class TestTokenExchangerUnit:
    def test_token_exchange_flow(self):
        from llm_service.auth import TokenExchanger

        exchanger = TokenExchanger(
            auth_url="https://auth.test/token",
            auth_token="j1-token",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "j2-token-value",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            token = exchanger.get_token()
            assert token == "j2-token-value"

            mock_client.post.reset_mock()
            token2 = exchanger.get_token()
            assert token2 == "j2-token-value"
            mock_client.post.assert_not_called()

    def test_is_expired_initially(self):
        from llm_service.auth import TokenExchanger

        exchanger = TokenExchanger(
            auth_url="https://auth.test/token",
            auth_token="j1",
        )
        assert exchanger.is_expired is True

    def test_clear_cache(self):
        from llm_service.auth import TokenExchanger

        exchanger = TokenExchanger(
            auth_url="https://auth.test/token",
            auth_token="j1",
        )
        exchanger._cached_token = "cached"
        exchanger._expires_at = 999999999999.0

        assert exchanger.is_expired is False
        exchanger.clear_cache()
        assert exchanger.is_expired is True
        assert exchanger._cached_token is None


class TestLLMClientDeprecated:
    def test_deprecation_warning(self):
        from llm_service.client import LLMClient

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LLMClient(config=LLMConfig())
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
