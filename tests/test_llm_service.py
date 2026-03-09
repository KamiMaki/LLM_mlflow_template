"""llm_service 單元測試。"""

from __future__ import annotations

import warnings
from unittest.mock import patch, MagicMock

from llm_service.config import AuthConfig, LLMConfig


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


class TestTokenExchange:
    def test_resolve_api_key_without_auth(self):
        """沒有 auth 設定時直接回傳 api_key。"""
        cfg = LLMConfig(api_key="direct-key")
        assert cfg.resolve_api_key() == "direct-key"

    def test_resolve_api_key_with_auth(self):
        """設定 auth 時應呼叫 TokenExchanger。"""
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
            mock_cls.assert_called_once_with(
                auth_url="https://auth.test/token",
                auth_token="j1-token",
                token_field="access_token",
                expires_field="expires_in",
                auth_method="bearer",
                extra_body={},
                extra_headers={},
                buffer_seconds=60,
            )

    def test_resolve_api_key_caches_exchanger(self):
        """TokenExchanger 應被快取，第二次呼叫不重建。"""
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
            # TokenExchanger 只建立一次
            mock_cls.assert_called_once()
            # get_token 呼叫兩次
            assert mock_exchanger.get_token.call_count == 2

    def test_from_yaml_with_auth(self, tmp_path):
        """from_yaml 應正確解析 auth 區段。"""
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
        """LLM_AUTH_TOKEN 環境變數應覆寫 auth.auth_token。"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(
            'auth:\n'
            '  auth_url: "https://auth.test/token"\n'
            '  auth_token: "yaml-j1"\n'
        )
        monkeypatch.setenv("LLM_AUTH_TOKEN", "env-j1")
        cfg = LLMConfig.from_yaml(str(yaml_file))
        assert cfg.auth.auth_token == "env-j1"

    def test_clear_token_cache(self):
        """clear_token_cache 應清除快取。"""
        cfg = LLMConfig(
            auth=AuthConfig(
                auth_url="https://auth.test/token",
                auth_token="j1",
            ),
        )
        with patch("llm_service.auth.TokenExchanger") as mock_cls:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "j2"
            mock_cls.return_value = mock_exchanger

            cfg.resolve_api_key()
            cfg.clear_token_cache()
            mock_exchanger.clear_cache.assert_called_once()


class TestTokenExchangerUnit:
    def test_token_exchange_flow(self):
        """TokenExchanger 應正確交換 token。"""
        from llm_service.auth import TokenExchanger
        import httpx

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

            # 快取生效，不再呼叫 post
            mock_client.post.reset_mock()
            token2 = exchanger.get_token()
            assert token2 == "j2-token-value"
            mock_client.post.assert_not_called()

    def test_is_expired_initially(self):
        """初始狀態應為 expired。"""
        from llm_service.auth import TokenExchanger

        exchanger = TokenExchanger(
            auth_url="https://auth.test/token",
            auth_token="j1",
        )
        assert exchanger.is_expired is True

    def test_clear_cache(self):
        """clear_cache 應重置 token。"""
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
        """LLMClient 應發出 DeprecationWarning。"""
        from llm_service.client import LLMClient

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LLMClient(config=LLMConfig())
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
