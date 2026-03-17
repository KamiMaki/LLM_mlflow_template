"""llm_service 單元測試 — LLMService + Config + Trace + Retry + AI Service。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_service.config import (
    LLMConfig,
    ModelConfig,
    ResolvedModelConfig,
    RetryConfig,
    ServiceConfig,
    SharedConfig,
)
from llm_service.models import AIServiceResponse, LLMResponse, TokenUsage
from llm_service.service import LLMService
from llm_service.trace import (
    is_sensitive_key,
    sanitize_completion_kwargs,
    sanitize_dict,
)


# === Trace / Sanitizer 測試 ===


class TestSanitizer:
    def test_is_sensitive_key_token(self):
        assert is_sensitive_key("Authorization") is True
        assert is_sensitive_key("x-auth-token") is True
        assert is_sensitive_key("api_key") is True
        assert is_sensitive_key("Bearer") is True
        assert is_sensitive_key("password") is True
        assert is_sensitive_key("secret_key") is True
        assert is_sensitive_key("credential") is True

    def test_is_sensitive_key_safe(self):
        assert is_sensitive_key("Content-Type") is False
        assert is_sensitive_key("X-Team-Id") is False
        assert is_sensitive_key("model") is False

    def test_sanitize_dict_masks_sensitive(self):
        data = {
            "Authorization": "Bearer sk-1234567890abcdef",
            "X-Team-Id": "team-123",
            "api_key": "long-secret-key-value",
        }
        result = sanitize_dict(data)
        assert "***REDACTED***" in result["Authorization"]
        assert result["Authorization"].startswith("Bear")
        assert result["X-Team-Id"] == "team-123"
        assert "***REDACTED***" in result["api_key"]

    def test_sanitize_dict_short_value(self):
        result = sanitize_dict({"token": "short"})
        assert result["token"] == "***REDACTED***"

    def test_sanitize_dict_nested(self):
        data = {"headers": {"Authorization": "Bearer long-token-value-here"}}
        result = sanitize_dict(data)
        assert "***REDACTED***" in result["headers"]["Authorization"]

    def test_sanitize_dict_list(self):
        data = [{"token": "secret-long-value-123"}]
        result = sanitize_dict(data)
        assert "***REDACTED***" in result[0]["token"]

    def test_sanitize_dict_non_dict(self):
        assert sanitize_dict("plain string") == "plain string"
        assert sanitize_dict(42) == 42

    def test_sanitize_completion_kwargs(self):
        kwargs = {
            "model": "openai/qwen3",
            "api_key": "j2-long-token-value-here",
            "api_base": "https://llm-dev/v1",
            "extra_headers": {
                "Authorization": "Bearer some-long-token",
                "X-Team-Id": "team-abc",
            },
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
        }
        result = sanitize_completion_kwargs(kwargs)
        assert result["model"] == "openai/qwen3"
        assert "***REDACTED***" in result["api_key"]
        assert "***REDACTED***" in result["extra_headers"]["Authorization"]
        assert result["extra_headers"]["X-Team-Id"] == "team-abc"
        assert result["messages"] == [{"role": "user", "content": "Hello"}]
        assert result["temperature"] == 0.7


# === RetryConfig 測試 ===


class TestRetryConfig:
    def test_defaults(self):
        rc = RetryConfig()
        assert rc.max_attempts == 1
        assert rc.wait_multiplier == 1.0
        assert rc.wait_min == 2.0
        assert rc.wait_max == 10.0

    def test_custom_values(self):
        rc = RetryConfig(max_attempts=3, wait_multiplier=2.0, wait_min=1.0, wait_max=30.0)
        assert rc.max_attempts == 3
        assert rc.wait_multiplier == 2.0


# === Config 測試 ===


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

    def test_j1_token_path_default(self):
        sc = SharedConfig(auth_urls={"DEV": "https://auth/token"})
        assert sc.j1_token_path == ""

    def test_j1_token_path_custom(self):
        sc = SharedConfig(
            auth_urls={"DEV": "https://auth/token"},
            j1_token_path="/var/run/secrets/j1-token",
        )
        assert sc.j1_token_path == "/var/run/secrets/j1-token"

    def test_retry_default(self):
        sc = SharedConfig(auth_urls={"DEV": "https://auth/token"})
        assert sc.retry.max_attempts == 1

    def test_retry_custom(self):
        sc = SharedConfig(
            auth_urls={"DEV": "https://auth/token"},
            retry=RetryConfig(max_attempts=3),
        )
        assert sc.retry.max_attempts == 3


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

    def test_get_api_endpoint_missing_zone(self):
        mc = ModelConfig(model_name="qwen3", api_endpoints={"DEV": "https://dev/v1"})
        with pytest.raises(ValueError, match="No api_endpoint configured"):
            mc.get_api_endpoint("PROD")

    def test_get_hyperparams(self):
        mc = ModelConfig(model_name="qwen3", temperature=0.5, max_tokens=2048, top_p=0.9)
        params = mc.get_hyperparams()
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 2048
        assert params["top_p"] == 0.9
        assert "top_k" not in params

    def test_hyperparams_exclude_none(self):
        mc = ModelConfig(model_name="qwen3")
        params = mc.get_hyperparams()
        assert "top_p" not in params
        assert "stop" not in params


# === ServiceConfig 測試 ===


class TestServiceConfig:
    def test_basic_service_config(self):
        sc = ServiceConfig(
            j1_token="svc-token",
            api_endpoints={"DEV": "https://svc-dev/v1"},
            timeout=60,
        )
        assert sc.j1_token == "svc-token"
        assert sc.timeout == 60

    def test_get_api_endpoint(self):
        sc = ServiceConfig(
            api_endpoints={"DEV": "https://dev/v1", "PROD": "https://prod/v1"},
        )
        assert sc.get_api_endpoint("DEV") == "https://dev/v1"

    def test_get_api_endpoint_missing_zone(self):
        sc = ServiceConfig(api_endpoints={"DEV": "https://dev/v1"})
        with pytest.raises(ValueError, match="No api_endpoint configured"):
            sc.get_api_endpoint("PROD")

    def test_default_timeout(self):
        sc = ServiceConfig(api_endpoints={"DEV": "https://dev/v1"})
        assert sc.timeout == 30

    def test_extra_headers_coerce_none(self):
        sc = ServiceConfig(api_endpoints={"DEV": "https://dev/v1"}, extra_headers=None)
        assert sc.extra_headers == {}


class TestLLMConfig:
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

    def test_default_model(self):
        cfg = self._make_config(default_model="QWEN3VL")
        assert cfg.default_model == "QWEN3VL"

    def test_default_model_invalid(self):
        with pytest.raises(ValueError, match="default_model.*not found"):
            self._make_config(default_model="NONEXISTENT")

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
            shared_config=SharedConfig(auth_urls={"DEV": "https://auth/token"}),
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

    def test_resolve_j1_from_file(self, tmp_path):
        token_file = tmp_path / "j1-token"
        token_file.write_text("file-j1-token")

        cfg = LLMConfig(
            shared_config=SharedConfig(
                auth_urls={"DEV": "https://auth/token"},
                j1_token_path=str(token_file),
            ),
            model_configs={
                "QWEN3": ModelConfig(
                    j1_token="",
                    model_name="qwen3",
                    api_endpoints={"DEV": "https://dev/v1"},
                ),
            },
        )
        with patch("llm_service.auth.TokenExchanger") as mock_cls:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "j2-from-file"
            mock_cls.return_value = mock_exchanger

            resolved = cfg.resolve("QWEN3")
            assert resolved.api_key == "j2-from-file"
            # 驗證 TokenExchanger 收到 file-j1-token
            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args
            assert call_kwargs.kwargs.get("auth_token") or call_kwargs.args[1] == "file-j1-token"

    def test_resolve_j1_env_overrides_file(self, tmp_path, monkeypatch):
        token_file = tmp_path / "j1-token"
        token_file.write_text("file-j1-token")
        monkeypatch.setenv("LLM_AUTH_TOKEN_QWEN3", "env-j1-token")

        cfg = LLMConfig(
            shared_config=SharedConfig(
                auth_urls={"DEV": "https://auth/token"},
                j1_token_path=str(token_file),
            ),
            model_configs={
                "QWEN3": ModelConfig(
                    j1_token="config-j1",
                    model_name="qwen3",
                    api_endpoints={"DEV": "https://dev/v1"},
                ),
            },
        )
        with patch("llm_service.auth.TokenExchanger") as mock_cls:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "j2"
            mock_cls.return_value = mock_exchanger

            cfg.resolve("QWEN3")
            # env var 優先於 config 和 file
            call_kwargs = mock_cls.call_args
            assert call_kwargs.kwargs.get("auth_token") == "env-j1-token"

    def test_read_j1_from_file_not_found(self):
        cfg = LLMConfig(
            shared_config=SharedConfig(
                auth_urls={"DEV": "https://auth/token"},
                j1_token_path="/nonexistent/path/j1-token",
            ),
            model_configs={},
        )
        assert cfg._read_j1_from_file() == ""

    def test_read_j1_from_file_empty_path(self):
        cfg = LLMConfig(
            shared_config=SharedConfig(auth_urls={"DEV": "https://auth/token"}),
            model_configs={},
        )
        assert cfg._read_j1_from_file() == ""

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

    def test_validation_requires_auth_urls_for_services(self):
        with pytest.raises(ValueError, match="auth_urls is required"):
            LLMConfig(
                shared_config=SharedConfig(auth_urls={}),
                service_configs={
                    "SVC": ServiceConfig(api_endpoints={"DEV": "https://dev/v1"}),
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
                        api_endpoints={"PROD": "https://prod/v1"},
                    ),
                },
            )

    def test_get_service_config(self):
        cfg = LLMConfig(
            shared_config=SharedConfig(auth_urls={"DEV": "https://auth/token"}),
            service_configs={
                "IMG": ServiceConfig(api_endpoints={"DEV": "https://img/v1"}),
            },
        )
        sc = cfg.get_service_config("IMG")
        assert sc.get_api_endpoint("DEV") == "https://img/v1"

    def test_get_service_config_not_found(self):
        cfg = LLMConfig(
            shared_config=SharedConfig(auth_urls={"DEV": "https://auth/token"}),
            service_configs={},
        )
        with pytest.raises(KeyError, match="not found"):
            cfg.get_service_config("NONEXISTENT")

    def test_resolve_service(self):
        cfg = LLMConfig(
            shared_config=SharedConfig(auth_urls={"DEV": "https://auth/token"}),
            service_configs={
                "IMG": ServiceConfig(
                    j1_token="svc-j1",
                    api_endpoints={"DEV": "https://img-dev/v1"},
                ),
            },
        )
        with patch("llm_service.auth.TokenExchanger") as mock_cls:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "svc-j2"
            mock_cls.return_value = mock_exchanger

            endpoint, j2, headers = cfg.resolve_service("IMG")
            assert endpoint == "https://img-dev/v1"
            assert j2 == "svc-j2"

    def test_clear_token_cache(self):
        cfg = self._make_config()
        with patch("llm_service.auth.TokenExchanger") as mock_cls:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "j2"
            mock_cls.return_value = mock_exchanger

            cfg.resolve("QWEN3")
            cfg.clear_token_cache("QWEN3")
            mock_exchanger.clear_cache.assert_called_once()

    def test_from_yaml(self, tmp_path):
        yaml_file = tmp_path / "llm_config.yaml"
        yaml_file.write_text(
            'default_model: "QWEN3"\n'
            'shared_config:\n'
            '  default_zone: "DEV"\n'
            '  auth_urls:\n'
            '    DEV: "https://auth-dev/token"\n'
            '  retry:\n'
            '    max_attempts: 3\n'
            '    wait_min: 1.0\n'
            'model_configs:\n'
            '  QWEN3:\n'
            '    j1_token: "yaml-j1"\n'
            '    model_name: "qwen3"\n'
            '    api_endpoints:\n'
            '      DEV: "https://llm-dev/v1"\n'
            '    temperature: 0.5\n'
        )
        cfg = LLMConfig.from_yaml(str(yaml_file))
        assert cfg.default_model == "QWEN3"
        assert "QWEN3" in cfg.model_configs
        assert cfg.model_configs["QWEN3"].temperature == 0.5
        assert cfg.shared_config.retry.max_attempts == 3
        assert cfg.shared_config.retry.wait_min == 1.0

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

    def test_from_yaml_service_token_env_override(self, tmp_path, monkeypatch):
        yaml_file = tmp_path / "llm_config.yaml"
        yaml_file.write_text(
            'shared_config:\n'
            '  auth_urls:\n'
            '    DEV: "https://auth/token"\n'
            'service_configs:\n'
            '  IMG:\n'
            '    j1_token: "yaml-svc-j1"\n'
            '    api_endpoints:\n'
            '      DEV: "https://img/v1"\n'
        )
        monkeypatch.setenv("LLM_AUTH_TOKEN_IMG", "env-svc-j1")
        cfg = LLMConfig.from_yaml(str(yaml_file))
        assert cfg.service_configs["IMG"].j1_token == "env-svc-j1"

    def test_from_yaml_missing_file(self):
        cfg = LLMConfig.from_yaml("nonexistent.yaml")
        assert cfg.model_configs == {}


# === LLMService 測試 ===


class TestLLMService:
    def _make_service(self, default_model="", retry_attempts=1, **model_overrides):
        """建立帶 mock config 的 LLMService。"""
        cfg = LLMConfig(
            default_model=default_model,
            shared_config=SharedConfig(
                default_zone="DEV",
                auth_urls={"DEV": "https://auth/token"},
                retry=RetryConfig(max_attempts=retry_attempts),
            ),
            model_configs={
                "QWEN3": ModelConfig(
                    j1_token="j1-qwen3",
                    model_name="qwen3",
                    api_endpoints={"DEV": "https://llm-dev/v1"},
                    temperature=0.7,
                    **model_overrides,
                ),
                "QWEN3VL": ModelConfig(
                    j1_token="j1-vl",
                    model_name="qwen3-vl",
                    api_endpoints={"DEV": "https://vl-dev/v1"},
                    temperature=0.3,
                ),
            },
        )
        return LLMService(config=cfg)

    def _mock_completion_response(self, content="Hello back!", reasoning=None):
        mock_message = MagicMock()
        mock_message.content = content
        mock_message.reasoning_content = reasoning
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        mock_resp.model = "qwen3"
        return mock_resp

    def test_init_default_model(self):
        service = self._make_service()
        assert service.current_model == "QWEN3"  # first model

    def test_init_explicit_default_model(self):
        service = self._make_service(default_model="QWEN3VL")
        assert service.current_model == "QWEN3VL"

    def test_set_model(self):
        service = self._make_service()
        result = service.set_model("QWEN3VL")
        assert service.current_model == "QWEN3VL"
        assert result is service  # chain call

    def test_set_model_invalid(self):
        service = self._make_service()
        with pytest.raises(KeyError, match="not found"):
            service.set_model("NONEXISTENT")

    def test_build_messages_text_only(self):
        service = self._make_service()
        messages = service._build_messages(
            user_prompt="Hello",
            system_prompt="Be helpful",
        )
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "Be helpful"}
        assert messages[1] == {"role": "user", "content": "Hello"}

    def test_build_messages_user_only(self):
        service = self._make_service()
        messages = service._build_messages(user_prompt="Hello")
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello"}

    def test_build_messages_system_only(self):
        service = self._make_service()
        messages = service._build_messages(system_prompt="Be helpful")
        assert len(messages) == 1
        assert messages[0] == {"role": "system", "content": "Be helpful"}

    def test_build_messages_no_input_raises(self):
        service = self._make_service()
        with pytest.raises(ValueError, match="Must provide"):
            service._build_messages()

    def test_build_messages_with_image(self):
        service = self._make_service()
        messages = service._build_messages(
            user_prompt="Describe",
            image_base64="abc123",
        )
        assert len(messages) == 1
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "Describe"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "data:image/png;base64,abc123"

    def test_build_messages_with_data_uri(self):
        service = self._make_service()
        messages = service._build_messages(
            user_prompt="Describe",
            image_base64="data:image/jpeg;base64,xyz",
        )
        url = messages[0]["content"][1]["image_url"]["url"]
        assert url == "data:image/jpeg;base64,xyz"

    def test_build_messages_with_multiple_images(self):
        service = self._make_service()
        messages = service._build_messages(
            user_prompt="Compare",
            image_base64=["img1", "img2"],
        )
        content = messages[0]["content"]
        assert len(content) == 3  # text + 2 images

    def test_build_messages_prompt_template(self):
        service = self._make_service()
        messages = service._build_messages(
            prompt_template="Check: {{ data }}",
            prompt_variables={"data": "test-data"},
            system_prompt="You are QA",
        )
        assert messages[0] == {"role": "system", "content": "You are QA"}
        assert messages[1] == {"role": "user", "content": "Check: test-data"}

    def test_build_messages_prompt_template_overrides_user_prompt(self):
        service = self._make_service()
        messages = service._build_messages(
            user_prompt="ignored",
            prompt_template="Template: {{ x }}",
            prompt_variables={"x": "value"},
        )
        assert messages[0]["content"] == "Template: value"

    @patch("litellm.completion")
    def test_call_llm(self, mock_completion):
        mock_completion.return_value = self._mock_completion_response()

        service = self._make_service()
        with patch("llm_service.auth.TokenExchanger") as mock_auth:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "j2-token"
            mock_auth.return_value = mock_exchanger

            response = service.call_llm(
                user_prompt="Hello",
                system_prompt="Be helpful",
            )

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello back!"
        assert response.model == "qwen3"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30

        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["model"] == "openai/qwen3"
        assert call_kwargs.kwargs["api_base"] == "https://llm-dev/v1"

    @patch("litellm.completion")
    def test_call_llm_with_overrides(self, mock_completion):
        mock_completion.return_value = self._mock_completion_response("Response")

        service = self._make_service()
        with patch("llm_service.auth.TokenExchanger") as mock_auth:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "j2"
            mock_auth.return_value = mock_exchanger

            service.call_llm(
                user_prompt="Test",
                temperature=0.1,
                max_tokens=8192,
            )

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["max_tokens"] == 8192

    @patch("litellm.completion")
    def test_call_llm_with_reasoning_content(self, mock_completion):
        mock_completion.return_value = self._mock_completion_response(
            "Final answer", reasoning="Let me think..."
        )

        service = self._make_service()
        with patch("llm_service.auth.TokenExchanger") as mock_auth:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "j2"
            mock_auth.return_value = mock_exchanger

            response = service.call_llm(user_prompt="Think about this")

        assert response.content == "Final answer"
        assert response.reasoning_content == "Let me think..."

    @patch("litellm.completion")
    def test_call_llm_model_switch(self, mock_completion):
        mock_resp = self._mock_completion_response("Response")
        mock_resp.model = "qwen3-vl"
        mock_completion.return_value = mock_resp

        service = self._make_service()
        with patch("llm_service.auth.TokenExchanger") as mock_auth:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "j2"
            mock_auth.return_value = mock_exchanger

            service.set_model("QWEN3VL")
            service.call_llm(user_prompt="Describe image", image_base64="abc")

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["model"] == "openai/qwen3-vl"
        assert call_kwargs["api_base"] == "https://vl-dev/v1"
        assert call_kwargs["temperature"] == 0.3

    @patch("litellm.completion")
    def test_call_llm_prompt_template(self, mock_completion):
        mock_completion.return_value = self._mock_completion_response("Checked")

        service = self._make_service()
        with patch("llm_service.auth.TokenExchanger") as mock_auth:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "j2"
            mock_auth.return_value = mock_exchanger

            service.call_llm(
                prompt_template="Check: {{ data }}",
                prompt_variables={"data": "test"},
                system_prompt="QA",
            )

        messages = mock_completion.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "QA"}
        assert messages[1] == {"role": "user", "content": "Check: test"}


# === Retry 測試 ===


class TestRetry:
    @patch("litellm.completion")
    def test_retry_on_failure(self, mock_completion):
        """測試 retry 機制：前兩次失敗，第三次成功。"""
        mock_message = MagicMock()
        mock_message.content = "Success"
        mock_message.reasoning_content = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = MagicMock(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        mock_resp.model = "qwen3"

        mock_completion.side_effect = [
            ConnectionError("Network error"),
            ConnectionError("Network error again"),
            mock_resp,
        ]

        cfg = LLMConfig(
            shared_config=SharedConfig(
                default_zone="DEV",
                auth_urls={"DEV": "https://auth/token"},
                retry=RetryConfig(max_attempts=3, wait_min=0.01, wait_max=0.02),
            ),
            model_configs={
                "QWEN3": ModelConfig(
                    j1_token="j1",
                    model_name="qwen3",
                    api_endpoints={"DEV": "https://dev/v1"},
                ),
            },
        )
        service = LLMService(config=cfg)

        with patch("llm_service.auth.TokenExchanger") as mock_auth:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "j2"
            mock_auth.return_value = mock_exchanger

            response = service.call_llm(user_prompt="Test retry")

        assert response.content == "Success"
        assert mock_completion.call_count == 3

    @patch("litellm.completion")
    def test_no_retry_when_disabled(self, mock_completion):
        """max_attempts=1 時不重試。"""
        mock_completion.side_effect = ConnectionError("fail")

        cfg = LLMConfig(
            shared_config=SharedConfig(
                default_zone="DEV",
                auth_urls={"DEV": "https://auth/token"},
                retry=RetryConfig(max_attempts=1),
            ),
            model_configs={
                "QWEN3": ModelConfig(
                    j1_token="j1",
                    model_name="qwen3",
                    api_endpoints={"DEV": "https://dev/v1"},
                ),
            },
        )
        service = LLMService(config=cfg)

        with patch("llm_service.auth.TokenExchanger") as mock_auth:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "j2"
            mock_auth.return_value = mock_exchanger

            with pytest.raises(ConnectionError):
                service.call_llm(user_prompt="Test no retry")

        assert mock_completion.call_count == 1

    @patch("litellm.completion")
    def test_retry_exhausted(self, mock_completion):
        """所有 retry 用完仍失敗時 raise。"""
        mock_completion.side_effect = ConnectionError("persistent failure")

        cfg = LLMConfig(
            shared_config=SharedConfig(
                default_zone="DEV",
                auth_urls={"DEV": "https://auth/token"},
                retry=RetryConfig(max_attempts=2, wait_min=0.01, wait_max=0.02),
            ),
            model_configs={
                "QWEN3": ModelConfig(
                    j1_token="j1",
                    model_name="qwen3",
                    api_endpoints={"DEV": "https://dev/v1"},
                ),
            },
        )
        service = LLMService(config=cfg)

        with patch("llm_service.auth.TokenExchanger") as mock_auth:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "j2"
            mock_auth.return_value = mock_exchanger

            with pytest.raises(ConnectionError, match="persistent failure"):
                service.call_llm(user_prompt="Test retry exhausted")

        assert mock_completion.call_count == 2


# === Async 測試 ===


class TestLLMServiceAsync:
    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_acall_llm(self, mock_acompletion):
        mock_message = MagicMock()
        mock_message.content = "Async response"
        mock_message.reasoning_content = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = MagicMock(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        mock_resp.model = "qwen3"
        mock_acompletion.return_value = mock_resp

        cfg = LLMConfig(
            shared_config=SharedConfig(
                auth_urls={"DEV": "https://auth/token"},
            ),
            model_configs={
                "QWEN3": ModelConfig(
                    j1_token="j1",
                    model_name="qwen3",
                    api_endpoints={"DEV": "https://dev/v1"},
                ),
            },
        )
        service = LLMService(config=cfg)

        with patch("llm_service.auth.TokenExchanger") as mock_auth:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "j2"
            mock_auth.return_value = mock_exchanger

            response = await service.acall_llm(user_prompt="Hello async")

        assert response.content == "Async response"
        mock_acompletion.assert_called_once()


# === call_service 測試 ===


class TestCallService:
    def _make_service_with_svc(self):
        cfg = LLMConfig(
            shared_config=SharedConfig(
                default_zone="DEV",
                auth_urls={"DEV": "https://auth/token"},
            ),
            model_configs={
                "QWEN3": ModelConfig(
                    j1_token="j1",
                    model_name="qwen3",
                    api_endpoints={"DEV": "https://dev/v1"},
                ),
            },
            service_configs={
                "IMG": ServiceConfig(
                    j1_token="svc-j1",
                    api_endpoints={"DEV": "https://img-dev/v1/extract"},
                    timeout=60,
                ),
            },
        )
        return LLMService(config=cfg)

    @patch("httpx.Client")
    def test_call_service_basic(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "extracted", "confidence": 0.95}
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        service = self._make_service_with_svc()
        with patch("llm_service.auth.TokenExchanger") as mock_auth:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "svc-j2"
            mock_auth.return_value = mock_exchanger

            result = service.call_service("IMG", payload={"image": "base64data"})

        assert isinstance(result, AIServiceResponse)
        assert result.status_code == 200
        assert result.data["result"] == "extracted"
        assert result.raw_response["confidence"] == 0.95

    @patch("httpx.Client")
    def test_call_service_with_parser(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": {"text": "parsed content"}}
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        service = self._make_service_with_svc()
        with patch("llm_service.auth.TokenExchanger") as mock_auth:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "svc-j2"
            mock_auth.return_value = mock_exchanger

            result = service.call_service(
                "IMG",
                payload={"image": "base64data"},
                response_parser=lambda r: r["data"]["text"],
            )

        assert result.data == "parsed content"

    def test_call_service_not_found(self):
        service = self._make_service_with_svc()
        with pytest.raises(KeyError, match="not found"):
            service.call_service("NONEXISTENT")


# === TokenExchanger 測試 ===


class TestTokenExchanger:
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


# === AIServiceResponse 測試 ===


class TestAIServiceResponse:
    def test_default_values(self):
        resp = AIServiceResponse()
        assert resp.data is None
        assert resp.status_code == 200
        assert resp.latency_ms == 0.0
        assert resp.raw_response == {}

    def test_custom_values(self):
        resp = AIServiceResponse(
            data={"text": "hello"},
            status_code=201,
            latency_ms=150.5,
            raw_response={"text": "hello", "meta": {}},
        )
        assert resp.data["text"] == "hello"
        assert resp.status_code == 201

    def test_frozen(self):
        resp = AIServiceResponse(data="test")
        with pytest.raises(AttributeError):
            resp.data = "modified"  # type: ignore[misc]
