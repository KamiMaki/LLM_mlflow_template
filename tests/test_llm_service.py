"""llm_service 單元測試 — LLMService + Config。"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from llm_service.config import (
    LLMConfig,
    ModelConfig,
    ResolvedModelConfig,
    SharedConfig,
)
from llm_service.models import LLMResponse, TokenUsage
from llm_service.service import LLMService


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
                        api_endpoints={"PROD": "https://prod/v1"},
                    ),
                },
            )

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

    def test_from_yaml_missing_file(self):
        cfg = LLMConfig.from_yaml("nonexistent.yaml")
        assert cfg.model_configs == {}


# === LLMService 測試 ===


class TestLLMService:
    def _make_service(self, default_model="", **model_overrides):
        """建立帶 mock config 的 LLMService。"""
        cfg = LLMConfig(
            default_model=default_model,
            shared_config=SharedConfig(
                default_zone="DEV",
                auth_urls={"DEV": "https://auth/token"},
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
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30

        mock_message = MagicMock()
        mock_message.content = "Hello back!"
        mock_message.reasoning_content = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = mock_usage
        mock_resp.model = "qwen3"
        mock_completion.return_value = mock_resp

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
        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_message.reasoning_content = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = MagicMock(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        mock_resp.model = "qwen3"
        mock_completion.return_value = mock_resp

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
        mock_message = MagicMock()
        mock_message.content = "Final answer"
        mock_message.reasoning_content = "Let me think..."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = MagicMock(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        mock_resp.model = "qwen3"
        mock_completion.return_value = mock_resp

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
        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_message.reasoning_content = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = MagicMock(prompt_tokens=5, completion_tokens=10, total_tokens=15)
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
        mock_message = MagicMock()
        mock_message.content = "Checked"
        mock_message.reasoning_content = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = MagicMock(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        mock_resp.model = "qwen3"
        mock_completion.return_value = mock_resp

        service = self._make_service()
        with patch("llm_service.auth.TokenExchanger") as mock_auth:
            mock_exchanger = MagicMock()
            mock_exchanger.get_token.return_value = "j2"
            mock_auth.return_value = mock_exchanger

            response = service.call_llm(
                prompt_template="Check: {{ data }}",
                prompt_variables={"data": "test"},
                system_prompt="QA",
            )

        messages = mock_completion.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "QA"}
        assert messages[1] == {"role": "user", "content": "Check: test"}


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
