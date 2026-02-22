"""Tests for LLM client module."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from llm_framework.llm_client import LLMClient, LLMError, LLMResponse, TokenUsage


class TestLLMClient:
    """Test LLM client initialization and basic operations."""

    def test_init_with_config(self, sample_config):
        """Test LLMClient initialization with explicit config."""
        client = LLMClient(config=sample_config)
        assert client._config == sample_config
        assert client._llm == sample_config.llm

    def test_init_with_global_config(self, sample_config):
        """Test LLMClient initialization using global config."""
        client = LLMClient()
        assert client._config is not None

    def test_http_client_headers(self, sample_config):
        """Test that HTTP client has correct auth headers."""
        client = LLMClient(config=sample_config)
        assert "Authorization" in client._http.headers
        assert client._http.headers["Authorization"] == f"Bearer {sample_config.llm.auth_token}"
        assert client._http.headers["Content-Type"] == "application/json"

    def test_http_client_timeout(self, sample_config):
        """Test that HTTP client uses config timeout."""
        client = LLMClient(config=sample_config)
        assert client._http.timeout.read == sample_config.llm.timeout


class TestChatMethod:
    """Test synchronous chat method."""

    def test_chat_success(self, sample_config, mock_httpx_response):
        """Test successful chat call."""
        client = LLMClient(config=sample_config)

        with patch.object(client._http, "post", return_value=mock_httpx_response):
            response = client.chat([{"role": "user", "content": "Hello"}])

            assert isinstance(response, LLMResponse)
            assert response.content == "Hello! How can I help you?"
            assert response.model == "gpt-4o"
            assert response.usage.prompt_tokens == 10
            assert response.usage.completion_tokens == 8
            assert response.usage.total_tokens == 18
            assert response.latency_ms > 0

    def test_chat_with_model_override(self, sample_config, mock_httpx_response):
        """Test chat with model parameter override."""
        client = LLMClient(config=sample_config)

        with patch.object(client._http, "post", return_value=mock_httpx_response) as mock_post:
            client.chat([{"role": "user", "content": "Test"}], model="gpt-3.5-turbo")

            # Verify the payload includes overridden model
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert payload["model"] == "gpt-3.5-turbo"

    def test_chat_with_temperature_override(self, sample_config, mock_httpx_response):
        """Test chat with temperature parameter override."""
        client = LLMClient(config=sample_config)

        with patch.object(client._http, "post", return_value=mock_httpx_response) as mock_post:
            client.chat([{"role": "user", "content": "Test"}], temperature=0.9)

            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert payload["temperature"] == 0.9

    def test_chat_with_max_tokens(self, sample_config, mock_httpx_response):
        """Test chat with max_tokens parameter."""
        client = LLMClient(config=sample_config)

        with patch.object(client._http, "post", return_value=mock_httpx_response) as mock_post:
            client.chat([{"role": "user", "content": "Test"}], max_tokens=100)

            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert payload["max_tokens"] == 100

    def test_chat_builds_correct_payload(self, sample_config, mock_httpx_response):
        """Test that chat builds correct API payload."""
        client = LLMClient(config=sample_config)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        with patch.object(client._http, "post", return_value=mock_httpx_response) as mock_post:
            client.chat(messages)

            call_args = mock_post.call_args
            payload = call_args[1]["json"]

            assert payload["model"] == sample_config.llm.default_model
            assert payload["messages"] == messages
            assert payload["temperature"] == sample_config.llm.temperature

    def test_chat_endpoint_extraction(self, sample_config, mock_httpx_response):
        """Test correct endpoint extraction from URL."""
        client = LLMClient(config=sample_config)

        with patch.object(client._http, "post", return_value=mock_httpx_response) as mock_post:
            client.chat([{"role": "user", "content": "Test"}])

            # Should call /chat/completions endpoint
            call_args = mock_post.call_args
            assert call_args[0][0] == "/chat/completions"


class TestChatRetry:
    """Test retry logic in chat method."""

    def test_chat_retry_on_http_error(self, sample_config):
        """Test that chat retries on HTTP errors."""
        client = LLMClient(config=sample_config)

        # Mock to fail twice, then succeed
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test",
            "model": "gpt-4o",
            "choices": [{"message": {"content": "Success"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        mock_response.raise_for_status.return_value = None

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.HTTPStatusError("Server error", request=MagicMock(), response=MagicMock())
            return mock_response

        with patch.object(client._http, "post", side_effect=side_effect):
            with patch("time.sleep"):  # Skip actual sleep
                response = client.chat([{"role": "user", "content": "Test"}])

        assert call_count == 2
        assert response.content == "Success"

    def test_chat_fails_after_max_retries(self, sample_config):
        """Test that chat raises LLMError after max retries."""
        client = LLMClient(config=sample_config)

        with patch.object(
            client._http,
            "post",
            side_effect=httpx.HTTPStatusError("Server error", request=MagicMock(), response=MagicMock()),
        ):
            with patch("time.sleep"):
                with pytest.raises(LLMError, match="failed after .* retries"):
                    client.chat([{"role": "user", "content": "Test"}])

    def test_chat_retry_backoff(self, sample_config):
        """Test exponential backoff in retry logic."""
        client = LLMClient(config=sample_config)

        sleep_times = []

        def mock_sleep(duration):
            sleep_times.append(duration)

        with patch.object(
            client._http,
            "post",
            side_effect=httpx.RequestError("Connection error", request=MagicMock()),
        ):
            with patch("time.sleep", side_effect=mock_sleep):
                with pytest.raises(LLMError):
                    client.chat([{"role": "user", "content": "Test"}])

        # Should have retried (max_retries - 1) times
        assert len(sleep_times) == sample_config.llm.max_retries - 1
        # Backoff should increase: 1.0, 2.0, 4.0, ...
        assert sleep_times[0] == 1.0
        assert sleep_times[1] == 2.0


class TestResponseParsing:
    """Test LLM response parsing."""

    def test_parse_response_with_full_data(self, sample_config):
        """Test parsing response with all fields."""
        client = LLMClient(config=sample_config)

        data = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Test response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 10,
                "total_tokens": 25,
            },
        }

        response = client._parse_response(data, latency_ms=200.0)

        assert response.content == "Test response"
        assert response.model == "gpt-4o"
        assert response.usage.prompt_tokens == 15
        assert response.usage.completion_tokens == 10
        assert response.usage.total_tokens == 25
        assert response.latency_ms == 200.0
        assert response.raw_response == data

    def test_parse_response_with_missing_fields(self, sample_config):
        """Test parsing response with missing optional fields."""
        client = LLMClient(config=sample_config)

        data = {
            "choices": [],
        }

        response = client._parse_response(data, latency_ms=100.0)

        assert response.content == ""
        assert response.model == "unknown"
        assert response.usage.prompt_tokens == 0
        assert response.usage.completion_tokens == 0
        assert response.usage.total_tokens == 0


class TestBackoffCalculation:
    """Test exponential backoff calculation."""

    def test_backoff_defaults(self):
        """Test backoff with default parameters."""
        assert LLMClient._backoff(1) == 1.0
        assert LLMClient._backoff(2) == 2.0
        assert LLMClient._backoff(3) == 4.0
        assert LLMClient._backoff(4) == 8.0

    def test_backoff_custom_base(self):
        """Test backoff with custom base."""
        assert LLMClient._backoff(1, base=2.0) == 2.0
        assert LLMClient._backoff(2, base=2.0) == 4.0
        assert LLMClient._backoff(3, base=2.0) == 8.0

    def test_backoff_custom_factor(self):
        """Test backoff with custom factor."""
        assert LLMClient._backoff(1, factor=3.0) == 1.0
        assert LLMClient._backoff(2, factor=3.0) == 3.0
        assert LLMClient._backoff(3, factor=3.0) == 9.0


class TestContextManager:
    """Test LLMClient as context manager."""

    def test_context_manager_closes_http(self, sample_config):
        """Test that context manager closes HTTP client."""
        with patch("httpx.Client") as mock_http_class:
            mock_http = MagicMock()
            mock_http_class.return_value = mock_http

            with LLMClient(config=sample_config) as client:
                pass

            mock_http.close.assert_called_once()
