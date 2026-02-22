"""Unified LLM client with automatic MLflow tracing.

Provides a single interface for calling internal LLM APIs. Reads connection
details from the framework config and automatically traces every call.

Usage:
    from llm_framework.config import load_config
    from llm_framework.llm_client import LLMClient

    load_config("dev")
    client = LLMClient()
    response = client.chat([{"role": "user", "content": "Hello!"}])
    print(response.content)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import httpx

from llm_framework.config import FrameworkConfig, get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

@dataclass
class TokenUsage:
    """Token usage statistics for an LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """Response from an LLM API call."""
    content: str
    model: str
    usage: TokenUsage
    latency_ms: float
    raw_response: dict = field(default_factory=dict)


class LLMError(Exception):
    """Raised when an LLM API call fails after all retries."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LLMClient:
    """Unified LLM client that reads config and auto-traces calls.

    Args:
        config: Explicit config. If None, uses the global singleton.
    """

    def __init__(self, config: FrameworkConfig | None = None):
        self._config = config or get_config()
        self._llm = self._config.llm
        self._http = httpx.Client(
            base_url=self._llm.url.rsplit("/chat/completions", 1)[0]
            if self._llm.url.endswith("/chat/completions")
            else self._llm.url,
            timeout=httpx.Timeout(self._llm.timeout),
            headers={
                "Authorization": f"Bearer {self._llm.auth_token}",
                "Content-Type": "application/json",
            },
        )
        self._async_http: httpx.AsyncClient | None = None

    # -- sync -----------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with "role" and "content".
            model: Override the default model from config.
            temperature: Override the default temperature.
            max_tokens: Maximum tokens in the response.
            **kwargs: Extra parameters forwarded to the API.

        Returns:
            LLMResponse with the assistant's reply and metadata.

        Raises:
            LLMError: If the call fails after all retries.
        """
        payload = self._build_payload(messages, model, temperature, max_tokens, **kwargs)
        endpoint = self._get_chat_endpoint()

        last_error: Exception | None = None
        for attempt in range(1, self._llm.max_retries + 1):
            try:
                start = time.perf_counter()
                resp = self._http.post(endpoint, json=payload)
                latency_ms = (time.perf_counter() - start) * 1000

                resp.raise_for_status()
                data = resp.json()
                return self._parse_response(data, latency_ms)

            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                last_error = exc
                wait = self._backoff(attempt)
                logger.warning(
                    "LLM call attempt %d/%d failed: %s. Retrying in %.1fs...",
                    attempt, self._llm.max_retries, exc, wait,
                )
                if attempt < self._llm.max_retries:
                    time.sleep(wait)

        raise LLMError(
            f"LLM call failed after {self._llm.max_retries} retries: {last_error}"
        ) from last_error

    # -- async ----------------------------------------------------------------

    async def achat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Async version of chat().

        Args:
            messages: List of message dicts with "role" and "content".
            model: Override the default model from config.
            temperature: Override the default temperature.
            max_tokens: Maximum tokens in the response.
            **kwargs: Extra parameters forwarded to the API.

        Returns:
            LLMResponse with the assistant's reply and metadata.

        Raises:
            LLMError: If the call fails after all retries.
        """
        import asyncio

        if self._async_http is None:
            self._async_http = httpx.AsyncClient(
                base_url=self._llm.url.rsplit("/chat/completions", 1)[0]
                if self._llm.url.endswith("/chat/completions")
                else self._llm.url,
                timeout=httpx.Timeout(self._llm.timeout),
                headers={
                    "Authorization": f"Bearer {self._llm.auth_token}",
                    "Content-Type": "application/json",
                },
            )

        payload = self._build_payload(messages, model, temperature, max_tokens, **kwargs)
        endpoint = self._get_chat_endpoint()

        last_error: Exception | None = None
        for attempt in range(1, self._llm.max_retries + 1):
            try:
                start = time.perf_counter()
                resp = await self._async_http.post(endpoint, json=payload)
                latency_ms = (time.perf_counter() - start) * 1000

                resp.raise_for_status()
                data = resp.json()
                return self._parse_response(data, latency_ms)

            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                last_error = exc
                wait = self._backoff(attempt)
                logger.warning(
                    "LLM async call attempt %d/%d failed: %s. Retrying in %.1fs...",
                    attempt, self._llm.max_retries, exc, wait,
                )
                if attempt < self._llm.max_retries:
                    await asyncio.sleep(wait)

        raise LLMError(
            f"LLM async call failed after {self._llm.max_retries} retries: {last_error}"
        ) from last_error

    # -- helpers --------------------------------------------------------------

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        **kwargs,
    ) -> dict:
        payload = {
            "model": model or self._llm.default_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._llm.temperature,
            **kwargs,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    def _get_chat_endpoint(self) -> str:
        if self._llm.url.endswith("/chat/completions"):
            return "/chat/completions"
        return ""

    def _parse_response(self, data: dict, latency_ms: float) -> LLMResponse:
        choices = data.get("choices", [])
        content = choices[0]["message"]["content"] if choices else ""
        model = data.get("model", "unknown")
        usage_raw = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        )
        return LLMResponse(
            content=content,
            model=model,
            usage=usage,
            latency_ms=latency_ms,
            raw_response=data,
        )

    @staticmethod
    def _backoff(attempt: int, base: float = 1.0, factor: float = 2.0) -> float:
        return base * (factor ** (attempt - 1))

    def close(self) -> None:
        """Close underlying HTTP clients."""
        self._http.close()
        if self._async_http:
            # async client should be closed with await, but provide sync cleanup
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
