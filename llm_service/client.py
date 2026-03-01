"""Mock LLM Client — 模擬 LLM SDK 行為。

此為開發用 mock 實作，提供與正式 llm_service SDK 相同的介面。
正式部署時替換為真正的 SDK import 即可。

Usage:
    from llm_service import LLMClient

    client = LLMClient(config={"base_url": "http://localhost:11434/v1", "model": "gpt-4o"})
    resp = client.chat(system_prompt="You are helpful.", user_prompt="Hello!")
    print(resp.content)
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .models import LLMResponse, TokenUsage

_DEFAULT_CONFIG: dict[str, Any] = {
    "base_url": "http://localhost:11434/v1",
    "model": "gpt-4o",
    "timeout": 30,
    "max_retries": 3,
    "temperature": 0.7,
    "auth_token": "",
}


class LLMClient:
    """LLM 呼叫客戶端（OpenAI-compatible API）。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = {**_DEFAULT_CONFIG, **(config or {})}
        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    # -- 設定 ----------------------------------------------------------------

    def load_config(self, config: dict[str, Any]) -> None:
        """載入或更新設定。"""
        self._config.update(config)
        self._close_clients()

    def set_model(self, model: str) -> None:
        """切換模型。"""
        self._config["model"] = model

    @property
    def model(self) -> str:
        return self._config["model"]

    # -- 同步呼叫 --------------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """同步呼叫 LLM chat completion。"""
        payload = self._build_payload(system_prompt, user_prompt, **kwargs)
        headers = self._build_headers()

        client = self._get_sync_client()
        start = time.perf_counter()

        resp = self._do_request(client, payload, headers)

        latency_ms = (time.perf_counter() - start) * 1000
        return self._parse_response(resp, latency_ms)

    # -- 非同步呼叫 ------------------------------------------------------------

    async def achat(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """非同步呼叫 LLM chat completion。"""
        payload = self._build_payload(system_prompt, user_prompt, **kwargs)
        headers = self._build_headers()

        client = self._get_async_client()
        start = time.perf_counter()

        resp = await self._do_async_request(client, payload, headers)

        latency_ms = (time.perf_counter() - start) * 1000
        return self._parse_response(resp, latency_ms)

    # -- 內部方法 --------------------------------------------------------------

    def _build_payload(
        self, system_prompt: str, user_prompt: str, **kwargs: Any
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return {
            "model": kwargs.get("model", self._config["model"]),
            "messages": messages,
            "temperature": kwargs.get("temperature", self._config["temperature"]),
            **{k: v for k, v in kwargs.items() if k not in ("model", "temperature")},
        }

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        token = self._config.get("auth_token", "")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _get_sync_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._config["timeout"])
        return self._client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self._config["timeout"])
        return self._async_client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _do_request(
        self, client: httpx.Client, payload: dict, headers: dict
    ) -> dict[str, Any]:
        url = f"{self._config['base_url']}/chat/completions"
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _do_async_request(
        self, client: httpx.AsyncClient, payload: dict, headers: dict
    ) -> dict[str, Any]:
        url = f"{self._config['base_url']}/chat/completions"
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()

    def _parse_response(self, data: dict[str, Any], latency_ms: float) -> LLMResponse:
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")

        usage_data = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return LLMResponse(
            content=content,
            model=data.get("model", self._config["model"]),
            usage=usage,
            latency_ms=latency_ms,
        )

    def _close_clients(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
        if self._async_client:
            # async client should be closed with await, but for config reload we just drop it
            self._async_client = None

    def close(self) -> None:
        """關閉所有 HTTP 連線。"""
        self._close_clients()
