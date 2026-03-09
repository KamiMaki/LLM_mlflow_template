"""Deprecated LLMClient — 向後相容封裝，內部使用 litellm.completion()。

請改用 factory functions:
    - get_langchain_llm()  → LangGraph workflow
    - get_adk_model()      → Google ADK agent
    - get_litellm_kwargs() → 直接呼叫 litellm
    - get_openai_client()  → OpenAI SDK

Usage (deprecated):
    from llm_service import LLMClient

    client = LLMClient()
    resp = client.chat("Hello!", system_prompt="You are helpful.")
    print(resp.content)
"""

from __future__ import annotations

import warnings
from typing import Any

import litellm

from .config import LLMConfig
from .factory import get_litellm_kwargs
from .models import LLMResponse, TokenUsage


class LLMClient:
    """Deprecated: 請改用 get_langchain_llm() / get_adk_model() / get_litellm_kwargs()。

    保留向後相容，內部改為呼叫 litellm.completion()。
    """

    def __init__(self, config: LLMConfig | dict[str, Any] | None = None) -> None:
        warnings.warn(
            "LLMClient is deprecated. Use factory functions instead: "
            "get_langchain_llm(), get_adk_model(), get_litellm_kwargs()",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(config, dict):
            config = LLMConfig(**config)
        self._config = config or LLMConfig.from_yaml()
        self._kwargs = get_litellm_kwargs(self._config)

    @property
    def model(self) -> str:
        return self._config.model

    def set_model(self, model: str) -> None:
        """切換模型。"""
        self._config.model = model
        self._kwargs = get_litellm_kwargs(self._config)

    def chat(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        **kwargs: Any,
    ) -> LLMResponse:
        """同步呼叫 LLM（內部使用 litellm.completion）。"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = litellm.completion(**self._kwargs, messages=messages, **kwargs)

        choice = resp.choices[0]
        content = choice.message.content or ""
        usage = resp.usage

        return LLMResponse(
            content=content,
            model=resp.model or self._config.model,
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens or 0,
                completion_tokens=usage.completion_tokens or 0,
                total_tokens=usage.total_tokens or 0,
            ),
            latency_ms=0.0,
        )

    async def achat(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        **kwargs: Any,
    ) -> LLMResponse:
        """非同步呼叫 LLM（內部使用 litellm.acompletion）。"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = await litellm.acompletion(**self._kwargs, messages=messages, **kwargs)

        choice = resp.choices[0]
        content = choice.message.content or ""
        usage = resp.usage

        return LLMResponse(
            content=content,
            model=resp.model or self._config.model,
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens or 0,
                completion_tokens=usage.completion_tokens or 0,
                total_tokens=usage.total_tokens or 0,
            ),
            latency_ms=0.0,
        )
