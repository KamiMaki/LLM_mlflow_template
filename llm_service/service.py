"""LLMService — 統一 LLM 呼叫入口。

封裝 config 載入、prompt 組裝、message 建構與 LLM 呼叫，
外部只需 call_llm() 即可完成所有操作。

Usage:
    from llm_service import LLMService

    service = LLMService()
    response = service.call_llm(
        user_prompt="請檢查這份資料",
        system_prompt="你是資料檢查助手",
    )
    print(response.content)

    # 切換模型
    service.set_model("QWEN3VL")
    response = service.call_llm(
        user_prompt="描述圖片",
        image_base64=img_b64,
    )

    # Prompt template
    response = service.call_llm(
        prompt_template="檢查：{{ data }}",
        prompt_variables={"data": "..."},
    )
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import logging

import litellm

# 抑制 LiteLLM 冗長的 console 輸出（Provider List、model cost map 等）
litellm.suppress_debug_info = True
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from .config import LLMConfig, ResolvedModelConfig
from .models import LLMResponse, TokenUsage


class LLMService:
    """統一 LLM 呼叫入口 — 封裝 config、prompt 組裝與 LLM 呼叫。"""

    def __init__(
        self,
        config: LLMConfig | None = None,
        config_path: str | Path = "llm_config.yaml",
        zone: str | None = None,
    ) -> None:
        """初始化 LLMService。

        Args:
            config: 直接傳入 LLMConfig，優先於 config_path。
            config_path: YAML 配置檔路徑，預設 llm_config.yaml。
            zone: 環境 zone（DEV/TEST/STG/PROD），None 時使用 config 預設值。
        """
        self._config = config or LLMConfig.from_yaml(config_path)
        self._zone = zone
        self._current_model: str = ""

        # 使用 default_model 或第一個 model
        if self._config.default_model:
            self._current_model = self._config.default_model
        elif self._config.model_configs:
            self._current_model = next(iter(self._config.model_configs))

    @property
    def current_model(self) -> str:
        """目前使用的模型別名。"""
        return self._current_model

    @property
    def config(self) -> LLMConfig:
        """取得 LLMConfig。"""
        return self._config

    def set_model(self, model_alias: str) -> LLMService:
        """設定當前模型。

        Args:
            model_alias: 模型別名（如 "QWEN3", "QWEN3VL"）。

        Returns:
            self，支援 chain call。
        """
        # 驗證 model 存在
        self._config.get_model_config(model_alias)
        self._current_model = model_alias
        return self

    def call_llm(
        self,
        user_prompt: str = "",
        system_prompt: str = "",
        image_base64: str | list[str] | None = None,
        prompt_template: str | None = None,
        prompt_variables: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """同步呼叫 LLM。

        Args:
            user_prompt: 使用者 prompt。
            system_prompt: 系統 prompt。
            image_base64: base64 圖片（單張或多張），可帶 data URI prefix。
            prompt_template: prompt 模板，使用 {{ var }} 語法。
            prompt_variables: 模板變數。
            **kwargs: 覆寫 config 中的超參數（temperature, max_tokens 等）。

        Returns:
            LLMResponse。
        """
        messages = self._build_messages(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            image_base64=image_base64,
            prompt_template=prompt_template,
            prompt_variables=prompt_variables,
        )
        resolved = self._resolve()
        completion_kwargs = self._build_completion_kwargs(resolved, **kwargs)

        start = time.perf_counter()
        resp = litellm.completion(messages=messages, **completion_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        return self._parse_response(resp, latency_ms)

    async def acall_llm(
        self,
        user_prompt: str = "",
        system_prompt: str = "",
        image_base64: str | list[str] | None = None,
        prompt_template: str | None = None,
        prompt_variables: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """非同步呼叫 LLM。參數同 call_llm。"""
        messages = self._build_messages(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            image_base64=image_base64,
            prompt_template=prompt_template,
            prompt_variables=prompt_variables,
        )
        resolved = self._resolve()
        completion_kwargs = self._build_completion_kwargs(resolved, **kwargs)

        start = time.perf_counter()
        resp = await litellm.acompletion(messages=messages, **completion_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        return self._parse_response(resp, latency_ms)

    # --- 內部方法 ---

    def _resolve(self) -> ResolvedModelConfig:
        """解析當前模型的 config。"""
        if not self._current_model:
            raise ValueError(
                "No model set. Call set_model() first or ensure model_configs "
                "is configured in llm_config.yaml."
            )
        return self._config.resolve(self._current_model, zone=self._zone)

    def _build_messages(
        self,
        user_prompt: str = "",
        system_prompt: str = "",
        image_base64: str | list[str] | None = None,
        prompt_template: str | None = None,
        prompt_variables: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """組裝 messages — 處理 prompt template 與 multimodal。"""
        # 1. Prompt template 格式化
        if prompt_template:
            text = prompt_template
            for key, value in (prompt_variables or {}).items():
                text = text.replace("{{ " + key + " }}", str(value))
            user_prompt = text

        if not user_prompt and not system_prompt:
            raise ValueError("Must provide user_prompt, system_prompt, or prompt_template.")

        # 2. 組裝 messages
        messages: list[dict[str, Any]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if not image_base64:
            if user_prompt:
                messages.append({"role": "user", "content": user_prompt})
        else:
            # Multimodal: text + images
            images = image_base64 if isinstance(image_base64, list) else [image_base64]
            content: list[dict[str, Any]] = []
            if user_prompt:
                content.append({"type": "text", "text": user_prompt})
            for img in images:
                if not img.startswith("data:"):
                    img = f"data:image/png;base64,{img}"
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img},
                })
            messages.append({"role": "user", "content": content})

        return messages

    def _build_completion_kwargs(
        self, resolved: ResolvedModelConfig, **overrides: Any
    ) -> dict[str, Any]:
        """建構 litellm.completion 參數。"""
        model_name = resolved.model_name
        if "/" not in model_name:
            model_name = f"openai/{model_name}"

        kwargs: dict[str, Any] = {
            "model": model_name,
            "api_base": resolved.api_base,
            "api_key": resolved.api_key,
            "temperature": overrides.pop("temperature", resolved.temperature),
            "max_tokens": overrides.pop("max_tokens", resolved.max_tokens),
        }

        if resolved.extra_headers:
            kwargs["extra_headers"] = resolved.extra_headers

        # 加入額外超參數
        for k, v in resolved.hyperparams.items():
            if k not in ("temperature", "max_tokens") and k not in kwargs:
                kwargs[k] = v

        # 使用者覆寫
        kwargs.update(overrides)

        return kwargs

    def _parse_response(self, resp: Any, latency_ms: float) -> LLMResponse:
        """解析 litellm response 為 LLMResponse。"""
        choice = resp.choices[0]
        content = choice.message.content or ""
        usage = resp.usage

        # 提取 reasoning_content（如果有）
        reasoning_content = getattr(choice.message, "reasoning_content", None)

        return LLMResponse(
            content=content,
            model=resp.model or self._current_model,
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens or 0,
                completion_tokens=usage.completion_tokens or 0,
                total_tokens=usage.total_tokens or 0,
            ),
            latency_ms=latency_ms,
            reasoning_content=reasoning_content,
        )
