"""LLMService — 統一 AI 服務呼叫入口。

封裝 config 載入、prompt 組裝、message 建構與 LLM / AI 服務呼叫，
支援 MLflow trace（含敏感資料過濾）、tenacity retry、自訂 AI 服務。

Usage:
    from llm_service import LLMService

    service = LLMService()

    # LLM 呼叫
    response = service.call_llm(
        user_prompt="請檢查這份資料",
        system_prompt="你是資料檢查助手",
    )
    print(response.content)
    print(response.reasoning_content)  # reasoning model 的思考過程

    # 切換模型
    service.set_model("QWEN3VL")
    response = service.call_llm(user_prompt="描述圖片", image_base64=img_b64)

    # 自訂 AI 服務
    result = service.call_service("IMAGE_EXTRACTION", payload={"image": img_b64})
    print(result.data)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import litellm
from loguru import logger
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    Retrying,
    stop_after_attempt,
    wait_exponential,
)

from .config import LLMConfig, ResolvedModelConfig
from .models import AIServiceResponse, LLMResponse, TokenUsage
from .trace import sanitize_completion_kwargs, sanitize_dict, trace_span

# 抑制 LiteLLM 冗長的 console 輸出（Provider List、model cost map 等）
litellm.suppress_debug_info = True
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def _log_before_retry(retry_state: RetryCallState) -> None:
    """Tenacity before_sleep callback — 每次 retry 前記錄 log。"""
    attempt = retry_state.attempt_number
    exc = retry_state.outcome.exception() if retry_state.outcome else "unknown"
    wait = retry_state.next_action.sleep if retry_state.next_action else 0  # type: ignore[union-attr]
    logger.warning(
        f"Retry attempt {attempt}, waiting {wait:.1f}s, error: {exc}"
    )


class LLMService:
    """統一 AI 服務呼叫入口 — 封裝 config、prompt 組裝、LLM 呼叫與自訂 AI 服務。"""

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

    # --- LLM 呼叫 ---

    def call_llm(
        self,
        user_prompt: str = "",
        system_prompt: str = "",
        image_base64: str | list[str] | None = None,
        prompt_template: str | None = None,
        prompt_variables: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """同步呼叫 LLM（含 retry + MLflow trace）。

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

        sanitized = sanitize_completion_kwargs(completion_kwargs)
        sanitized["messages"] = messages

        with trace_span("LLMService.call_llm", inputs=sanitized) as span:
            start = time.perf_counter()
            resp = self._execute_with_retry(
                litellm.completion, messages=messages, **completion_kwargs
            )
            latency_ms = (time.perf_counter() - start) * 1000

            result = self._parse_response(resp, latency_ms)

            if span:
                outputs: dict[str, Any] = {
                    "content": result.content,
                    "model": result.model,
                    "usage": {
                        "prompt_tokens": result.usage.prompt_tokens,
                        "completion_tokens": result.usage.completion_tokens,
                        "total_tokens": result.usage.total_tokens,
                    },
                    "latency_ms": result.latency_ms,
                }
                if result.reasoning_content:
                    outputs["reasoning_content"] = result.reasoning_content
                span.set_outputs(outputs)

        return result

    async def acall_llm(
        self,
        user_prompt: str = "",
        system_prompt: str = "",
        image_base64: str | list[str] | None = None,
        prompt_template: str | None = None,
        prompt_variables: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """非同步呼叫 LLM（含 retry + MLflow trace）。參數同 call_llm。"""
        messages = self._build_messages(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            image_base64=image_base64,
            prompt_template=prompt_template,
            prompt_variables=prompt_variables,
        )
        resolved = self._resolve()
        completion_kwargs = self._build_completion_kwargs(resolved, **kwargs)

        sanitized = sanitize_completion_kwargs(completion_kwargs)
        sanitized["messages"] = messages

        with trace_span("LLMService.acall_llm", inputs=sanitized) as span:
            start = time.perf_counter()
            resp = await self._aexecute_with_retry(
                litellm.acompletion, messages=messages, **completion_kwargs
            )
            latency_ms = (time.perf_counter() - start) * 1000

            result = self._parse_response(resp, latency_ms)

            if span:
                outputs: dict[str, Any] = {
                    "content": result.content,
                    "model": result.model,
                    "usage": {
                        "prompt_tokens": result.usage.prompt_tokens,
                        "completion_tokens": result.usage.completion_tokens,
                        "total_tokens": result.usage.total_tokens,
                    },
                    "latency_ms": result.latency_ms,
                }
                if result.reasoning_content:
                    outputs["reasoning_content"] = result.reasoning_content
                span.set_outputs(outputs)

        return result

    # --- 自訂 AI 服務呼叫 ---

    def call_service(
        self,
        service_name: str,
        *,
        payload: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        response_parser: Callable[[dict[str, Any]], Any] | None = None,
        **kwargs: Any,
    ) -> AIServiceResponse:
        """同步呼叫自訂 AI 服務（含 retry + MLflow trace）。

        Args:
            service_name: 服務名稱（對應 service_configs 中的 key）。
            payload: JSON request body。
            files: 檔案上傳（httpx files 格式）。
            response_parser: 自訂回應解析函式，接收 response dict，回傳解析後的資料。
            **kwargs: 額外 httpx 請求參數。

        Returns:
            AIServiceResponse。
        """
        endpoint, j2_token, headers = self._config.resolve_service(
            service_name, zone=self._zone
        )
        scfg = self._config.get_service_config(service_name)
        headers["Authorization"] = f"Bearer {j2_token}"

        safe_inputs = sanitize_dict({
            "service": service_name,
            "endpoint": endpoint,
            "payload": payload,
            "headers": headers,
        })

        with trace_span(f"AIService.{service_name}", inputs=safe_inputs) as span:
            start = time.perf_counter()

            def _do_request() -> httpx.Response:
                with httpx.Client(timeout=scfg.timeout) as client:
                    if files:
                        resp = client.post(
                            endpoint, data=payload, files=files, headers=headers, **kwargs
                        )
                    else:
                        resp = client.post(
                            endpoint, json=payload, headers=headers, **kwargs
                        )
                    resp.raise_for_status()
                    return resp

            http_resp = self._execute_with_retry(_do_request)
            latency_ms = (time.perf_counter() - start) * 1000

            raw = http_resp.json()
            parsed = response_parser(raw) if response_parser else raw

            result = AIServiceResponse(
                data=parsed,
                status_code=http_resp.status_code,
                latency_ms=latency_ms,
                raw_response=raw,
            )

            if span:
                span.set_outputs({
                    "status_code": result.status_code,
                    "latency_ms": result.latency_ms,
                    "data_preview": str(result.data)[:500],
                })

        return result

    async def acall_service(
        self,
        service_name: str,
        *,
        payload: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        response_parser: Callable[[dict[str, Any]], Any] | None = None,
        **kwargs: Any,
    ) -> AIServiceResponse:
        """非同步呼叫自訂 AI 服務（含 retry + MLflow trace）。參數同 call_service。"""
        endpoint, j2_token, headers = self._config.resolve_service(
            service_name, zone=self._zone
        )
        scfg = self._config.get_service_config(service_name)
        headers["Authorization"] = f"Bearer {j2_token}"

        safe_inputs = sanitize_dict({
            "service": service_name,
            "endpoint": endpoint,
            "payload": payload,
            "headers": headers,
        })

        with trace_span(f"AIService.{service_name}", inputs=safe_inputs) as span:
            start = time.perf_counter()

            async def _do_request() -> httpx.Response:
                async with httpx.AsyncClient(timeout=scfg.timeout) as client:
                    if files:
                        resp = await client.post(
                            endpoint, data=payload, files=files, headers=headers, **kwargs
                        )
                    else:
                        resp = await client.post(
                            endpoint, json=payload, headers=headers, **kwargs
                        )
                    resp.raise_for_status()
                    return resp

            http_resp = await self._aexecute_with_retry(_do_request)
            latency_ms = (time.perf_counter() - start) * 1000

            raw = http_resp.json()
            parsed = response_parser(raw) if response_parser else raw

            result = AIServiceResponse(
                data=parsed,
                status_code=http_resp.status_code,
                latency_ms=latency_ms,
                raw_response=raw,
            )

            if span:
                span.set_outputs({
                    "status_code": result.status_code,
                    "latency_ms": result.latency_ms,
                    "data_preview": str(result.data)[:500],
                })

        return result

    # --- 內部方法 ---

    def _resolve(self) -> ResolvedModelConfig:
        """解析當前模型的 config。"""
        if not self._current_model:
            raise ValueError(
                "No model set. Call set_model() first or ensure model_configs "
                "is configured in llm_config.yaml."
            )
        return self._config.resolve(self._current_model, zone=self._zone)

    def _execute_with_retry(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """使用 tenacity retry 執行函式。"""
        retry_cfg = self._config.shared_config.retry
        if retry_cfg.max_attempts <= 1:
            return fn(*args, **kwargs)

        retryer = Retrying(
            stop=stop_after_attempt(retry_cfg.max_attempts),
            wait=wait_exponential(
                multiplier=retry_cfg.wait_multiplier,
                min=retry_cfg.wait_min,
                max=retry_cfg.wait_max,
            ),
            before_sleep=_log_before_retry,
            reraise=True,
        )
        return retryer(fn, *args, **kwargs)

    async def _aexecute_with_retry(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """使用 tenacity retry 執行非同步函式。"""
        retry_cfg = self._config.shared_config.retry
        if retry_cfg.max_attempts <= 1:
            return await fn(*args, **kwargs)

        retryer = AsyncRetrying(
            stop=stop_after_attempt(retry_cfg.max_attempts),
            wait=wait_exponential(
                multiplier=retry_cfg.wait_multiplier,
                min=retry_cfg.wait_min,
                max=retry_cfg.wait_max,
            ),
            before_sleep=_log_before_retry,
            reraise=True,
        )
        return await retryer(fn, *args, **kwargs)

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
