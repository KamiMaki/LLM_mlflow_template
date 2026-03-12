"""Model Factory — 輸出各框架原生物件，支援多模型切換與 base64 圖片。

統一使用 LiteLLM 作為底層，支援 LangGraph、Google ADK、直接呼叫等場景。
新版支援 LLMConfig.resolve(model_alias) 取得已解析設定。

Usage:
    from llm_service import get_langchain_llm, get_litellm_kwargs

    # 多模型模式
    cfg = LLMConfig.from_yaml()
    llm = get_langchain_llm(config=cfg, model_alias="QWEN3")

    # 帶圖片呼叫
    from llm_service.factory import build_multimodal_messages
    messages = build_multimodal_messages(
        user_text="描述這張圖片",
        image_base64="data:image/png;base64,...",
        system_prompt="你是助手",
    )
"""

from __future__ import annotations

from typing import Any

from llm_service.config import LLMConfig, ResolvedModelConfig


def _resolve_config(
    config: LLMConfig | None,
    model_alias: str | None,
    zone: str | None,
    **overrides: Any,
) -> tuple[LLMConfig, ResolvedModelConfig | None]:
    """內部：解析 config + model_alias，回傳 (cfg, resolved_or_None)。"""
    cfg = config or LLMConfig.from_yaml()
    resolved = None
    if model_alias and cfg.model_configs:
        resolved = cfg.resolve(model_alias, zone=zone)
    elif cfg.model_configs and not model_alias:
        first = next(iter(cfg.model_configs))
        resolved = cfg.resolve(first, zone=zone)
    return cfg, resolved


def get_langchain_llm(
    config: LLMConfig | None = None,
    *,
    model_alias: str | None = None,
    zone: str | None = None,
    **overrides: Any,
):
    """回傳 ChatLiteLLM，直接用於 LangGraph。

    Args:
        config: LLMConfig，None 時自動從 llm_config.yaml 載入。
        model_alias: 模型別名（如 "QWEN3"），用於多模型模式。
        zone: 環境 zone（DEV/TEST/STG/PROD）。
        **overrides: 覆寫 config 中的參數（model, temperature, max_tokens）。

    Returns:
        ChatLiteLLM 實例，可直接用於 LangGraph StateGraph。
    """
    from langchain_litellm import ChatLiteLLM

    cfg, resolved = _resolve_config(config, model_alias, zone, **overrides)

    if resolved:
        return ChatLiteLLM(
            model=overrides.get("model", resolved.model_name),
            api_base=resolved.api_base,
            api_key=resolved.api_key,
            temperature=overrides.get("temperature", resolved.temperature),
            max_tokens=overrides.get("max_tokens", resolved.max_tokens),
            model_kwargs={"extra_headers": resolved.extra_headers} if resolved.extra_headers else {},
        )

    # 向後相容
    return ChatLiteLLM(
        model=overrides.get("model", cfg.model),
        api_base=cfg.api_base,
        api_key=cfg.resolve_api_key(),
        temperature=overrides.get("temperature", cfg.temperature),
        max_tokens=overrides.get("max_tokens", cfg.max_tokens),
        model_kwargs={"extra_headers": cfg.extra_headers} if cfg.extra_headers else {},
    )


def get_adk_model(
    config: LLMConfig | None = None,
    *,
    model_alias: str | None = None,
    zone: str | None = None,
    **overrides: Any,
):
    """回傳 LiteLlm，直接用於 Google ADK Agent。

    Args:
        config: LLMConfig，None 時自動從 llm_config.yaml 載入。
        model_alias: 模型別名。
        zone: 環境 zone。
        **overrides: 覆寫 config 中的參數（model）。

    Returns:
        google.adk.models.lite_llm.LiteLlm 實例。
    """
    from google.adk.models.lite_llm import LiteLlm

    cfg, resolved = _resolve_config(config, model_alias, zone, **overrides)

    if resolved:
        model_name = overrides.get("model", resolved.model_name)
        if "/" not in model_name:
            model_name = f"openai/{model_name}"
        headers = {**resolved.extra_headers}
        if resolved.api_key:
            headers["Authorization"] = f"Bearer {resolved.api_key}"
        return LiteLlm(model=model_name, api_base=resolved.api_base, extra_headers=headers)

    # 向後相容
    model_name = overrides.get("model", cfg.model)
    if "/" not in model_name:
        model_name = f"openai/{model_name}"
    api_key = cfg.resolve_api_key()
    headers = {**cfg.extra_headers}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return LiteLlm(model=model_name, api_base=cfg.api_base, extra_headers=headers)


def get_litellm_kwargs(
    config: LLMConfig | None = None,
    *,
    model_alias: str | None = None,
    zone: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """回傳 dict，用於 litellm.completion(**kwargs, messages=[...])。

    Args:
        config: LLMConfig，None 時自動從 llm_config.yaml 載入。
        model_alias: 模型別名。
        zone: 環境 zone。
        **overrides: 覆寫 config 中的參數。

    Returns:
        可直接展開傳入 litellm.completion() 的 dict。
    """
    cfg, resolved = _resolve_config(config, model_alias, zone, **overrides)

    if resolved:
        model_name = overrides.get("model", resolved.model_name)
        if "/" not in model_name:
            model_name = f"openai/{model_name}"
        result = {
            "model": model_name,
            "api_base": resolved.api_base,
            "api_key": resolved.api_key,
            "temperature": overrides.get("temperature", resolved.temperature),
            "max_tokens": overrides.get("max_tokens", resolved.max_tokens),
            "extra_headers": resolved.extra_headers,
        }
        for k, v in resolved.hyperparams.items():
            if k not in ("temperature", "max_tokens") and k not in result:
                result[k] = v
        return result

    # 向後相容
    model_name = overrides.get("model", cfg.model)
    if "/" not in model_name:
        model_name = f"openai/{model_name}"
    return {
        "model": model_name,
        "api_base": cfg.api_base,
        "api_key": cfg.resolve_api_key(),
        "temperature": overrides.get("temperature", cfg.temperature),
        "max_tokens": overrides.get("max_tokens", cfg.max_tokens),
        "extra_headers": cfg.extra_headers,
    }


def get_openai_client(
    config: LLMConfig | None = None,
    *,
    model_alias: str | None = None,
    zone: str | None = None,
):
    """回傳 OpenAI client，用於直接使用 OpenAI SDK。

    Args:
        config: LLMConfig，None 時自動從 llm_config.yaml 載入。
        model_alias: 模型別名。
        zone: 環境 zone。

    Returns:
        openai.OpenAI 實例。
    """
    from openai import OpenAI

    cfg, resolved = _resolve_config(config, model_alias, zone)

    if resolved:
        return OpenAI(
            base_url=resolved.api_base,
            api_key=resolved.api_key,
            default_headers=resolved.extra_headers or None,
        )

    return OpenAI(
        base_url=cfg.api_base,
        api_key=cfg.resolve_api_key(),
        default_headers=cfg.extra_headers or None,
    )


# --- Base64 圖片支援 ---

def build_multimodal_messages(
    user_text: str,
    image_base64: str | list[str] | None = None,
    system_prompt: str = "",
) -> list[dict[str, Any]]:
    """建構包含 base64 圖片的 OpenAI-compatible messages。

    支援單張或多張圖片，圖片可帶 data URI prefix 或純 base64。

    Args:
        user_text: 使用者文字訊息。
        image_base64: base64 圖片字串或列表，可帶 "data:image/png;base64,..." 前綴。
        system_prompt: 系統提示詞。

    Returns:
        OpenAI messages 格式的 list[dict]。
    """
    messages: list[dict[str, Any]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if not image_base64:
        messages.append({"role": "user", "content": user_text})
        return messages

    images = image_base64 if isinstance(image_base64, list) else [image_base64]
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]

    for img in images:
        if not img.startswith("data:"):
            img = f"data:image/png;base64,{img}"
        content.append({
            "type": "image_url",
            "image_url": {"url": img},
        })

    messages.append({"role": "user", "content": content})
    return messages
