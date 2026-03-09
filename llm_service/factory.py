"""Model Factory — 輸出各框架原生物件。

統一使用 LiteLLM 作為底層，支援 LangGraph、Google ADK、直接呼叫等場景。

Usage:
    from llm_service import get_langchain_llm, get_adk_model, get_litellm_kwargs

    # LangGraph workflow
    llm = get_langchain_llm()
    graph_builder.add_node("llm", llm)

    # Google ADK agent
    model = get_adk_model()
    agent = Agent(model=model, ...)

    # 直接呼叫 litellm
    kwargs = get_litellm_kwargs()
    response = litellm.completion(**kwargs, messages=[...])
"""

from __future__ import annotations

from typing import Any

from llm_service.config import LLMConfig


def get_langchain_llm(config: LLMConfig | None = None, **overrides: Any):
    """回傳 ChatLiteLLM，直接用於 LangGraph。

    ChatLiteLLM 是 LangChain 的 BaseChatModel 實作，底層使用 litellm，
    支援所有 litellm 支援的 provider（OpenAI、Azure、Anthropic、Ollama 等）。

    Args:
        config: LLMConfig，None 時自動從 llm_config.yaml 載入。
        **overrides: 覆寫 config 中的參數（model, temperature, max_tokens）。

    Returns:
        ChatLiteLLM 實例，可直接用於 LangGraph StateGraph。
    """
    from langchain_litellm import ChatLiteLLM

    cfg = config or LLMConfig.from_yaml()
    return ChatLiteLLM(
        model=overrides.get("model", cfg.model),
        api_base=cfg.api_base,
        api_key=cfg.resolve_api_key(),
        temperature=overrides.get("temperature", cfg.temperature),
        max_tokens=overrides.get("max_tokens", cfg.max_tokens),
        model_kwargs={"extra_headers": cfg.extra_headers} if cfg.extra_headers else {},
    )


def get_adk_model(config: LLMConfig | None = None, **overrides: Any):
    """回傳 LiteLlm，直接用於 Google ADK Agent。

    Args:
        config: LLMConfig，None 時自動從 llm_config.yaml 載入。
        **overrides: 覆寫 config 中的參數（model）。

    Returns:
        google.adk.models.lite_llm.LiteLlm 實例。
    """
    from google.adk.models.lite_llm import LiteLlm

    cfg = config or LLMConfig.from_yaml()
    model_name = overrides.get("model", cfg.model)

    # LiteLlm 需要 provider/model 格式（如 openai/gpt-4o）
    if "/" not in model_name:
        model_name = f"openai/{model_name}"

    api_key = cfg.resolve_api_key()
    headers = {**cfg.extra_headers}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return LiteLlm(
        model=model_name,
        api_base=cfg.api_base,
        extra_headers=headers,
    )


def get_litellm_kwargs(config: LLMConfig | None = None, **overrides: Any) -> dict[str, Any]:
    """回傳 dict，用於 litellm.completion(**kwargs, messages=[...])。

    Args:
        config: LLMConfig，None 時自動從 llm_config.yaml 載入。
        **overrides: 覆寫 config 中的參數。

    Returns:
        可直接展開傳入 litellm.completion() 的 dict。
    """
    cfg = config or LLMConfig.from_yaml()
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


def get_openai_client(config: LLMConfig | None = None):
    """回傳 OpenAI client，用於直接使用 OpenAI SDK。

    Args:
        config: LLMConfig，None 時自動從 llm_config.yaml 載入。

    Returns:
        openai.OpenAI 實例。
    """
    from openai import OpenAI

    cfg = config or LLMConfig.from_yaml()
    return OpenAI(
        base_url=cfg.api_base,
        api_key=cfg.resolve_api_key(),
        default_headers=cfg.extra_headers or None,
    )
