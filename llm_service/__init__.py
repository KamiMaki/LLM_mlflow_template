"""LLM Service — Config 管理 + Model Factory。

提供統一配置，輸出各框架原生物件（ChatLiteLLM、Google ADK LiteLlm、litellm kwargs 等）。
支援多模型 + 多環境（DEV/TEST/STG/PROD）配置與 J1→J2 token exchange。
"""

from .config import (
    AuthConfig,
    LLMConfig,
    ModelConfig,
    ResolvedModelConfig,
    SharedConfig,
)
from .factory import (
    build_multimodal_messages,
    get_adk_model,
    get_langchain_llm,
    get_litellm_kwargs,
    get_openai_client,
)
from .models import LLMResponse, TokenUsage

# Deprecated — 向後相容
from .client import LLMClient

__all__ = [
    "AuthConfig",
    "LLMConfig",
    "ModelConfig",
    "ResolvedModelConfig",
    "SharedConfig",
    "build_multimodal_messages",
    "get_langchain_llm",
    "get_adk_model",
    "get_litellm_kwargs",
    "get_openai_client",
    "LLMClient",
    "LLMResponse",
    "TokenUsage",
]
