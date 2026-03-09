"""LLM Service — Config 管理 + Model Factory。

提供統一配置，輸出各框架原生物件（ChatLiteLLM、Google ADK LiteLlm、litellm kwargs 等）。
"""

from .config import LLMConfig
from .factory import get_adk_model, get_langchain_llm, get_litellm_kwargs, get_openai_client
from .models import LLMResponse, TokenUsage

# Deprecated — 向後相容
from .client import LLMClient

__all__ = [
    "LLMConfig",
    "get_langchain_llm",
    "get_adk_model",
    "get_litellm_kwargs",
    "get_openai_client",
    "LLMClient",
    "LLMResponse",
    "TokenUsage",
]
