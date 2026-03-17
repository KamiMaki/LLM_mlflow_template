"""LLM / AI Service — 統一呼叫入口。

Usage:
    from llm_service import LLMService

    service = LLMService()

    # LLM 呼叫
    response = service.call_llm(user_prompt="Hello", system_prompt="Be helpful")
    print(response.content)

    # 自訂 AI 服務
    result = service.call_service("IMAGE_EXTRACTION", payload={"image": img_b64})
    print(result.data)
"""

from .config import LLMConfig, RetryConfig, ServiceConfig
from .models import AIServiceResponse, LLMResponse, TokenUsage
from .service import LLMService

__all__ = [
    "AIServiceResponse",
    "LLMConfig",
    "LLMResponse",
    "LLMService",
    "RetryConfig",
    "ServiceConfig",
    "TokenUsage",
]
