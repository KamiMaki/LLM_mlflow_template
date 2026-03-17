"""LLM / AI Service 資料模型。

定義 LLM 回應與通用 AI 服務回應的結構化資料型別。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TokenUsage:
    """Token 使用量統計。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class LLMResponse:
    """LLM 回應結果。"""

    content: str = ""
    model: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    reasoning_content: str | None = None


@dataclass(frozen=True)
class AIServiceResponse:
    """通用 AI 服務回應結果（非 LLM）。

    用於圖片辨識、文件擷取等自訂 AI 服務。
    """

    data: Any = None
    status_code: int = 200
    latency_ms: float = 0.0
    raw_response: dict[str, Any] = field(default_factory=dict)
