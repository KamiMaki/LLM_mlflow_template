"""LLM Service 資料模型。

定義 LLM 回應的結構化資料型別。
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
