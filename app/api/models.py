"""FastAPI base request/response models — 使用者可繼承擴充。"""

from __future__ import annotations

from pydantic import BaseModel


class LLMRequest(BaseModel):
    """基礎 LLM 請求模型。使用者可繼承並新增欄位。"""
    system_prompt: str | None = None
    user_prompt: str


class LLMResponseModel(BaseModel):
    """基礎 LLM 回應模型。使用者可繼承並新增欄位。"""
    content: str
    model: str = ""
    latency_ms: float = 0.0
