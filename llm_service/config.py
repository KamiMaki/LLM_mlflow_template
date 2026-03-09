"""LLM 配置核心 — 使用 Pydantic 管理 LLM 連線與參數設定。

支援 YAML 檔案載入 + 環境變數覆寫（ENV 優先級最高）。

Usage:
    from llm_service.config import LLMConfig

    cfg = LLMConfig.from_yaml("llm_config.yaml")
    cfg = LLMConfig(api_base="https://...", api_key="...", model="gpt-4o")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class LLMConfig(BaseModel):
    """LLM 服務連線與參數配置。"""

    api_base: str = Field(default="http://localhost:11434/v1", description="API base URL")
    api_key: str = Field(default="", description="認證 token")
    model: str = Field(default="gpt-4o", description="預設模型名稱")
    temperature: float = 0.7
    max_tokens: int = 4096
    extra_headers: dict[str, str] = Field(default_factory=dict, description="額外 HTTP headers")

    @field_validator("extra_headers", mode="before")
    @classmethod
    def _coerce_extra_headers(cls, v):
        return v or {}

    @classmethod
    def from_yaml(cls, path: str | Path = "llm_config.yaml") -> LLMConfig:
        """從 YAML 載入設定，再用環境變數覆寫。

        環境變數對應:
            LLM_API_BASE  -> api_base
            LLM_API_KEY   -> api_key
            LLM_MODEL     -> model
            LLM_TEMPERATURE -> temperature
            LLM_MAX_TOKENS  -> max_tokens
        """
        data: dict[str, Any] = {}
        path = Path(path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        env_map = {
            "LLM_API_BASE": "api_base",
            "LLM_API_KEY": "api_key",
            "LLM_MODEL": "model",
            "LLM_TEMPERATURE": "temperature",
            "LLM_MAX_TOKENS": "max_tokens",
        }
        for env_key, field_name in env_map.items():
            val = os.getenv(env_key)
            if val:
                data[field_name] = val

        return cls(**data)
