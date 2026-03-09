"""LLM 配置核心 — 使用 Pydantic 管理 LLM 連線與參數設定。

支援 YAML 檔案載入 + 環境變數覆寫（ENV 優先級最高）。
支援 Token Exchange（J1→J2）：設定 auth 區段即可自動換 token。

Usage:
    from llm_service.config import LLMConfig

    cfg = LLMConfig.from_yaml("llm_config.yaml")
    cfg = LLMConfig(api_base="https://...", api_key="...", model="gpt-4o")

    # 需要 token exchange 時
    cfg = LLMConfig(
        api_base="https://llm-api.internal.com/v1",
        auth=AuthConfig(
            auth_url="https://auth.internal.com/token",
            auth_token="your-j1-token",
        ),
        model="gpt-4o",
    )
    api_key = cfg.resolve_api_key()  # 自動用 J1 換 J2
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AuthConfig(BaseModel):
    """Token Exchange 配置 — 用 J1 token 換取 J2 token。"""

    auth_url: str = Field(description="驗證端點 URL")
    auth_token: str = Field(default="", description="初始 token (J1)，可用 LLM_AUTH_TOKEN 環境變數覆寫")
    token_field: str = Field(default="access_token", description="回應 JSON 中 token 欄位名稱")
    expires_field: str = Field(default="expires_in", description="回應 JSON 中過期秒數欄位名稱")
    auth_method: str = Field(default="bearer", description="認證方式: bearer 或 body")
    extra_body: dict[str, Any] = Field(default_factory=dict, description="額外 POST body")
    extra_headers: dict[str, str] = Field(default_factory=dict, description="額外 request headers")
    buffer_seconds: int = Field(default=60, description="提前幾秒視為過期")

    @field_validator("extra_body", mode="before")
    @classmethod
    def _coerce_extra_body(cls, v):
        return v or {}

    @field_validator("extra_headers", mode="before")
    @classmethod
    def _coerce_auth_extra_headers(cls, v):
        return v or {}


class LLMConfig(BaseModel):
    """LLM 服務連線與參數配置。"""

    api_base: str = Field(default="http://localhost:11434/v1", description="API base URL")
    api_key: str = Field(default="", description="直接使用的 API key（不需 token exchange 時）")
    model: str = Field(default="gpt-4o", description="預設模型名稱")
    temperature: float = 0.7
    max_tokens: int = 4096
    extra_headers: dict[str, str] = Field(default_factory=dict, description="額外 HTTP headers")

    # Token Exchange（可選）
    auth: AuthConfig | None = Field(default=None, description="Token exchange 設定，設定後自動用 J1 換 J2")

    # 內部快取 TokenExchanger（不序列化）
    _exchanger: Any = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("extra_headers", mode="before")
    @classmethod
    def _coerce_extra_headers(cls, v):
        return v or {}

    def resolve_api_key(self) -> str:
        """取得有效的 API key。

        若設定了 auth（token exchange），自動用 J1 換取 J2 並回傳。
        否則直接回傳 api_key。

        Returns:
            可用於 LLM API 呼叫的 token 字串。
        """
        if self.auth is None:
            return self.api_key

        if self._exchanger is None:
            from llm_service.auth import TokenExchanger
            self._exchanger = TokenExchanger(
                auth_url=self.auth.auth_url,
                auth_token=self.auth.auth_token,
                token_field=self.auth.token_field,
                expires_field=self.auth.expires_field,
                auth_method=self.auth.auth_method,
                extra_body=self.auth.extra_body,
                extra_headers=self.auth.extra_headers,
                buffer_seconds=self.auth.buffer_seconds,
            )

        return self._exchanger.get_token()

    def clear_token_cache(self) -> None:
        """清除 token exchange 快取，下次會重新交換。"""
        if self._exchanger is not None:
            self._exchanger.clear_cache()

    @classmethod
    def from_yaml(cls, path: str | Path = "llm_config.yaml") -> LLMConfig:
        """從 YAML 載入設定，再用環境變數覆寫。

        環境變數對應:
            LLM_API_BASE    -> api_base
            LLM_API_KEY     -> api_key
            LLM_MODEL       -> model
            LLM_TEMPERATURE -> temperature
            LLM_MAX_TOKENS  -> max_tokens
            LLM_AUTH_TOKEN  -> auth.auth_token（J1 token）
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

        # auth.auth_token 環境變數覆寫
        auth_token_env = os.getenv("LLM_AUTH_TOKEN")
        if auth_token_env and "auth" in data and isinstance(data["auth"], dict):
            data["auth"]["auth_token"] = auth_token_env

        return cls(**data)
