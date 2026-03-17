"""LLM 配置核心 — 多模型 + 多環境（DEV/TEST/STG/PROD）支援。

使用 Pydantic 管理 LLM 連線與參數設定，支援 J1→J2 token exchange。

Usage:
    from llm_service.config import LLMConfig

    cfg = LLMConfig.from_yaml("llm_config.yaml")
    resolved = cfg.resolve("QWEN3")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

VALID_ZONES = {"DEV", "TEST", "STG", "PROD"}


class RetryConfig(BaseModel):
    """Retry 設定 — 搭配 tenacity 使用。"""

    max_attempts: int = Field(default=1, description="最大重試次數（1 = 不重試）")
    wait_multiplier: float = Field(default=1.0, description="指數退避乘數")
    wait_min: float = Field(default=2.0, description="最小等待秒數")
    wait_max: float = Field(default=10.0, description="最大等待秒數")


class SharedConfig(BaseModel):
    """全域共用設定 — J1→J2 auth URL（分 zone）與 token exchange 參數。"""

    default_zone: str = Field(default="DEV", description="預設 zone（DEV/TEST/STG/PROD）")
    auth_urls: dict[str, str] = Field(
        default_factory=dict,
        description="各 zone 的 J1→J2 Token Exchange 端點",
    )

    token_field: str = Field(default="access_token")
    expires_field: str = Field(default="expires_in")
    auth_method: str = Field(default="bearer")
    buffer_seconds: int = Field(default=60)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    extra_headers: dict[str, str] = Field(default_factory=dict)

    j1_token_path: str = Field(
        default="",
        description="J1 token 檔案路徑（非 DEV 環境從 pod 讀取）",
    )
    retry: RetryConfig = Field(default_factory=RetryConfig)

    @field_validator("default_zone", mode="before")
    @classmethod
    def _normalize_zone(cls, v: str) -> str:
        v = str(v).upper()
        if v not in VALID_ZONES:
            raise ValueError(f"Invalid zone '{v}', must be one of {VALID_ZONES}")
        return v

    @field_validator("extra_body", mode="before")
    @classmethod
    def _coerce_extra_body(cls, v):
        return v or {}

    @field_validator("extra_headers", mode="before")
    @classmethod
    def _coerce_extra_headers(cls, v):
        return v or {}

    def get_auth_url(self, zone: str | None = None) -> str:
        """取得指定 zone 的 auth URL。"""
        z = (zone or self.default_zone).upper()
        if z not in self.auth_urls:
            raise ValueError(
                f"No auth_url configured for zone '{z}'. "
                f"Available zones: {list(self.auth_urls.keys())}"
            )
        return self.auth_urls[z]


class ModelConfig(BaseModel):
    """單一模型設定 — J1 token、各 zone endpoint、模型名稱與超參數。"""

    j1_token: str = Field(default="", description="J1 token，建議用環境變數傳入")
    model_name: str = Field(description="LiteLLM 模型名稱")

    api_endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="各 zone 的 API endpoint",
    )

    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.7)
    top_p: float | None = Field(default=None)
    top_k: int | None = Field(default=None)
    frequency_penalty: float | None = Field(default=None)
    presence_penalty: float | None = Field(default=None)
    stop: list[str] | None = Field(default=None)

    extra_headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("extra_headers", mode="before")
    @classmethod
    def _coerce_extra_headers(cls, v):
        return v or {}

    def get_api_endpoint(self, zone: str) -> str:
        """取得指定 zone 的 API endpoint。"""
        z = zone.upper()
        if z not in self.api_endpoints:
            raise ValueError(
                f"No api_endpoint configured for zone '{z}'. "
                f"Available zones: {list(self.api_endpoints.keys())}"
            )
        return self.api_endpoints[z]

    def get_hyperparams(self) -> dict[str, Any]:
        """取得所有已設定的超參數（排除 None）。"""
        params: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.top_k is not None:
            params["top_k"] = self.top_k
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        if self.stop is not None:
            params["stop"] = self.stop
        return params


class ServiceConfig(BaseModel):
    """自訂 AI 服務設定（非 LLM），如圖片辨識、文件擷取等。"""

    j1_token: str = Field(default="", description="J1 token，建議用環境變數傳入")
    api_endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="各 zone 的 API endpoint",
    )
    timeout: int = Field(default=30, description="HTTP 請求超時秒數")
    extra_headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("extra_headers", mode="before")
    @classmethod
    def _coerce_extra_headers(cls, v):
        return v or {}

    def get_api_endpoint(self, zone: str) -> str:
        """取得指定 zone 的 API endpoint。"""
        z = zone.upper()
        if z not in self.api_endpoints:
            raise ValueError(
                f"No api_endpoint configured for zone '{z}'. "
                f"Available zones: {list(self.api_endpoints.keys())}"
            )
        return self.api_endpoints[z]


class ResolvedModelConfig(BaseModel):
    """已解析的模型設定 — 可直接用於 litellm.completion()。"""

    model_name: str
    api_base: str
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    extra_headers: dict[str, str] = Field(default_factory=dict)
    hyperparams: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LLMConfig(BaseModel):
    """LLM 服務配置 — 多模型 + 多環境。"""

    default_model: str = Field(default="", description="預設使用的模型別名")
    shared_config: SharedConfig = Field(default_factory=SharedConfig)
    model_configs: dict[str, ModelConfig] = Field(default_factory=dict)
    service_configs: dict[str, ServiceConfig] = Field(default_factory=dict)

    _exchangers: dict[str, Any] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_config(self):
        """驗證設定。"""
        if self.model_configs or self.service_configs:
            if not self.shared_config.auth_urls:
                raise ValueError(
                    "shared_config.auth_urls is required when using model_configs or service_configs. "
                    "At least one zone auth URL must be configured."
                )
        if self.model_configs:
            default_zone = self.shared_config.default_zone
            for name, mcfg in self.model_configs.items():
                if default_zone not in mcfg.api_endpoints:
                    raise ValueError(
                        f"Model '{name}' missing api_endpoint for default zone '{default_zone}'. "
                        f"Available zones: {list(mcfg.api_endpoints.keys())}"
                    )
            if self.default_model and self.default_model not in self.model_configs:
                raise ValueError(
                    f"default_model '{self.default_model}' not found in model_configs. "
                    f"Available models: {list(self.model_configs.keys())}"
                )
        return self

    def list_models(self) -> list[str]:
        """列出所有已設定的模型別名。"""
        return list(self.model_configs.keys())

    def get_model_config(self, model_alias: str) -> ModelConfig:
        """取得指定模型的原始設定。"""
        if model_alias not in self.model_configs:
            raise KeyError(
                f"Model '{model_alias}' not found. "
                f"Available models: {self.list_models()}"
            )
        return self.model_configs[model_alias]

    def get_service_config(self, service_name: str) -> ServiceConfig:
        """取得指定 AI 服務的原始設定。"""
        if service_name not in self.service_configs:
            raise KeyError(
                f"Service '{service_name}' not found. "
                f"Available services: {list(self.service_configs.keys())}"
            )
        return self.service_configs[service_name]

    def _read_j1_from_file(self) -> str:
        """從檔案路徑讀取 J1 token（非 DEV 環境使用 pod 掛載路徑）。"""
        path = self.shared_config.j1_token_path
        if not path:
            return ""
        try:
            token = Path(path).read_text(encoding="utf-8").strip()
            if token:
                logger.debug(f"J1 token read from file: {path}")
            return token
        except (OSError, FileNotFoundError):
            logger.debug(f"J1 token file not found or unreadable: {path}")
            return ""

    def _resolve_j1_token(self, alias: str, token_from_config: str) -> str:
        """從多個來源解析 J1 token。

        優先順序：env var > config value > file path。
        """
        return (
            os.getenv(f"LLM_AUTH_TOKEN_{alias}")
            or os.getenv("LLM_AUTH_TOKEN")
            or token_from_config
            or self._read_j1_from_file()
        )

    def resolve(self, model_alias: str, zone: str | None = None) -> ResolvedModelConfig:
        """解析指定模型 + zone 的完整設定，自動執行 J1→J2 token exchange。"""
        z = (zone or os.getenv("LLM_ZONE") or self.shared_config.default_zone).upper()

        mcfg = self.get_model_config(model_alias)
        api_base = mcfg.get_api_endpoint(z)
        auth_url = self.shared_config.get_auth_url(z)

        j1_token = self._resolve_j1_token(model_alias, mcfg.j1_token)
        if not j1_token:
            raise ValueError(
                f"No J1 token for model '{model_alias}'. "
                f"Set LLM_AUTH_TOKEN_{model_alias} or LLM_AUTH_TOKEN env var, "
                f"configure j1_token in llm_config.yaml, "
                f"or set shared_config.j1_token_path for file-based token."
            )

        api_key = self._exchange_token(model_alias, auth_url, j1_token)
        merged_headers = {**self.shared_config.extra_headers, **mcfg.extra_headers}

        return ResolvedModelConfig(
            model_name=mcfg.model_name,
            api_base=api_base,
            api_key=api_key,
            temperature=mcfg.temperature,
            max_tokens=mcfg.max_tokens,
            extra_headers=merged_headers,
            hyperparams=mcfg.get_hyperparams(),
        )

    def _exchange_token(self, cache_key: str, auth_url: str, j1_token: str) -> str:
        """執行 J1→J2 token exchange（自動快取）。"""
        from llm_service.auth import TokenExchanger

        if cache_key not in self._exchangers:
            self._exchangers[cache_key] = TokenExchanger(
                auth_url=auth_url,
                auth_token=j1_token,
                token_field=self.shared_config.token_field,
                expires_field=self.shared_config.expires_field,
                auth_method=self.shared_config.auth_method,
                extra_body=self.shared_config.extra_body,
                extra_headers=self.shared_config.extra_headers,
                buffer_seconds=self.shared_config.buffer_seconds,
            )

        return self._exchangers[cache_key].get_token()

    def resolve_service(
        self, service_name: str, zone: str | None = None
    ) -> tuple[str, str, dict[str, str]]:
        """解析 AI 服務的 endpoint、J2 token、headers。

        Returns:
            (endpoint, j2_token, merged_headers)
        """
        z = (zone or os.getenv("LLM_ZONE") or self.shared_config.default_zone).upper()
        scfg = self.get_service_config(service_name)
        endpoint = scfg.get_api_endpoint(z)
        auth_url = self.shared_config.get_auth_url(z)

        j1_token = self._resolve_j1_token(service_name, scfg.j1_token)
        if not j1_token:
            raise ValueError(
                f"No J1 token for service '{service_name}'. "
                f"Set LLM_AUTH_TOKEN_{service_name} or LLM_AUTH_TOKEN env var, "
                f"or configure j1_token in service_configs."
            )

        j2_token = self._exchange_token(f"svc_{service_name}", auth_url, j1_token)
        merged_headers = {**self.shared_config.extra_headers, **scfg.extra_headers}
        return endpoint, j2_token, merged_headers

    def clear_token_cache(self, model_alias: str | None = None) -> None:
        """清除 token exchange 快取。"""
        if model_alias:
            if model_alias in self._exchangers:
                self._exchangers[model_alias].clear_cache()
        else:
            for exchanger in self._exchangers.values():
                exchanger.clear_cache()
            self._exchangers.clear()

    @classmethod
    def from_yaml(cls, path: str | Path = "llm_config.yaml") -> LLMConfig:
        """從 YAML 載入設定，環境變數可覆寫。

        環境變數:
            LLM_ZONE              -> shared_config.default_zone
            LLM_AUTH_TOKEN        -> 各模型的 fallback J1 token
            LLM_AUTH_TOKEN_XXX    -> 指定模型 XXX 的 J1 token
        """
        data: dict[str, Any] = {}
        path = Path(path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        zone_env = os.getenv("LLM_ZONE")
        if zone_env:
            if "shared_config" not in data:
                data["shared_config"] = {}
            data["shared_config"]["default_zone"] = zone_env

        if "model_configs" in data:
            for model_alias in data["model_configs"]:
                token_env = os.getenv(f"LLM_AUTH_TOKEN_{model_alias}")
                if token_env:
                    data["model_configs"][model_alias]["j1_token"] = token_env

        if "service_configs" in data:
            for svc_name in data["service_configs"]:
                token_env = os.getenv(f"LLM_AUTH_TOKEN_{svc_name}")
                if token_env:
                    data["service_configs"][svc_name]["j1_token"] = token_env

        return cls(**data)
