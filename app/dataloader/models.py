"""Dataloader 資料模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class LoaderConfig(BaseModel):
    """Data loader 設定。"""
    base_path: str = "./data"
    encoding: str = "utf-8"
    extra: dict[str, Any] = {}


class LoadedData(BaseModel):
    """載入資料的包裝物件。"""
    content: Any
    source: str
    content_type: str  # "json", "csv", "text", "markdown"
    metadata: dict[str, Any] = {}
