"""設定管理 — 載入 YAML 設定並提供 singleton 存取。

Usage:
    from app.utils.config import init_config, get_config

    cfg = init_config()                    # 載入預設 config/config.yaml
    cfg = init_config("path/to/my.yaml")   # 載入指定檔案
    print(cfg.api.port)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


class AppConfig(dict):
    """支援 dot-notation 存取的 config dict。

    Example:
        cfg = AppConfig({"api": {"port": 8000}})
        cfg.api.port  # 8000
        cfg["api"]["port"]  # 8000
    """

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")
        if isinstance(value, dict) and not isinstance(value, AppConfig):
            value = AppConfig(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


_active_config: AppConfig | None = None


def _resolve_env_vars(data: Any) -> Any:
    """遞迴解析 ${ENV_VAR} 和 ${ENV_VAR:default} 語法。"""
    if isinstance(data, str):
        def _replace(match: re.Match) -> str:
            expr = match.group(1)
            if ":" in expr:
                var, default = expr.split(":", 1)
                return os.environ.get(var.strip(), default.strip())
            return os.environ.get(expr.strip(), match.group(0))
        return re.sub(r"\$\{([^}]+)\}", _replace, data)
    elif isinstance(data, dict):
        return {k: _resolve_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_resolve_env_vars(item) for item in data]
    return data


def init_config(config_path: str | None = None) -> AppConfig:
    """載入 YAML 設定檔。

    Args:
        config_path: YAML 檔案路徑。預設為專案根目錄下的 config/config.yaml。

    Returns:
        AppConfig（支援 dot-notation 的 dict）。
    """
    global _active_config

    if config_path is None:
        root = Path(__file__).resolve().parent.parent.parent
        config_path = str(root / "config" / "config.yaml")

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    resolved = _resolve_env_vars(raw)
    _active_config = AppConfig(resolved)
    return _active_config


def get_config() -> AppConfig:
    """取得目前的設定 singleton。

    Returns:
        目前生效的 AppConfig。

    Raises:
        RuntimeError: 若尚未呼叫 init_config()。
    """
    if _active_config is None:
        raise RuntimeError(
            "設定尚未初始化。請先呼叫 init_config() 或在 FastAPI lifespan 中初始化。"
        )
    return _active_config


def reset_config() -> None:
    """重置設定 singleton（僅供測試使用）。"""
    global _active_config
    _active_config = None
