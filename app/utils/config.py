"""Hydra 設定管理 — 載入、組合及提供設定給各模組。

使用 Hydra Compose API 在程式啟動時載入設定，
並提供 singleton 存取方式給其他模組使用。

Usage:
    from app.utils.config import init_config, get_config

    cfg = init_config(overrides=["env=prod"])
    llm_cfg = get_config().llm
"""

from __future__ import annotations

import os
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

_active_config: DictConfig | None = None


def init_config(
    config_dir: str | None = None,
    config_name: str = "config",
    overrides: list[str] | None = None,
) -> DictConfig:
    """使用 Hydra Compose API 初始化設定。

    Args:
        config_dir: 設定目錄的絕對路徑。預設為專案根目錄下的 config/。
        config_name: 主設定檔名（不含 .yaml）。
        overrides: Hydra override 列表，例如 ["env=prod", "llm.timeout=60"]。

    Returns:
        合併後的 OmegaConf DictConfig。
    """
    global _active_config

    if config_dir is None:
        config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config")

    config_dir = os.path.abspath(config_dir)

    # 清除先前的 Hydra 狀態
    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides or [])

    _active_config = cfg
    return cfg


def get_config() -> DictConfig:
    """取得目前的設定 singleton。

    Returns:
        目前生效的 DictConfig。

    Raises:
        RuntimeError: 若尚未呼叫 init_config()。
    """
    if _active_config is None:
        raise RuntimeError(
            "設定尚未初始化。請先呼叫 init_config() 或在 FastAPI lifespan 中初始化。"
        )
    return _active_config


def get_llm_config() -> dict[str, Any]:
    """提取 LLM config 區段為純 dict，供 llm_service SDK 使用。

    Returns:
        解析後的 LLM 設定 dict。
    """
    cfg = get_config()
    return OmegaConf.to_container(cfg.llm, resolve=True)


def reset_config() -> None:
    """重置設定 singleton（僅供測試使用）。"""
    global _active_config
    _active_config = None
    GlobalHydra.instance().clear()
