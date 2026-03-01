"""Loguru + MLflow 整合日誌設定。

- Loguru: 處理所有 process info 及錯誤訊息
- MLflow: 在 DEV 環境記錄完整 LLM response（透過 tracking/mlflow_logger.py）

Usage:
    from app.logger import setup_logging, get_logger

    setup_logging(cfg)  # 啟動時呼叫一次
    logger = get_logger(__name__)
    logger.info("Processing started")
"""

from __future__ import annotations

import sys

from loguru import logger as _loguru_logger

_initialized = False


def setup_logging(cfg=None) -> None:
    """初始化 Loguru 日誌系統。

    Args:
        cfg: Hydra DictConfig，需包含 cfg.logging.level。
             若為 None 則使用 INFO level。
    """
    global _initialized
    if _initialized:
        return

    # 取得 log level
    level = "INFO"
    if cfg is not None:
        level = getattr(cfg.logging, "level", "INFO") if hasattr(cfg, "logging") else "INFO"

    # 移除預設 handler
    _loguru_logger.remove()

    # stderr handler
    _loguru_logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    _initialized = True


def get_logger(name: str):
    """取得帶有模組名稱 context 的 logger。

    Args:
        name: 模組名稱，通常傳入 __name__。

    Returns:
        Bound loguru logger。
    """
    return _loguru_logger.bind(name=name)
