"""Loguru 日誌 + MLflow 初始化。

Usage:
    from app.logger import setup_logging, get_logger, init_mlflow, is_mlflow_available

    setup_logging(cfg)
    init_mlflow(cfg)
    logger = get_logger(__name__)
"""

from __future__ import annotations

import sys
import warnings

from loguru import logger as _loguru_logger

_initialized = False
_mlflow_available = False


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


# --- MLflow ---

def is_mlflow_available() -> bool:
    """檢查 MLflow 是否已初始化且可用。"""
    return _mlflow_available


def init_mlflow(cfg=None) -> None:
    """初始化 MLflow：設定 tracking URI、experiment、active model，並啟用 autolog。

    Args:
        cfg: 設定物件，需包含 cfg.mlflow 區段。
    """
    global _mlflow_available

    if cfg is None:
        _mlflow_available = False
        return

    mlflow_cfg = getattr(cfg, "mlflow", None)
    if mlflow_cfg is None or not getattr(mlflow_cfg, "enabled", False):
        _loguru_logger.info("MLflow is disabled in config")
        _mlflow_available = False
        return

    try:
        import mlflow

        tracking_uri = getattr(mlflow_cfg, "tracking_uri", "")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            _loguru_logger.info(f"MLflow tracking URI: {tracking_uri}")

        experiment_name = getattr(mlflow_cfg, "experiment_name", "default")
        if experiment_name:
            mlflow.set_experiment(experiment_name)
            _loguru_logger.info(f"MLflow experiment: {experiment_name}")

        model_name = getattr(mlflow_cfg, "model_name", "")
        if model_name:
            mlflow.set_active_model(name=model_name)
            _loguru_logger.info(f"MLflow active model: {model_name}")

        try:
            mlflow.litellm.autolog()
            _loguru_logger.info("MLflow LiteLLM autolog enabled")
        except Exception:
            _loguru_logger.debug("LiteLLM autolog not available")

        try:
            mlflow.langchain.autolog()
            _loguru_logger.info("MLflow LangChain autolog enabled")
        except Exception:
            _loguru_logger.debug("LangChain autolog not available")

        _mlflow_available = True
        _loguru_logger.info("MLflow initialized successfully")

    except ImportError:
        warnings.warn("mlflow package not installed. MLflow features disabled.", stacklevel=2)
        _mlflow_available = False
    except Exception as e:
        warnings.warn(f"Failed to initialize MLflow: {e}. MLflow features disabled.", stacklevel=2)
        _loguru_logger.error(f"MLflow init error: {e}")
        _mlflow_available = False
