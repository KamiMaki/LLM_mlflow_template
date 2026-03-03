"""MLflow 初始化 — autolog + set_active_model。

Usage:
    from app.tracking.setup import init_mlflow, is_mlflow_available

    init_mlflow(cfg)
    if is_mlflow_available():
        ...
"""

from __future__ import annotations

import warnings

from app.logger import get_logger

logger = get_logger(__name__)

_mlflow_available = False


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
        logger.info("MLflow is disabled in config")
        _mlflow_available = False
        return

    try:
        import mlflow

        # Tracking URI
        tracking_uri = getattr(mlflow_cfg, "tracking_uri", "")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI: {tracking_uri}")

        # Experiment
        experiment_name = getattr(mlflow_cfg, "experiment_name", "default")
        if experiment_name:
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment: {experiment_name}")

        # Active Model（Model-Centric 設計）
        model_name = getattr(mlflow_cfg, "model_name", "")
        if model_name:
            mlflow.set_active_model(name=model_name)
            logger.info(f"MLflow active model: {model_name}")

        # 啟用 LangChain/LangGraph autolog
        try:
            mlflow.langchain.autolog()
            logger.info("MLflow LangChain autolog enabled")
        except Exception:
            logger.debug("LangChain autolog not available")

        _mlflow_available = True
        logger.info("MLflow initialized successfully")

    except ImportError:
        warnings.warn("mlflow package not installed. MLflow features disabled.", stacklevel=2)
        _mlflow_available = False
    except Exception as e:
        warnings.warn(f"Failed to initialize MLflow: {e}. MLflow features disabled.", stacklevel=2)
        logger.error(f"MLflow init error: {e}")
        _mlflow_available = False
