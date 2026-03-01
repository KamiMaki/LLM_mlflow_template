"""MLflow 初始化與 experiment 管理。

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
    """檢查 MLflow 是否可用。"""
    return _mlflow_available


def init_mlflow(cfg=None) -> None:
    """初始化 MLflow tracking。

    Args:
        cfg: Hydra DictConfig，需包含 cfg.mlflow 區段。
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

        tracking_uri = getattr(mlflow_cfg, "tracking_uri", "")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI: {tracking_uri}")

        experiment_name = getattr(mlflow_cfg, "experiment_name", "default")
        if experiment_name:
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment: {experiment_name}")

        _mlflow_available = True
        logger.info("MLflow initialized successfully")

    except ImportError:
        warnings.warn("mlflow package not installed. MLflow features disabled.", stacklevel=2)
        _mlflow_available = False
    except Exception as e:
        warnings.warn(f"Failed to initialize MLflow: {e}. MLflow features disabled.", stacklevel=2)
        logger.error(f"MLflow init error: {e}")
        _mlflow_available = False
