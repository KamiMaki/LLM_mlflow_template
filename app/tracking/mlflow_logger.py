"""MLflow logging — 記錄 LLM 呼叫、參數、指標與 artifacts。

所有函式在 MLflow 不可用時自動降級為 no-op。

Usage:
    from app.tracking.mlflow_logger import log_llm_call, log_params, log_metrics

    log_llm_call(prompt="...", response="...", model="gpt-4o",
                 token_usage={"total": 100}, latency_ms=500.0)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from app.logger import get_logger
from app.tracking.setup import is_mlflow_available

logger = get_logger(__name__)


def log_llm_call(
    prompt: str,
    response: str,
    model: str,
    token_usage: dict[str, int],
    latency_ms: float,
    params: dict[str, Any] | None = None,
) -> None:
    """記錄一次 LLM 呼叫的完整資訊到 MLflow。"""
    if not is_mlflow_available():
        return

    try:
        import mlflow

        if params:
            mlflow.log_params({f"llm.{k}": v for k, v in params.items()})
        mlflow.log_param("llm.model", model)
        for key, value in token_usage.items():
            mlflow.log_metric(f"tokens.{key}", value)
        mlflow.log_metric("latency_ms", latency_ms)
        mlflow.log_text(prompt, "prompt.txt")
        mlflow.log_text(response, "response.txt")
        logger.debug(f"Logged LLM call: model={model}, latency={latency_ms}ms")
    except Exception as e:
        warnings.warn(f"Failed to log LLM call: {e}", stacklevel=2)


def log_params(params: dict[str, Any]) -> None:
    """記錄參數到 MLflow。"""
    if not is_mlflow_available() or not params:
        return
    try:
        import mlflow
        mlflow.log_params(params)
        logger.debug(f"Logged {len(params)} params")
    except Exception as e:
        warnings.warn(f"Failed to log params: {e}", stacklevel=2)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """記錄指標到 MLflow。"""
    if not is_mlflow_available() or not metrics:
        return
    try:
        import mlflow
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged {len(metrics)} metrics")
    except Exception as e:
        warnings.warn(f"Failed to log metrics: {e}", stacklevel=2)


def log_artifact(local_path: str | Path, artifact_path: str | None = None) -> None:
    """記錄本地檔案為 MLflow artifact。"""
    if not is_mlflow_available():
        return
    try:
        import mlflow
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
        logger.debug(f"Logged artifact: {local_path}")
    except Exception as e:
        warnings.warn(f"Failed to log artifact: {e}", stacklevel=2)


def log_dict_artifact(data: dict[str, Any], filename: str) -> None:
    """記錄 dict 為 JSON artifact。"""
    if not is_mlflow_available():
        return
    try:
        import mlflow
        mlflow.log_dict(data, filename)
        logger.debug(f"Logged dict artifact: {filename}")
    except Exception as e:
        warnings.warn(f"Failed to log dict artifact: {e}", stacklevel=2)
