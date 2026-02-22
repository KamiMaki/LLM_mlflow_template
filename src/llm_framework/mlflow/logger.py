"""MLflow logging wrapper with graceful degradation.

Provides convenient functions for logging LLM calls, parameters, metrics,
and artifacts to MLflow. All functions are no-ops if MLflow is disabled
or unavailable, ensuring the framework continues to work without MLflow.

Usage:
    from llm_framework.mlflow.logger import log_llm_call, log_params, log_metrics

    # Log an LLM call with all relevant details
    log_llm_call(
        prompt="What is the capital of France?",
        response="The capital of France is Paris.",
        model="gpt-4o",
        token_usage={"prompt": 10, "completion": 8, "total": 18},
        latency_ms=245.3,
        params={"temperature": 0.7, "max_tokens": 100}
    )

    # Log multiple parameters at once
    log_params({"model": "gpt-4o", "temperature": 0.7})

    # Log metrics
    log_metrics({"accuracy": 0.95, "f1_score": 0.92}, step=1)

    # Log artifacts
    log_artifact("output.json", artifact_path="predictions")
    log_dict_artifact({"results": [1, 2, 3]}, "results.json")
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _is_mlflow_available() -> bool:
    """Check if MLflow is both enabled in config and importable.

    Returns:
        True if MLflow can be used, False otherwise.
    """
    try:
        from llm_framework.config import get_config

        config = get_config()
        if not config.mlflow.enabled:
            return False

        # Try importing mlflow
        import mlflow
        return True
    except ImportError:
        return False
    except Exception as e:
        logger.debug(f"MLflow availability check failed: {e}")
        return False


def log_llm_call(
    prompt: str,
    response: str,
    model: str,
    token_usage: dict[str, int],
    latency_ms: float,
    params: dict[str, Any] | None = None,
) -> None:
    """Log an LLM call's details to the active MLflow run.

    Records prompt, response, model info, token usage, latency, and optional
    parameters. If no active run exists, logs a warning and returns.

    Args:
        prompt: The input prompt sent to the LLM.
        response: The LLM's response text.
        model: Model identifier (e.g., "gpt-4o", "claude-3").
        token_usage: Dictionary with token counts, e.g.:
            {"prompt": 10, "completion": 8, "total": 18}
        latency_ms: Response latency in milliseconds.
        params: Optional dictionary of LLM parameters (temperature, max_tokens, etc.).

    Example:
        log_llm_call(
            prompt="Summarize this text...",
            response="Here is a summary...",
            model="gpt-4o",
            token_usage={"prompt": 150, "completion": 50, "total": 200},
            latency_ms=1234.5,
            params={"temperature": 0.7, "max_tokens": 100}
        )
    """
    if not _is_mlflow_available():
        return

    try:
        import mlflow

        # Log parameters if provided
        if params:
            mlflow.log_params({f"llm.{k}": v for k, v in params.items()})

        # Log model info
        mlflow.log_param("llm.model", model)

        # Log token usage metrics
        for key, value in token_usage.items():
            mlflow.log_metric(f"tokens.{key}", value)

        # Log latency
        mlflow.log_metric("latency_ms", latency_ms)

        # Log prompt and response as text artifacts
        mlflow.log_text(prompt, "prompt.txt")
        mlflow.log_text(response, "response.txt")

        logger.debug(f"Logged LLM call: model={model}, latency={latency_ms}ms, tokens={token_usage.get('total', 0)}")

    except Exception as e:
        warnings.warn(
            f"Failed to log LLM call to MLflow: {e}. Continuing without MLflow logging.",
            stacklevel=2
        )
        logger.debug(f"MLflow logging error details: {e}", exc_info=True)


def log_params(params: dict[str, Any]) -> None:
    """Log multiple parameters to the active MLflow run.

    Parameters are key-value pairs that describe the configuration
    of an experiment (e.g., model hyperparameters, feature flags).

    Args:
        params: Dictionary of parameter names to values. Values will be
            converted to strings if necessary.

    Example:
        log_params({
            "model": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 1000,
            "use_cache": True
        })
    """
    if not _is_mlflow_available():
        return

    if not params:
        return

    try:
        import mlflow
        mlflow.log_params(params)
        logger.debug(f"Logged {len(params)} parameters to MLflow")
    except Exception as e:
        warnings.warn(
            f"Failed to log parameters to MLflow: {e}. Continuing without MLflow logging.",
            stacklevel=2
        )
        logger.debug(f"MLflow logging error details: {e}", exc_info=True)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log multiple metrics to the active MLflow run.

    Metrics are numerical measurements that track experiment performance
    over time (e.g., accuracy, loss, throughput).

    Args:
        metrics: Dictionary of metric names to numeric values.
        step: Optional step number for time-series metrics.

    Example:
        log_metrics({"accuracy": 0.95, "loss": 0.05}, step=10)
        log_metrics({"throughput": 1234.5, "error_rate": 0.01})
    """
    if not _is_mlflow_available():
        return

    if not metrics:
        return

    try:
        import mlflow
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged {len(metrics)} metrics to MLflow" + (f" at step {step}" if step else ""))
    except Exception as e:
        warnings.warn(
            f"Failed to log metrics to MLflow: {e}. Continuing without MLflow logging.",
            stacklevel=2
        )
        logger.debug(f"MLflow logging error details: {e}", exc_info=True)


def log_artifact(local_path: str | Path, artifact_path: str | None = None) -> None:
    """Log a local file as an artifact in the active MLflow run.

    Artifacts are output files from your run (e.g., models, plots, datasets).
    They are stored in the MLflow artifact store.

    Args:
        local_path: Path to the local file to log.
        artifact_path: Optional subdirectory path within the run's artifact
            directory. If None, the file is logged at the root.

    Example:
        log_artifact("model.pkl")
        log_artifact("results/plot.png", artifact_path="visualizations")
    """
    if not _is_mlflow_available():
        return

    try:
        import mlflow
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
        logger.debug(f"Logged artifact: {local_path}" + (f" to {artifact_path}" if artifact_path else ""))
    except Exception as e:
        warnings.warn(
            f"Failed to log artifact to MLflow: {e}. Continuing without MLflow logging.",
            stacklevel=2
        )
        logger.debug(f"MLflow logging error details: {e}", exc_info=True)


def log_dict_artifact(data: dict[str, Any], filename: str) -> None:
    """Log a dictionary as a JSON artifact in the active MLflow run.

    Convenience function for logging structured data. The dictionary is
    serialized to JSON and logged as an artifact.

    Args:
        data: Dictionary to log (must be JSON-serializable).
        filename: Name of the artifact file (should end with .json).

    Example:
        log_dict_artifact(
            {"results": [1, 2, 3], "metadata": {"version": "1.0"}},
            "output.json"
        )
    """
    if not _is_mlflow_available():
        return

    try:
        import mlflow
        mlflow.log_dict(data, filename)
        logger.debug(f"Logged dictionary artifact: {filename}")
    except Exception as e:
        warnings.warn(
            f"Failed to log dictionary artifact to MLflow: {e}. Continuing without MLflow logging.",
            stacklevel=2
        )
        logger.debug(f"MLflow logging error details: {e}", exc_info=True)
