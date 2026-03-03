"""Evaluation Runner — 使用 MLflow GenAI evaluate 執行評估。

Usage:
    from app.evaluator.runner import run_evaluation

    results = run_evaluation(
        predict_fn=my_app,
        eval_data=eval_data,
        scorers=[Correctness(), response_not_empty],
    )
"""

from __future__ import annotations

from typing import Any, Callable

import mlflow
from mlflow.genai.scorers import Correctness, RelevanceToQuery

from app.logger import get_logger

logger = get_logger(__name__)


def run_evaluation(
    predict_fn: Callable,
    eval_data: list[dict[str, Any]],
    scorers: list | None = None,
    run_name: str = "evaluation",
) -> Any:
    """執行 MLflow GenAI 評估。

    Args:
        predict_fn: 預測函式，接受 eval_data 中 inputs 的 key-value 作為參數。
        eval_data: 評估資料列表，每筆包含 inputs 和可選的 expectations。
        scorers: 評分器列表，預設使用 Correctness + RelevanceToQuery。
        run_name: MLflow run 名稱。

    Returns:
        mlflow.genai.evaluate 的結果物件。
    """
    if scorers is None:
        scorers = [Correctness(), RelevanceToQuery()]

    logger.info(f"Running evaluation: {len(eval_data)} cases, {len(scorers)} scorers")

    with mlflow.start_run(run_name=run_name):
        results = mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=predict_fn,
            scorers=scorers,
        )

    logger.info("Evaluation complete")
    return results


def run_trace_evaluation(
    data: list[dict[str, Any]],
    scorers: list,
    run_name: str = "trace-evaluation",
) -> Any:
    """對已存在的 traces/資料執行評估（不需 predict_fn）。

    Args:
        data: 包含 outputs 的評估資料（已有預測結果）。
        scorers: 評分器列表。
        run_name: MLflow run 名稱。

    Returns:
        mlflow.genai.evaluate 的結果物件。
    """
    logger.info(f"Running trace evaluation: {len(data)} entries, {len(scorers)} scorers")

    with mlflow.start_run(run_name=run_name):
        results = mlflow.genai.evaluate(
            data=data,
            scorers=scorers,
        )

    logger.info("Trace evaluation complete")
    return results
