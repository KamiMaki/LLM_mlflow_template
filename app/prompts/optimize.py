"""Prompt Optimization — 使用 MLflow GenAI optimize_prompts 自動優化 prompt。

基於 GEPA 演算法（DSPy MIPROv2），從訓練資料中學習最佳 prompt 版本。

Usage:
    from app.prompts.optimize import optimize_prompt

    result = optimize_prompt(
        predict_fn=my_predict,
        train_data=train_data,
        prompt_name="my_prompt",
        scorers=[my_scorer],
    )
    print(result.prompt.uri)  # 優化後的 prompt URI
"""

from __future__ import annotations

from typing import Any, Callable

import mlflow
from mlflow.genai.optimize import GepaPromptOptimizer, LLMParams

from app.logger import get_logger

logger = get_logger(__name__)


def optimize_prompt(
    predict_fn: Callable,
    train_data: list[dict[str, Any]],
    prompt_name: str,
    scorers: list,
    *,
    eval_data: list[dict[str, Any]] | None = None,
    reflection_model: str = "openai:/gpt-4o",
    prompt_version: str | None = None,
) -> Any:
    """執行 prompt 優化。

    predict_fn 內部必須使用 mlflow.genai.load_prompt() 載入 prompt，
    optimize_prompts 會自動替換不同版本來搜尋最佳結果。

    Args:
        predict_fn: 預測函式，內部需透過 load_prompt 載入指定 prompt。
        train_data: 訓練資料列表，每筆包含 inputs 和 expectations。
        prompt_name: 要優化的 prompt 名稱（已在 Registry 中註冊）。
        scorers: 評分器列表。
        eval_data: 評估資料列表（可選，用於驗證優化結果）。
        reflection_model: 用於反思與生成候選 prompt 的模型。
        prompt_version: 指定起始版本，None 為最新版。

    Returns:
        OptimizationResult，包含 .prompt（優化後的 PromptVersion）。
    """
    if prompt_version:
        prompt_uri = f"prompts:/{prompt_name}/{prompt_version}"
    else:
        prompt_uri = f"prompts:/{prompt_name}@latest"

    logger.info(f"Starting prompt optimization for '{prompt_name}' with {len(train_data)} training examples")

    optimizer = GepaPromptOptimizer(
        reflection_model=reflection_model,
    )

    kwargs: dict[str, Any] = {
        "predict_fn": predict_fn,
        "train_data": train_data,
        "prompt_uris": [prompt_uri],
        "optimizer": optimizer,
        "scorers": scorers,
    }

    if eval_data:
        kwargs["eval_data"] = eval_data

    result = mlflow.genai.optimize_prompts(**kwargs)

    logger.info(f"Prompt optimization complete. New prompt URI: {result.prompt.uri}")
    return result
