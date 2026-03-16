"""Evaluation Runner — 使用 MLflow GenAI evaluate 執行評估。

提供 run_evaluation / run_trace_evaluation 以及
make_workflow_predict_fn 將 LangGraph workflow 轉為 predict_fn。

Usage:
    from app.evaluator import run_evaluation, make_workflow_predict_fn

    predict_fn = make_workflow_predict_fn(compiled_graph)
    results = run_evaluation(
        predict_fn=predict_fn,
        eval_data=eval_data,
        scorers=[response_not_empty],
    )
"""

from __future__ import annotations

from typing import Any, Callable

import mlflow

from app.logger import get_logger

logger = get_logger(__name__)


def make_workflow_predict_fn(
    graph: Any,
    *,
    input_key: str = "question",
    state_builder: Callable[..., dict] | None = None,
    output_parser: Callable[[dict], str] | None = None,
) -> Callable:
    """將 LangGraph compiled graph 轉為 mlflow.genai.evaluate 的 predict_fn。

    mlflow.genai.evaluate 會以 predict_fn(**inputs) 呼叫，
    此函式將 inputs 轉為 graph.invoke() 所需的 state，
    並從回傳的 state 中提取最終回應字串。

    Args:
        graph: LangGraph compiled graph（graph.compile() 的回傳值）。
        input_key: eval_data inputs 中的主要輸入欄位名稱，預設 "question"。
        state_builder: 自訂函式，接收 **inputs 回傳 graph state dict。
            若為 None，預設將 inputs[input_key] 包裝為 MessagesState。
        output_parser: 自訂函式，從 graph 回傳的 state dict 提取回應字串。
            若為 None，預設取最後一則 AI message 的 content。

    Returns:
        可直接傳入 run_evaluation 的 predict_fn。

    Example:
        # 最簡單用法
        predict_fn = make_workflow_predict_fn(compiled_graph)

        # 自訂 state 建構
        predict_fn = make_workflow_predict_fn(
            compiled_graph,
            state_builder=lambda question, **kw: {
                "messages": [("user", question)],
                "language": "zh-TW",
            },
        )

        # 自訂 output 解析
        predict_fn = make_workflow_predict_fn(
            compiled_graph,
            output_parser=lambda state: state["summary"],
        )
    """

    def _default_state_builder(**inputs: Any) -> dict:
        text = inputs.get(input_key, str(inputs))
        return {"messages": [("user", text)]}

    def _default_output_parser(state: dict) -> str:
        messages = state.get("messages", [])
        if not messages:
            return ""
        last = messages[-1]
        if hasattr(last, "content"):
            return last.content
        if isinstance(last, dict):
            return last.get("content", str(last))
        if isinstance(last, tuple):
            return last[1]
        return str(last)

    builder = state_builder or _default_state_builder
    parser = output_parser or _default_output_parser

    @mlflow.trace
    def predict_fn(**inputs: Any) -> str:
        state = builder(**inputs)
        result = graph.invoke(state)
        return parser(result)

    return predict_fn


def run_evaluation(
    predict_fn: Callable,
    eval_data: list[dict[str, Any]],
    scorers: list,
    run_name: str = "evaluation",
) -> Any:
    """執行 MLflow GenAI 評估。

    Args:
        predict_fn: 預測函式，接受 eval_data 中 inputs 的 key-value 作為參數。
        eval_data: 評估資料列表，每筆包含 inputs 和可選的 expectations。
        scorers: 評分器列表。
        run_name: MLflow run 名稱。

    Returns:
        mlflow.genai.evaluate 的結果物件。
    """
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
