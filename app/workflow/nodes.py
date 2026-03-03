"""通用 LangGraph node functions。

提供可直接在 StateGraph 中使用的 node，包含 LLM 呼叫與 MLflow span 追蹤。

Usage:
    from app.workflow.nodes import create_call_llm_node

    call_llm = create_call_llm_node(client)
    graph.add_node("call_llm", call_llm)
"""

from __future__ import annotations

import time
from typing import Any, Callable

from app.logger import get_logger

logger = get_logger(__name__)

MLFLOW_AVAILABLE = False
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    pass


def create_call_llm_node(
    client: Any,
    *,
    system_prompt_key: str = "system_prompt",
    default_system_prompt: str = "You are a helpful assistant.",
) -> Callable[[dict], dict]:
    """建立 call_llm node function。

    使用 mlflow.start_span 記錄 LLM 呼叫細節（reasoning、token usage 等）。

    Args:
        client: LLM client，需有 chat(system_prompt, user_prompt, **kwargs) 方法。
        system_prompt_key: 從 state.metadata 中取得 system prompt 的 key。
        default_system_prompt: 預設 system prompt。

    Returns:
        可直接作為 LangGraph node 的 function。
    """

    def call_llm(state: dict) -> dict:
        messages = state.get("messages", [])
        metadata = state.get("metadata", {})

        # 取得最後一則 user message 作為 prompt
        user_message = ""
        for msg in reversed(messages):
            role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "type", "")
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if role in ("user", "human"):
                user_message = content
                break

        system_prompt = metadata.get(system_prompt_key, default_system_prompt)

        if MLFLOW_AVAILABLE:
            return _call_with_span(client, system_prompt, user_message, metadata)
        return _call_plain(client, system_prompt, user_message, metadata)

    return call_llm


def _call_with_span(client: Any, system_prompt: str, user_prompt: str, metadata: dict) -> dict:
    """呼叫 LLM 並用 MLflow span 記錄細節。"""
    with mlflow.start_span("call_llm") as span:
        span.set_inputs({"system_prompt": system_prompt, "user_prompt": user_prompt})

        start = time.perf_counter()
        try:
            kwargs = {k: v for k, v in metadata.items() if k in ("temperature", "max_tokens", "model")}
            response = client.chat(system_prompt, user_prompt, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000

            token_usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }

            span.set_outputs({"content": response.content})
            span.set_attributes({
                "model": response.model,
                "latency_ms": latency_ms,
                "prompt_tokens": token_usage["prompt_tokens"],
                "completion_tokens": token_usage["completion_tokens"],
                "total_tokens": token_usage["total_tokens"],
            })

            return {
                "llm_response": response.content,
                "token_usage": token_usage,
                "model": response.model,
                "messages": [{"role": "assistant", "content": response.content}],
                "error": None,
            }

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            span.set_attributes({"error": str(e), "latency_ms": latency_ms})
            logger.error(f"LLM call failed: {e}")
            return {"error": str(e), "llm_response": "", "token_usage": {}, "model": ""}


def _call_plain(client: Any, system_prompt: str, user_prompt: str, metadata: dict) -> dict:
    """無 MLflow 時的 plain LLM 呼叫。"""
    try:
        kwargs = {k: v for k, v in metadata.items() if k in ("temperature", "max_tokens", "model")}
        response = client.chat(system_prompt, user_prompt, **kwargs)

        token_usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            "total_tokens": getattr(response.usage, "total_tokens", 0),
        }

        return {
            "llm_response": response.content,
            "token_usage": token_usage,
            "model": response.model,
            "messages": [{"role": "assistant", "content": response.content}],
            "error": None,
        }

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {"error": str(e), "llm_response": "", "token_usage": {}, "model": ""}
