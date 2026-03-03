"""預建 LangGraph workflow 範例。

直接使用 LangGraph StateGraph API，不再包裝額外抽象層。

Usage:
    from app.workflow.build_workflow import build_simple_chain

    graph = build_simple_chain(client)
    result = graph.invoke(create_llm_state(messages=[...]))
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, StateGraph

from app.workflow.nodes import create_call_llm_node
from app.workflow.state import LLMState


def build_simple_chain(
    client: Any,
    *,
    system_prompt: str = "You are a helpful assistant.",
) -> Any:
    """建構最簡單的 LLM chain：接收 user message → 呼叫 LLM → 回傳結果。

    Args:
        client: LLM client（需有 chat() 方法）。
        system_prompt: 預設 system prompt。

    Returns:
        已 compile 的 LangGraph CompiledGraph。
    """
    call_llm = create_call_llm_node(client, default_system_prompt=system_prompt)

    graph = StateGraph(LLMState)
    graph.add_node("call_llm", call_llm)
    graph.set_entry_point("call_llm")
    graph.add_edge("call_llm", END)

    return graph.compile()


def build_chain_with_preprocessing(
    client: Any,
    preprocess_fn: Callable[[dict], dict],
    *,
    system_prompt: str = "You are a helpful assistant.",
) -> Any:
    """建構帶前處理的 LLM chain：preprocess → call_llm → END。

    Args:
        client: LLM client。
        preprocess_fn: 前處理 node function，接受 state dict 回傳 state updates。
        system_prompt: 預設 system prompt。

    Returns:
        已 compile 的 LangGraph CompiledGraph。
    """
    call_llm = create_call_llm_node(client, default_system_prompt=system_prompt)

    graph = StateGraph(LLMState)
    graph.add_node("preprocess", preprocess_fn)
    graph.add_node("call_llm", call_llm)
    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "call_llm")
    graph.add_edge("call_llm", END)

    return graph.compile()
