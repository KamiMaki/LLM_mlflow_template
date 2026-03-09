"""預建 LangGraph workflow。

使用 llm_service factory 自動注入 ChatLiteLLM，直接使用 LangGraph StateGraph API。

Usage:
    from app.workflow.build_workflow import build_simple_chain

    graph = build_simple_chain(system_prompt="你是一個助手")
    result = graph.invoke({"messages": [("user", "Hello")]})
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, StateGraph

from app.workflow.nodes import create_call_llm_node
from app.workflow.state import BaseState


def build_simple_chain(
    *,
    system_prompt: str = "You are a helpful assistant.",
    tools: list | None = None,
    llm: Any | None = None,
    **llm_overrides: Any,
) -> Any:
    """建構最簡單的 LLM chain：接收 user message → 呼叫 LLM → 回傳結果。

    Args:
        system_prompt: 系統提示詞。
        tools: 要綁定的工具列表。
        llm: 自訂 LLM（BaseChatModel），None 時自動從 factory 取得 ChatLiteLLM。
        **llm_overrides: 傳給 get_langchain_llm 的覆寫參數。

    Returns:
        已 compile 的 LangGraph CompiledGraph。
    """
    call_llm = create_call_llm_node(
        system_prompt=system_prompt, tools=tools, llm=llm, **llm_overrides
    )

    graph = StateGraph(BaseState)
    graph.add_node("call_llm", call_llm)
    graph.set_entry_point("call_llm")
    graph.add_edge("call_llm", END)

    return graph.compile()


def build_chain_with_preprocessing(
    preprocess_fn: Callable[[dict], dict],
    *,
    system_prompt: str = "You are a helpful assistant.",
    llm: Any | None = None,
    **llm_overrides: Any,
) -> Any:
    """建構帶前處理的 LLM chain：preprocess → call_llm → END。

    Args:
        preprocess_fn: 前處理 node function，接受 state dict 回傳 state updates。
        system_prompt: 系統提示詞。
        llm: 自訂 LLM（BaseChatModel），None 時自動從 factory 取得。
        **llm_overrides: 傳給 get_langchain_llm 的覆寫參數。

    Returns:
        已 compile 的 LangGraph CompiledGraph。
    """
    call_llm = create_call_llm_node(
        system_prompt=system_prompt, llm=llm, **llm_overrides
    )

    graph = StateGraph(BaseState)
    graph.add_node("preprocess", preprocess_fn)
    graph.add_node("call_llm", call_llm)
    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "call_llm")
    graph.add_edge("call_llm", END)

    return graph.compile()
