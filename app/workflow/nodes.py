"""通用 LangGraph node functions。

使用 llm_service factory 取得 ChatLiteLLM，直接在 StateGraph 中使用。
MLflow autolog 自動追蹤 LangChain/LiteLLM 呼叫，不需手動 span。

Usage:
    from app.workflow.nodes import create_call_llm_node

    call_llm = create_call_llm_node(system_prompt="你是一個助手")
    graph.add_node("call_llm", call_llm)
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import SystemMessage

from app.logger import get_logger

logger = get_logger(__name__)


def create_call_llm_node(
    *,
    system_prompt: str = "You are a helpful assistant.",
    tools: list | None = None,
    llm: Any | None = None,
    **llm_overrides: Any,
) -> Callable[[dict], dict]:
    """建立 call_llm node function。

    使用 ChatLiteLLM（透過 llm_service.get_langchain_llm）作為 LLM，
    MLflow autolog 會自動追蹤所有 LangChain 呼叫。

    Args:
        system_prompt: 系統提示詞。
        tools: 要綁定的工具列表（LangChain Tool）。
        llm: 自訂 LLM 實例（BaseChatModel），None 時自動從 factory 取得。
        **llm_overrides: 傳給 get_langchain_llm 的覆寫參數（model, temperature 等）。

    Returns:
        可直接作為 LangGraph node 的 function。
    """
    if llm is None:
        from llm_service import get_langchain_llm
        llm = get_langchain_llm(**llm_overrides)

    if tools:
        llm = llm.bind_tools(tools)

    def call_llm(state: dict) -> dict:
        messages = list(state.get("messages", []))
        if system_prompt:
            messages = [SystemMessage(content=system_prompt)] + messages

        response = llm.invoke(messages)
        return {"messages": [response]}

    return call_llm
