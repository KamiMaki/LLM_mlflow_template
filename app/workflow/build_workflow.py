"""預建 LangGraph workflow — 支援多模型切換與圖片輸入。

使用 llm_service factory 自動注入 ChatLiteLLM，直接使用 LangGraph StateGraph API。

Usage:
    from app.workflow.build_workflow import build_simple_chain, build_multimodel_chain

    # 簡單 chain
    graph = build_simple_chain(system_prompt="你是一個助手")
    result = graph.invoke({"messages": [("user", "Hello")]})

    # 多模型 chain（帶圖片支援）
    from llm_service import LLMConfig
    cfg = LLMConfig.from_yaml()
    graph = build_multimodel_chain(
        system_prompt="分析以下內容",
        support_images=True,
    )
    result = graph.invoke({
        "messages": [("user", "分析這張圖表")],
        "llm_config": cfg,
        "model_alias": "QWEN3VL",
        "image_base64": "...",
    })
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, StateGraph

from app.workflow.nodes import (
    create_call_llm_node,
    create_prompt_assembly_node,
    create_set_model_node,
)
from app.workflow.state import BaseState, WorkflowState


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


def build_multimodel_chain(
    *,
    system_prompt: str = "You are a helpful assistant.",
    support_images: bool = False,
    prompt_name: str | None = None,
    prompt_template: str | None = None,
    preprocess_fn: Callable[[dict], dict] | None = None,
) -> Any:
    """建構多模型 workflow chain，支援動態模型切換、prompt 組裝與圖片。

    流程: [preprocess] → [assemble_prompt] → call_llm → END

    state 需使用 WorkflowState，在 invoke 時傳入 llm_config、model_alias 等。

    Args:
        system_prompt: 系統提示詞。
        support_images: 是否支援 base64 圖片。
        prompt_name: MLflow prompt 名稱（自動載入並格式化）。
        prompt_template: 直接提供的 prompt 模板。
        preprocess_fn: 可選的前處理 node function。

    Returns:
        已 compile 的 LangGraph CompiledGraph（使用 WorkflowState）。
    """
    graph = StateGraph(WorkflowState)

    entry_point = "call_llm"

    # 可選：prompt 組裝
    if prompt_name or prompt_template:
        assemble = create_prompt_assembly_node(
            prompt_name=prompt_name,
            prompt_template=prompt_template,
        )
        graph.add_node("assemble_prompt", assemble)
        entry_point = "assemble_prompt"

    # 可選：前處理
    if preprocess_fn:
        graph.add_node("preprocess", preprocess_fn)
        if prompt_name or prompt_template:
            graph.add_edge("preprocess", "assemble_prompt")
        else:
            graph.add_edge("preprocess", "call_llm")
        entry_point = "preprocess"

    # LLM 呼叫
    call_llm = create_call_llm_node(
        system_prompt=system_prompt,
        support_images=support_images,
    )
    graph.add_node("call_llm", call_llm)

    if (prompt_name or prompt_template) and not preprocess_fn:
        graph.add_edge("assemble_prompt", "call_llm")
    elif (prompt_name or prompt_template) and preprocess_fn:
        graph.add_edge("assemble_prompt", "call_llm")

    graph.add_edge("call_llm", END)
    graph.set_entry_point(entry_point)

    return graph.compile()
