"""LangGraph state schemas — 支援多模型切換與 prompt template。

BaseState 保持最精簡（只有 messages）。
WorkflowState 擴展 BaseState，加入 llm_config、prompt_template、
model_alias、image 等欄位，支援在不同 node 間切換模型與組裝 prompt。

Usage:
    from app.workflow.state import WorkflowState

    graph = StateGraph(WorkflowState)

    # 在 node 中切換模型
    def my_node(state: dict) -> dict:
        return {"model_alias": "QWEN3VL"}
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import MessagesState


class BaseState(MessagesState):
    """最小狀態 — 只有 messages（由 MessagesState 提供）。"""
    pass


class WorkflowState(MessagesState):
    """多模型 workflow 狀態 — 支援 llm_config、prompt template 與圖片輸入。

    Attributes:
        llm_config: LLMConfig 實例（在 workflow 啟動時注入）。
        prompt_template: 當前 prompt 模板字串。
        prompt_variables: prompt 模板變數。
        model_alias: 當前使用的模型別名（如 "QWEN3", "QWEN3VL"）。
        zone: 當前環境 zone。
        image_base64: base64 編碼的圖片（單張或多張）。
        metadata: 額外 metadata（自由欄位）。
    """

    llm_config: Any = None
    prompt_template: str = ""
    prompt_variables: dict[str, Any] = {}
    model_alias: str = ""
    zone: str = ""
    image_base64: str | list[str] | None = None
    metadata: dict[str, Any] = {}
