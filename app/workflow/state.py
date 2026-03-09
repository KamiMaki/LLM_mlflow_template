"""LangGraph state schemas。

使用 LangGraph 內建的 MessagesState 管理訊息，BaseState 保持最精簡。
使用者依需求自行擴展 state（如加入 image_base64、llm_kwargs 等）。

Usage:
    from app.workflow.state import BaseState

    # 直接使用
    graph = StateGraph(BaseState)

    # 或擴展
    class MyState(BaseState):
        image_base64: str
        llm_kwargs: dict[str, Any]
"""

from __future__ import annotations

from langgraph.graph import MessagesState


class BaseState(MessagesState):
    """最小狀態 — 只有 messages（由 MessagesState 提供）。"""
    pass
