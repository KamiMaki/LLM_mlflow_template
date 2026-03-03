"""LangGraph TypedDict state schemas。

Usage:
    from app.workflow.state import LLMState, create_llm_state

    initial = create_llm_state(messages=[{"role": "user", "content": "Hello"}])
"""

from __future__ import annotations

from typing import Annotated, Any

from typing_extensions import TypedDict

try:
    from langgraph.graph.message import add_messages
except ImportError:
    def add_messages(left: list, right: list) -> list:
        """Fallback message reducer."""
        return left + right


class BaseState(TypedDict):
    """最小狀態，所有 workflow 共用。"""
    messages: Annotated[list, add_messages]
    metadata: dict[str, Any]


class LLMState(BaseState):
    """LLM 呼叫 workflow 狀態。"""
    llm_response: str
    token_usage: dict[str, int]
    model: str
    error: str | None


def create_base_state(
    messages: list[dict[str, str]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> BaseState:
    return BaseState(messages=messages or [], metadata=metadata or {})


def create_llm_state(
    messages: list[dict[str, str]] | None = None,
    metadata: dict[str, Any] | None = None,
    llm_response: str = "",
    token_usage: dict[str, int] | None = None,
    model: str = "",
    error: str | None = None,
) -> LLMState:
    return LLMState(
        messages=messages or [],
        metadata=metadata or {},
        llm_response=llm_response,
        token_usage=token_usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        model=model,
        error=error,
    )
