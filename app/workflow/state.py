"""LangGraph TypedDict state schemas 與 factory functions。

Usage:
    from app.workflow.state import WorkflowState, create_workflow_state

    initial = create_workflow_state(messages=[{"role": "user", "content": "Hello"}])
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
    """單一 LLM 呼叫的狀態。"""
    llm_response: str
    token_usage: dict[str, int]
    error: str | None


class WorkflowState(BaseState):
    """多步驟 workflow 狀態。"""
    current_step: str
    results: dict[str, Any]
    retry_count: int
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
    error: str | None = None,
) -> LLMState:
    return LLMState(
        messages=messages or [],
        metadata=metadata or {},
        llm_response=llm_response,
        token_usage=token_usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        error=error,
    )


def create_workflow_state(
    messages: list[dict[str, str]] | None = None,
    metadata: dict[str, Any] | None = None,
    current_step: str = "",
    results: dict[str, Any] | None = None,
    retry_count: int = 0,
    error: str | None = None,
) -> WorkflowState:
    return WorkflowState(
        messages=messages or [],
        metadata=metadata or {},
        current_step=current_step,
        results=results or {},
        retry_count=retry_count,
        error=error,
    )
