"""Pydantic state models for LangGraph workflows.

Defines TypedDict state schemas following LangGraph conventions. Each state
type includes a factory function for creating initial state instances.

Usage:
    from llm_framework.workflow.state import WorkflowState, create_workflow_state

    initial = create_workflow_state(messages=[{"role": "user", "content": "Hello"}])
"""

from __future__ import annotations

from typing import Annotated, Any

from typing_extensions import TypedDict

try:
    from langgraph.graph.message import add_messages
except ImportError:
    # Fallback if langgraph not installed
    def add_messages(left: list, right: list) -> list:
        """Fallback message reducer that concatenates lists."""
        return left + right


# ---------------------------------------------------------------------------
# State schemas
# ---------------------------------------------------------------------------


class BaseState(TypedDict):
    """Minimal state shared by all workflows.

    Attributes:
        messages: List of chat messages with automatic concatenation.
        metadata: Arbitrary metadata dict for workflow-specific data.
    """
    messages: Annotated[list, add_messages]
    metadata: dict[str, Any]


class LLMState(BaseState):
    """State for single-LLM-call workflows.

    Extends BaseState with fields specific to simple LLM request/response
    patterns without complex multi-step orchestration.

    Attributes:
        messages: Inherited from BaseState.
        metadata: Inherited from BaseState.
        llm_response: The text response from the LLM.
        token_usage: Token counts (prompt, completion, total).
        error: Error message if the LLM call failed, None otherwise.
    """
    llm_response: str
    token_usage: dict[str, int]
    error: str | None


class WorkflowState(BaseState):
    """State for multi-step workflows.

    Extends BaseState with fields for tracking progress through complex
    workflows with multiple nodes and conditional routing.

    Attributes:
        messages: Inherited from BaseState.
        metadata: Inherited from BaseState.
        current_step: Name of the currently executing workflow step.
        results: Dictionary storing outputs from each workflow node.
        retry_count: Number of retry attempts for the current operation.
        error: Error message if a step failed, None otherwise.
    """
    current_step: str
    results: dict[str, Any]
    retry_count: int
    error: str | None


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_base_state(
    messages: list[dict[str, str]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> BaseState:
    """Create an initial BaseState instance.

    Args:
        messages: Initial message list. Defaults to empty list.
        metadata: Initial metadata dict. Defaults to empty dict.

    Returns:
        BaseState with provided or default values.
    """
    return BaseState(
        messages=messages or [],
        metadata=metadata or {},
    )


def create_llm_state(
    messages: list[dict[str, str]] | None = None,
    metadata: dict[str, Any] | None = None,
    llm_response: str = "",
    token_usage: dict[str, int] | None = None,
    error: str | None = None,
) -> LLMState:
    """Create an initial LLMState instance.

    Args:
        messages: Initial message list. Defaults to empty list.
        metadata: Initial metadata dict. Defaults to empty dict.
        llm_response: Initial LLM response. Defaults to empty string.
        token_usage: Initial token usage. Defaults to zero counts.
        error: Initial error state. Defaults to None.

    Returns:
        LLMState with provided or default values.
    """
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
    """Create an initial WorkflowState instance.

    Args:
        messages: Initial message list. Defaults to empty list.
        metadata: Initial metadata dict. Defaults to empty dict.
        current_step: Initial step name. Defaults to empty string.
        results: Initial results dict. Defaults to empty dict.
        retry_count: Initial retry count. Defaults to 0.
        error: Initial error state. Defaults to None.

    Returns:
        WorkflowState with provided or default values.
    """
    return WorkflowState(
        messages=messages or [],
        metadata=metadata or {},
        current_step=current_step,
        results=results or {},
        retry_count=retry_count,
        error=error,
    )
