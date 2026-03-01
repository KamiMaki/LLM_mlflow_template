"""app.workflow 單元測試。"""

from __future__ import annotations

import pytest

from app.workflow import BaseWorkflow, WorkflowError, WorkflowState, create_workflow_state


def passthrough_node(state: dict) -> dict:
    """簡單的 pass-through node，用於測試。"""
    return {"results": {**state.get("results", {}), "done": True}}


class TestBaseWorkflow:
    def test_build_and_run(self):
        wf = (
            BaseWorkflow("test-wf", WorkflowState)
            .add_node("step1", passthrough_node)
            .set_entry("step1")
            .set_finish("step1")
            .compile()
        )
        initial = create_workflow_state(
            messages=[{"role": "user", "content": "hi"}],
        )
        result = wf.run(initial)
        assert result["results"]["done"] is True

    def test_run_without_compile_raises(self):
        wf = (
            BaseWorkflow("test-wf", WorkflowState)
            .add_node("step1", passthrough_node)
            .set_entry("step1")
            .set_finish("step1")
        )
        with pytest.raises(WorkflowError):
            wf.run(create_workflow_state())

    def test_compile_without_entry_raises(self):
        wf = BaseWorkflow("test-wf", WorkflowState).add_node("step1", passthrough_node)
        with pytest.raises(WorkflowError):
            wf.compile()

    def test_multi_node_workflow(self):
        def node_a(state: dict) -> dict:
            return {"results": {**state.get("results", {}), "a": True}}

        def node_b(state: dict) -> dict:
            return {"results": {**state.get("results", {}), "b": True}}

        wf = (
            BaseWorkflow("multi", WorkflowState)
            .add_node("a", node_a)
            .add_node("b", node_b)
            .add_edge("a", "b")
            .set_entry("a")
            .set_finish("b")
            .compile()
        )
        result = wf.run(create_workflow_state())
        assert result["results"]["a"] is True
        assert result["results"]["b"] is True


class TestStateFactories:
    def test_create_workflow_state_defaults(self):
        state = create_workflow_state()
        assert state["messages"] == []
        assert state["metadata"] == {}
        assert state["current_step"] == ""
        assert state["retry_count"] == 0
        assert state["error"] is None
