"""Tests for workflow state TypedDicts and factory functions."""
import pytest

from llm_framework.workflow.state import (
    BaseState,
    LLMState,
    WorkflowState,
    create_base_state,
    create_llm_state,
    create_workflow_state,
)


class TestBaseState:
    """Test BaseState TypedDict and factory function."""

    def test_create_base_state_with_defaults(self):
        """Test creating BaseState with default values."""
        state = create_base_state()

        assert state["messages"] == []
        assert state["metadata"] == {}

    def test_create_base_state_with_messages(self):
        """Test creating BaseState with custom messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        state = create_base_state(messages=messages)

        assert state["messages"] == messages
        assert state["metadata"] == {}

    def test_create_base_state_with_metadata(self):
        """Test creating BaseState with custom metadata."""
        metadata = {"user_id": "123", "session": "abc"}
        state = create_base_state(metadata=metadata)

        assert state["messages"] == []
        assert state["metadata"] == metadata

    def test_create_base_state_with_all_fields(self):
        """Test creating BaseState with all fields specified."""
        messages = [{"role": "user", "content": "Test"}]
        metadata = {"key": "value"}

        state = create_base_state(messages=messages, metadata=metadata)

        assert state["messages"] == messages
        assert state["metadata"] == metadata


class TestLLMState:
    """Test LLMState TypedDict and factory function."""

    def test_create_llm_state_with_defaults(self):
        """Test creating LLMState with default values."""
        state = create_llm_state()

        assert state["messages"] == []
        assert state["metadata"] == {}
        assert state["llm_response"] == ""
        assert state["token_usage"] == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        assert state["error"] is None

    def test_create_llm_state_with_llm_response(self):
        """Test creating LLMState with custom LLM response."""
        state = create_llm_state(llm_response="Generated text")

        assert state["llm_response"] == "Generated text"
        assert state["error"] is None

    def test_create_llm_state_with_token_usage(self):
        """Test creating LLMState with custom token usage."""
        token_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
        state = create_llm_state(token_usage=token_usage)

        assert state["token_usage"] == token_usage

    def test_create_llm_state_with_error(self):
        """Test creating LLMState with error."""
        state = create_llm_state(error="API timeout")

        assert state["error"] == "API timeout"
        assert state["llm_response"] == ""

    def test_create_llm_state_with_all_fields(self):
        """Test creating LLMState with all fields specified."""
        messages = [{"role": "user", "content": "Hello"}]
        metadata = {"request_id": "req123"}
        llm_response = "Response text"
        token_usage = {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
        error = None

        state = create_llm_state(
            messages=messages,
            metadata=metadata,
            llm_response=llm_response,
            token_usage=token_usage,
            error=error
        )

        assert state["messages"] == messages
        assert state["metadata"] == metadata
        assert state["llm_response"] == llm_response
        assert state["token_usage"] == token_usage
        assert state["error"] is None


class TestWorkflowState:
    """Test WorkflowState TypedDict and factory function."""

    def test_create_workflow_state_with_defaults(self):
        """Test creating WorkflowState with default values."""
        state = create_workflow_state()

        assert state["messages"] == []
        assert state["metadata"] == {}
        assert state["current_step"] == ""
        assert state["results"] == {}
        assert state["retry_count"] == 0
        assert state["error"] is None

    def test_create_workflow_state_with_current_step(self):
        """Test creating WorkflowState with custom current step."""
        state = create_workflow_state(current_step="preprocessing")

        assert state["current_step"] == "preprocessing"
        assert state["results"] == {}
        assert state["retry_count"] == 0

    def test_create_workflow_state_with_results(self):
        """Test creating WorkflowState with custom results."""
        results = {
            "step1": {"output": "data1"},
            "step2": {"output": "data2"}
        }
        state = create_workflow_state(results=results)

        assert state["results"] == results
        assert state["current_step"] == ""

    def test_create_workflow_state_with_retry_count(self):
        """Test creating WorkflowState with custom retry count."""
        state = create_workflow_state(retry_count=3)

        assert state["retry_count"] == 3
        assert state["error"] is None

    def test_create_workflow_state_with_error(self):
        """Test creating WorkflowState with error."""
        state = create_workflow_state(error="Validation failed")

        assert state["error"] == "Validation failed"
        assert state["retry_count"] == 0

    def test_create_workflow_state_with_all_fields(self):
        """Test creating WorkflowState with all fields specified."""
        messages = [{"role": "system", "content": "System prompt"}]
        metadata = {"workflow_id": "wf123"}
        current_step = "inference"
        results = {"preprocess": {"cleaned": True}}
        retry_count = 2
        error = "Temporary failure"

        state = create_workflow_state(
            messages=messages,
            metadata=metadata,
            current_step=current_step,
            results=results,
            retry_count=retry_count,
            error=error
        )

        assert state["messages"] == messages
        assert state["metadata"] == metadata
        assert state["current_step"] == current_step
        assert state["results"] == results
        assert state["retry_count"] == retry_count
        assert state["error"] == error


class TestStateTypeChecking:
    """Test that states conform to TypedDict expectations."""

    def test_base_state_has_required_keys(self):
        """Test that BaseState has all required keys."""
        state = create_base_state()

        assert "messages" in state
        assert "metadata" in state

    def test_llm_state_has_required_keys(self):
        """Test that LLMState has all required keys."""
        state = create_llm_state()

        assert "messages" in state
        assert "metadata" in state
        assert "llm_response" in state
        assert "token_usage" in state
        assert "error" in state

    def test_workflow_state_has_required_keys(self):
        """Test that WorkflowState has all required keys."""
        state = create_workflow_state()

        assert "messages" in state
        assert "metadata" in state
        assert "current_step" in state
        assert "results" in state
        assert "retry_count" in state
        assert "error" in state

    def test_states_are_mutable_dicts(self):
        """Test that state objects are mutable dictionaries."""
        base_state = create_base_state()
        base_state["metadata"]["new_key"] = "new_value"
        assert base_state["metadata"]["new_key"] == "new_value"

        llm_state = create_llm_state()
        llm_state["llm_response"] = "Updated response"
        assert llm_state["llm_response"] == "Updated response"

        workflow_state = create_workflow_state()
        workflow_state["current_step"] = "step2"
        assert workflow_state["current_step"] == "step2"


class TestStateInheritance:
    """Test that state types maintain inheritance relationships."""

    def test_llm_state_extends_base_state(self):
        """Test that LLMState includes all BaseState fields."""
        llm_state = create_llm_state(
            messages=[{"role": "user", "content": "test"}],
            metadata={"key": "value"}
        )

        # BaseState fields
        assert "messages" in llm_state
        assert "metadata" in llm_state
        assert llm_state["messages"][0]["content"] == "test"
        assert llm_state["metadata"]["key"] == "value"

        # LLMState-specific fields
        assert "llm_response" in llm_state
        assert "token_usage" in llm_state
        assert "error" in llm_state

    def test_workflow_state_extends_base_state(self):
        """Test that WorkflowState includes all BaseState fields."""
        workflow_state = create_workflow_state(
            messages=[{"role": "system", "content": "system"}],
            metadata={"id": "123"}
        )

        # BaseState fields
        assert "messages" in workflow_state
        assert "metadata" in workflow_state
        assert workflow_state["messages"][0]["role"] == "system"
        assert workflow_state["metadata"]["id"] == "123"

        # WorkflowState-specific fields
        assert "current_step" in workflow_state
        assert "results" in workflow_state
        assert "retry_count" in workflow_state
        assert "error" in workflow_state
