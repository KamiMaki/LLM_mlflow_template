"""Tests for mlflow tracer module."""
import functools
from unittest.mock import patch, MagicMock, call
import pytest
import warnings

from llm_framework.mlflow.tracer import (
    trace_llm_call,
    trace_node,
    trace_workflow,
    span,
    _is_tracing_available,
)


class TestTracingAvailability:
    """Test MLflow tracing availability checks."""

    def test_is_tracing_available_when_disabled(self, disabled_mlflow_config):
        """Test that tracing is unavailable when MLflow is disabled in config."""
        with patch("llm_framework.mlflow.tracer.get_config", return_value=disabled_mlflow_config):
            assert not _is_tracing_available()

    def test_is_tracing_available_when_mlflow_not_installed(self, sample_config):
        """Test that tracing is unavailable when MLflow is not installed."""
        with patch("llm_framework.mlflow.tracer.get_config", return_value=sample_config):
            with patch.dict("sys.modules", {"mlflow": None}):
                with patch("llm_framework.mlflow.tracer.import", side_effect=ImportError):
                    assert not _is_tracing_available()

    def test_is_tracing_available_when_enabled_and_installed(self, sample_config):
        """Test that tracing is available when MLflow is enabled and has tracing support."""
        mock_mlflow = MagicMock()
        mock_mlflow.start_span = MagicMock()

        with patch("llm_framework.mlflow.tracer.get_config", return_value=sample_config):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                assert _is_tracing_available()


class TestTraceLLMCall:
    """Test trace_llm_call decorator."""

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function name, docstring, etc."""
        @trace_llm_call
        def example_function(x, y):
            """This is a docstring."""
            return x + y

        assert example_function.__name__ == "example_function"
        assert example_function.__doc__ == "This is a docstring."
        assert functools.WRAPPER_ASSIGNMENTS

    def test_trace_llm_call_as_noop_when_unavailable(self, disabled_mlflow_config):
        """Test that trace_llm_call is a no-op when MLflow is unavailable."""
        call_count = 0

        @trace_llm_call
        def test_func(value):
            nonlocal call_count
            call_count += 1
            return value * 2

        with patch("llm_framework.mlflow.tracer.get_config", return_value=disabled_mlflow_config):
            with patch("llm_framework.mlflow.tracer._is_tracing_available", return_value=False):
                result = test_func(5)

        assert result == 10
        assert call_count == 1

    def test_trace_llm_call_executes_without_mlflow(self, disabled_mlflow_config):
        """Test that function executes normally when tracing is unavailable."""
        @trace_llm_call
        def simple_func(a, b):
            return a + b

        with patch("llm_framework.mlflow.tracer._is_tracing_available", return_value=False):
            result = simple_func(3, 4)

        assert result == 7

    def test_trace_llm_call_handles_exceptions(self, disabled_mlflow_config):
        """Test that exceptions in the wrapped function are propagated."""
        @trace_llm_call
        def failing_func():
            raise ValueError("Test error")

        with patch("llm_framework.mlflow.tracer._is_tracing_available", return_value=False):
            with pytest.raises(ValueError, match="Test error"):
                failing_func()


class TestTraceNode:
    """Test trace_node decorator."""

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""
        @trace_node("test_node")
        def example_function(state):
            """Node function."""
            return state

        assert example_function.__name__ == "example_function"
        assert example_function.__doc__ == "Node function."

    def test_trace_node_as_noop_when_unavailable(self, disabled_mlflow_config):
        """Test that trace_node is a no-op when MLflow is unavailable."""
        call_count = 0

        @trace_node("processing_node")
        def process_node(data):
            nonlocal call_count
            call_count += 1
            return {"result": data["value"] * 2}

        with patch("llm_framework.mlflow.tracer._is_tracing_available", return_value=False):
            result = process_node({"value": 5})

        assert result == {"result": 10}
        assert call_count == 1

    def test_trace_node_executes_without_mlflow(self):
        """Test that node function executes when tracing is unavailable."""
        @trace_node("my_node")
        def node_func(state):
            return {"current_step": "processed"}

        with patch("llm_framework.mlflow.tracer._is_tracing_available", return_value=False):
            result = node_func({"data": "test"})

        assert result == {"current_step": "processed"}


class TestTraceWorkflow:
    """Test trace_workflow decorator."""

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""
        @trace_workflow("test_workflow")
        def example_workflow(docs):
            """Workflow function."""
            return docs

        assert example_workflow.__name__ == "example_workflow"
        assert example_workflow.__doc__ == "Workflow function."

    def test_trace_workflow_as_noop_when_unavailable(self, disabled_mlflow_config):
        """Test that trace_workflow is a no-op when MLflow is unavailable."""
        call_count = 0

        @trace_workflow("document_pipeline")
        def run_pipeline(documents):
            nonlocal call_count
            call_count += 1
            return [doc.upper() for doc in documents]

        with patch("llm_framework.mlflow.tracer._is_tracing_available", return_value=False):
            result = run_pipeline(["doc1", "doc2"])

        assert result == ["DOC1", "DOC2"]
        assert call_count == 1

    def test_trace_workflow_executes_without_mlflow(self):
        """Test that workflow executes when tracing is unavailable."""
        @trace_workflow("my_workflow")
        def workflow_func(items):
            return len(items)

        with patch("llm_framework.mlflow.tracer._is_tracing_available", return_value=False):
            result = workflow_func([1, 2, 3])

        assert result == 3


class TestSpanContextManager:
    """Test span context manager."""

    def test_span_yields_none_when_unavailable(self, disabled_mlflow_config):
        """Test that span context manager yields None when MLflow is unavailable."""
        with patch("llm_framework.mlflow.tracer._is_tracing_available", return_value=False):
            with span("test_span") as span_obj:
                assert span_obj is None

    def test_span_executes_code_block_when_unavailable(self):
        """Test that code inside span executes normally when tracing is unavailable."""
        result = None

        with patch("llm_framework.mlflow.tracer._is_tracing_available", return_value=False):
            with span("database_query", span_type="TOOL"):
                result = 42

        assert result == 42

    def test_span_with_different_span_types(self):
        """Test span with different span type parameters."""
        results = []

        with patch("llm_framework.mlflow.tracer._is_tracing_available", return_value=False):
            with span("llm_span", span_type="LLM"):
                results.append("llm")

            with span("tool_span", span_type="TOOL"):
                results.append("tool")

            with span("retriever_span", span_type="RETRIEVER"):
                results.append("retriever")

        assert results == ["llm", "tool", "retriever"]

    def test_span_handles_exceptions(self):
        """Test that exceptions inside span are propagated."""
        with patch("llm_framework.mlflow.tracer._is_tracing_available", return_value=False):
            with pytest.raises(ValueError, match="Test span error"):
                with span("error_span"):
                    raise ValueError("Test span error")


class TestTracingWithMLflowEnabled:
    """Test tracing behavior when MLflow is enabled and available."""

    def test_trace_llm_call_creates_span(self, sample_config):
        """Test that trace_llm_call creates MLflow span when available."""
        mock_mlflow = MagicMock()
        mock_span = MagicMock()
        mock_mlflow.start_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_mlflow.start_span.return_value.__exit__ = MagicMock(return_value=False)

        @trace_llm_call
        def llm_call(prompt):
            return f"Response to: {prompt}"

        with patch("llm_framework.mlflow.tracer.get_config", return_value=sample_config):
            with patch("llm_framework.mlflow.tracer._is_tracing_available", return_value=True):
                with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                    with patch("llm_framework.mlflow.tracer.mlflow", mock_mlflow):
                        result = llm_call("test prompt")

        assert result == "Response to: test prompt"
        mock_mlflow.start_span.assert_called_once()
        assert mock_span.set_inputs.called
        assert mock_span.set_outputs.called

    def test_trace_llm_call_logs_attributes(self, sample_config):
        """Test that trace_llm_call logs latency attributes."""
        mock_mlflow = MagicMock()
        mock_span = MagicMock()
        mock_mlflow.start_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_mlflow.start_span.return_value.__exit__ = MagicMock(return_value=False)

        @trace_llm_call
        def llm_call():
            return "result"

        with patch("llm_framework.mlflow.tracer.get_config", return_value=sample_config):
            with patch("llm_framework.mlflow.tracer._is_tracing_available", return_value=True):
                with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                    with patch("llm_framework.mlflow.tracer.mlflow", mock_mlflow):
                        llm_call()

        # Verify latency_ms was logged
        calls = mock_span.set_attributes.call_args_list
        assert any("latency_ms" in str(call) for call in calls)

    def test_trace_llm_call_handles_errors_with_mlflow(self, sample_config):
        """Test that trace_llm_call logs errors to MLflow span."""
        mock_mlflow = MagicMock()
        mock_span = MagicMock()
        mock_mlflow.start_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_mlflow.start_span.return_value.__exit__ = MagicMock(return_value=False)

        @trace_llm_call
        def failing_llm_call():
            raise RuntimeError("LLM error")

        with patch("llm_framework.mlflow.tracer.get_config", return_value=sample_config):
            with patch("llm_framework.mlflow.tracer._is_tracing_available", return_value=True):
                with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                    with patch("llm_framework.mlflow.tracer.mlflow", mock_mlflow):
                        with pytest.raises(RuntimeError, match="LLM error"):
                            failing_llm_call()

        # Verify error was logged
        mock_span.set_status.assert_called_with("ERROR")
        calls = mock_span.set_attributes.call_args_list
        assert any("error" in str(call) for call in calls)

    def test_tracing_failure_falls_back_gracefully(self, sample_config):
        """Test that tracing failures don't break the function execution."""
        mock_mlflow = MagicMock()
        mock_mlflow.start_span.side_effect = Exception("MLflow error")

        @trace_llm_call
        def llm_call():
            return "success"

        with patch("llm_framework.mlflow.tracer.get_config", return_value=sample_config):
            with patch("llm_framework.mlflow.tracer._is_tracing_available", return_value=True):
                with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                    with patch("llm_framework.mlflow.tracer.mlflow", mock_mlflow):
                        with warnings.catch_warnings(record=True) as w:
                            warnings.simplefilter("always")
                            result = llm_call()

        assert result == "success"
        assert len(w) == 1
        assert "Failed to trace" in str(w[0].message)
