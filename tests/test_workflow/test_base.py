"""Tests for base workflow wrapper around LangGraph."""
from unittest.mock import MagicMock, patch
import pytest

from llm_framework.workflow.base import (
    BaseWorkflow,
    WorkflowError,
)
from llm_framework.workflow.state import WorkflowState


class TestBaseWorkflowWithoutLangGraph:
    """Test BaseWorkflow behavior when LangGraph is not installed."""

    def test_constructor_raises_error_without_langgraph(self):
        """Test that BaseWorkflow raises error when LangGraph is not available."""
        with patch("llm_framework.workflow.base.LANGGRAPH_AVAILABLE", False):
            with pytest.raises(WorkflowError, match="LangGraph is not installed"):
                BaseWorkflow("test_workflow")


class TestBaseWorkflowWithLangGraph:
    """Test BaseWorkflow with mocked LangGraph."""

    @pytest.fixture
    def mock_state_graph(self):
        """Mock StateGraph class."""
        mock_graph_class = MagicMock()
        mock_graph_instance = MagicMock()
        mock_graph_class.return_value = mock_graph_instance
        return mock_graph_class, mock_graph_instance

    def test_initialization(self, mock_state_graph):
        """Test BaseWorkflow initialization."""
        mock_graph_class, mock_graph_instance = mock_state_graph

        with patch("llm_framework.workflow.base.LANGGRAPH_AVAILABLE", True):
            with patch("llm_framework.workflow.base.StateGraph", mock_graph_class):
                workflow = BaseWorkflow("test_workflow", WorkflowState)

        assert workflow.name == "test_workflow"
        assert workflow.state_schema == WorkflowState
        assert workflow._compiled is None
        assert workflow._entry_node is None
        assert workflow._finish_nodes == set()
        mock_graph_class.assert_called_once_with(WorkflowState)

    def test_add_node_returns_self(self, mock_state_graph):
        """Test that add_node returns self for method chaining."""
        mock_graph_class, mock_graph_instance = mock_state_graph

        def node_func(state):
            return state

        with patch("llm_framework.workflow.base.LANGGRAPH_AVAILABLE", True):
            with patch("llm_framework.workflow.base.StateGraph", mock_graph_class):
                workflow = BaseWorkflow("test")
                result = workflow.add_node("node1", node_func)

        assert result is workflow
        mock_graph_instance.add_node.assert_called_once_with("node1", node_func)

    def test_add_edge_returns_self(self, mock_state_graph):
        """Test that add_edge returns self for method chaining."""
        mock_graph_class, mock_graph_instance = mock_state_graph

        with patch("llm_framework.workflow.base.LANGGRAPH_AVAILABLE", True):
            with patch("llm_framework.workflow.base.StateGraph", mock_graph_class):
                workflow = BaseWorkflow("test")
                result = workflow.add_edge("node1", "node2")

        assert result is workflow
        mock_graph_instance.add_edge.assert_called_once_with("node1", "node2")

    def test_add_conditional_edge_returns_self(self, mock_state_graph):
        """Test that add_conditional_edge returns self for method chaining."""
        mock_graph_class, mock_graph_instance = mock_state_graph

        def condition(state):
            return "target1"

        targets = {"target1": "node1", "target2": "node2"}

        with patch("llm_framework.workflow.base.LANGGRAPH_AVAILABLE", True):
            with patch("llm_framework.workflow.base.StateGraph", mock_graph_class):
                workflow = BaseWorkflow("test")
                result = workflow.add_conditional_edge("source", condition, targets)

        assert result is workflow
        mock_graph_instance.add_conditional_edges.assert_called_once_with(
            "source", condition, targets
        )

    def test_set_entry_returns_self(self, mock_state_graph):
        """Test that set_entry returns self for method chaining."""
        mock_graph_class, mock_graph_instance = mock_state_graph

        with patch("llm_framework.workflow.base.LANGGRAPH_AVAILABLE", True):
            with patch("llm_framework.workflow.base.StateGraph", mock_graph_class):
                workflow = BaseWorkflow("test")
                result = workflow.set_entry("start_node")

        assert result is workflow
        assert workflow._entry_node == "start_node"
        mock_graph_instance.set_entry_point.assert_called_once_with("start_node")

    def test_set_finish_returns_self(self, mock_state_graph):
        """Test that set_finish returns self for method chaining."""
        mock_graph_class, mock_graph_instance = mock_state_graph

        with patch("llm_framework.workflow.base.LANGGRAPH_AVAILABLE", True):
            with patch("llm_framework.workflow.base.StateGraph", mock_graph_class):
                with patch("llm_framework.workflow.base.END", "__end__"):
                    workflow = BaseWorkflow("test")
                    result = workflow.set_finish("end_node")

        assert result is workflow
        assert "end_node" in workflow._finish_nodes
        mock_graph_instance.add_edge.assert_called_once_with("end_node", "__end__")

    def test_method_chaining(self, mock_state_graph):
        """Test that methods can be chained together."""
        mock_graph_class, mock_graph_instance = mock_state_graph

        def node_func(state):
            return state

        with patch("llm_framework.workflow.base.LANGGRAPH_AVAILABLE", True):
            with patch("llm_framework.workflow.base.StateGraph", mock_graph_class):
                with patch("llm_framework.workflow.base.END", "__end__"):
                    workflow = (
                        BaseWorkflow("test")
                        .add_node("node1", node_func)
                        .add_node("node2", node_func)
                        .set_entry("node1")
                        .add_edge("node1", "node2")
                        .set_finish("node2")
                    )

        assert workflow.name == "test"
        assert workflow._entry_node == "node1"
        assert "node2" in workflow._finish_nodes

    def test_compile_without_entry_raises_error(self, mock_state_graph):
        """Test that compile raises error when entry point is not set."""
        mock_graph_class, mock_graph_instance = mock_state_graph

        with patch("llm_framework.workflow.base.LANGGRAPH_AVAILABLE", True):
            with patch("llm_framework.workflow.base.StateGraph", mock_graph_class):
                workflow = BaseWorkflow("test")

                with pytest.raises(WorkflowError, match="without entry point"):
                    workflow.compile()

    def test_compile_returns_self(self, mock_state_graph):
        """Test that compile returns self for method chaining."""
        mock_graph_class, mock_graph_instance = mock_state_graph
        mock_graph_instance.compile.return_value = MagicMock()

        with patch("llm_framework.workflow.base.LANGGRAPH_AVAILABLE", True):
            with patch("llm_framework.workflow.base.StateGraph", mock_graph_class):
                workflow = BaseWorkflow("test")
                workflow.set_entry("start")
                result = workflow.compile()

        assert result is workflow
        assert workflow._compiled is not None
        mock_graph_instance.compile.assert_called_once()

    def test_compile_handles_errors(self, mock_state_graph):
        """Test that compile raises WorkflowError on failure."""
        mock_graph_class, mock_graph_instance = mock_state_graph
        mock_graph_instance.compile.side_effect = Exception("Compilation failed")

        with patch("llm_framework.workflow.base.LANGGRAPH_AVAILABLE", True):
            with patch("llm_framework.workflow.base.StateGraph", mock_graph_class):
                workflow = BaseWorkflow("test")
                workflow.set_entry("start")

                with pytest.raises(WorkflowError, match="Failed to compile"):
                    workflow.compile()

    def test_add_node_after_compile_raises_error(self, mock_state_graph):
        """Test that adding node after compile raises error."""
        mock_graph_class, mock_graph_instance = mock_state_graph
        mock_graph_instance.compile.return_value = MagicMock()

        def node_func(state):
            return state

        with patch("llm_framework.workflow.base.LANGGRAPH_AVAILABLE", True):
            with patch("llm_framework.workflow.base.StateGraph", mock_graph_class):
                workflow = BaseWorkflow("test")
                workflow.set_entry("start")
                workflow.compile()

                with pytest.raises(WorkflowError, match="already compiled"):
                    workflow.add_node("new_node", node_func)

    def test_add_edge_after_compile_raises_error(self, mock_state_graph):
        """Test that adding edge after compile raises error."""
        mock_graph_class, mock_graph_instance = mock_state_graph
        mock_graph_instance.compile.return_value = MagicMock()

        with patch("llm_framework.workflow.base.LANGGRAPH_AVAILABLE", True):
            with patch("llm_framework.workflow.base.StateGraph", mock_graph_class):
                workflow = BaseWorkflow("test")
                workflow.set_entry("start")
                workflow.compile()

                with pytest.raises(WorkflowError, match="already compiled"):
                    workflow.add_edge("node1", "node2")

    def test_set_entry_after_compile_raises_error(self, mock_state_graph):
        """Test that setting entry after compile raises error."""
        mock_graph_class, mock_graph_instance = mock_state_graph
        mock_graph_instance.compile.return_value = MagicMock()

        with patch("llm_framework.workflow.base.LANGGRAPH_AVAILABLE", True):
            with patch("llm_framework.workflow.base.StateGraph", mock_graph_class):
                workflow = BaseWorkflow("test")
                workflow.set_entry("start")
                workflow.compile()

                with pytest.raises(WorkflowError, match="already compiled"):
                    workflow.set_entry("new_start")

    def test_run_without_compile_raises_error(self, mock_state_graph):
        """Test that run raises error when workflow is not compiled."""
        mock_graph_class, mock_graph_instance = mock_state_graph

        with patch("llm_framework.workflow.base.LANGGRAPH_AVAILABLE", True):
            with patch("llm_framework.workflow.base.StateGraph", mock_graph_class):
                workflow = BaseWorkflow("test")

                with pytest.raises(WorkflowError, match="uncompiled workflow"):
                    workflow.run({})

    def test_run_executes_workflow(self, mock_state_graph):
        """Test that run executes the compiled workflow."""
        mock_graph_class, mock_graph_instance = mock_state_graph
        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {"result": "success"}
        mock_graph_instance.compile.return_value = mock_compiled

        with patch("llm_framework.workflow.base.LANGGRAPH_AVAILABLE", True):
            with patch("llm_framework.workflow.base.StateGraph", mock_graph_class):
                with patch("llm_framework.workflow.base.MLFLOW_AVAILABLE", False):
                    workflow = BaseWorkflow("test")
                    workflow.set_entry("start")
                    workflow.compile()

                    input_state = {"messages": [], "metadata": {}}
                    result = workflow.run(input_state)

        assert result == {"result": "success"}
        mock_compiled.invoke.assert_called_once_with(input_state, config=None)

    def test_run_with_config(self, mock_state_graph):
        """Test that run passes config to compiled graph."""
        mock_graph_class, mock_graph_instance = mock_state_graph
        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {"result": "success"}
        mock_graph_instance.compile.return_value = mock_compiled

        with patch("llm_framework.workflow.base.LANGGRAPH_AVAILABLE", True):
            with patch("llm_framework.workflow.base.StateGraph", mock_graph_class):
                with patch("llm_framework.workflow.base.MLFLOW_AVAILABLE", False):
                    workflow = BaseWorkflow("test")
                    workflow.set_entry("start")
                    workflow.compile()

                    config = {"recursion_limit": 10}
                    workflow.run({}, config=config)

        call_args = mock_compiled.invoke.call_args
        assert call_args[1]["config"] == config

    def test_run_handles_execution_errors(self, mock_state_graph):
        """Test that run raises WorkflowError on execution failure."""
        mock_graph_class, mock_graph_instance = mock_state_graph
        mock_compiled = MagicMock()
        mock_compiled.invoke.side_effect = Exception("Execution failed")
        mock_graph_instance.compile.return_value = mock_compiled

        with patch("llm_framework.workflow.base.LANGGRAPH_AVAILABLE", True):
            with patch("llm_framework.workflow.base.StateGraph", mock_graph_class):
                with patch("llm_framework.workflow.base.MLFLOW_AVAILABLE", False):
                    workflow = BaseWorkflow("test")
                    workflow.set_entry("start")
                    workflow.compile()

                    with pytest.raises(WorkflowError, match="execution failed"):
                        workflow.run({})

    def test_run_with_mlflow_tracing(self, mock_state_graph):
        """Test that run uses MLflow tracing when available."""
        mock_graph_class, mock_graph_instance = mock_state_graph
        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {"result": "success"}
        mock_graph_instance.compile.return_value = mock_compiled

        mock_mlflow = MagicMock()
        mock_span = MagicMock()
        mock_mlflow.start_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_mlflow.start_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("llm_framework.workflow.base.LANGGRAPH_AVAILABLE", True):
            with patch("llm_framework.workflow.base.StateGraph", mock_graph_class):
                with patch("llm_framework.workflow.base.MLFLOW_AVAILABLE", True):
                    with patch("llm_framework.workflow.base.mlflow", mock_mlflow):
                        workflow = BaseWorkflow("test_workflow", WorkflowState)
                        workflow.set_entry("start")
                        workflow.compile()

                        workflow.run({})

        mock_mlflow.start_span.assert_called_once()
        assert mock_span.set_attribute.called


class TestAsyncWorkflow:
    """Test async workflow execution."""

    @pytest.fixture
    def mock_state_graph(self):
        """Mock StateGraph class."""
        mock_graph_class = MagicMock()
        mock_graph_instance = MagicMock()
        mock_graph_class.return_value = mock_graph_instance
        return mock_graph_class, mock_graph_instance

    @pytest.mark.asyncio
    async def test_arun_without_compile_raises_error(self, mock_state_graph):
        """Test that arun raises error when workflow is not compiled."""
        mock_graph_class, mock_graph_instance = mock_state_graph

        with patch("llm_framework.workflow.base.LANGGRAPH_AVAILABLE", True):
            with patch("llm_framework.workflow.base.StateGraph", mock_graph_class):
                workflow = BaseWorkflow("test")

                with pytest.raises(WorkflowError, match="uncompiled workflow"):
                    await workflow.arun({})

    @pytest.mark.asyncio
    async def test_arun_executes_workflow(self, mock_state_graph):
        """Test that arun executes the compiled workflow asynchronously."""
        mock_graph_class, mock_graph_instance = mock_state_graph
        mock_compiled = MagicMock()

        async def mock_ainvoke(state, config=None):
            return {"result": "async_success"}

        mock_compiled.ainvoke = mock_ainvoke
        mock_graph_instance.compile.return_value = mock_compiled

        with patch("llm_framework.workflow.base.LANGGRAPH_AVAILABLE", True):
            with patch("llm_framework.workflow.base.StateGraph", mock_graph_class):
                with patch("llm_framework.workflow.base.MLFLOW_AVAILABLE", False):
                    workflow = BaseWorkflow("test")
                    workflow.set_entry("start")
                    workflow.compile()

                    result = await workflow.arun({})

        assert result == {"result": "async_success"}
