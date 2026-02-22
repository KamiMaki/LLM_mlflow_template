"""Base workflow wrapper around LangGraph's StateGraph.

Provides a high-level API for building LangGraph workflows with automatic
MLflow tracing integration and a chainable builder pattern.

Usage:
    from llm_framework.workflow.base import BaseWorkflow
    from llm_framework.workflow.state import WorkflowState

    def process_node(state: WorkflowState) -> dict:
        return {"current_step": "processed"}

    workflow = (
        BaseWorkflow("my_workflow", WorkflowState)
        .add_node("process", process_node)
        .set_entry("process")
        .set_finish("process")
        .compile()
    )

    result = workflow.run({"messages": [], "metadata": {}})
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from llm_framework.workflow.state import WorkflowState

logger = logging.getLogger(__name__)

# Optional MLflow integration
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END, START
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # Define placeholder types for type hints
    StateGraph = None
    END = "__end__"
    START = "__start__"


class WorkflowError(Exception):
    """Raised when workflow construction or execution fails."""


class BaseWorkflow:
    """Wrapper around LangGraph's StateGraph with builder pattern.

    Provides a chainable API for constructing workflows and automatic
    MLflow tracing when available.

    Args:
        name: Workflow identifier for logging and tracing.
        state_schema: TypedDict class defining the workflow state structure.

    Raises:
        WorkflowError: If LangGraph is not installed.
    """

    def __init__(self, name: str, state_schema: type = WorkflowState):
        if not LANGGRAPH_AVAILABLE:
            raise WorkflowError(
                "LangGraph is not installed. Install with: pip install langgraph"
            )

        self.name = name
        self.state_schema = state_schema
        self._graph = StateGraph(state_schema)
        self._compiled: Any = None  # CompiledGraph from LangGraph
        self._entry_node: str | None = None
        self._finish_nodes: set[str] = set()

        logger.debug(f"Initialized workflow '{name}' with state schema {state_schema.__name__}")

    def add_node(
        self,
        name: str,
        func: Callable,
        **kwargs,
    ) -> BaseWorkflow:
        """Add a node to the workflow graph.

        Args:
            name: Unique node identifier.
            func: Callable that takes state dict and returns state updates.
            **kwargs: Additional arguments passed to StateGraph.add_node().

        Returns:
            Self for method chaining.

        Raises:
            WorkflowError: If the graph is already compiled.
        """
        if self._compiled is not None:
            raise WorkflowError(
                f"Cannot add node '{name}' to already compiled workflow '{self.name}'"
            )

        self._graph.add_node(name, func, **kwargs)
        logger.debug(f"Added node '{name}' to workflow '{self.name}'")
        return self

    def add_edge(self, source: str, target: str) -> BaseWorkflow:
        """Add a directed edge between two nodes.

        Args:
            source: Source node name.
            target: Target node name (or END constant).

        Returns:
            Self for method chaining.

        Raises:
            WorkflowError: If the graph is already compiled.
        """
        if self._compiled is not None:
            raise WorkflowError(
                f"Cannot add edge to already compiled workflow '{self.name}'"
            )

        self._graph.add_edge(source, target)
        logger.debug(f"Added edge '{source}' -> '{target}' in workflow '{self.name}'")
        return self

    def add_conditional_edge(
        self,
        source: str,
        condition: Callable,
        targets: dict[str, str],
    ) -> BaseWorkflow:
        """Add a conditional edge with routing logic.

        Args:
            source: Source node name.
            condition: Function that takes state and returns a key from targets dict.
            targets: Mapping from condition return values to target node names.

        Returns:
            Self for method chaining.

        Raises:
            WorkflowError: If the graph is already compiled.
        """
        if self._compiled is not None:
            raise WorkflowError(
                f"Cannot add conditional edge to already compiled workflow '{self.name}'"
            )

        self._graph.add_conditional_edges(source, condition, targets)
        logger.debug(
            f"Added conditional edge from '{source}' with {len(targets)} targets "
            f"in workflow '{self.name}'"
        )
        return self

    def set_entry(self, node_name: str) -> BaseWorkflow:
        """Set the entry point for the workflow.

        Args:
            node_name: Name of the node where execution begins.

        Returns:
            Self for method chaining.

        Raises:
            WorkflowError: If the graph is already compiled.
        """
        if self._compiled is not None:
            raise WorkflowError(
                f"Cannot set entry point on already compiled workflow '{self.name}'"
            )

        self._graph.set_entry_point(node_name)
        self._entry_node = node_name
        logger.debug(f"Set entry point to '{node_name}' in workflow '{self.name}'")
        return self

    def set_finish(self, node_name: str) -> BaseWorkflow:
        """Set a node to connect to the END state.

        Args:
            node_name: Name of the node that should terminate the workflow.

        Returns:
            Self for method chaining.

        Raises:
            WorkflowError: If the graph is already compiled.
        """
        if self._compiled is not None:
            raise WorkflowError(
                f"Cannot set finish point on already compiled workflow '{self.name}'"
            )

        self._graph.add_edge(node_name, END)
        self._finish_nodes.add(node_name)
        logger.debug(f"Set finish point at '{node_name}' in workflow '{self.name}'")
        return self

    def compile(self) -> BaseWorkflow:
        """Compile the workflow graph into an executable form.

        Returns:
            Self with compiled graph ready for execution.

        Raises:
            WorkflowError: If entry point is not set or compilation fails.
        """
        if self._entry_node is None:
            raise WorkflowError(
                f"Cannot compile workflow '{self.name}' without entry point. "
                "Call set_entry() first."
            )

        try:
            self._compiled = self._graph.compile()
            logger.info(f"Successfully compiled workflow '{self.name}'")
        except Exception as exc:
            raise WorkflowError(
                f"Failed to compile workflow '{self.name}': {exc}"
            ) from exc

        return self

    def run(
        self,
        input_state: dict,
        config: dict[str, Any] | None = None,
    ) -> dict:
        """Execute the workflow synchronously.

        Automatically traces execution with MLflow if available.

        Args:
            input_state: Initial state dict matching the state schema.
            config: Optional LangGraph runtime configuration.

        Returns:
            Final state dict after workflow completion.

        Raises:
            WorkflowError: If the workflow is not compiled or execution fails.
        """
        if self._compiled is None:
            raise WorkflowError(
                f"Cannot run uncompiled workflow '{self.name}'. Call compile() first."
            )

        logger.info(f"Running workflow '{self.name}'")

        # Run with optional MLflow tracing
        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_span(name=f"workflow.{self.name}") as span:
                    span.set_attribute("workflow.name", self.name)
                    span.set_attribute("workflow.state_schema", self.state_schema.__name__)
                    result = self._compiled.invoke(input_state, config=config)
                    logger.info(f"Workflow '{self.name}' completed successfully")
                    return result
            except Exception as exc:
                logger.error(f"Workflow '{self.name}' failed: {exc}")
                raise WorkflowError(
                    f"Workflow '{self.name}' execution failed: {exc}"
                ) from exc
        else:
            try:
                result = self._compiled.invoke(input_state, config=config)
                logger.info(f"Workflow '{self.name}' completed successfully")
                return result
            except Exception as exc:
                logger.error(f"Workflow '{self.name}' failed: {exc}")
                raise WorkflowError(
                    f"Workflow '{self.name}' execution failed: {exc}"
                ) from exc

    async def arun(
        self,
        input_state: dict,
        config: dict[str, Any] | None = None,
    ) -> dict:
        """Execute the workflow asynchronously.

        Automatically traces execution with MLflow if available.

        Args:
            input_state: Initial state dict matching the state schema.
            config: Optional LangGraph runtime configuration.

        Returns:
            Final state dict after workflow completion.

        Raises:
            WorkflowError: If the workflow is not compiled or execution fails.
        """
        if self._compiled is None:
            raise WorkflowError(
                f"Cannot run uncompiled workflow '{self.name}'. Call compile() first."
            )

        logger.info(f"Running workflow '{self.name}' (async)")

        # Run with optional MLflow tracing
        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_span(name=f"workflow.{self.name}") as span:
                    span.set_attribute("workflow.name", self.name)
                    span.set_attribute("workflow.state_schema", self.state_schema.__name__)
                    result = await self._compiled.ainvoke(input_state, config=config)
                    logger.info(f"Workflow '{self.name}' completed successfully (async)")
                    return result
            except Exception as exc:
                logger.error(f"Workflow '{self.name}' failed (async): {exc}")
                raise WorkflowError(
                    f"Workflow '{self.name}' async execution failed: {exc}"
                ) from exc
        else:
            try:
                result = await self._compiled.ainvoke(input_state, config=config)
                logger.info(f"Workflow '{self.name}' completed successfully (async)")
                return result
            except Exception as exc:
                logger.error(f"Workflow '{self.name}' failed (async): {exc}")
                raise WorkflowError(
                    f"Workflow '{self.name}' async execution failed: {exc}"
                ) from exc
