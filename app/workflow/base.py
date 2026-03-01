"""BaseWorkflow — LangGraph StateGraph builder pattern wrapper。

Usage:
    from app.workflow.base import BaseWorkflow
    from app.workflow.state import WorkflowState

    workflow = (
        BaseWorkflow("my_workflow", WorkflowState)
        .add_node("process", process_fn)
        .set_entry("process")
        .set_finish("process")
        .compile()
    )
    result = workflow.run({"messages": [], "metadata": {}})
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.logger import get_logger
from app.tracking.setup import is_mlflow_available
from app.workflow.state import WorkflowState

logger = get_logger(__name__)

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None  # type: ignore
    END = "__end__"


class WorkflowError(Exception):
    """Workflow 建構或執行失敗時拋出。"""


class BaseWorkflow:
    """LangGraph StateGraph 的高階 wrapper，支援 builder pattern 與 MLflow tracing。

    Args:
        name: Workflow 識別名稱。
        state_schema: TypedDict class 定義 workflow state 結構。
    """

    def __init__(self, name: str, state_schema: type = WorkflowState):
        if not LANGGRAPH_AVAILABLE:
            raise WorkflowError("LangGraph is not installed. Install with: pip install langgraph")

        self.name = name
        self.state_schema = state_schema
        self._graph = StateGraph(state_schema)
        self._compiled: Any = None
        self._entry_node: str | None = None
        self._finish_nodes: set[str] = set()
        logger.debug(f"Initialized workflow '{name}'")

    def add_node(self, name: str, func: Callable, **kwargs) -> BaseWorkflow:
        """新增 node 到 workflow graph。"""
        if self._compiled is not None:
            raise WorkflowError(f"Cannot modify compiled workflow '{self.name}'")
        self._graph.add_node(name, func, **kwargs)
        logger.debug(f"Added node '{name}' to '{self.name}'")
        return self

    def add_edge(self, source: str, target: str) -> BaseWorkflow:
        """新增有向邊。"""
        if self._compiled is not None:
            raise WorkflowError(f"Cannot modify compiled workflow '{self.name}'")
        self._graph.add_edge(source, target)
        logger.debug(f"Added edge '{source}' -> '{target}'")
        return self

    def add_conditional_edge(self, source: str, condition: Callable, targets: dict[str, str]) -> BaseWorkflow:
        """新增條件邊。"""
        if self._compiled is not None:
            raise WorkflowError(f"Cannot modify compiled workflow '{self.name}'")
        self._graph.add_conditional_edges(source, condition, targets)
        logger.debug(f"Added conditional edge from '{source}'")
        return self

    def set_entry(self, node_name: str) -> BaseWorkflow:
        """設定入口節點。"""
        if self._compiled is not None:
            raise WorkflowError(f"Cannot modify compiled workflow '{self.name}'")
        self._graph.set_entry_point(node_name)
        self._entry_node = node_name
        logger.debug(f"Entry point: '{node_name}'")
        return self

    def set_finish(self, node_name: str) -> BaseWorkflow:
        """設定終止節點（連接到 END）。"""
        if self._compiled is not None:
            raise WorkflowError(f"Cannot modify compiled workflow '{self.name}'")
        self._graph.add_edge(node_name, END)
        self._finish_nodes.add(node_name)
        logger.debug(f"Finish point: '{node_name}'")
        return self

    def compile(self) -> BaseWorkflow:
        """編譯 workflow graph。"""
        if self._entry_node is None:
            raise WorkflowError(f"No entry point set for workflow '{self.name}'")
        try:
            self._compiled = self._graph.compile()
            logger.info(f"Compiled workflow '{self.name}'")
        except Exception as exc:
            raise WorkflowError(f"Failed to compile '{self.name}': {exc}") from exc
        return self

    def run(self, input_state: dict, config: dict[str, Any] | None = None) -> dict:
        """同步執行 workflow，自動以 MLflow tracing 包裝。"""
        if self._compiled is None:
            raise WorkflowError(f"Workflow '{self.name}' not compiled")

        logger.info(f"Running workflow '{self.name}'")
        try:
            if is_mlflow_available():
                import mlflow
                with mlflow.start_span(name=f"workflow.{self.name}") as span_obj:
                    span_obj.set_attribute("workflow.name", self.name)
                    result = self._compiled.invoke(input_state, config=config)
            else:
                result = self._compiled.invoke(input_state, config=config)
            logger.info(f"Workflow '{self.name}' completed")
            return result
        except Exception as exc:
            logger.error(f"Workflow '{self.name}' failed: {exc}")
            raise WorkflowError(f"Workflow '{self.name}' failed: {exc}") from exc

    async def arun(self, input_state: dict, config: dict[str, Any] | None = None) -> dict:
        """非同步執行 workflow。"""
        if self._compiled is None:
            raise WorkflowError(f"Workflow '{self.name}' not compiled")

        logger.info(f"Running workflow '{self.name}' (async)")
        try:
            if is_mlflow_available():
                import mlflow
                with mlflow.start_span(name=f"workflow.{self.name}") as span_obj:
                    span_obj.set_attribute("workflow.name", self.name)
                    result = await self._compiled.ainvoke(input_state, config=config)
            else:
                result = await self._compiled.ainvoke(input_state, config=config)
            logger.info(f"Workflow '{self.name}' completed (async)")
            return result
        except Exception as exc:
            logger.error(f"Workflow '{self.name}' failed (async): {exc}")
            raise WorkflowError(f"Workflow '{self.name}' async failed: {exc}") from exc
