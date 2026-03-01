"""MLflow tracing decorators — 為 LLM 呼叫、workflow node、整個 workflow 提供 trace。

所有 tracing 在 MLflow 不可用時自動降級為 no-op。

Usage:
    from app.tracking.tracer import trace_llm_call, trace_node, trace_workflow, span

    @trace_workflow("qa_pipeline")
    def run_pipeline(docs, question):
        ...

    @trace_node("preprocess")
    def preprocess(state):
        ...

    @trace_llm_call
    def call_llm(prompt):
        ...
"""

from __future__ import annotations

import functools
import time
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Generator, TypeVar

from app.logger import get_logger
from app.tracking.setup import is_mlflow_available

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def trace_llm_call(func: F) -> F:
    """Decorator: 將函式包裝為 MLflow LLM span。"""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not is_mlflow_available():
            return func(*args, **kwargs)
        try:
            import mlflow

            with mlflow.start_span(name=f"llm_call:{func.__name__}") as span_obj:
                span_obj.set_inputs({"args": args, "kwargs": kwargs})
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    elapsed = (time.perf_counter() - start) * 1000
                    span_obj.set_outputs(result)
                    span_obj.set_attributes({"latency_ms": elapsed})
                    return result
                except Exception as e:
                    elapsed = (time.perf_counter() - start) * 1000
                    span_obj.set_attributes({"latency_ms": elapsed, "error": str(e), "error_type": type(e).__name__})
                    span_obj.set_status("ERROR")
                    raise
        except Exception as e:
            warnings.warn(f"Tracing failed for {func.__name__}: {e}", stacklevel=2)
            return func(*args, **kwargs)

    return wrapper  # type: ignore


def trace_node(node_name: str) -> Callable[[F], F]:
    """Decorator: 將函式包裝為 workflow node span。"""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_mlflow_available():
                return func(*args, **kwargs)
            try:
                import mlflow

                with mlflow.start_span(name=f"node:{node_name}") as span_obj:
                    span_obj.set_inputs({"args": args, "kwargs": kwargs})
                    start = time.perf_counter()
                    try:
                        result = func(*args, **kwargs)
                        elapsed = (time.perf_counter() - start) * 1000
                        span_obj.set_outputs(result)
                        span_obj.set_attributes({"latency_ms": elapsed, "node_name": node_name})
                        return result
                    except Exception as e:
                        elapsed = (time.perf_counter() - start) * 1000
                        span_obj.set_attributes({
                            "latency_ms": elapsed,
                            "node_name": node_name,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        })
                        span_obj.set_status("ERROR")
                        raise
            except Exception as e:
                warnings.warn(f"Tracing failed for node {node_name}: {e}", stacklevel=2)
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def trace_workflow(workflow_name: str) -> Callable[[F], F]:
    """Decorator: 將函式包裝為 workflow 頂層 span。"""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_mlflow_available():
                return func(*args, **kwargs)
            try:
                import mlflow

                with mlflow.start_span(name=f"workflow:{workflow_name}") as span_obj:
                    span_obj.set_inputs({"args": args, "kwargs": kwargs})
                    start = time.perf_counter()
                    try:
                        result = func(*args, **kwargs)
                        elapsed = (time.perf_counter() - start) * 1000
                        span_obj.set_outputs(result)
                        span_obj.set_attributes({"latency_ms": elapsed, "workflow_name": workflow_name})
                        return result
                    except Exception as e:
                        elapsed = (time.perf_counter() - start) * 1000
                        span_obj.set_attributes({
                            "latency_ms": elapsed,
                            "workflow_name": workflow_name,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        })
                        span_obj.set_status("ERROR")
                        raise
            except Exception as e:
                warnings.warn(f"Tracing failed for workflow {workflow_name}: {e}", stacklevel=2)
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


@contextmanager
def span(name: str, span_type: str = "CHAIN") -> Generator[Any, None, None]:
    """Context manager: 建立自訂 trace span。

    Args:
        name: Span 名稱。
        span_type: Span 類型 ("LLM", "CHAIN", "TOOL", "RETRIEVER")。
    """
    if not is_mlflow_available():
        yield None
        return

    try:
        import mlflow

        with mlflow.start_span(name=name) as span_obj:
            span_obj.set_attributes({"span_type": span_type})
            start = time.perf_counter()
            try:
                yield span_obj
                elapsed = (time.perf_counter() - start) * 1000
                span_obj.set_attributes({"latency_ms": elapsed})
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                span_obj.set_attributes({"latency_ms": elapsed, "error": str(e), "error_type": type(e).__name__})
                span_obj.set_status("ERROR")
                raise
    except Exception as e:
        warnings.warn(f"Failed to create span {name}: {e}", stacklevel=2)
        yield None
