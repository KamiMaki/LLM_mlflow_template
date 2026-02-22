"""MLflow tracing decorators and context managers for LLM workflows.

Provides decorators and utilities for tracing LLM calls, workflow nodes,
and entire workflows using MLflow's tracing API. All tracing functions
gracefully degrade to no-ops if MLflow is disabled or unavailable.

Usage:
    from llm_framework.mlflow.tracer import trace_llm_call, trace_node, trace_workflow, span

    @trace_workflow("document_processing")
    def process_documents(docs):
        results = []
        for doc in docs:
            results.append(extract_entities(doc))
        return results

    @trace_node("entity_extraction")
    def extract_entities(document):
        with span("llm_call", span_type="LLM"):
            entities = llm.extract(document)
        return entities

    @trace_llm_call
    def call_llm(prompt, model="gpt-4o"):
        # Your LLM call logic here
        response = client.chat(prompt, model=model)
        return response
"""

from __future__ import annotations

import functools
import logging
import time
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Generator, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


def _is_tracing_available() -> bool:
    """Check if MLflow tracing is available.

    Returns:
        True if MLflow is enabled, importable, and supports tracing.
    """
    try:
        from llm_framework.config import get_config

        config = get_config()
        if not config.mlflow.enabled:
            return False

        # Try importing mlflow and check for tracing support
        import mlflow
        # Verify tracing API exists
        return hasattr(mlflow, 'start_span')
    except ImportError:
        return False
    except Exception as e:
        logger.debug(f"MLflow tracing availability check failed: {e}")
        return False


def trace_llm_call(func: F) -> F:
    """Decorator that traces a function as an LLM span.

    Wraps the function execution in an MLflow span with type "LLM",
    capturing inputs, outputs, execution time, and any exceptions.

    Args:
        func: Function to trace (typically an LLM API call).

    Returns:
        Wrapped function that traces execution.

    Example:
        @trace_llm_call
        def call_openai(prompt: str, model: str = "gpt-4o") -> str:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
    """
    if not _is_tracing_available():
        return func

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            import mlflow

            span_name = f"llm_call:{func.__name__}"
            with mlflow.start_span(name=span_name) as span_obj:
                # Log inputs
                span_obj.set_inputs({"args": args, "kwargs": kwargs})

                # Execute function and measure time
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000

                    # Log outputs and metrics
                    span_obj.set_outputs(result)
                    span_obj.set_attributes({"latency_ms": elapsed_ms})

                    return result
                except Exception as e:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    span_obj.set_attributes({
                        "latency_ms": elapsed_ms,
                        "error": str(e),
                        "error_type": type(e).__name__
                    })
                    span_obj.set_status("ERROR")
                    raise

        except Exception as e:
            # If tracing fails, log warning and execute function normally
            warnings.warn(
                f"Failed to trace LLM call {func.__name__}: {e}. Executing without tracing.",
                stacklevel=2
            )
            logger.debug(f"MLflow tracing error details: {e}", exc_info=True)
            return func(*args, **kwargs)

    return wrapper  # type: ignore


def trace_node(node_name: str) -> Callable[[F], F]:
    """Decorator that traces a workflow node as a span.

    Wraps the function execution in an MLflow span with type "CHAIN",
    representing a node in a workflow or processing pipeline.

    Args:
        node_name: Name of the workflow node for display in traces.

    Returns:
        Decorator function.

    Example:
        @trace_node("preprocess")
        def preprocess_data(raw_data: list[str]) -> list[str]:
            return [text.lower().strip() for text in raw_data]

        @trace_node("inference")
        def run_inference(data: list[str]) -> list[dict]:
            return [model.predict(item) for item in data]
    """
    if not _is_tracing_available():
        def decorator(func: F) -> F:
            return func
        return decorator

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                import mlflow

                span_name = f"node:{node_name}"
                with mlflow.start_span(name=span_name) as span_obj:
                    # Log inputs
                    span_obj.set_inputs({"args": args, "kwargs": kwargs})

                    # Execute function and measure time
                    start_time = time.perf_counter()
                    try:
                        result = func(*args, **kwargs)
                        elapsed_ms = (time.perf_counter() - start_time) * 1000

                        # Log outputs and metrics
                        span_obj.set_outputs(result)
                        span_obj.set_attributes({
                            "latency_ms": elapsed_ms,
                            "node_name": node_name
                        })

                        return result
                    except Exception as e:
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        span_obj.set_attributes({
                            "latency_ms": elapsed_ms,
                            "node_name": node_name,
                            "error": str(e),
                            "error_type": type(e).__name__
                        })
                        span_obj.set_status("ERROR")
                        raise

            except Exception as e:
                warnings.warn(
                    f"Failed to trace node {node_name}: {e}. Executing without tracing.",
                    stacklevel=2
                )
                logger.debug(f"MLflow tracing error details: {e}", exc_info=True)
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def trace_workflow(workflow_name: str) -> Callable[[F], F]:
    """Decorator that traces an entire workflow as a parent span.

    Creates a top-level trace span for the workflow, under which all
    nested operations (nodes, LLM calls) will be traced as child spans.

    Args:
        workflow_name: Name of the workflow for display in traces.

    Returns:
        Decorator function.

    Example:
        @trace_workflow("document_qa_pipeline")
        def run_qa_pipeline(documents: list[str], question: str) -> str:
            # All nested trace_node and trace_llm_call operations
            # will appear as children of this workflow span
            context = retrieve_context(documents, question)
            answer = generate_answer(context, question)
            return answer
    """
    if not _is_tracing_available():
        def decorator(func: F) -> F:
            return func
        return decorator

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                import mlflow

                span_name = f"workflow:{workflow_name}"
                with mlflow.start_span(name=span_name) as span_obj:
                    # Log inputs
                    span_obj.set_inputs({"args": args, "kwargs": kwargs})

                    # Execute workflow and measure time
                    start_time = time.perf_counter()
                    try:
                        result = func(*args, **kwargs)
                        elapsed_ms = (time.perf_counter() - start_time) * 1000

                        # Log outputs and metrics
                        span_obj.set_outputs(result)
                        span_obj.set_attributes({
                            "latency_ms": elapsed_ms,
                            "workflow_name": workflow_name
                        })

                        return result
                    except Exception as e:
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        span_obj.set_attributes({
                            "latency_ms": elapsed_ms,
                            "workflow_name": workflow_name,
                            "error": str(e),
                            "error_type": type(e).__name__
                        })
                        span_obj.set_status("ERROR")
                        raise

            except Exception as e:
                warnings.warn(
                    f"Failed to trace workflow {workflow_name}: {e}. Executing without tracing.",
                    stacklevel=2
                )
                logger.debug(f"MLflow tracing error details: {e}", exc_info=True)
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


@contextmanager
def span(name: str, span_type: str = "CHAIN") -> Generator[Any, None, None]:
    """Context manager for creating custom trace spans.

    Creates a span for a specific block of code. Useful for tracing
    operations that don't fit into function boundaries.

    Args:
        name: Name of the span for display in traces.
        span_type: Type of span. Options: "LLM", "CHAIN", "TOOL", "RETRIEVER".
            Defaults to "CHAIN".

    Yields:
        Span object (if tracing is available), or None.

    Example:
        with span("database_query", span_type="TOOL"):
            results = database.query("SELECT * FROM users")

        with span("embedding_generation", span_type="LLM"):
            embeddings = model.embed(texts)

        with span("vector_search", span_type="RETRIEVER"):
            docs = vector_store.similarity_search(query, k=5)
    """
    if not _is_tracing_available():
        # No-op context manager
        yield None
        return

    try:
        import mlflow

        with mlflow.start_span(name=name) as span_obj:
            # Set span type as an attribute
            span_obj.set_attributes({"span_type": span_type})
            start_time = time.perf_counter()

            try:
                yield span_obj
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                span_obj.set_attributes({"latency_ms": elapsed_ms})
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                span_obj.set_attributes({
                    "latency_ms": elapsed_ms,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                span_obj.set_status("ERROR")
                raise

    except Exception as e:
        warnings.warn(
            f"Failed to create span {name}: {e}. Continuing without tracing.",
            stacklevel=2
        )
        logger.debug(f"MLflow tracing error details: {e}", exc_info=True)
        yield None
