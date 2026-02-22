"""MLflow integration modules for logging, tracing, evaluation, and experiment management.

This package provides comprehensive MLflow integration for LLM workflows:

- logger: Log LLM calls, parameters, metrics, and artifacts
- tracer: Trace LLM calls and workflows with hierarchical spans
- evaluator: Evaluate LLM outputs with built-in and custom metrics
- experiment: Manage experiments, runs, and run history

All modules gracefully degrade when MLflow is disabled or unavailable,
ensuring the framework continues to work without MLflow dependencies.

Usage:
    from llm_framework.mlflow import (
        log_llm_call, log_params, log_metrics,
        trace_llm_call, trace_workflow,
        Evaluator, ExperimentManager
    )
"""

from llm_framework.mlflow.evaluator import EvaluationResult, Evaluator
from llm_framework.mlflow.experiment import ExperimentManager
from llm_framework.mlflow.logger import (
    log_artifact,
    log_dict_artifact,
    log_llm_call,
    log_metrics,
    log_params,
)
from llm_framework.mlflow.tracer import (
    span,
    trace_llm_call,
    trace_node,
    trace_workflow,
)

__all__ = [
    # Logger functions
    "log_llm_call",
    "log_params",
    "log_metrics",
    "log_artifact",
    "log_dict_artifact",
    # Tracer decorators and context managers
    "trace_llm_call",
    "trace_node",
    "trace_workflow",
    "span",
    # Evaluator
    "Evaluator",
    "EvaluationResult",
    # Experiment manager
    "ExperimentManager",
]
