"""MLflow evaluation tools for LLM outputs.

Provides an Evaluator class for running evaluations on LLM outputs using
MLflow's evaluation API. Supports both built-in metrics and custom scoring
functions, with graceful degradation when MLflow is unavailable.

Usage:
    from llm_framework.mlflow.evaluator import Evaluator
    import pandas as pd

    # Create evaluator
    evaluator = Evaluator(experiment_name="qa_evaluation")

    # Prepare evaluation data
    data = pd.DataFrame({
        "question": ["What is the capital of France?", "Who wrote Hamlet?"],
        "response": ["Paris", "William Shakespeare"],
        "expected": ["Paris", "Shakespeare"]
    })

    # Run evaluation with built-in metrics
    result = evaluator.evaluate(
        data=data,
        metrics=["exact_match", "token_count"],
        model_output_col="response",
        target_col="expected"
    )

    print(f"Metrics: {result.metrics}")
    print(f"Per-row scores:\n{result.per_row}")

    # Compare multiple runs
    comparison = evaluator.compare(
        run_ids=["run1", "run2", "run3"],
        metric_keys=["exact_match", "f1_score"]
    )
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of an MLflow evaluation.

    Attributes:
        metrics: Dictionary of aggregate metrics (e.g., {"accuracy": 0.95}).
        per_row: DataFrame with per-row evaluation scores.
        run_id: MLflow run ID where the evaluation was logged.
    """
    metrics: dict[str, float]
    per_row: pd.DataFrame
    run_id: str | None = None


def _is_mlflow_available() -> bool:
    """Check if MLflow is both enabled in config and importable.

    Returns:
        True if MLflow can be used, False otherwise.
    """
    try:
        from llm_framework.config import get_config

        config = get_config()
        if not config.mlflow.enabled:
            return False

        import mlflow
        return True
    except ImportError:
        return False
    except Exception as e:
        logger.debug(f"MLflow availability check failed: {e}")
        return False


class Evaluator:
    """MLflow evaluator for LLM outputs.

    Provides methods to evaluate LLM responses against ground truth or
    using custom metrics, and to compare results across multiple runs.
    """

    def __init__(self, experiment_name: str | None = None) -> None:
        """Initialize the Evaluator.

        Args:
            experiment_name: Name of the MLflow experiment for logging
                evaluation results. If None, uses the default from config.
        """
        self.experiment_name = experiment_name
        self._mlflow_available = _is_mlflow_available()

        if self._mlflow_available:
            try:
                import mlflow
                from llm_framework.config import get_config

                config = get_config()
                if experiment_name:
                    mlflow.set_experiment(experiment_name)
                elif config.mlflow.experiment_name:
                    mlflow.set_experiment(config.mlflow.experiment_name)

                logger.debug(f"Evaluator initialized with experiment: {experiment_name or config.mlflow.experiment_name}")
            except Exception as e:
                warnings.warn(
                    f"Failed to initialize MLflow experiment: {e}. Evaluator will run with limited functionality.",
                    stacklevel=2
                )
                self._mlflow_available = False

    def evaluate(
        self,
        data: pd.DataFrame,
        metrics: list[str | Callable],
        model_output_col: str = "response",
        target_col: str = "expected",
    ) -> EvaluationResult:
        """Run evaluation on a dataset.

        Evaluates model outputs against targets using the specified metrics.
        Supports both built-in MLflow metric names (strings) and custom
        scoring functions.

        Args:
            data: DataFrame with columns for model outputs and targets.
            metrics: List of metric names or custom scoring callables.
                Built-in metrics include: "exact_match", "token_count",
                "toxicity", "flesch_kincaid_grade_level", etc.
                Custom metrics should be callables that take a DataFrame row
                and return a numeric score.
            model_output_col: Name of the column containing model outputs.
            target_col: Name of the column containing ground truth targets.

        Returns:
            EvaluationResult with aggregate metrics, per-row scores, and run ID.

        Example:
            # Built-in metrics
            result = evaluator.evaluate(
                data=test_df,
                metrics=["exact_match", "token_count"],
                model_output_col="prediction",
                target_col="ground_truth"
            )

            # Custom metric
            def custom_scorer(row):
                return len(row["prediction"].split())

            result = evaluator.evaluate(
                data=test_df,
                metrics=[custom_scorer],
                model_output_col="prediction"
            )
        """
        if not self._mlflow_available:
            warnings.warn(
                "MLflow is not available. Evaluation will run with basic metrics only.",
                stacklevel=2
            )
            return self._basic_evaluate(data, metrics, model_output_col, target_col)

        try:
            import mlflow

            # Prepare evaluation data
            eval_data = data.copy()

            # Build metric list
            eval_metrics = []
            for metric in metrics:
                if isinstance(metric, str):
                    # Use built-in metric name
                    eval_metrics.append(metric)
                elif callable(metric):
                    # Wrap custom function as MLflow metric
                    try:
                        from mlflow.metrics import make_metric

                        metric_name = getattr(metric, '__name__', 'custom_metric')
                        custom_metric = make_metric(
                            eval_fn=metric,
                            greater_is_better=True,
                            name=metric_name
                        )
                        eval_metrics.append(custom_metric)
                    except Exception as e:
                        logger.warning(f"Failed to create custom metric {metric}: {e}")
                        continue

            # Run evaluation
            with mlflow.start_run() as run:
                result = mlflow.evaluate(
                    data=eval_data,
                    targets=target_col,
                    predictions=model_output_col,
                    extra_metrics=eval_metrics if eval_metrics else None,
                )

                # Extract results
                metrics_dict = result.metrics if hasattr(result, 'metrics') else {}
                per_row_df = result.tables['eval_results_table'] if hasattr(result, 'tables') else pd.DataFrame()

                logger.info(f"Evaluation complete. Metrics: {metrics_dict}")

                return EvaluationResult(
                    metrics=metrics_dict,
                    per_row=per_row_df,
                    run_id=run.info.run_id
                )

        except Exception as e:
            warnings.warn(
                f"MLflow evaluation failed: {e}. Falling back to basic evaluation.",
                stacklevel=2
            )
            logger.debug(f"MLflow evaluation error details: {e}", exc_info=True)
            return self._basic_evaluate(data, metrics, model_output_col, target_col)

    def compare(
        self,
        run_ids: list[str],
        metric_keys: list[str],
    ) -> pd.DataFrame:
        """Compare metrics across multiple MLflow runs.

        Retrieves the specified metrics from each run and returns a
        comparison DataFrame.

        Args:
            run_ids: List of MLflow run IDs to compare.
            metric_keys: List of metric names to compare.

        Returns:
            DataFrame with runs as rows and metrics as columns.

        Example:
            comparison = evaluator.compare(
                run_ids=["abc123", "def456", "ghi789"],
                metric_keys=["accuracy", "f1_score", "latency_ms"]
            )
            print(comparison)
        """
        if not self._mlflow_available:
            warnings.warn(
                "MLflow is not available. Cannot compare runs.",
                stacklevel=2
            )
            return pd.DataFrame()

        try:
            import mlflow

            results = []
            for run_id in run_ids:
                try:
                    run = mlflow.get_run(run_id)
                    row = {"run_id": run_id, "run_name": run.info.run_name}

                    for metric_key in metric_keys:
                        row[metric_key] = run.data.metrics.get(metric_key, None)

                    results.append(row)
                except Exception as e:
                    logger.warning(f"Failed to retrieve run {run_id}: {e}")
                    continue

            comparison_df = pd.DataFrame(results)
            logger.info(f"Compared {len(results)} runs across {len(metric_keys)} metrics")
            return comparison_df

        except Exception as e:
            warnings.warn(
                f"Failed to compare runs: {e}. Returning empty DataFrame.",
                stacklevel=2
            )
            logger.debug(f"MLflow comparison error details: {e}", exc_info=True)
            return pd.DataFrame()

    def _basic_evaluate(
        self,
        data: pd.DataFrame,
        metrics: list[str | Callable],
        model_output_col: str,
        target_col: str,
    ) -> EvaluationResult:
        """Fallback evaluation without MLflow.

        Computes basic metrics when MLflow is unavailable.

        Args:
            data: DataFrame with model outputs and targets.
            metrics: List of metric names or callables.
            model_output_col: Column with model outputs.
            target_col: Column with ground truth.

        Returns:
            EvaluationResult with computed metrics.
        """
        per_row = data.copy()
        aggregate_metrics = {}

        # Compute exact match if available
        if target_col in data.columns and model_output_col in data.columns:
            per_row['exact_match'] = (
                data[model_output_col].astype(str) == data[target_col].astype(str)
            ).astype(int)
            aggregate_metrics['exact_match'] = per_row['exact_match'].mean()

        # Compute token count
        if model_output_col in data.columns:
            per_row['token_count'] = data[model_output_col].astype(str).str.split().str.len()
            aggregate_metrics['avg_token_count'] = per_row['token_count'].mean()

        # Apply custom callable metrics
        for metric in metrics:
            if callable(metric):
                try:
                    metric_name = getattr(metric, '__name__', 'custom')
                    per_row[metric_name] = data.apply(metric, axis=1)
                    aggregate_metrics[f"avg_{metric_name}"] = per_row[metric_name].mean()
                except Exception as e:
                    logger.warning(f"Failed to compute custom metric: {e}")

        logger.info(f"Basic evaluation complete. Metrics: {aggregate_metrics}")

        return EvaluationResult(
            metrics=aggregate_metrics,
            per_row=per_row,
            run_id=None
        )
