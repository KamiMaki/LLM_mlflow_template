"""MLflow experiment management for organizing and tracking runs.

Provides an ExperimentManager class for creating experiments, managing runs,
and querying run history. All operations gracefully degrade when MLflow
is unavailable.

Usage:
    from llm_framework.mlflow.experiment import ExperimentManager

    # Initialize manager
    manager = ExperimentManager()

    # Get or create experiment
    exp_id = manager.get_or_create_experiment("my_experiment")

    # Start a run and log data
    with manager.start_run(run_name="baseline_v1", tags={"version": "1.0"}):
        # Your code here
        from llm_framework.mlflow.logger import log_params, log_metrics

        log_params({"model": "gpt-4o", "temperature": 0.7})
        log_metrics({"accuracy": 0.95})

    # List all runs in an experiment
    runs_df = manager.list_runs(experiment_name="my_experiment")
    print(runs_df[["run_id", "run_name", "metrics.accuracy"]])

    # Find the best run by metric
    best_run = manager.get_best_run(
        metric="accuracy",
        experiment_name="my_experiment"
    )
    print(f"Best run: {best_run['run_id']} with accuracy {best_run['metrics.accuracy']}")
"""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from typing import Any, Generator

import pandas as pd

logger = logging.getLogger(__name__)


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


class ExperimentManager:
    """Manager for MLflow experiments and runs.

    Handles experiment creation, run management, and querying run history.
    Provides a high-level interface for organizing ML experiments.
    """

    def __init__(self, config: Any | None = None) -> None:
        """Initialize the ExperimentManager.

        Args:
            config: Optional FrameworkConfig instance. If None, uses the
                global config from get_config().
        """
        self._mlflow_available = _is_mlflow_available()
        self._config = config

        if self._mlflow_available:
            try:
                import mlflow
                from llm_framework.config import get_config

                cfg = config if config else get_config()

                # Set tracking URI if provided
                if cfg.mlflow.tracking_uri:
                    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
                    logger.debug(f"MLflow tracking URI set to: {cfg.mlflow.tracking_uri}")

                # Set default experiment
                if cfg.mlflow.experiment_name:
                    mlflow.set_experiment(cfg.mlflow.experiment_name)
                    logger.debug(f"Default experiment set to: {cfg.mlflow.experiment_name}")

            except Exception as e:
                warnings.warn(
                    f"Failed to initialize MLflow: {e}. Experiment manager will run with limited functionality.",
                    stacklevel=2
                )
                self._mlflow_available = False

    def get_or_create_experiment(self, name: str | None = None) -> str | None:
        """Get or create an MLflow experiment by name.

        If the experiment exists, returns its ID. Otherwise, creates it
        and returns the new ID. If name is None, uses the default from config.

        Args:
            name: Experiment name. If None, uses config default.

        Returns:
            Experiment ID as a string, or None if MLflow is unavailable.

        Example:
            exp_id = manager.get_or_create_experiment("document_qa")
            print(f"Experiment ID: {exp_id}")
        """
        if not self._mlflow_available:
            return None

        try:
            import mlflow
            from llm_framework.config import get_config

            cfg = self._config if self._config else get_config()
            experiment_name = name if name else cfg.mlflow.experiment_name

            if not experiment_name:
                logger.warning("No experiment name provided and no default in config")
                return None

            # Try to get existing experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)

            if experiment is not None:
                logger.debug(f"Found existing experiment: {experiment_name} (ID: {experiment.experiment_id})")
                return experiment.experiment_id

            # Create new experiment
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
            return experiment_id

        except Exception as e:
            warnings.warn(
                f"Failed to get or create experiment: {e}. Continuing without experiment.",
                stacklevel=2
            )
            logger.debug(f"MLflow experiment error details: {e}", exc_info=True)
            return None

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Generator[Any, None, None]:
        """Context manager for starting an MLflow run.

        Creates a new MLflow run within the current experiment. All logging
        operations within the context will be associated with this run.

        Args:
            run_name: Optional name for the run (for easier identification).
            tags: Optional dictionary of tags to attach to the run.

        Yields:
            MLflow run object (if available), or None.

        Example:
            with manager.start_run(run_name="experiment_1", tags={"type": "baseline"}):
                log_params({"learning_rate": 0.001})
                # Train model
                log_metrics({"loss": 0.05, "accuracy": 0.95})
        """
        if not self._mlflow_available:
            # No-op context manager
            logger.debug("MLflow unavailable, starting no-op run context")
            yield None
            return

        try:
            import mlflow

            with mlflow.start_run(run_name=run_name, tags=tags) as run:
                logger.debug(f"Started MLflow run: {run.info.run_id}" + (f" ({run_name})" if run_name else ""))
                yield run

        except Exception as e:
            warnings.warn(
                f"Failed to start MLflow run: {e}. Continuing without run tracking.",
                stacklevel=2
            )
            logger.debug(f"MLflow run start error details: {e}", exc_info=True)
            yield None

    def list_runs(
        self,
        experiment_name: str | None = None,
        filter_string: str | None = None,
    ) -> pd.DataFrame:
        """List all runs in an experiment.

        Retrieves run metadata, parameters, and metrics as a DataFrame.
        Useful for analyzing experiment history and comparing runs.

        Args:
            experiment_name: Name of the experiment to query. If None,
                uses the default from config.
            filter_string: Optional MLflow filter string to narrow results.
                Example: "metrics.accuracy > 0.9" or "params.model = 'gpt-4o'"

        Returns:
            DataFrame with columns for run metadata, parameters, and metrics.
            Returns empty DataFrame if MLflow is unavailable.

        Example:
            # List all runs
            runs = manager.list_runs("my_experiment")

            # Filter by metric
            good_runs = manager.list_runs(
                "my_experiment",
                filter_string="metrics.accuracy > 0.9"
            )

            # Display key columns
            print(runs[["run_id", "run_name", "metrics.accuracy", "params.model"]])
        """
        if not self._mlflow_available:
            warnings.warn(
                "MLflow is not available. Cannot list runs.",
                stacklevel=2
            )
            return pd.DataFrame()

        try:
            import mlflow
            from llm_framework.config import get_config

            cfg = self._config if self._config else get_config()
            exp_name = experiment_name if experiment_name else cfg.mlflow.experiment_name

            if not exp_name:
                logger.warning("No experiment name provided")
                return pd.DataFrame()

            # Get experiment
            experiment = mlflow.get_experiment_by_name(exp_name)
            if experiment is None:
                logger.warning(f"Experiment not found: {exp_name}")
                return pd.DataFrame()

            # Search runs
            runs_df = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string,
                order_by=["start_time DESC"]
            )

            logger.debug(f"Found {len(runs_df)} runs in experiment {exp_name}")
            return runs_df

        except Exception as e:
            warnings.warn(
                f"Failed to list runs: {e}. Returning empty DataFrame.",
                stacklevel=2
            )
            logger.debug(f"MLflow list runs error details: {e}", exc_info=True)
            return pd.DataFrame()

    def get_best_run(
        self,
        metric: str,
        experiment_name: str | None = None,
        ascending: bool = False,
    ) -> dict[str, Any] | None:
        """Get the run with the best value for a specific metric.

        Finds the run with the highest (or lowest, if ascending=True) value
        for the specified metric.

        Args:
            metric: Name of the metric to optimize (e.g., "accuracy", "loss").
            experiment_name: Name of the experiment to search. If None,
                uses the default from config.
            ascending: If True, lower metric values are better (e.g., for loss).
                If False (default), higher values are better (e.g., for accuracy).

        Returns:
            Dictionary with run information, or None if no runs found.

        Example:
            # Get best accuracy
            best_run = manager.get_best_run("accuracy", "my_experiment")
            if best_run:
                print(f"Best run: {best_run['run_id']}")
                print(f"Accuracy: {best_run['metrics.accuracy']}")

            # Get lowest loss
            best_run = manager.get_best_run("loss", "my_experiment", ascending=True)
        """
        runs_df = self.list_runs(experiment_name=experiment_name)

        if runs_df.empty:
            logger.warning("No runs found in experiment")
            return None

        metric_col = f"metrics.{metric}"
        if metric_col not in runs_df.columns:
            logger.warning(f"Metric '{metric}' not found in runs")
            return None

        # Drop rows with NaN values for the metric
        valid_runs = runs_df.dropna(subset=[metric_col])

        if valid_runs.empty:
            logger.warning(f"No runs have valid values for metric '{metric}'")
            return None

        # Sort and get best
        sorted_runs = valid_runs.sort_values(by=metric_col, ascending=ascending)
        best_run = sorted_runs.iloc[0]

        logger.info(f"Best run for metric '{metric}': {best_run['run_id']} (value: {best_run[metric_col]})")
        return best_run.to_dict()
