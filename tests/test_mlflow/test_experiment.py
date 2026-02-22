"""Tests for mlflow experiment manager module."""
import warnings
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from llm_framework.mlflow.experiment import (
    ExperimentManager,
    _is_mlflow_available,
)


class TestMLflowAvailability:
    """Test MLflow availability checks."""

    def test_is_mlflow_available_when_disabled(self, disabled_mlflow_config):
        """Test that MLflow is unavailable when disabled in config."""
        with patch("llm_framework.mlflow.experiment.get_config", return_value=disabled_mlflow_config):
            assert not _is_mlflow_available()

    def test_is_mlflow_available_when_enabled(self, sample_config):
        """Test that MLflow is available when enabled."""
        mock_mlflow = MagicMock()

        with patch("llm_framework.mlflow.experiment.get_config", return_value=sample_config):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                assert _is_mlflow_available()


class TestExperimentManagerWithMLflowUnavailable:
    """Test ExperimentManager when MLflow is unavailable."""

    def test_initialization_when_disabled(self, disabled_mlflow_config):
        """Test ExperimentManager initialization when MLflow is disabled."""
        with patch("llm_framework.mlflow.experiment._is_mlflow_available", return_value=False):
            manager = ExperimentManager(config=disabled_mlflow_config)

        assert not manager._mlflow_available

    def test_get_or_create_experiment_returns_none(self):
        """Test that get_or_create_experiment returns None when MLflow unavailable."""
        with patch("llm_framework.mlflow.experiment._is_mlflow_available", return_value=False):
            manager = ExperimentManager()
            result = manager.get_or_create_experiment("test_experiment")

        assert result is None

    def test_start_run_yields_none(self):
        """Test that start_run yields None when MLflow unavailable."""
        with patch("llm_framework.mlflow.experiment._is_mlflow_available", return_value=False):
            manager = ExperimentManager()

            with manager.start_run(run_name="test_run") as run:
                assert run is None

    def test_start_run_executes_code_block(self):
        """Test that code inside start_run executes when MLflow unavailable."""
        with patch("llm_framework.mlflow.experiment._is_mlflow_available", return_value=False):
            manager = ExperimentManager()

            result = None
            with manager.start_run():
                result = 42

        assert result == 42

    def test_list_runs_returns_empty_dataframe(self):
        """Test that list_runs returns empty DataFrame when MLflow unavailable."""
        with patch("llm_framework.mlflow.experiment._is_mlflow_available", return_value=False):
            manager = ExperimentManager()

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = manager.list_runs(experiment_name="test")

        assert len(w) == 1
        assert "MLflow is not available" in str(w[0].message)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_get_best_run_returns_none(self):
        """Test that get_best_run returns None when MLflow unavailable."""
        with patch("llm_framework.mlflow.experiment._is_mlflow_available", return_value=False):
            manager = ExperimentManager()

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = manager.get_best_run(metric="accuracy")

        # list_runs warns, and it returns empty DataFrame, so get_best_run returns None
        assert result is None


class TestExperimentManagerWithMLflowEnabled:
    """Test ExperimentManager when MLflow is enabled."""

    def test_initialization_sets_tracking_uri(self, sample_config):
        """Test that initialization sets MLflow tracking URI."""
        mock_mlflow = MagicMock()

        with patch("llm_framework.mlflow.experiment._is_mlflow_available", return_value=True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.experiment.mlflow", mock_mlflow):
                    with patch("llm_framework.mlflow.experiment.get_config", return_value=sample_config):
                        manager = ExperimentManager(config=sample_config)

        mock_mlflow.set_tracking_uri.assert_called_with(sample_config.mlflow.tracking_uri)
        mock_mlflow.set_experiment.assert_called_with(sample_config.mlflow.experiment_name)

    def test_initialization_failure_disables_mlflow(self, sample_config):
        """Test that initialization failure disables MLflow gracefully."""
        mock_mlflow = MagicMock()
        mock_mlflow.set_tracking_uri.side_effect = Exception("Connection error")

        with patch("llm_framework.mlflow.experiment._is_mlflow_available", return_value=True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.experiment.mlflow", mock_mlflow):
                    with patch("llm_framework.mlflow.experiment.get_config", return_value=sample_config):
                        with warnings.catch_warnings(record=True) as w:
                            warnings.simplefilter("always")
                            manager = ExperimentManager(config=sample_config)

        assert len(w) == 1
        assert "Failed to initialize MLflow" in str(w[0].message)
        assert not manager._mlflow_available

    def test_get_or_create_experiment_gets_existing(self, sample_config):
        """Test getting an existing experiment."""
        mock_mlflow = MagicMock()
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        with patch("llm_framework.mlflow.experiment._is_mlflow_available", return_value=True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.experiment.mlflow", mock_mlflow):
                    with patch("llm_framework.mlflow.experiment.get_config", return_value=sample_config):
                        manager = ExperimentManager()
                        result = manager.get_or_create_experiment("existing_exp")

        assert result == "exp123"
        mock_mlflow.get_experiment_by_name.assert_called_with("existing_exp")

    def test_get_or_create_experiment_creates_new(self, sample_config):
        """Test creating a new experiment."""
        mock_mlflow = MagicMock()
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "new_exp456"

        with patch("llm_framework.mlflow.experiment._is_mlflow_available", return_value=True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.experiment.mlflow", mock_mlflow):
                    with patch("llm_framework.mlflow.experiment.get_config", return_value=sample_config):
                        manager = ExperimentManager()
                        result = manager.get_or_create_experiment("new_exp")

        assert result == "new_exp456"
        mock_mlflow.create_experiment.assert_called_with("new_exp")

    def test_get_or_create_experiment_handles_errors(self, sample_config):
        """Test that get_or_create_experiment handles errors gracefully."""
        mock_mlflow = MagicMock()
        mock_mlflow.get_experiment_by_name.side_effect = Exception("API error")

        with patch("llm_framework.mlflow.experiment._is_mlflow_available", return_value=True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.experiment.mlflow", mock_mlflow):
                    with patch("llm_framework.mlflow.experiment.get_config", return_value=sample_config):
                        manager = ExperimentManager()

                        with warnings.catch_warnings(record=True) as w:
                            warnings.simplefilter("always")
                            result = manager.get_or_create_experiment("test")

        assert len(w) == 1
        assert "Failed to get or create experiment" in str(w[0].message)
        assert result is None

    def test_start_run_creates_mlflow_run(self, sample_config):
        """Test that start_run creates an MLflow run."""
        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "run123"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch("llm_framework.mlflow.experiment._is_mlflow_available", return_value=True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.experiment.mlflow", mock_mlflow):
                    manager = ExperimentManager()

                    with manager.start_run(run_name="test_run", tags={"env": "test"}) as run:
                        assert run.info.run_id == "run123"

        mock_mlflow.start_run.assert_called_once_with(run_name="test_run", tags={"env": "test"})

    def test_start_run_handles_errors(self, sample_config):
        """Test that start_run handles errors gracefully."""
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.side_effect = Exception("Run start error")

        with patch("llm_framework.mlflow.experiment._is_mlflow_available", return_value=True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.experiment.mlflow", mock_mlflow):
                    manager = ExperimentManager()

                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        with manager.start_run() as run:
                            assert run is None

        assert len(w) == 1
        assert "Failed to start MLflow run" in str(w[0].message)

    def test_list_runs_returns_dataframe(self, sample_config):
        """Test that list_runs returns a DataFrame of runs."""
        mock_mlflow = MagicMock()
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        mock_runs_df = pd.DataFrame({
            "run_id": ["run1", "run2"],
            "run_name": ["exp_1", "exp_2"],
            "metrics.accuracy": [0.95, 0.90]
        })
        mock_mlflow.search_runs.return_value = mock_runs_df

        with patch("llm_framework.mlflow.experiment._is_mlflow_available", return_value=True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.experiment.mlflow", mock_mlflow):
                    with patch("llm_framework.mlflow.experiment.get_config", return_value=sample_config):
                        manager = ExperimentManager()
                        result = manager.list_runs(experiment_name="test_exp")

        assert len(result) == 2
        assert "run_id" in result.columns
        mock_mlflow.search_runs.assert_called_once()

    def test_list_runs_with_filter(self, sample_config):
        """Test list_runs with filter string."""
        mock_mlflow = MagicMock()
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        mock_mlflow.search_runs.return_value = pd.DataFrame()

        with patch("llm_framework.mlflow.experiment._is_mlflow_available", return_value=True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.experiment.mlflow", mock_mlflow):
                    with patch("llm_framework.mlflow.experiment.get_config", return_value=sample_config):
                        manager = ExperimentManager()
                        manager.list_runs(
                            experiment_name="test_exp",
                            filter_string="metrics.accuracy > 0.9"
                        )

        call_args = mock_mlflow.search_runs.call_args
        assert call_args[1]["filter_string"] == "metrics.accuracy > 0.9"

    def test_list_runs_handles_missing_experiment(self, sample_config):
        """Test list_runs when experiment doesn't exist."""
        mock_mlflow = MagicMock()
        mock_mlflow.get_experiment_by_name.return_value = None

        with patch("llm_framework.mlflow.experiment._is_mlflow_available", return_value=True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.experiment.mlflow", mock_mlflow):
                    with patch("llm_framework.mlflow.experiment.get_config", return_value=sample_config):
                        manager = ExperimentManager()
                        result = manager.list_runs(experiment_name="missing_exp")

        assert result.empty

    def test_get_best_run_finds_highest_metric(self, sample_config):
        """Test get_best_run finds the run with highest metric value."""
        mock_mlflow = MagicMock()
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        mock_runs_df = pd.DataFrame({
            "run_id": ["run1", "run2", "run3"],
            "metrics.accuracy": [0.90, 0.95, 0.85]
        })
        mock_mlflow.search_runs.return_value = mock_runs_df

        with patch("llm_framework.mlflow.experiment._is_mlflow_available", return_value=True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.experiment.mlflow", mock_mlflow):
                    with patch("llm_framework.mlflow.experiment.get_config", return_value=sample_config):
                        manager = ExperimentManager()
                        result = manager.get_best_run(
                            metric="accuracy",
                            experiment_name="test_exp"
                        )

        assert result["run_id"] == "run2"
        assert result["metrics.accuracy"] == 0.95

    def test_get_best_run_finds_lowest_metric(self, sample_config):
        """Test get_best_run finds the run with lowest metric value when ascending=True."""
        mock_mlflow = MagicMock()
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        mock_runs_df = pd.DataFrame({
            "run_id": ["run1", "run2", "run3"],
            "metrics.loss": [0.10, 0.05, 0.15]
        })
        mock_mlflow.search_runs.return_value = mock_runs_df

        with patch("llm_framework.mlflow.experiment._is_mlflow_available", return_value=True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.experiment.mlflow", mock_mlflow):
                    with patch("llm_framework.mlflow.experiment.get_config", return_value=sample_config):
                        manager = ExperimentManager()
                        result = manager.get_best_run(
                            metric="loss",
                            experiment_name="test_exp",
                            ascending=True
                        )

        assert result["run_id"] == "run2"
        assert result["metrics.loss"] == 0.05

    def test_get_best_run_with_missing_metric(self, sample_config):
        """Test get_best_run when metric doesn't exist."""
        mock_mlflow = MagicMock()
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        mock_runs_df = pd.DataFrame({
            "run_id": ["run1", "run2"],
            "metrics.accuracy": [0.90, 0.95]
        })
        mock_mlflow.search_runs.return_value = mock_runs_df

        with patch("llm_framework.mlflow.experiment._is_mlflow_available", return_value=True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.experiment.mlflow", mock_mlflow):
                    with patch("llm_framework.mlflow.experiment.get_config", return_value=sample_config):
                        manager = ExperimentManager()
                        result = manager.get_best_run(
                            metric="missing_metric",
                            experiment_name="test_exp"
                        )

        assert result is None

    def test_get_best_run_with_empty_runs(self):
        """Test get_best_run when no runs exist."""
        with patch("llm_framework.mlflow.experiment._is_mlflow_available", return_value=False):
            manager = ExperimentManager()
            result = manager.get_best_run(metric="accuracy")

        assert result is None
