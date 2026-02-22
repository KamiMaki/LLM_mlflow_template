"""Tests for mlflow evaluator module."""
import warnings
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from llm_framework.mlflow.evaluator import (
    EvaluationResult,
    Evaluator,
    _is_mlflow_available,
)


class TestEvaluationResult:
    """Test EvaluationResult dataclass."""

    def test_evaluation_result_creation(self):
        """Test creating an EvaluationResult instance."""
        metrics = {"accuracy": 0.95, "f1_score": 0.92}
        per_row = pd.DataFrame({"score": [1, 0, 1]})
        run_id = "test-run-123"

        result = EvaluationResult(
            metrics=metrics,
            per_row=per_row,
            run_id=run_id
        )

        assert result.metrics == metrics
        assert result.per_row.equals(per_row)
        assert result.run_id == run_id

    def test_evaluation_result_without_run_id(self):
        """Test EvaluationResult with None run_id."""
        metrics = {"accuracy": 0.85}
        per_row = pd.DataFrame()

        result = EvaluationResult(
            metrics=metrics,
            per_row=per_row,
            run_id=None
        )

        assert result.metrics == metrics
        assert result.per_row.empty
        assert result.run_id is None


class TestMLflowAvailability:
    """Test MLflow availability checks."""

    def test_is_mlflow_available_when_disabled(self, disabled_mlflow_config):
        """Test that MLflow is unavailable when disabled in config."""
        with patch("llm_framework.mlflow.evaluator.get_config", return_value=disabled_mlflow_config):
            assert not _is_mlflow_available()

    def test_is_mlflow_available_when_enabled(self, sample_config):
        """Test that MLflow is available when enabled and importable."""
        mock_mlflow = MagicMock()

        with patch("llm_framework.mlflow.evaluator.get_config", return_value=sample_config):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                assert _is_mlflow_available()


class TestBasicEvaluate:
    """Test _basic_evaluate fallback method."""

    def test_basic_evaluate_exact_match(self, disabled_mlflow_config):
        """Test basic evaluation computes exact match correctly."""
        evaluator = Evaluator()

        data = pd.DataFrame({
            "prediction": ["Paris", "London", "Berlin"],
            "ground_truth": ["Paris", "Madrid", "Berlin"]
        })

        with patch("llm_framework.mlflow.evaluator._is_mlflow_available", return_value=False):
            result = evaluator._basic_evaluate(
                data=data,
                metrics=["exact_match"],
                model_output_col="prediction",
                target_col="ground_truth"
            )

        assert result.metrics["exact_match"] == 2/3
        assert "exact_match" in result.per_row.columns
        assert result.per_row["exact_match"].tolist() == [1, 0, 1]
        assert result.run_id is None

    def test_basic_evaluate_token_count(self):
        """Test basic evaluation computes token count correctly."""
        evaluator = Evaluator()

        data = pd.DataFrame({
            "response": ["Hello world", "This is a test response", "Short"]
        })

        with patch("llm_framework.mlflow.evaluator._is_mlflow_available", return_value=False):
            result = evaluator._basic_evaluate(
                data=data,
                metrics=["token_count"],
                model_output_col="response",
                target_col="expected"
            )

        assert "avg_token_count" in result.metrics
        assert result.metrics["avg_token_count"] == (2 + 5 + 1) / 3
        assert "token_count" in result.per_row.columns

    def test_basic_evaluate_custom_metric(self):
        """Test basic evaluation with custom callable metric."""
        evaluator = Evaluator()

        data = pd.DataFrame({
            "response": ["ABC", "ABCDEF", "AB"]
        })

        def string_length(row):
            return len(row["response"])

        with patch("llm_framework.mlflow.evaluator._is_mlflow_available", return_value=False):
            result = evaluator._basic_evaluate(
                data=data,
                metrics=[string_length],
                model_output_col="response",
                target_col="expected"
            )

        assert "avg_string_length" in result.metrics
        assert result.metrics["avg_string_length"] == (3 + 6 + 2) / 3
        assert "string_length" in result.per_row.columns

    def test_basic_evaluate_handles_custom_metric_errors(self):
        """Test that custom metric errors are logged but don't crash."""
        evaluator = Evaluator()

        data = pd.DataFrame({"response": ["test"]})

        def failing_metric(row):
            raise ValueError("Metric error")

        with patch("llm_framework.mlflow.evaluator._is_mlflow_available", return_value=False):
            result = evaluator._basic_evaluate(
                data=data,
                metrics=[failing_metric],
                model_output_col="response",
                target_col="expected"
            )

        # Should complete without raising
        assert isinstance(result, EvaluationResult)


class TestEvaluatorWithMLflowDisabled:
    """Test Evaluator with MLflow disabled."""

    def test_evaluator_initialization_when_disabled(self, disabled_mlflow_config):
        """Test Evaluator initializes with MLflow disabled."""
        with patch("llm_framework.mlflow.evaluator.get_config", return_value=disabled_mlflow_config):
            with patch("llm_framework.mlflow.evaluator._is_mlflow_available", return_value=False):
                evaluator = Evaluator(experiment_name="test")

        assert not evaluator._mlflow_available
        assert evaluator.experiment_name == "test"

    def test_evaluate_falls_back_to_basic_when_disabled(self, disabled_mlflow_config):
        """Test that evaluate uses basic evaluation when MLflow is disabled."""
        with patch("llm_framework.mlflow.evaluator._is_mlflow_available", return_value=False):
            evaluator = Evaluator()

            data = pd.DataFrame({
                "response": ["answer1", "answer2"],
                "expected": ["answer1", "answer3"]
            })

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = evaluator.evaluate(
                    data=data,
                    metrics=["exact_match"],
                    model_output_col="response",
                    target_col="expected"
                )

        assert len(w) == 1
        assert "MLflow is not available" in str(w[0].message)
        assert result.metrics["exact_match"] == 0.5
        assert result.run_id is None

    def test_compare_returns_empty_when_disabled(self):
        """Test that compare returns empty DataFrame when MLflow is disabled."""
        with patch("llm_framework.mlflow.evaluator._is_mlflow_available", return_value=False):
            evaluator = Evaluator()

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = evaluator.compare(
                    run_ids=["run1", "run2"],
                    metric_keys=["accuracy"]
                )

        assert len(w) == 1
        assert "MLflow is not available" in str(w[0].message)
        assert result.empty

    def test_evaluate_with_no_target_column(self):
        """Test evaluate when target column is missing."""
        with patch("llm_framework.mlflow.evaluator._is_mlflow_available", return_value=False):
            evaluator = Evaluator()

            data = pd.DataFrame({
                "response": ["answer1", "answer2"]
            })

            result = evaluator.evaluate(
                data=data,
                metrics=["token_count"],
                model_output_col="response",
                target_col="missing_column"
            )

        # Should still compute token count
        assert "avg_token_count" in result.metrics

    def test_evaluate_with_empty_dataframe(self):
        """Test evaluate with empty DataFrame."""
        with patch("llm_framework.mlflow.evaluator._is_mlflow_available", return_value=False):
            evaluator = Evaluator()

            data = pd.DataFrame()

            result = evaluator.evaluate(
                data=data,
                metrics=["exact_match"],
                model_output_col="response",
                target_col="expected"
            )

        assert isinstance(result, EvaluationResult)
        assert result.per_row.empty


class TestEvaluatorWithMLflowEnabled:
    """Test Evaluator behavior when MLflow is enabled."""

    def test_evaluator_sets_experiment(self, sample_config):
        """Test that Evaluator sets MLflow experiment on initialization."""
        mock_mlflow = MagicMock()

        with patch("llm_framework.mlflow.evaluator.get_config", return_value=sample_config):
            with patch("llm_framework.mlflow.evaluator._is_mlflow_available", return_value=True):
                with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                    with patch("llm_framework.mlflow.evaluator.mlflow", mock_mlflow):
                        evaluator = Evaluator(experiment_name="custom_experiment")

        mock_mlflow.set_experiment.assert_called_with("custom_experiment")

    def test_evaluate_with_mlflow_failure_falls_back(self, sample_config):
        """Test that evaluate falls back to basic when MLflow evaluation fails."""
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.side_effect = Exception("MLflow error")

        with patch("llm_framework.mlflow.evaluator.get_config", return_value=sample_config):
            with patch("llm_framework.mlflow.evaluator._is_mlflow_available", return_value=True):
                with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                    with patch("llm_framework.mlflow.evaluator.mlflow", mock_mlflow):
                        evaluator = Evaluator()

                        data = pd.DataFrame({
                            "response": ["test"],
                            "expected": ["test"]
                        })

                        with warnings.catch_warnings(record=True) as w:
                            warnings.simplefilter("always")
                            result = evaluator.evaluate(
                                data=data,
                                metrics=["exact_match"],
                                model_output_col="response",
                                target_col="expected"
                            )

        assert len(w) == 1
        assert "MLflow evaluation failed" in str(w[0].message)
        assert result.metrics["exact_match"] == 1.0

    def test_compare_handles_missing_runs(self, sample_config):
        """Test that compare handles missing run IDs gracefully."""
        mock_mlflow = MagicMock()
        mock_mlflow.get_run.side_effect = Exception("Run not found")

        with patch("llm_framework.mlflow.evaluator._is_mlflow_available", return_value=True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.evaluator.mlflow", mock_mlflow):
                    evaluator = Evaluator()

                    result = evaluator.compare(
                        run_ids=["missing_run"],
                        metric_keys=["accuracy"]
                    )

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_compare_with_successful_runs(self, sample_config):
        """Test compare with successful run retrieval."""
        mock_mlflow = MagicMock()

        # Mock two runs
        mock_run1 = MagicMock()
        mock_run1.info.run_id = "run1"
        mock_run1.info.run_name = "experiment_1"
        mock_run1.data.metrics = {"accuracy": 0.95, "f1": 0.93}

        mock_run2 = MagicMock()
        mock_run2.info.run_id = "run2"
        mock_run2.info.run_name = "experiment_2"
        mock_run2.data.metrics = {"accuracy": 0.92, "f1": 0.90}

        mock_mlflow.get_run.side_effect = [mock_run1, mock_run2]

        with patch("llm_framework.mlflow.evaluator._is_mlflow_available", return_value=True):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.evaluator.mlflow", mock_mlflow):
                    evaluator = Evaluator()

                    result = evaluator.compare(
                        run_ids=["run1", "run2"],
                        metric_keys=["accuracy", "f1"]
                    )

        assert len(result) == 2
        assert result.iloc[0]["accuracy"] == 0.95
        assert result.iloc[1]["accuracy"] == 0.92
        assert "run_id" in result.columns
        assert "run_name" in result.columns
