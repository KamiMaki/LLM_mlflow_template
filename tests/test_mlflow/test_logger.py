"""Tests for MLflow logger module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestMlflowAvailability:
    """Test MLflow availability checking."""

    def test_is_mlflow_available_when_enabled(self, sample_config):
        """Test that MLflow is detected as available when enabled."""
        from llm_framework.mlflow.logger import _is_mlflow_available

        with patch("llm_framework.mlflow.logger.get_config", return_value=sample_config):
            with patch.dict("sys.modules", {"mlflow": MagicMock()}):
                assert _is_mlflow_available() is True

    def test_is_mlflow_available_when_disabled(self, disabled_mlflow_config):
        """Test that MLflow is not available when disabled in config."""
        from llm_framework.mlflow.logger import _is_mlflow_available

        with patch("llm_framework.mlflow.logger.get_config", return_value=disabled_mlflow_config):
            assert _is_mlflow_available() is False

    def test_is_mlflow_available_when_import_fails(self, sample_config):
        """Test that MLflow is not available when import fails."""
        from llm_framework.mlflow.logger import _is_mlflow_available

        with patch("llm_framework.mlflow.logger.get_config", return_value=sample_config):
            with patch.dict("sys.modules", {"mlflow": None}):
                with patch("builtins.__import__", side_effect=ImportError):
                    assert _is_mlflow_available() is False


class TestLogLLMCall:
    """Test log_llm_call function."""

    def test_log_llm_call_success(self, sample_config):
        """Test logging LLM call with MLflow enabled."""
        from llm_framework.mlflow.logger import log_llm_call

        mock_mlflow = MagicMock()

        with patch("llm_framework.mlflow.logger.get_config", return_value=sample_config):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.logger._is_mlflow_available", return_value=True):
                    import sys
                    sys.modules["mlflow"] = mock_mlflow

                    log_llm_call(
                        prompt="What is AI?",
                        response="AI is artificial intelligence.",
                        model="gpt-4o",
                        token_usage={"prompt": 5, "completion": 10, "total": 15},
                        latency_ms=250.5,
                        params={"temperature": 0.7, "max_tokens": 100},
                    )

                    # Verify params logged
                    mock_mlflow.log_params.assert_called_once()
                    logged_params = mock_mlflow.log_params.call_args[0][0]
                    assert "llm.temperature" in logged_params
                    assert logged_params["llm.temperature"] == 0.7

                    # Verify model logged
                    mock_mlflow.log_param.assert_called()

                    # Verify metrics logged
                    assert mock_mlflow.log_metric.call_count >= 4  # tokens + latency

                    # Verify text artifacts logged
                    assert mock_mlflow.log_text.call_count == 2  # prompt + response

    def test_log_llm_call_no_params(self, sample_config):
        """Test logging LLM call without params."""
        from llm_framework.mlflow.logger import log_llm_call

        mock_mlflow = MagicMock()

        with patch("llm_framework.mlflow.logger.get_config", return_value=sample_config):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.logger._is_mlflow_available", return_value=True):
                    import sys
                    sys.modules["mlflow"] = mock_mlflow

                    log_llm_call(
                        prompt="Test",
                        response="Response",
                        model="gpt-4o",
                        token_usage={"prompt": 1, "completion": 1, "total": 2},
                        latency_ms=100.0,
                    )

                    # Should not call log_params if params=None
                    mock_mlflow.log_params.assert_not_called()

    def test_log_llm_call_when_disabled(self, disabled_mlflow_config):
        """Test that log_llm_call is no-op when MLflow disabled."""
        from llm_framework.mlflow.logger import log_llm_call

        mock_mlflow = MagicMock()

        with patch("llm_framework.mlflow.logger.get_config", return_value=disabled_mlflow_config):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                log_llm_call(
                    prompt="Test",
                    response="Response",
                    model="gpt-4o",
                    token_usage={"prompt": 1, "completion": 1, "total": 2},
                    latency_ms=100.0,
                )

                # Should not call any MLflow functions
                mock_mlflow.log_params.assert_not_called()
                mock_mlflow.log_param.assert_not_called()
                mock_mlflow.log_metric.assert_not_called()

    def test_log_llm_call_handles_exceptions(self, sample_config):
        """Test that log_llm_call handles MLflow exceptions gracefully."""
        from llm_framework.mlflow.logger import log_llm_call

        mock_mlflow = MagicMock()
        mock_mlflow.log_param.side_effect = Exception("MLflow error")

        with patch("llm_framework.mlflow.logger.get_config", return_value=sample_config):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.logger._is_mlflow_available", return_value=True):
                    import sys
                    sys.modules["mlflow"] = mock_mlflow

                    # Should not raise, just warn
                    with pytest.warns(UserWarning, match="Failed to log LLM call"):
                        log_llm_call(
                            prompt="Test",
                            response="Response",
                            model="gpt-4o",
                            token_usage={"prompt": 1, "completion": 1, "total": 2},
                            latency_ms=100.0,
                        )


class TestLogParams:
    """Test log_params function."""

    def test_log_params_success(self, sample_config):
        """Test logging parameters."""
        from llm_framework.mlflow.logger import log_params

        mock_mlflow = MagicMock()

        with patch("llm_framework.mlflow.logger.get_config", return_value=sample_config):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.logger._is_mlflow_available", return_value=True):
                    import sys
                    sys.modules["mlflow"] = mock_mlflow

                    params = {"model": "gpt-4o", "temperature": 0.7}
                    log_params(params)

                    mock_mlflow.log_params.assert_called_once_with(params)

    def test_log_params_empty_dict(self, sample_config):
        """Test that empty params dict is no-op."""
        from llm_framework.mlflow.logger import log_params

        mock_mlflow = MagicMock()

        with patch("llm_framework.mlflow.logger.get_config", return_value=sample_config):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.logger._is_mlflow_available", return_value=True):
                    log_params({})
                    mock_mlflow.log_params.assert_not_called()


class TestLogMetrics:
    """Test log_metrics function."""

    def test_log_metrics_success(self, sample_config):
        """Test logging metrics."""
        from llm_framework.mlflow.logger import log_metrics

        mock_mlflow = MagicMock()

        with patch("llm_framework.mlflow.logger.get_config", return_value=sample_config):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.logger._is_mlflow_available", return_value=True):
                    import sys
                    sys.modules["mlflow"] = mock_mlflow

                    metrics = {"accuracy": 0.95, "f1_score": 0.92}
                    log_metrics(metrics)

                    mock_mlflow.log_metrics.assert_called_once_with(metrics, step=None)

    def test_log_metrics_with_step(self, sample_config):
        """Test logging metrics with step parameter."""
        from llm_framework.mlflow.logger import log_metrics

        mock_mlflow = MagicMock()

        with patch("llm_framework.mlflow.logger.get_config", return_value=sample_config):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.logger._is_mlflow_available", return_value=True):
                    import sys
                    sys.modules["mlflow"] = mock_mlflow

                    metrics = {"loss": 0.05}
                    log_metrics(metrics, step=10)

                    mock_mlflow.log_metrics.assert_called_once_with(metrics, step=10)


class TestLogArtifact:
    """Test log_artifact function."""

    def test_log_artifact_success(self, sample_config, tmp_path):
        """Test logging artifact file."""
        from llm_framework.mlflow.logger import log_artifact

        mock_mlflow = MagicMock()

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        with patch("llm_framework.mlflow.logger.get_config", return_value=sample_config):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.logger._is_mlflow_available", return_value=True):
                    import sys
                    sys.modules["mlflow"] = mock_mlflow

                    log_artifact(test_file)

                    mock_mlflow.log_artifact.assert_called_once_with(str(test_file), artifact_path=None)

    def test_log_artifact_with_path(self, sample_config, tmp_path):
        """Test logging artifact with artifact_path."""
        from llm_framework.mlflow.logger import log_artifact

        mock_mlflow = MagicMock()

        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        with patch("llm_framework.mlflow.logger.get_config", return_value=sample_config):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.logger._is_mlflow_available", return_value=True):
                    import sys
                    sys.modules["mlflow"] = mock_mlflow

                    log_artifact(test_file, artifact_path="models")

                    mock_mlflow.log_artifact.assert_called_once_with(str(test_file), artifact_path="models")


class TestLogDictArtifact:
    """Test log_dict_artifact function."""

    def test_log_dict_artifact_success(self, sample_config):
        """Test logging dictionary as artifact."""
        from llm_framework.mlflow.logger import log_dict_artifact

        mock_mlflow = MagicMock()

        with patch("llm_framework.mlflow.logger.get_config", return_value=sample_config):
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                with patch("llm_framework.mlflow.logger._is_mlflow_available", return_value=True):
                    import sys
                    sys.modules["mlflow"] = mock_mlflow

                    data = {"key": "value", "number": 42}
                    log_dict_artifact(data, "output.json")

                    mock_mlflow.log_dict.assert_called_once_with(data, "output.json")
