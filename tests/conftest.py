"""Shared test fixtures and mocks for the LLM framework test suite."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_framework.config import (
    FrameworkConfig,
    LLMConfig,
    LoggingConfig,
    MLflowConfig,
    load_config_from_dict,
    reset_config,
)
from llm_framework.llm_client import LLMResponse, TokenUsage


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config_dict():
    """Raw config dict matching YAML structure."""
    return {
        "llm": {
            "url": "https://test-llm.company.com/v1/chat/completions",
            "auth_token": "test-token-123",
            "default_model": "gpt-4o",
            "timeout": 10,
            "max_retries": 2,
            "temperature": 0.0,
        },
        "mlflow": {
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "test-experiment",
            "enabled": True,
        },
        "logging": {
            "level": "DEBUG",
            "format": "%(message)s",
        },
    }


@pytest.fixture
def sample_config(sample_config_dict):
    """A FrameworkConfig instance for testing."""
    config = load_config_from_dict(sample_config_dict, env="test")
    yield config
    reset_config()


@pytest.fixture
def disabled_mlflow_config():
    """Config with MLflow disabled."""
    config = load_config_from_dict(
        {
            "llm": {
                "url": "https://test-llm.company.com/v1/chat/completions",
                "auth_token": "test-token",
            },
            "mlflow": {"enabled": False},
        },
        env="test",
    )
    yield config
    reset_config()


@pytest.fixture(autouse=True)
def _reset_config_after_test():
    """Ensure config is reset after every test."""
    yield
    reset_config()


# ---------------------------------------------------------------------------
# LLM mock fixtures
# ---------------------------------------------------------------------------

MOCK_LLM_RESPONSE_DATA = {
    "id": "chatcmpl-test123",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello! How can I help you?"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "total_tokens": 18,
    },
}


@pytest.fixture
def mock_llm_response_data():
    """Raw API response dict from the LLM."""
    return MOCK_LLM_RESPONSE_DATA.copy()


@pytest.fixture
def mock_llm_response():
    """An LLMResponse object for testing."""
    return LLMResponse(
        content="Hello! How can I help you?",
        model="gpt-4o",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=8, total_tokens=18),
        latency_ms=150.0,
        raw_response=MOCK_LLM_RESPONSE_DATA.copy(),
    )


@pytest.fixture
def mock_httpx_response(mock_llm_response_data):
    """A mock httpx.Response for LLM API calls."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = mock_llm_response_data
    response.raise_for_status.return_value = None
    return response


# ---------------------------------------------------------------------------
# MLflow mock fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_mlflow():
    """Mock the mlflow module to avoid needing a real server."""
    with patch.dict("sys.modules", {"mlflow": MagicMock()}):
        import sys
        mock = sys.modules["mlflow"]
        mock.start_run.return_value.__enter__ = MagicMock()
        mock.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock.active_run.return_value = MagicMock()
        mock.active_run.return_value.info.run_id = "test-run-id"
        yield mock


# ---------------------------------------------------------------------------
# Temp file fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_config_dir(tmp_path):
    """Create a temporary config directory with YAML files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    dev_config = {
        "llm": {
            "url": "https://dev-llm.company.com/v1/chat/completions",
            "auth_token": "dev-token",
            "default_model": "gpt-4o",
            "timeout": 30,
            "max_retries": 3,
            "temperature": 0.7,
        },
        "mlflow": {
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "dev-experiment",
            "enabled": True,
        },
    }

    import yaml
    with open(config_dir / "dev.yaml", "w") as f:
        yaml.dump(dev_config, f)

    return config_dir


@pytest.fixture
def env_vars():
    """Set and clean up environment variables for testing."""
    original = {}

    def _set(**kwargs):
        for key, value in kwargs.items():
            original[key] = os.environ.get(key)
            os.environ[key] = value

    yield _set

    for key, value in original.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
