"""Tests for config module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from llm_framework.config import (
    ConfigError,
    FrameworkConfig,
    LLMConfig,
    LoggingConfig,
    MLflowConfig,
    get_config,
    load_config,
    load_config_from_dict,
    reset_config,
)


class TestConfigDataclasses:
    """Test configuration dataclass structures."""

    def test_llm_config_defaults(self):
        """Test LLMConfig default values."""
        config = LLMConfig(url="https://api.example.com", auth_token="test-token")
        assert config.url == "https://api.example.com"
        assert config.auth_token == "test-token"
        assert config.default_model == "gpt-4o"
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.temperature == 0.7

    def test_llm_config_custom_values(self):
        """Test LLMConfig with custom values."""
        config = LLMConfig(
            url="https://custom.api.com",
            auth_token="custom-token",
            default_model="gpt-3.5-turbo",
            timeout=60,
            max_retries=5,
            temperature=0.9,
        )
        assert config.default_model == "gpt-3.5-turbo"
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.temperature == 0.9

    def test_mlflow_config_defaults(self):
        """Test MLflowConfig default values."""
        config = MLflowConfig()
        assert config.tracking_uri == ""
        assert config.experiment_name == "default"
        assert config.enabled is True

    def test_logging_config_defaults(self):
        """Test LoggingConfig default values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert "asctime" in config.format

    def test_framework_config_immutable(self):
        """Test that FrameworkConfig is immutable."""
        config = FrameworkConfig(
            llm=LLMConfig(url="https://test.com", auth_token="token"),
            env="test",
        )
        with pytest.raises(Exception):  # frozen dataclass raises on modification
            config.env = "prod"  # type: ignore


class TestLoadConfigFromDict:
    """Test loading config from dictionary."""

    def test_load_config_from_dict_minimal(self):
        """Test loading config with minimal required fields."""
        config_dict = {
            "llm": {
                "url": "https://llm.example.com/v1/chat/completions",
                "auth_token": "test-token",
            }
        }
        config = load_config_from_dict(config_dict, env="test")

        assert config.llm.url == "https://llm.example.com/v1/chat/completions"
        assert config.llm.auth_token == "test-token"
        assert config.llm.default_model == "gpt-4o"
        assert config.env == "test"
        assert config.mlflow.enabled is True
        assert config.logging.level == "INFO"

    def test_load_config_from_dict_full(self, sample_config_dict):
        """Test loading config with all fields."""
        config = load_config_from_dict(sample_config_dict, env="dev")

        assert config.llm.url == "https://test-llm.company.com/v1/chat/completions"
        assert config.llm.auth_token == "test-token-123"
        assert config.llm.default_model == "gpt-4o"
        assert config.llm.timeout == 10
        assert config.llm.max_retries == 2
        assert config.llm.temperature == 0.0

        assert config.mlflow.tracking_uri == "http://localhost:5000"
        assert config.mlflow.experiment_name == "test-experiment"
        assert config.mlflow.enabled is True

        assert config.logging.level == "DEBUG"
        assert config.logging.format == "%(message)s"

        assert config.env == "dev"

    def test_load_config_from_dict_missing_url(self):
        """Test that missing llm.url raises ConfigError."""
        config_dict = {"llm": {"auth_token": "test-token"}}

        with pytest.raises(ConfigError, match="llm.url.*required"):
            load_config_from_dict(config_dict)

    def test_load_config_from_dict_empty(self):
        """Test that empty dict raises ConfigError."""
        with pytest.raises(ConfigError, match="llm.url.*required"):
            load_config_from_dict({})


class TestEnvVarResolution:
    """Test environment variable resolution in config."""

    def test_env_var_resolution(self, env_vars):
        """Test ${VAR_NAME} replacement in config values."""
        env_vars(TEST_LLM_URL="https://env-llm.example.com", TEST_AUTH="env-token-456")

        config_dict = {
            "llm": {
                "url": "${TEST_LLM_URL}/chat/completions",
                "auth_token": "${TEST_AUTH}",
            }
        }

        config = load_config_from_dict(config_dict)
        assert config.llm.url == "https://env-llm.example.com/chat/completions"
        assert config.llm.auth_token == "env-token-456"

    def test_env_var_resolution_missing_var(self, env_vars):
        """Test that missing env var raises ConfigError."""
        config_dict = {
            "llm": {
                "url": "https://llm.example.com",
                "auth_token": "${MISSING_ENV_VAR}",
            }
        }

        with pytest.raises(ConfigError, match="MISSING_ENV_VAR.*not set"):
            load_config_from_dict(config_dict)

    def test_env_var_resolution_nested(self, env_vars):
        """Test env var resolution in nested config."""
        env_vars(MLFLOW_URI="http://mlflow.example.com:5000")

        config_dict = {
            "llm": {"url": "https://llm.example.com", "auth_token": "token"},
            "mlflow": {"tracking_uri": "${MLFLOW_URI}", "enabled": True},
        }

        config = load_config_from_dict(config_dict)
        assert config.mlflow.tracking_uri == "http://mlflow.example.com:5000"


class TestLoadConfigFromFile:
    """Test loading config from YAML files."""

    def test_load_config_from_yaml_file(self, tmp_config_dir, env_vars):
        """Test loading config from YAML file."""
        env_vars(LLM_ENV="dev")
        config = load_config("dev", config_dir=tmp_config_dir)

        assert config.llm.url == "https://dev-llm.company.com/v1/chat/completions"
        assert config.llm.auth_token == "dev-token"
        assert config.llm.default_model == "gpt-4o"
        assert config.llm.timeout == 30
        assert config.mlflow.experiment_name == "dev-experiment"
        assert config.env == "dev"

    def test_load_config_missing_file(self, tmp_path):
        """Test that missing config file raises ConfigError."""
        with pytest.raises(ConfigError, match="Config file not found"):
            load_config("nonexistent", config_dir=tmp_path)

    def test_load_config_default_env(self, tmp_config_dir):
        """Test loading config with default env from LLM_ENV."""
        # LLM_ENV not set, should default to "dev"
        # But dev.yaml exists, so it should work
        config = load_config(config_dir=tmp_config_dir)
        assert config.env == "dev"

    def test_load_config_custom_env(self, tmp_config_dir):
        """Test loading config with explicit env parameter."""
        config = load_config("dev", config_dir=tmp_config_dir)
        assert config.env == "dev"


class TestConfigSingleton:
    """Test global config singleton behavior."""

    def test_get_config_before_load(self):
        """Test that get_config raises error before load_config is called."""
        reset_config()
        with pytest.raises(ConfigError, match="Configuration not loaded"):
            get_config()

    def test_get_config_after_load(self, sample_config_dict):
        """Test get_config returns the loaded config."""
        reset_config()
        loaded = load_config_from_dict(sample_config_dict, env="test")
        retrieved = get_config()

        assert retrieved is loaded
        assert retrieved.llm.url == loaded.llm.url
        assert retrieved.env == "test"

    def test_reset_config(self, sample_config_dict):
        """Test reset_config clears the singleton."""
        load_config_from_dict(sample_config_dict, env="test")
        assert get_config() is not None

        reset_config()
        with pytest.raises(ConfigError, match="Configuration not loaded"):
            get_config()

    def test_config_singleton_persistence(self, sample_config_dict):
        """Test that config persists across get_config calls."""
        reset_config()
        load_config_from_dict(sample_config_dict, env="test")

        config1 = get_config()
        config2 = get_config()

        assert config1 is config2


class TestConfigTypeConversion:
    """Test type conversion in config loading."""

    def test_int_conversion(self):
        """Test that string numbers are converted to ints."""
        config_dict = {
            "llm": {
                "url": "https://llm.example.com",
                "auth_token": "token",
                "timeout": "45",  # string
                "max_retries": "10",  # string
            }
        }

        config = load_config_from_dict(config_dict)
        assert isinstance(config.llm.timeout, int)
        assert config.llm.timeout == 45
        assert isinstance(config.llm.max_retries, int)
        assert config.llm.max_retries == 10

    def test_float_conversion(self):
        """Test that string floats are converted to floats."""
        config_dict = {
            "llm": {
                "url": "https://llm.example.com",
                "auth_token": "token",
                "temperature": "0.5",  # string
            }
        }

        config = load_config_from_dict(config_dict)
        assert isinstance(config.llm.temperature, float)
        assert config.llm.temperature == 0.5

    def test_bool_conversion(self):
        """Test that various values are converted to bools."""
        config_dict = {
            "llm": {"url": "https://llm.example.com", "auth_token": "token"},
            "mlflow": {"enabled": "true"},  # string
        }

        config = load_config_from_dict(config_dict)
        assert isinstance(config.mlflow.enabled, bool)
        assert config.mlflow.enabled is True
