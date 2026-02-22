"""Configuration management with multi-environment support.

Loads YAML config files, resolves environment variables, and provides
a global singleton for consistent access across the framework.

Usage:
    from llm_framework.config import load_config, get_config

    # Load config for a specific environment
    config = load_config("dev")

    # Access config anywhere in the codebase
    config = get_config()
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMConfig:
    """LLM service configuration."""
    url: str
    auth_token: str
    default_model: str = "gpt-4o"
    timeout: int = 30
    max_retries: int = 3
    temperature: float = 0.7


@dataclass(frozen=True)
class MLflowConfig:
    """MLflow tracking configuration."""
    tracking_uri: str = ""
    experiment_name: str = "default"
    enabled: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass(frozen=True)
class FrameworkConfig:
    """Top-level framework configuration. Immutable after creation."""
    llm: LLMConfig
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    env: str = "dev"


# ---------------------------------------------------------------------------
# Environment variable resolution
# ---------------------------------------------------------------------------

_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ${VAR_NAME} placeholders in config values."""
    if isinstance(value, str):
        def _replace(match: re.Match) -> str:
            var_name = match.group(1)
            env_value = os.environ.get(var_name)
            if env_value is None:
                raise ConfigError(
                    f"Environment variable '{var_name}' is not set. "
                    f"Please set it or update your config file."
                )
            return env_value
        return _ENV_VAR_PATTERN.sub(_replace, value)
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _build_config(raw: dict, env: str) -> FrameworkConfig:
    """Build a FrameworkConfig from a raw (already env-resolved) dict."""
    llm_raw = raw.get("llm", {})
    mlflow_raw = raw.get("mlflow", {})
    logging_raw = raw.get("logging", {})

    if not llm_raw.get("url"):
        raise ConfigError("'llm.url' is required in config.")

    return FrameworkConfig(
        llm=LLMConfig(
            url=llm_raw["url"],
            auth_token=llm_raw.get("auth_token", ""),
            default_model=llm_raw.get("default_model", "gpt-4o"),
            timeout=int(llm_raw.get("timeout", 30)),
            max_retries=int(llm_raw.get("max_retries", 3)),
            temperature=float(llm_raw.get("temperature", 0.7)),
        ),
        mlflow=MLflowConfig(
            tracking_uri=mlflow_raw.get("tracking_uri", ""),
            experiment_name=mlflow_raw.get("experiment_name", "default"),
            enabled=bool(mlflow_raw.get("enabled", True)),
        ),
        logging=LoggingConfig(
            level=logging_raw.get("level", "INFO"),
            format=logging_raw.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        ),
        env=env,
    )


def load_config(
    env: str | None = None,
    config_dir: str | Path = "config",
) -> FrameworkConfig:
    """Load configuration for the specified environment.

    Args:
        env: Environment name (dev/test/stg/prod). If None, reads from
             the LLM_ENV environment variable, defaulting to "dev".
        config_dir: Path to the directory containing YAML config files.

    Returns:
        FrameworkConfig instance.

    Raises:
        ConfigError: If the config file is missing or invalid.
    """
    global _active_config

    if env is None:
        env = os.environ.get("LLM_ENV", "dev")

    config_path = Path(config_dir) / f"{env}.yaml"
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    resolved = _resolve_env_vars(raw)
    config = _build_config(resolved, env)
    _active_config = config
    return config


def load_config_from_dict(raw: dict, env: str = "custom") -> FrameworkConfig:
    """Load configuration from a dictionary (useful for testing).

    Args:
        raw: Dictionary with the same structure as a YAML config file.
        env: Environment label.

    Returns:
        FrameworkConfig instance.
    """
    global _active_config
    resolved = _resolve_env_vars(raw)
    config = _build_config(resolved, env)
    _active_config = config
    return config


def get_config() -> FrameworkConfig:
    """Get the currently active configuration.

    Returns:
        The FrameworkConfig loaded by the most recent load_config() call.

    Raises:
        ConfigError: If load_config() has not been called yet.
    """
    if _active_config is None:
        raise ConfigError(
            "Configuration not loaded. Call load_config() first."
        )
    return _active_config


def reset_config() -> None:
    """Reset the global config singleton. Primarily for testing."""
    global _active_config
    _active_config = None


# Module-level singleton
_active_config: FrameworkConfig | None = None
