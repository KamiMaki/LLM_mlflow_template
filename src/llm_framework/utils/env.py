"""Environment variable helpers."""

from __future__ import annotations

import os


def get_env(name: str, default: str | None = None) -> str:
    """Get an environment variable, raising if not set and no default given."""
    value = os.environ.get(name, default)
    if value is None:
        raise EnvironmentError(f"Required environment variable '{name}' is not set.")
    return value


def get_current_env() -> str:
    """Get the current environment name from LLM_ENV, defaulting to 'dev'."""
    return os.environ.get("LLM_ENV", "dev")
