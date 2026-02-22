"""LLM Framework - A universal LLM development framework.

Quick start:
    from llm_framework import load_config, LLMClient

    config = load_config("dev")
    client = LLMClient()
    response = client.chat([{"role": "user", "content": "Hello!"}])
    print(response.content)
"""

__version__ = "0.1.0"

from llm_framework.config import (
    FrameworkConfig,
    LLMConfig,
    MLflowConfig,
    load_config,
    load_config_from_dict,
    get_config,
)
from llm_framework.llm_client import LLMClient, LLMResponse, TokenUsage, LLMError

__all__ = [
    # Config
    "load_config",
    "load_config_from_dict",
    "get_config",
    "FrameworkConfig",
    "LLMConfig",
    "MLflowConfig",
    # LLM Client
    "LLMClient",
    "LLMResponse",
    "TokenUsage",
    "LLMError",
]
