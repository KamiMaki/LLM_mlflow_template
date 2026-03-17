"""Trace 工具 — 敏感資料過濾與 MLflow span 輔助。

提供：
1. 敏感欄位辨識與遮蔽（token、auth、secret 等）
2. MLflow trace span context manager，安全記錄 I/O

Usage:
    from llm_service.trace import sanitize_dict, trace_span

    safe = sanitize_dict({"Authorization": "Bearer sk-xxx..."})
    with trace_span("my_call", inputs=safe) as span:
        result = do_something()
        if span:
            span.set_outputs({"result": result})
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any

from loguru import logger

# --- 敏感欄位辨識 ---

SENSITIVE_KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i).*token.*"),
    re.compile(r"(?i).*auth.*"),
    re.compile(r"(?i).*secret.*"),
    re.compile(r"(?i).*password.*"),
    re.compile(r"(?i).*api[_-]?key.*"),
    re.compile(r"(?i).*credential.*"),
    re.compile(r"(?i).*bearer.*"),
)

REDACTED = "***REDACTED***"


def is_sensitive_key(key: str) -> bool:
    """判斷 key 是否為敏感欄位。"""
    return any(p.match(str(key)) for p in SENSITIVE_KEY_PATTERNS)


def _mask_value(key: str, value: Any) -> Any:
    """遮蔽敏感值：保留前 4 字元 + REDACTED。"""
    if not is_sensitive_key(key):
        return value
    if isinstance(value, str) and len(value) > 8:
        return value[:4] + "..." + REDACTED
    return REDACTED


def sanitize_dict(data: Any) -> Any:
    """遞迴遮蔽 dict 中的敏感欄位值。"""
    if isinstance(data, dict):
        return {
            k: sanitize_dict(v) if isinstance(v, (dict, list)) else _mask_value(k, v)
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [sanitize_dict(item) for item in data]
    return data


def sanitize_completion_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """針對 litellm completion kwargs 進行敏感資料遮蔽。

    - extra_headers: 遮蔽含 token/auth 的 header 值
    - api_key: 遮蔽
    - messages: 保留原樣（使用者的 prompt）
    """
    safe: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k == "extra_headers" and isinstance(v, dict):
            safe[k] = sanitize_dict(v)
        elif k == "api_key":
            safe[k] = _mask_value(k, v)
        elif k == "messages":
            safe[k] = v
        elif isinstance(v, dict):
            safe[k] = sanitize_dict(v)
        else:
            safe[k] = v
    return safe


# --- MLflow Trace Span ---

try:
    import mlflow

    _mlflow_ok = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    _mlflow_ok = False


@contextmanager
def trace_span(name: str, inputs: dict[str, Any] | None = None):
    """建立 MLflow trace span，自動處理 MLflow 未安裝或未啟用的情況。

    Usage:
        with trace_span("call_llm", inputs={...}) as span:
            result = do_something()
            if span:
                span.set_outputs({...})
    """
    if not _mlflow_ok:
        yield None
        return

    span = None
    try:
        span = mlflow.start_span(name=name)
        if inputs is not None:
            span.set_inputs(inputs)
    except Exception as e:
        logger.debug(f"Failed to start MLflow span: {e}")
        span = None

    try:
        yield span
    finally:
        if span is not None:
            try:
                span.end()
            except Exception:
                pass
