"""Retry utilities with exponential backoff.

Provides decorators and helpers for retrying operations with configurable
backoff strategies, useful for handling transient failures in LLM calls
and external API requests.

Usage:
    from llm_framework.workflow.tools.retry import with_retry

    @with_retry(max_retries=5, backoff_factor=2.0)
    def call_external_api():
        response = requests.get("https://api.example.com/data")
        response.raise_for_status()
        return response.json()

    # Or use without parentheses for defaults
    @with_retry
    def simple_retry_example():
        return risky_operation()
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


def with_retry(
    func: Callable[..., T] | None = None,
    *,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[..., T]:
    """Decorator for retrying functions with exponential backoff.

    Can be used with or without parentheses:
        @with_retry
        @with_retry(max_retries=5)

    Args:
        func: Function to wrap (auto-provided when used without parentheses).
        max_retries: Maximum number of retry attempts.
        backoff_base: Base delay in seconds for first retry.
        backoff_factor: Multiplier for each subsequent retry delay.
        retryable_exceptions: Tuple of exception types that trigger retries.

    Returns:
        Wrapped function that retries on failure.

    Examples:
        >>> @with_retry(max_retries=3, backoff_base=0.5)
        ... def flaky_function():
        ...     return external_call()

        >>> @with_retry
        ... def simple_retry():
        ...     return might_fail()
    """
    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(1, max_retries + 1):
                try:
                    return f(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exception = exc
                    if attempt >= max_retries:
                        logger.error(
                            f"Function {f.__name__} failed after {max_retries} retries: {exc}"
                        )
                        raise

                    wait_time = backoff_base * (backoff_factor ** (attempt - 1))
                    logger.warning(
                        f"Attempt {attempt}/{max_retries} failed for {f.__name__}: {exc}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    time.sleep(wait_time)

            # This should never be reached, but satisfies type checker
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Unexpected state in retry logic for {f.__name__}")

        return wrapper

    # Handle decorator usage with and without parentheses
    if func is None:
        # Called with parentheses: @with_retry(...)
        return decorator
    else:
        # Called without parentheses: @with_retry
        return decorator(func)


def with_async_retry(
    func: Callable[..., Any] | None = None,
    *,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[..., Any]:
    """Decorator for retrying async functions with exponential backoff.

    Async version of with_retry decorator.

    Args:
        func: Async function to wrap (auto-provided when used without parentheses).
        max_retries: Maximum number of retry attempts.
        backoff_base: Base delay in seconds for first retry.
        backoff_factor: Multiplier for each subsequent retry delay.
        retryable_exceptions: Tuple of exception types that trigger retries.

    Returns:
        Wrapped async function that retries on failure.

    Examples:
        >>> @with_async_retry(max_retries=5)
        ... async def async_api_call():
        ...     return await external_async_call()
    """
    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(f)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None

            for attempt in range(1, max_retries + 1):
                try:
                    return await f(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exception = exc
                    if attempt >= max_retries:
                        logger.error(
                            f"Async function {f.__name__} failed after {max_retries} retries: {exc}"
                        )
                        raise

                    wait_time = backoff_base * (backoff_factor ** (attempt - 1))
                    logger.warning(
                        f"Async attempt {attempt}/{max_retries} failed for {f.__name__}: {exc}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    await asyncio.sleep(wait_time)

            # This should never be reached, but satisfies type checker
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Unexpected state in async retry logic for {f.__name__}")

        return wrapper

    # Handle decorator usage with and without parentheses
    if func is None:
        return decorator
    else:
        return decorator(func)


async def async_with_retry(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    **kwargs: Any,
) -> Any:
    """Async retry wrapper for calling async functions with retry logic.

    Alternative to decorator pattern, useful when you need to apply retry
    logic to a function you don't control.

    Args:
        func: Async function to call with retry logic.
        *args: Positional arguments to pass to func.
        max_retries: Maximum number of retry attempts.
        backoff_base: Base delay in seconds for first retry.
        backoff_factor: Multiplier for each subsequent retry delay.
        retryable_exceptions: Tuple of exception types that trigger retries.
        **kwargs: Keyword arguments to pass to func.

    Returns:
        Result from successful function call.

    Raises:
        Exception: The last exception if all retries fail.

    Examples:
        >>> result = await async_with_retry(
        ...     api_client.fetch_data,
        ...     user_id=123,
        ...     max_retries=5
        ... )
    """
    last_exception: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except retryable_exceptions as exc:
            last_exception = exc
            if attempt >= max_retries:
                logger.error(
                    f"Async function {func.__name__} failed after {max_retries} retries: {exc}"
                )
                raise

            wait_time = backoff_base * (backoff_factor ** (attempt - 1))
            logger.warning(
                f"Async attempt {attempt}/{max_retries} failed for {func.__name__}: {exc}. "
                f"Retrying in {wait_time:.2f}s..."
            )
            await asyncio.sleep(wait_time)

    # This should never be reached
    if last_exception:
        raise last_exception
    raise RuntimeError(f"Unexpected state in async_with_retry for {func.__name__}")


def calculate_backoff(attempt: int, base: float = 1.0, factor: float = 2.0) -> float:
    """Calculate exponential backoff delay for a given attempt number.

    Utility function for manually implementing retry logic.

    Args:
        attempt: Current attempt number (1-indexed).
        base: Base delay in seconds.
        factor: Exponential multiplier.

    Returns:
        Delay in seconds for this attempt.

    Examples:
        >>> calculate_backoff(1, base=1.0, factor=2.0)
        1.0
        >>> calculate_backoff(2, base=1.0, factor=2.0)
        2.0
        >>> calculate_backoff(3, base=1.0, factor=2.0)
        4.0
    """
    return base * (factor ** (attempt - 1))
