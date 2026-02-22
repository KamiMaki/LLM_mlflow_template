"""Structured output extraction from LLM responses.

Combines LLM calls with JSON parsing and Pydantic validation to reliably
extract structured data from language models. Automatically retries with
error feedback when parsing or validation fails.

Usage:
    from pydantic import BaseModel
    from llm_framework.llm_client import LLMClient
    from llm_framework.workflow.tools.structured_output import get_structured_output

    class UserProfile(BaseModel):
        name: str
        age: int
        email: str

    client = LLMClient()
    messages = [{"role": "user", "content": "Extract user info: John, 30, john@example.com"}]

    profile = get_structured_output(client, messages, UserProfile, max_retries=3)
    print(profile.name)  # John
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from llm_framework.llm_client import LLMClient, LLMError
from llm_framework.workflow.tools.parser import parse_json, JSONParseError
from llm_framework.workflow.tools.validator import validate_output, get_validation_errors

logger = logging.getLogger(__name__)


class StructuredOutputError(Exception):
    """Raised when structured output extraction fails after all retries."""


def get_structured_output(
    client: LLMClient,
    messages: list[dict[str, str]],
    output_schema: type[BaseModel],
    max_retries: int = 3,
    model: str | None = None,
    temperature: float | None = None,
    **llm_kwargs: Any,
) -> BaseModel:
    """Call LLM and parse response into a Pydantic model with automatic retries.

    Sends messages to the LLM, attempts to parse the response as JSON,
    and validates against the provided Pydantic schema. If parsing or
    validation fails, appends error feedback to the conversation and
    retries up to max_retries times.

    Args:
        client: LLMClient instance for making API calls.
        messages: List of message dicts (role, content) for the conversation.
        output_schema: Pydantic model class defining expected output structure.
        max_retries: Maximum retry attempts on parse/validation failures.
        model: Optional model override (defaults to client config).
        temperature: Optional temperature override (defaults to client config).
        **llm_kwargs: Additional arguments passed to client.chat().

    Returns:
        Validated Pydantic model instance.

    Raises:
        StructuredOutputError: If extraction fails after all retries.
        LLMError: If the LLM API call fails.

    Examples:
        >>> from pydantic import BaseModel
        >>> class Answer(BaseModel):
        ...     result: int
        ...     explanation: str
        >>> client = LLMClient()
        >>> messages = [
        ...     {"role": "system", "content": "Return JSON only."},
        ...     {"role": "user", "content": "What is 2+2? Return {result, explanation}"}
        ... ]
        >>> answer = get_structured_output(client, messages, Answer)
        >>> print(answer.result)
        4
    """
    # Create a working copy of messages to avoid mutating the input
    working_messages = list(messages)

    # Add schema instruction to the last user message or system prompt
    schema_json = output_schema.model_json_schema()
    schema_instruction = (
        f"\n\nIMPORTANT: Your response must be valid JSON matching this schema:\n"
        f"{schema_json}\n\n"
        f"Return ONLY the JSON object, no additional text or markdown."
    )

    # Append to last user message if exists, otherwise add system message
    if working_messages and working_messages[-1]["role"] == "user":
        working_messages[-1]["content"] += schema_instruction
    else:
        working_messages.append({
            "role": "system",
            "content": schema_instruction
        })

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            # Call LLM
            logger.debug(f"Structured output attempt {attempt}/{max_retries}")
            response = client.chat(
                working_messages,
                model=model,
                temperature=temperature,
                **llm_kwargs,
            )

            # Parse JSON from response
            try:
                parsed_data = parse_json(response.content, fix_common_errors=True)
            except JSONParseError as exc:
                raise StructuredOutputError(
                    f"Failed to parse LLM response as JSON: {exc}"
                ) from exc

            # Validate against schema
            try:
                validated = validate_output(parsed_data, output_schema)
                logger.info(
                    f"Successfully extracted structured output of type {output_schema.__name__}"
                )
                return validated
            except ValidationError as exc:
                # Get detailed error messages for feedback
                error_messages = get_validation_errors(parsed_data, output_schema)
                raise StructuredOutputError(
                    f"LLM response failed validation: {error_messages}"
                ) from exc

        except StructuredOutputError as exc:
            last_error = exc

            if attempt >= max_retries:
                logger.error(
                    f"Structured output extraction failed after {max_retries} attempts: {exc}"
                )
                raise StructuredOutputError(
                    f"Failed to extract structured output after {max_retries} retries. "
                    f"Last error: {exc}"
                ) from exc

            # Provide error feedback for retry
            error_feedback = (
                f"Your previous response had an error: {exc}\n\n"
                f"Please try again, ensuring your response is valid JSON "
                f"matching the exact schema provided."
            )
            working_messages.append({
                "role": "assistant",
                "content": response.content if 'response' in locals() else "[parsing failed]"
            })
            working_messages.append({
                "role": "user",
                "content": error_feedback
            })

            logger.warning(
                f"Structured output attempt {attempt} failed: {exc}. "
                f"Retrying with error feedback..."
            )

    # Should never reach here, but satisfy type checker
    if last_error:
        raise last_error
    raise StructuredOutputError("Unexpected error in retry logic")


async def aget_structured_output(
    client: LLMClient,
    messages: list[dict[str, str]],
    output_schema: type[BaseModel],
    max_retries: int = 3,
    model: str | None = None,
    temperature: float | None = None,
    **llm_kwargs: Any,
) -> BaseModel:
    """Async version of get_structured_output.

    Args:
        client: LLMClient instance for making API calls.
        messages: List of message dicts (role, content) for the conversation.
        output_schema: Pydantic model class defining expected output structure.
        max_retries: Maximum retry attempts on parse/validation failures.
        model: Optional model override (defaults to client config).
        temperature: Optional temperature override (defaults to client config).
        **llm_kwargs: Additional arguments passed to client.achat().

    Returns:
        Validated Pydantic model instance.

    Raises:
        StructuredOutputError: If extraction fails after all retries.
        LLMError: If the LLM API call fails.
    """
    # Create a working copy of messages to avoid mutating the input
    working_messages = list(messages)

    # Add schema instruction
    schema_json = output_schema.model_json_schema()
    schema_instruction = (
        f"\n\nIMPORTANT: Your response must be valid JSON matching this schema:\n"
        f"{schema_json}\n\n"
        f"Return ONLY the JSON object, no additional text or markdown."
    )

    if working_messages and working_messages[-1]["role"] == "user":
        working_messages[-1]["content"] += schema_instruction
    else:
        working_messages.append({
            "role": "system",
            "content": schema_instruction
        })

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            # Call LLM asynchronously
            logger.debug(f"Async structured output attempt {attempt}/{max_retries}")
            response = await client.achat(
                working_messages,
                model=model,
                temperature=temperature,
                **llm_kwargs,
            )

            # Parse JSON from response
            try:
                parsed_data = parse_json(response.content, fix_common_errors=True)
            except JSONParseError as exc:
                raise StructuredOutputError(
                    f"Failed to parse LLM response as JSON: {exc}"
                ) from exc

            # Validate against schema
            try:
                validated = validate_output(parsed_data, output_schema)
                logger.info(
                    f"Successfully extracted structured output of type {output_schema.__name__} (async)"
                )
                return validated
            except ValidationError as exc:
                error_messages = get_validation_errors(parsed_data, output_schema)
                raise StructuredOutputError(
                    f"LLM response failed validation: {error_messages}"
                ) from exc

        except StructuredOutputError as exc:
            last_error = exc

            if attempt >= max_retries:
                logger.error(
                    f"Async structured output extraction failed after {max_retries} attempts: {exc}"
                )
                raise StructuredOutputError(
                    f"Failed to extract structured output after {max_retries} retries. "
                    f"Last error: {exc}"
                ) from exc

            # Provide error feedback for retry
            error_feedback = (
                f"Your previous response had an error: {exc}\n\n"
                f"Please try again, ensuring your response is valid JSON "
                f"matching the exact schema provided."
            )
            working_messages.append({
                "role": "assistant",
                "content": response.content if 'response' in locals() else "[parsing failed]"
            })
            working_messages.append({
                "role": "user",
                "content": error_feedback
            })

            logger.warning(
                f"Async structured output attempt {attempt} failed: {exc}. "
                f"Retrying with error feedback..."
            )

    # Should never reach here
    if last_error:
        raise last_error
    raise StructuredOutputError("Unexpected error in async retry logic")
