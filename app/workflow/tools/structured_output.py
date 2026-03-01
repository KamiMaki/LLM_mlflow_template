"""結構化 LLM 輸出 — 結合 LLM 呼叫、JSON 解析與 Pydantic 驗證。

自動重試並提供錯誤回饋。使用 tenacity 進行重試控制。

Usage:
    from app.workflow.tools.structured_output import get_structured_output

    result = get_structured_output(client, messages, MySchema, max_retries=3)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from app.logger import get_logger
from app.workflow.tools.parser import JSONParseError, parse_json
from app.workflow.tools.validator import get_validation_errors, validate_output

logger = get_logger(__name__)


class StructuredOutputError(Exception):
    """結構化輸出提取失敗。"""


def get_structured_output(
    client: Any,
    messages: list[dict[str, str]],
    output_schema: type[BaseModel],
    max_retries: int = 3,
    model: str | None = None,
    temperature: float | None = None,
    **llm_kwargs: Any,
) -> BaseModel:
    """呼叫 LLM 並解析為 Pydantic model，支援自動重試。

    Args:
        client: LLM client（需有 chat() 方法回傳含 .content 的 response）。
        messages: 對話 messages 列表。
        output_schema: 預期輸出的 Pydantic model class。
        max_retries: 最大重試次數。
        model: 可選 model override。
        temperature: 可選 temperature override。

    Returns:
        驗證通過的 Pydantic model instance。

    Raises:
        StructuredOutputError: 所有重試都失敗時。
    """
    working_messages = list(messages)

    schema_json = output_schema.model_json_schema()
    schema_instruction = (
        f"\n\nIMPORTANT: Your response must be valid JSON matching this schema:\n"
        f"{schema_json}\n\n"
        f"Return ONLY the JSON object, no additional text or markdown."
    )

    if working_messages and working_messages[-1]["role"] == "user":
        working_messages[-1] = {**working_messages[-1], "content": working_messages[-1]["content"] + schema_instruction}
    else:
        working_messages.append({"role": "system", "content": schema_instruction})

    last_error: Exception | None = None
    response_content = ""

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"Structured output attempt {attempt}/{max_retries}")

            chat_kwargs: dict[str, Any] = {}
            if model is not None:
                chat_kwargs["model"] = model
            if temperature is not None:
                chat_kwargs["temperature"] = temperature
            chat_kwargs.update(llm_kwargs)

            response = client.chat(working_messages, **chat_kwargs)
            response_content = response.content

            try:
                parsed = parse_json(response_content, fix_common_errors=True)
            except JSONParseError as exc:
                raise StructuredOutputError(f"JSON parse failed: {exc}") from exc

            try:
                validated = validate_output(parsed, output_schema)
                logger.info(f"Extracted {output_schema.__name__} successfully")
                return validated
            except ValidationError as exc:
                errors = get_validation_errors(parsed, output_schema)
                raise StructuredOutputError(f"Validation failed: {errors}") from exc

        except StructuredOutputError as exc:
            last_error = exc
            if attempt >= max_retries:
                logger.error(f"Structured output failed after {max_retries} attempts: {exc}")
                raise StructuredOutputError(f"Failed after {max_retries} retries: {exc}") from exc

            working_messages.append({"role": "assistant", "content": response_content or "[parse failed]"})
            working_messages.append({
                "role": "user",
                "content": f"Error: {exc}\nPlease try again with valid JSON matching the schema.",
            })
            logger.warning(f"Attempt {attempt} failed: {exc}, retrying...")

    if last_error:
        raise last_error
    raise StructuredOutputError("Unexpected error in retry logic")


async def aget_structured_output(
    client: Any,
    messages: list[dict[str, str]],
    output_schema: type[BaseModel],
    max_retries: int = 3,
    model: str | None = None,
    temperature: float | None = None,
    **llm_kwargs: Any,
) -> BaseModel:
    """非同步版本的 get_structured_output。"""
    working_messages = list(messages)

    schema_json = output_schema.model_json_schema()
    schema_instruction = (
        f"\n\nIMPORTANT: Your response must be valid JSON matching this schema:\n"
        f"{schema_json}\n\n"
        f"Return ONLY the JSON object, no additional text or markdown."
    )

    if working_messages and working_messages[-1]["role"] == "user":
        working_messages[-1] = {**working_messages[-1], "content": working_messages[-1]["content"] + schema_instruction}
    else:
        working_messages.append({"role": "system", "content": schema_instruction})

    last_error: Exception | None = None
    response_content = ""

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"Async structured output attempt {attempt}/{max_retries}")

            chat_kwargs: dict[str, Any] = {}
            if model is not None:
                chat_kwargs["model"] = model
            if temperature is not None:
                chat_kwargs["temperature"] = temperature
            chat_kwargs.update(llm_kwargs)

            response = await client.achat(working_messages, **chat_kwargs)
            response_content = response.content

            try:
                parsed = parse_json(response_content, fix_common_errors=True)
            except JSONParseError as exc:
                raise StructuredOutputError(f"JSON parse failed: {exc}") from exc

            try:
                validated = validate_output(parsed, output_schema)
                logger.info(f"Extracted {output_schema.__name__} successfully (async)")
                return validated
            except ValidationError as exc:
                errors = get_validation_errors(parsed, output_schema)
                raise StructuredOutputError(f"Validation failed: {errors}") from exc

        except StructuredOutputError as exc:
            last_error = exc
            if attempt >= max_retries:
                logger.error(f"Async structured output failed after {max_retries} attempts: {exc}")
                raise StructuredOutputError(f"Failed after {max_retries} retries: {exc}") from exc

            working_messages.append({"role": "assistant", "content": response_content or "[parse failed]"})
            working_messages.append({
                "role": "user",
                "content": f"Error: {exc}\nPlease try again with valid JSON matching the schema.",
            })
            logger.warning(f"Async attempt {attempt} failed: {exc}, retrying...")

    if last_error:
        raise last_error
    raise StructuredOutputError("Unexpected error in async retry logic")
