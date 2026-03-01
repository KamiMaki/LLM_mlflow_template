"""Pydantic 驗證工具 — 驗證 LLM 輸出是否符合指定 schema。

Usage:
    from pydantic import BaseModel
    from app.workflow.tools.validator import validate_output

    class UserInfo(BaseModel):
        name: str
        age: int

    user = validate_output({"name": "Alice", "age": 30}, UserInfo)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from app.logger import get_logger

logger = get_logger(__name__)


def validate_output(data: dict | list | Any, schema: type[BaseModel]) -> BaseModel:
    """驗證並解析為 Pydantic model。"""
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        logger.error(f"Validation failed for {schema.__name__}: {exc}")
        raise


def validate_or_none(data: dict | list | Any, schema: type[BaseModel]) -> BaseModel | None:
    """驗證，失敗時回傳 None。"""
    try:
        return validate_output(data, schema)
    except ValidationError as exc:
        logger.warning(f"Validation failed for {schema.__name__}, returning None: {exc}")
        return None


def validate_list(
    data: list[dict],
    schema: type[BaseModel],
    skip_invalid: bool = False,
) -> list[BaseModel]:
    """驗證 list 中的每個 item。"""
    results: list[BaseModel] = []
    for idx, item in enumerate(data):
        try:
            results.append(validate_output(item, schema))
        except ValidationError as exc:
            if skip_invalid:
                logger.warning(f"Skipping invalid item at index {idx}: {exc}")
                continue
            raise
    return results


def get_validation_errors(data: dict | list | Any, schema: type[BaseModel]) -> list[str]:
    """取得驗證錯誤訊息列表（不拋出例外）。"""
    try:
        schema.model_validate(data)
        return []
    except ValidationError as exc:
        return [str(error) for error in exc.errors()]
