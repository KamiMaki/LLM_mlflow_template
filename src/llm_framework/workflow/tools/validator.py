"""Output validation using Pydantic schemas.

Validates and parses LLM outputs into strongly-typed Pydantic models,
ensuring data conforms to expected schemas before downstream processing.

Usage:
    from pydantic import BaseModel
    from llm_framework.workflow.tools.validator import validate_output

    class UserInfo(BaseModel):
        name: str
        age: int

    data = {"name": "Alice", "age": 30}
    user = validate_output(data, UserInfo)
    print(user.name)  # Alice
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


def validate_output(data: dict | list | Any, schema: type[BaseModel]) -> BaseModel:
    """Validate and parse LLM output into a Pydantic model.

    Args:
        data: Raw data to validate (typically from parsed JSON).
        schema: Pydantic model class defining the expected structure.

    Returns:
        Validated and parsed Pydantic model instance.

    Raises:
        ValidationError: If data does not conform to the schema.

    Examples:
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> validate_output({"name": "Bob", "age": 25}, Person)
        Person(name='Bob', age=25)
    """
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        logger.error(
            f"Validation failed for schema {schema.__name__}: {exc}\n"
            f"Data: {data}"
        )
        raise


def validate_or_none(data: dict | list | Any, schema: type[BaseModel]) -> BaseModel | None:
    """Validate LLM output, returns None on failure instead of raising.

    Convenience wrapper around validate_output that catches validation
    errors and returns None, useful for optional or best-effort parsing.

    Args:
        data: Raw data to validate.
        schema: Pydantic model class defining the expected structure.

    Returns:
        Validated Pydantic model instance, or None if validation fails.

    Examples:
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> validate_or_none({"name": "Carol"}, Person)
        None
        >>> validate_or_none({"name": "David", "age": 35}, Person)
        Person(name='David', age=35)
    """
    try:
        return validate_output(data, schema)
    except ValidationError as exc:
        logger.warning(
            f"Validation failed for schema {schema.__name__}, returning None: {exc}"
        )
        return None


def validate_list(
    data: list[dict],
    schema: type[BaseModel],
    skip_invalid: bool = False,
) -> list[BaseModel]:
    """Validate a list of items against a Pydantic schema.

    Args:
        data: List of raw data items to validate.
        schema: Pydantic model class defining the expected structure.
        skip_invalid: If True, skip invalid items instead of raising.
                     If False, raise on first validation error.

    Returns:
        List of validated Pydantic model instances.

    Raises:
        ValidationError: If skip_invalid=False and any item is invalid.

    Examples:
        >>> from pydantic import BaseModel
        >>> class Item(BaseModel):
        ...     id: int
        ...     name: str
        >>> items = [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]
        >>> validate_list(items, Item)
        [Item(id=1, name='A'), Item(id=2, name='B')]
    """
    results: list[BaseModel] = []

    for idx, item in enumerate(data):
        try:
            validated = validate_output(item, schema)
            results.append(validated)
        except ValidationError as exc:
            if skip_invalid:
                logger.warning(
                    f"Skipping invalid item at index {idx} in list: {exc}"
                )
                continue
            else:
                logger.error(
                    f"Validation failed for item at index {idx}: {exc}\n"
                    f"Item: {item}"
                )
                raise

    return results


def get_validation_errors(data: dict | list | Any, schema: type[BaseModel]) -> list[str]:
    """Get human-readable validation error messages without raising.

    Useful for debugging or providing feedback to LLMs about what
    went wrong with their output format.

    Args:
        data: Raw data to validate.
        schema: Pydantic model class defining the expected structure.

    Returns:
        List of error message strings, empty if validation succeeds.

    Examples:
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> get_validation_errors({"name": "Eve"}, Person)
        ['Field required [type=missing, input_value={...}, input_type=dict]']
    """
    try:
        schema.model_validate(data)
        return []
    except ValidationError as exc:
        return [str(error) for error in exc.errors()]
