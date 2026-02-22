"""JSON parsing utilities for LLM outputs.

Handles common formatting issues in LLM-generated JSON, including markdown
code fences, trailing commas, single quotes, and unquoted keys.

Usage:
    from llm_framework.workflow.tools.parser import parse_json

    llm_output = '''```json
    {"name": "example", "value": 42,}
    ```'''

    data = parse_json(llm_output, fix_common_errors=True)
    print(data)  # {"name": "example", "value": 42}
"""

from __future__ import annotations

import json
import re
from typing import Any


class JSONParseError(Exception):
    """Raised when JSON parsing fails after all fix attempts."""


def extract_json(text: str) -> str:
    """Extract JSON from text that may have markdown fences or surrounding text.

    Looks for JSON content within markdown code blocks (```json ... ```)
    or standalone JSON objects/arrays. Returns the first valid-looking
    JSON structure found.

    Args:
        text: Raw text that may contain JSON.

    Returns:
        Extracted JSON string, or the original text if no JSON pattern found.

    Examples:
        >>> extract_json('```json\\n{"key": "value"}\\n```')
        '{"key": "value"}'
        >>> extract_json('Here is the data: {"key": "value"}')
        '{"key": "value"}'
    """
    # Try markdown code fence extraction first
    fence_pattern = r'```(?:json)?\s*\n(.*?)\n```'
    fence_match = re.search(fence_pattern, text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    # Try to find standalone JSON object
    obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    obj_match = re.search(obj_pattern, text, re.DOTALL)
    if obj_match:
        return obj_match.group(0).strip()

    # Try to find standalone JSON array
    arr_pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
    arr_match = re.search(arr_pattern, text, re.DOTALL)
    if arr_match:
        return arr_match.group(0).strip()

    # No JSON pattern found, return original text
    return text.strip()


def parse_json(text: str, fix_common_errors: bool = True) -> dict | list:
    """Parse JSON from LLM output with optional error fixing.

    First extracts JSON from markdown fences or surrounding text, then
    attempts to parse. If parsing fails and fix_common_errors is True,
    applies common fixes like removing trailing commas, converting single
    quotes to double quotes, and adding quotes to unquoted keys.

    Args:
        text: Raw text containing JSON (may include markdown or prose).
        fix_common_errors: Whether to attempt automatic fixes on parse errors.

    Returns:
        Parsed JSON as a dict or list.

    Raises:
        JSONParseError: If parsing fails after all fix attempts.

    Examples:
        >>> parse_json('{"key": "value"}')
        {'key': 'value'}
        >>> parse_json("{'key': 'value',}", fix_common_errors=True)
        {'key': 'value'}
    """
    # Extract JSON content
    json_str = extract_json(text)

    # Try direct parse first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        if not fix_common_errors:
            raise JSONParseError(
                f"Failed to parse JSON: {exc}. Set fix_common_errors=True to attempt fixes."
            ) from exc

    # Apply common fixes
    fixed = _fix_common_json_errors(json_str)

    # Try parsing again
    try:
        return json.loads(fixed)
    except json.JSONDecodeError as exc:
        raise JSONParseError(
            f"Failed to parse JSON after applying fixes: {exc}\n"
            f"Original: {json_str[:200]}...\n"
            f"Fixed: {fixed[:200]}..."
        ) from exc


def _fix_common_json_errors(json_str: str) -> str:
    """Apply common fixes to malformed JSON strings.

    Fixes applied:
    - Remove trailing commas before closing braces/brackets
    - Convert single quotes to double quotes (carefully)
    - Add double quotes to unquoted keys
    - Remove comments (// and /* */)

    Args:
        json_str: Potentially malformed JSON string.

    Returns:
        Fixed JSON string (may still be invalid).
    """
    # Remove comments
    # Single-line comments: // ...
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    # Multi-line comments: /* ... */
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

    # Remove trailing commas before } or ]
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

    # Convert single quotes to double quotes (naive approach)
    # This is fragile but handles simple cases
    json_str = json_str.replace("'", '"')

    # Add quotes to unquoted keys (e.g., {key: "value"} -> {"key": "value"})
    # Match pattern: word followed by colon, not already quoted
    json_str = re.sub(
        r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:',
        r'\1 "\2":',
        json_str
    )

    return json_str


def safe_parse_json(text: str, default: Any = None) -> dict | list | Any:
    """Parse JSON with a fallback default value on failure.

    Convenience wrapper around parse_json that returns a default value
    instead of raising an exception.

    Args:
        text: Raw text containing JSON.
        default: Value to return if parsing fails. Defaults to None.

    Returns:
        Parsed JSON or the default value.

    Examples:
        >>> safe_parse_json('invalid json', default={})
        {}
    """
    try:
        return parse_json(text, fix_common_errors=True)
    except JSONParseError:
        return default
