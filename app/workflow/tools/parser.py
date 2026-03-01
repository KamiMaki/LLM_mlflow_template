"""JSON 解析工具 — 處理 LLM 輸出中常見的 JSON 格式問題。

Usage:
    from app.workflow.tools.parser import parse_json

    data = parse_json('```json\\n{"key": "value",}\\n```')
"""

from __future__ import annotations

import json
import re
from typing import Any


class JSONParseError(Exception):
    """JSON 解析失敗時拋出。"""


def extract_json(text: str) -> str:
    """從可能包含 markdown 或其他文字的內容中提取 JSON。"""
    # Markdown code fence
    fence_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    # Standalone JSON object
    obj_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if obj_match:
        return obj_match.group(0).strip()

    # Standalone JSON array
    arr_match = re.search(r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]', text, re.DOTALL)
    if arr_match:
        return arr_match.group(0).strip()

    return text.strip()


def parse_json(text: str, fix_common_errors: bool = True) -> dict | list:
    """解析 LLM 輸出中的 JSON，可自動修復常見錯誤。"""
    json_str = extract_json(text)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        if not fix_common_errors:
            raise JSONParseError(f"JSON parse failed: {exc}") from exc

    fixed = _fix_common_json_errors(json_str)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError as exc:
        raise JSONParseError(
            f"JSON parse failed after fixes: {exc}\n"
            f"Original: {json_str[:200]}...\nFixed: {fixed[:200]}..."
        ) from exc


def _fix_common_json_errors(json_str: str) -> str:
    """修復常見 JSON 錯誤：trailing commas、single quotes、unquoted keys、comments。"""
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    json_str = json_str.replace("'", '"')
    json_str = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', json_str)
    return json_str


def safe_parse_json(text: str, default: Any = None) -> dict | list | Any:
    """安全解析 JSON，失敗時回傳 default。"""
    try:
        return parse_json(text, fix_common_errors=True)
    except JSONParseError:
        return default
