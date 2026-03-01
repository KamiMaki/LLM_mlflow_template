"""內建 Scorers — exact_match, contains, json_valid。"""

from __future__ import annotations

import json

from app.evaluator.scorers.base import BaseScorer


class ExactMatchScorer(BaseScorer):
    """完全匹配（忽略前後空白）。"""

    def score(self, output: str, expected: str) -> dict[str, float | str]:
        match = output.strip() == expected.strip()
        return {"score": 1.0 if match else 0.0, "reason": "Exact match" if match else "No match"}


class ContainsScorer(BaseScorer):
    """檢查 output 是否包含 expected 中的關鍵字（逗號分隔）。"""

    def score(self, output: str, expected: str) -> dict[str, float | str]:
        keywords = [k.strip() for k in expected.split(",") if k.strip()]
        if not keywords:
            return {"score": 1.0, "reason": "No keywords to check"}
        found = sum(1 for k in keywords if k in output)
        ratio = found / len(keywords)
        return {"score": ratio, "reason": f"Found {found}/{len(keywords)} keywords"}


class JsonValidScorer(BaseScorer):
    """檢查 output 是否為合法 JSON。"""

    def score(self, output: str, expected: str) -> dict[str, float | str]:
        try:
            json.loads(output)
            return {"score": 1.0, "reason": "Valid JSON"}
        except (json.JSONDecodeError, TypeError) as e:
            return {"score": 0.0, "reason": f"Invalid JSON: {e}"}
