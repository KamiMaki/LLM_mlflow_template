"""LLM-as-Judge Scorer — 使用 LLM 進行評分。

Usage:
    from app.evaluator.scorers.llm_scorer import LLMJudgeScorer

    scorer = LLMJudgeScorer(client=my_llm_client)
    result = scorer.score(output="...", expected="...")
"""

from __future__ import annotations

from typing import Any

from app.evaluator.scorers.base import BaseScorer
from app.logger import get_logger
from app.workflow.tools.parser import safe_parse_json

logger = get_logger(__name__)

DEFAULT_JUDGE_PROMPT = """You are an evaluation judge. Score the following output against the expected result.

Expected: {expected}
Actual Output: {output}

Return a JSON object with:
- "score": a float between 0.0 and 1.0
- "reason": a brief explanation

Return ONLY the JSON object."""


class LLMJudgeScorer(BaseScorer):
    """使用 LLM 作為 judge 進行評分。

    Args:
        client: LLM client（需有 chat(system_prompt, user_prompt) 方法）。
        judge_prompt: 自定義 judge prompt 模板。
    """

    def __init__(self, client: Any, judge_prompt: str | None = None):
        self._client = client
        self._judge_prompt = judge_prompt or DEFAULT_JUDGE_PROMPT

    def score(self, output: str, expected: str) -> dict[str, float | str]:
        try:
            prompt = self._judge_prompt.format(output=output, expected=expected)
            response = self._client.chat(
                system_prompt="You are a strict evaluation judge. Return only JSON.",
                user_prompt=prompt,
            )
            parsed = safe_parse_json(response.content, default=None)
            if parsed and isinstance(parsed, dict):
                return {
                    "score": float(parsed.get("score", 0.0)),
                    "reason": str(parsed.get("reason", "")),
                }
            return {"score": 0.0, "reason": f"Failed to parse judge response: {response.content[:200]}"}
        except Exception as e:
            logger.error(f"LLM judge scoring failed: {e}")
            return {"score": 0.0, "reason": f"Judge error: {e}"}
