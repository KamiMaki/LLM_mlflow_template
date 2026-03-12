"""MLflow GenAI Scorers — 自定義評分器與 LLM Judge。

Rule-based scorers 使用 MLflow @scorer decorator。
LLM Judge 使用 litellm 進行評分。

Usage:
    from app.evaluator.scorers import response_not_empty, create_quality_judge

    results = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=my_app,
        scorers=[response_not_empty, create_quality_judge()],
    )
"""

from __future__ import annotations

from typing import Any

from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback

# Re-export 常用內建 scorers
from mlflow.genai.scorers import (  # noqa: F401
    Correctness,
    RelevanceToQuery,
    Safety,
)


# --- Rule-based Scorers ---

@scorer
def response_not_empty(outputs: str) -> bool:
    """檢查輸出是否非空。"""
    return len(outputs.strip()) > 0


@scorer
def response_length_check(outputs: str) -> Feedback:
    """檢查輸出長度是否合理（50~2000 字元）。"""
    length = len(outputs)
    if length < 50:
        return Feedback(value=0.3, rationale=f"Response too short ({length} chars)")
    elif length > 2000:
        return Feedback(value=0.5, rationale=f"Response too long ({length} chars)")
    return Feedback(value=1.0, rationale=f"Response length appropriate ({length} chars)")


@scorer
def exact_match(outputs: str, expectations: dict) -> bool:
    """精確匹配（忽略前後空白）。"""
    expected = expectations.get("expected", expectations.get("answer", ""))
    return outputs.strip() == str(expected).strip()


@scorer
def contains_keywords(outputs: str, expectations: dict) -> Feedback:
    """檢查輸出是否包含所有預期關鍵字。"""
    keywords_str = expectations.get("keywords", expectations.get("expected", ""))
    if isinstance(keywords_str, list):
        keywords = keywords_str
    else:
        keywords = [k.strip() for k in str(keywords_str).split(",") if k.strip()]

    if not keywords:
        return Feedback(value=1.0, rationale="No keywords to check")

    found = sum(1 for kw in keywords if kw.lower() in outputs.lower())
    score = found / len(keywords)
    return Feedback(
        value=score,
        rationale=f"Found {found}/{len(keywords)} keywords",
    )


# --- LLM Judge（使用 litellm）---

def create_llm_judge(
    *,
    name: str,
    instructions: str,
    config: Any | None = None,
    **llm_overrides: Any,
) -> Any:
    """建立基於 litellm 的 LLM Judge scorer。

    Args:
        name: Judge 名稱（同時作為 scorer 名稱）。
        instructions: Judge 評分指示，可使用 {inputs} 和 {outputs} 佔位符。
        config: LLMConfig，None 時自動從 llm_config.yaml 載入。
        **llm_overrides: 傳給 LLMService.call_llm 的覆寫參數。

    Returns:
        MLflow @scorer 裝飾的評分函式。
    """
    import json
    from llm_service import LLMService

    service = LLMService(config=config) if config else LLMService()

    @scorer
    def llm_judge(inputs: dict | str, outputs: str) -> Feedback:
        prompt = instructions.format(
            inputs=inputs if isinstance(inputs, str) else str(inputs),
            outputs=outputs,
        )

        response = service.call_llm(
            user_prompt=prompt,
            system_prompt="You are a strict evaluator. Respond with a JSON object containing 'score' (float 0-1) and 'rationale' (string).",
            **llm_overrides,
        )
        content = response.content

        try:
            result = json.loads(content)
            return Feedback(
                value=float(result.get("score", 0.0)),
                rationale=result.get("rationale", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            content_lower = content.strip().lower()
            if content_lower in ("yes", "true"):
                return Feedback(value=1.0, rationale=content)
            elif content_lower in ("no", "false"):
                return Feedback(value=0.0, rationale=content)
            return Feedback(value=0.5, rationale=f"Unparseable judge response: {content}")

    llm_judge.__name__ = name
    return llm_judge


def create_tone_judge(**kwargs: Any) -> Any:
    """建立專業語調 LLM Judge。"""
    return create_llm_judge(
        name="professional_tone",
        instructions=(
            "Evaluate if the following response maintains a professional tone.\n"
            "Response: {outputs}\n"
            "Return a JSON with 'score' (1.0 if professional, 0.0 if not) and 'rationale'."
        ),
        **kwargs,
    )


def create_quality_judge(**kwargs: Any) -> Any:
    """建立回答品質 LLM Judge。"""
    return create_llm_judge(
        name="answer_quality",
        instructions=(
            "Evaluate if the response correctly and completely answers the question.\n"
            "Question: {inputs}\n"
            "Response: {outputs}\n"
            "Return a JSON with 'score' (1.0 if correct and complete, 0.0 if not) and 'rationale'."
        ),
        **kwargs,
    )
