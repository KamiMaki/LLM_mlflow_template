"""MLflow GenAI Scorers — 自定義評分器與 LLM Judge。

使用 MLflow 3.x @scorer decorator 和 make_judge() 建立評分器。

Usage:
    from app.evaluator.scorers import response_not_empty, response_length_check, tone_judge

    results = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=my_app,
        scorers=[response_not_empty, response_length_check, tone_judge],
    )
"""

from __future__ import annotations

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


# --- LLM Judge（需要 LLM API key 才能使用）---

def create_tone_judge(model: str = "openai:/gpt-4o-mini"):
    """建立專業語調 LLM judge。"""
    from typing import Literal
    from mlflow.genai.judges import make_judge

    return make_judge(
        name="professional_tone",
        instructions=(
            "Evaluate if the response maintains a professional tone.\n"
            "Output: {{ outputs }}\n"
            "Return 'yes' if professional, 'no' otherwise."
        ),
        feedback_value_type=Literal["yes", "no"],
        model=model,
    )


def create_quality_judge(model: str = "openai:/gpt-4o-mini"):
    """建立回答品質 LLM judge。"""
    from typing import Literal
    from mlflow.genai.judges import make_judge

    return make_judge(
        name="answer_quality",
        instructions=(
            "Evaluate if the response correctly and completely answers the question.\n"
            "Question: {{ inputs }}\n"
            "Response: {{ outputs }}\n"
            "Return 'yes' if correct and complete, 'no' otherwise."
        ),
        feedback_value_type=Literal["yes", "no"],
        model=model,
    )
