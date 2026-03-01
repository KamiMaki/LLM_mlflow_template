"""Evaluator 資料模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class TestCase(BaseModel):
    """單一測試案例。"""
    input: dict[str, Any]       # 包含 system_prompt, user_prompt 等
    expected: str               # 預期輸出或關鍵字
    metadata: dict[str, Any] = {}


class ScorerResult(BaseModel):
    """單一 scorer 對單筆 test case 的評分結果。"""
    scorer_name: str
    score: float
    reason: str = ""


class EvalResult(BaseModel):
    """完整 evaluation 結果。"""
    metrics: dict[str, float]           # 彙總指標 (avg_score, pass_rate 等)
    details: list[dict[str, Any]]       # 每筆 test case 的詳細結果
    run_id: str | None = None           # MLflow run ID
