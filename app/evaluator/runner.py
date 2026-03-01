"""EvaluationRunner — 執行 workflow 驗證並記錄到 MLflow。

Usage:
    from app.evaluator.runner import EvaluationRunner
    from app.evaluator.scorers.builtin import ContainsScorer

    runner = EvaluationRunner()
    results = runner.evaluate(
        workflow_fn=my_workflow,
        test_cases="data/eval/cases.json",
        scorers=[ContainsScorer()],
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import yaml

from app.evaluator.models import EvalResult, ScorerResult, TestCase
from app.evaluator.scorers.base import BaseScorer
from app.logger import get_logger
from app.tracking.setup import is_mlflow_available

logger = get_logger(__name__)


class EvaluationRunner:
    """執行 evaluation pipeline 並記錄結果到 MLflow。"""

    def evaluate(
        self,
        workflow_fn: Callable[[dict], str],
        test_cases: str | list[dict[str, Any]],
        scorers: list[BaseScorer | Callable[[str, str], dict[str, float | str]]],
        run_name: str | None = None,
    ) -> EvalResult:
        """執行驗證。

        Args:
            workflow_fn: 接受 test case input dict，回傳 str 的 callable。
            test_cases: JSON/YAML 檔案路徑或 test case dict 列表。
            scorers: BaseScorer 實例或 callable(output, expected) -> dict 的列表。
            run_name: MLflow run 名稱。

        Returns:
            EvalResult 包含指標與詳細結果。
        """
        cases = self._load_test_cases(test_cases)
        logger.info(f"Running evaluation with {len(cases)} test cases and {len(scorers)} scorers")

        details: list[dict[str, Any]] = []
        all_scores: list[float] = []

        for idx, case in enumerate(cases):
            logger.debug(f"Evaluating test case {idx + 1}/{len(cases)}")

            # 執行 workflow
            try:
                output = workflow_fn(case.input)
            except Exception as e:
                logger.error(f"Workflow failed for test case {idx}: {e}")
                details.append({
                    "test_case_idx": idx,
                    "input": case.input,
                    "expected": case.expected,
                    "output": None,
                    "error": str(e),
                    "scores": [],
                })
                all_scores.append(0.0)
                continue

            # 執行所有 scorers
            case_scores: list[ScorerResult] = []
            for scorer in scorers:
                try:
                    if isinstance(scorer, BaseScorer):
                        result = scorer.score(output, case.expected)
                        scorer_name = scorer.name
                    else:
                        result = scorer(output, case.expected)
                        scorer_name = getattr(scorer, "__name__", "custom_scorer")
                    case_scores.append(ScorerResult(
                        scorer_name=scorer_name,
                        score=float(result.get("score", 0.0)),
                        reason=str(result.get("reason", "")),
                    ))
                except Exception as e:
                    logger.warning(f"Scorer failed: {e}")
                    case_scores.append(ScorerResult(scorer_name="error", score=0.0, reason=str(e)))

            avg_case_score = sum(s.score for s in case_scores) / max(len(case_scores), 1)
            all_scores.append(avg_case_score)

            details.append({
                "test_case_idx": idx,
                "input": case.input,
                "expected": case.expected,
                "output": output,
                "scores": [s.model_dump() for s in case_scores],
                "avg_score": avg_case_score,
                "metadata": case.metadata,
            })

        # 計算彙總指標
        avg_score = sum(all_scores) / max(len(all_scores), 1)
        pass_rate = sum(1 for s in all_scores if s >= 0.5) / max(len(all_scores), 1)
        metrics = {
            "avg_score": round(avg_score, 4),
            "pass_rate": round(pass_rate, 4),
            "total_cases": float(len(cases)),
            "passed_cases": float(sum(1 for s in all_scores if s >= 0.5)),
        }

        # 記錄到 MLflow
        run_id = self._log_to_mlflow(metrics, details, run_name)

        logger.info(f"Evaluation complete: avg_score={avg_score:.4f}, pass_rate={pass_rate:.4f}")
        return EvalResult(metrics=metrics, details=details, run_id=run_id)

    def _load_test_cases(self, source: str | list[dict[str, Any]]) -> list[TestCase]:
        """載入 test cases。"""
        if isinstance(source, list):
            return [TestCase(**tc) for tc in source]

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Test case file not found: {path}")

        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            raw = yaml.safe_load(text)
        else:
            raw = json.loads(text)

        if not isinstance(raw, list):
            raise ValueError(f"Test cases file must contain a JSON/YAML list, got {type(raw).__name__}")

        return [TestCase(**tc) for tc in raw]

    def _log_to_mlflow(
        self,
        metrics: dict[str, float],
        details: list[dict[str, Any]],
        run_name: str | None,
    ) -> str | None:
        """將評估結果記錄到 MLflow。"""
        if not is_mlflow_available():
            logger.debug("MLflow unavailable, skipping result logging")
            return None

        try:
            import mlflow

            with mlflow.start_run(run_name=run_name or "evaluation") as run:
                mlflow.log_metrics(metrics)
                mlflow.log_dict({"results": details}, "evaluation_details.json")
                logger.info(f"Evaluation results logged to MLflow run: {run.info.run_id}")
                return run.info.run_id

        except Exception as e:
            logger.error(f"Failed to log evaluation to MLflow: {e}")
            return None
