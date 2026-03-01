"""app.evaluator 單元測試。"""

from __future__ import annotations

from pathlib import Path

from app.evaluator import EvaluationRunner
from app.evaluator.scorers import ContainsScorer, ExactMatchScorer, JsonValidScorer


class TestExactMatchScorer:
    def test_exact_match(self):
        scorer = ExactMatchScorer()
        result = scorer.score("hello", "hello")
        assert result["score"] == 1.0

    def test_no_match(self):
        scorer = ExactMatchScorer()
        result = scorer.score("hello", "world")
        assert result["score"] == 0.0

    def test_whitespace_insensitive(self):
        scorer = ExactMatchScorer()
        result = scorer.score("  hello  ", "hello")
        assert result["score"] == 1.0


class TestContainsScorer:
    def test_all_keywords_found(self):
        scorer = ContainsScorer()
        result = scorer.score("hello world foo", "hello,world")
        assert result["score"] == 1.0

    def test_partial_keywords(self):
        scorer = ContainsScorer()
        result = scorer.score("hello bar", "hello,world")
        assert result["score"] == 0.5

    def test_no_keywords_found(self):
        scorer = ContainsScorer()
        result = scorer.score("nothing", "hello,world")
        assert result["score"] == 0.0


class TestJsonValidScorer:
    def test_valid_json(self):
        scorer = JsonValidScorer()
        result = scorer.score('{"key": "value"}', "")
        assert result["score"] == 1.0

    def test_invalid_json(self):
        scorer = JsonValidScorer()
        result = scorer.score("not json", "")
        assert result["score"] == 0.0

    def test_json_array(self):
        scorer = JsonValidScorer()
        result = scorer.score("[1, 2, 3]", "")
        assert result["score"] == 1.0


class TestEvaluationRunner:
    def test_evaluate_with_list(self):
        runner = EvaluationRunner()

        def mock_workflow(inputs: dict) -> str:
            return "hello world"

        test_cases = [
            {"input": {"user_prompt": "test"}, "expected": "hello", "metadata": {}},
        ]

        result = runner.evaluate(
            workflow_fn=mock_workflow,
            test_cases=test_cases,
            scorers=[ContainsScorer()],
        )
        assert result.metrics is not None
        assert len(result.details) == 1

    def test_evaluate_from_file(self, test_cases_file: Path):
        runner = EvaluationRunner()

        def mock_workflow(inputs: dict) -> str:
            return "hello goodbye"

        result = runner.evaluate(
            workflow_fn=mock_workflow,
            test_cases=str(test_cases_file),
            scorers=[ContainsScorer()],
        )
        assert len(result.details) == 2

    def test_evaluate_multiple_scorers(self):
        runner = EvaluationRunner()

        def mock_workflow(inputs: dict) -> str:
            return '{"result": "hello"}'

        test_cases = [
            {"input": {"user_prompt": "test"}, "expected": "hello", "metadata": {}},
        ]

        result = runner.evaluate(
            workflow_fn=mock_workflow,
            test_cases=test_cases,
            scorers=[ContainsScorer(), JsonValidScorer()],
        )
        assert len(result.details) == 1
