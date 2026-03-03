"""app.evaluator 單元測試。

注意：MLflow GenAI evaluate 需要 MLflow server，
此處僅測試 scorers 的邏輯，不測試完整 evaluate pipeline。
"""

from __future__ import annotations

from app.evaluator.scorers import (
    response_not_empty,
    response_length_check,
    exact_match,
    contains_keywords,
)


class TestResponseNotEmpty:
    def test_non_empty(self):
        result = response_not_empty._original_func(outputs="hello")
        assert result is True

    def test_empty(self):
        result = response_not_empty._original_func(outputs="")
        assert result is False

    def test_whitespace_only(self):
        result = response_not_empty._original_func(outputs="   ")
        assert result is False


class TestResponseLengthCheck:
    def test_appropriate_length(self):
        result = response_length_check._original_func(outputs="x" * 100)
        assert result.value == 1.0

    def test_too_short(self):
        result = response_length_check._original_func(outputs="hi")
        assert result.value == 0.3

    def test_too_long(self):
        result = response_length_check._original_func(outputs="x" * 3000)
        assert result.value == 0.5


class TestExactMatch:
    def test_match(self):
        result = exact_match._original_func(outputs="hello", expectations={"expected": "hello"})
        assert result is True

    def test_no_match(self):
        result = exact_match._original_func(outputs="hello", expectations={"expected": "world"})
        assert result is False

    def test_whitespace_insensitive(self):
        result = exact_match._original_func(outputs="  hello  ", expectations={"expected": "hello"})
        assert result is True

    def test_answer_key(self):
        result = exact_match._original_func(outputs="42", expectations={"answer": "42"})
        assert result is True


class TestContainsKeywords:
    def test_all_keywords(self):
        result = contains_keywords._original_func(
            outputs="hello world foo",
            expectations={"keywords": "hello,world"},
        )
        assert result.value == 1.0

    def test_partial_keywords(self):
        result = contains_keywords._original_func(
            outputs="hello bar",
            expectations={"keywords": "hello,world"},
        )
        assert result.value == 0.5

    def test_no_keywords(self):
        result = contains_keywords._original_func(
            outputs="nothing",
            expectations={"keywords": "hello,world"},
        )
        assert result.value == 0.0

    def test_keywords_as_list(self):
        result = contains_keywords._original_func(
            outputs="hello world",
            expectations={"keywords": ["hello", "world"]},
        )
        assert result.value == 1.0
