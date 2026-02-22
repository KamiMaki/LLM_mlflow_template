"""Tests for workflow tools: parser, validator, retry, prompt_template."""
import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch
import time

import pytest
from pydantic import BaseModel, ValidationError

from llm_framework.workflow.tools.parser import (
    extract_json,
    parse_json,
    safe_parse_json,
    JSONParseError,
)
from llm_framework.workflow.tools.validator import (
    validate_output,
    validate_or_none,
    validate_list,
    get_validation_errors,
)
from llm_framework.workflow.tools.retry import (
    with_retry,
    with_async_retry,
    calculate_backoff,
)
from llm_framework.workflow.tools.prompt_template import (
    PromptTemplate,
    PromptTemplateError,
    render_prompt,
    load_prompt,
    create_chat_messages,
    render_chat_messages,
)


# Test schemas for validation tests
class Person(BaseModel):
    name: str
    age: int


class Item(BaseModel):
    id: int
    name: str
    price: float


class TestExtractJson:
    """Test extract_json function."""

    def test_extract_from_markdown_fence(self):
        """Test extracting JSON from markdown code fence."""
        text = '```json\n{"key": "value"}\n```'
        result = extract_json(text)
        assert result == '{"key": "value"}'

    def test_extract_from_markdown_fence_without_language(self):
        """Test extracting JSON from markdown fence without language specifier."""
        text = '```\n{"key": "value"}\n```'
        result = extract_json(text)
        assert result == '{"key": "value"}'

    def test_extract_json_object_from_text(self):
        """Test extracting JSON object from surrounding text."""
        text = 'Here is the data: {"key": "value"} and more text'
        result = extract_json(text)
        assert result == '{"key": "value"}'

    def test_extract_json_array_from_text(self):
        """Test extracting JSON array from text."""
        text = 'The list is: [1, 2, 3] end'
        result = extract_json(text)
        assert result == '[1, 2, 3]'

    def test_extract_returns_original_if_no_json(self):
        """Test that extract_json returns original text if no JSON found."""
        text = 'This is plain text without JSON'
        result = extract_json(text)
        assert result == text.strip()

    def test_extract_nested_json(self):
        """Test extracting nested JSON objects."""
        text = '```json\n{"outer": {"inner": "value"}}\n```'
        result = extract_json(text)
        assert '{"outer": {"inner": "value"}}' in result


class TestParseJson:
    """Test parse_json function."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        text = '{"name": "Alice", "age": 30}'
        result = parse_json(text)
        assert result == {"name": "Alice", "age": 30}

    def test_parse_json_with_trailing_comma(self):
        """Test parsing JSON with trailing comma when fix_common_errors=True."""
        text = '{"name": "Bob", "age": 25,}'
        result = parse_json(text, fix_common_errors=True)
        assert result == {"name": "Bob", "age": 25}

    def test_parse_json_with_single_quotes(self):
        """Test parsing JSON with single quotes when fix_common_errors=True."""
        text = "{'name': 'Carol', 'age': 28}"
        result = parse_json(text, fix_common_errors=True)
        assert result == {"name": "Carol", "age": 28}

    def test_parse_json_with_unquoted_keys(self):
        """Test parsing JSON with unquoted keys when fix_common_errors=True."""
        text = '{name: "Dave", age: 35}'
        result = parse_json(text, fix_common_errors=True)
        assert result == {"name": "Dave", "age": 35}

    def test_parse_json_with_comments(self):
        """Test parsing JSON with comments when fix_common_errors=True."""
        text = '''
        {
            "name": "Eve", // This is a comment
            "age": 40
        }
        '''
        result = parse_json(text, fix_common_errors=True)
        assert result == {"name": "Eve", "age": 40}

    def test_parse_json_from_markdown(self):
        """Test parsing JSON extracted from markdown fence."""
        text = '```json\n{"key": "value"}\n```'
        result = parse_json(text)
        assert result == {"key": "value"}

    def test_parse_invalid_json_raises_error(self):
        """Test that invalid JSON raises JSONParseError."""
        text = '{"invalid": json}'
        with pytest.raises(JSONParseError):
            parse_json(text, fix_common_errors=False)

    def test_parse_broken_json_with_fixes(self):
        """Test parsing broken JSON with common fixes applied."""
        text = "{'key': 'value',}"  # Single quotes and trailing comma
        result = parse_json(text, fix_common_errors=True)
        assert result == {"key": "value"}

    def test_parse_json_array(self):
        """Test parsing JSON array."""
        text = '[1, 2, 3, 4]'
        result = parse_json(text)
        assert result == [1, 2, 3, 4]


class TestSafeParseJson:
    """Test safe_parse_json function."""

    def test_safe_parse_valid_json(self):
        """Test safe parsing of valid JSON."""
        text = '{"key": "value"}'
        result = safe_parse_json(text)
        assert result == {"key": "value"}

    def test_safe_parse_invalid_json_returns_default(self):
        """Test that invalid JSON returns default value."""
        text = 'invalid json'
        result = safe_parse_json(text, default={})
        assert result == {}

    def test_safe_parse_with_custom_default(self):
        """Test safe parsing with custom default value."""
        text = 'not json'
        result = safe_parse_json(text, default={"error": True})
        assert result == {"error": True}

    def test_safe_parse_returns_none_by_default(self):
        """Test that safe_parse_json returns None by default on error."""
        text = 'bad json'
        result = safe_parse_json(text)
        assert result is None


class TestValidateOutput:
    """Test validate_output function."""

    def test_validate_valid_data(self):
        """Test validating data that conforms to schema."""
        data = {"name": "Alice", "age": 30}
        result = validate_output(data, Person)
        assert isinstance(result, Person)
        assert result.name == "Alice"
        assert result.age == 30

    def test_validate_invalid_data_raises_error(self):
        """Test that invalid data raises ValidationError."""
        data = {"name": "Bob"}  # Missing 'age'
        with pytest.raises(ValidationError):
            validate_output(data, Person)

    def test_validate_with_wrong_types(self):
        """Test validation with wrong data types."""
        data = {"name": "Carol", "age": "thirty"}  # age should be int
        with pytest.raises(ValidationError):
            validate_output(data, Person)


class TestValidateOrNone:
    """Test validate_or_none function."""

    def test_validate_or_none_with_valid_data(self):
        """Test that valid data is validated successfully."""
        data = {"name": "Dave", "age": 35}
        result = validate_or_none(data, Person)
        assert isinstance(result, Person)
        assert result.name == "Dave"

    def test_validate_or_none_with_invalid_data(self):
        """Test that invalid data returns None."""
        data = {"name": "Eve"}  # Missing age
        result = validate_or_none(data, Person)
        assert result is None

    def test_validate_or_none_with_type_errors(self):
        """Test that type errors return None."""
        data = {"name": 123, "age": "invalid"}
        result = validate_or_none(data, Person)
        assert result is None


class TestValidateList:
    """Test validate_list function."""

    def test_validate_list_all_valid(self):
        """Test validating a list where all items are valid."""
        data = [
            {"id": 1, "name": "Item1", "price": 10.0},
            {"id": 2, "name": "Item2", "price": 20.0}
        ]
        result = validate_list(data, Item)
        assert len(result) == 2
        assert all(isinstance(item, Item) for item in result)

    def test_validate_list_with_invalid_raises_error(self):
        """Test that invalid item raises error when skip_invalid=False."""
        data = [
            {"id": 1, "name": "Item1", "price": 10.0},
            {"id": 2, "name": "Item2"}  # Missing price
        ]
        with pytest.raises(ValidationError):
            validate_list(data, Item, skip_invalid=False)

    def test_validate_list_skip_invalid(self):
        """Test that invalid items are skipped when skip_invalid=True."""
        data = [
            {"id": 1, "name": "Item1", "price": 10.0},
            {"id": 2, "name": "Item2"},  # Missing price
            {"id": 3, "name": "Item3", "price": 30.0}
        ]
        result = validate_list(data, Item, skip_invalid=True)
        assert len(result) == 2
        assert result[0].id == 1
        assert result[1].id == 3

    def test_validate_empty_list(self):
        """Test validating an empty list."""
        result = validate_list([], Item)
        assert result == []


class TestGetValidationErrors:
    """Test get_validation_errors function."""

    def test_get_validation_errors_with_valid_data(self):
        """Test that valid data returns empty error list."""
        data = {"name": "Alice", "age": 30}
        errors = get_validation_errors(data, Person)
        assert errors == []

    def test_get_validation_errors_with_missing_field(self):
        """Test getting errors for missing required field."""
        data = {"name": "Bob"}
        errors = get_validation_errors(data, Person)
        assert len(errors) > 0
        assert any("required" in error.lower() for error in errors)

    def test_get_validation_errors_with_wrong_type(self):
        """Test getting errors for wrong data type."""
        data = {"name": "Carol", "age": "thirty"}
        errors = get_validation_errors(data, Person)
        assert len(errors) > 0

    def test_get_validation_errors_multiple_issues(self):
        """Test getting errors for multiple validation issues."""
        data = {"age": "not_an_int"}  # Missing name, wrong type for age
        errors = get_validation_errors(data, Person)
        assert len(errors) >= 2


class TestWithRetry:
    """Test with_retry decorator."""

    def test_with_retry_succeeds_first_attempt(self):
        """Test that successful function doesn't retry."""
        call_count = 0

        @with_retry(max_retries=3)
        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = success_func()
        assert result == "success"
        assert call_count == 1

    def test_with_retry_counts_attempts(self):
        """Test that with_retry counts retry attempts correctly."""
        call_count = 0

        @with_retry(max_retries=3, backoff_base=0.01)
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"

        result = failing_func()
        assert result == "success"
        assert call_count == 3

    def test_with_retry_exhausts_retries(self):
        """Test that function fails after max retries."""
        call_count = 0

        @with_retry(max_retries=2, backoff_base=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            always_fails()

        assert call_count == 2

    def test_with_retry_without_parentheses(self):
        """Test using decorator without parentheses."""
        @with_retry
        def simple_func():
            return 42

        result = simple_func()
        assert result == 42

    def test_with_retry_specific_exceptions(self):
        """Test that only specified exceptions trigger retries."""
        call_count = 0

        @with_retry(max_retries=3, retryable_exceptions=(ValueError,))
        def raises_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Wrong exception type")

        with pytest.raises(TypeError):
            raises_type_error()

        # Should fail immediately without retries
        assert call_count == 1


class TestBackoffCalculation:
    """Test calculate_backoff function."""

    def test_calculate_backoff_first_attempt(self):
        """Test backoff calculation for first attempt."""
        result = calculate_backoff(1, base=1.0, factor=2.0)
        assert result == 1.0

    def test_calculate_backoff_second_attempt(self):
        """Test backoff calculation for second attempt."""
        result = calculate_backoff(2, base=1.0, factor=2.0)
        assert result == 2.0

    def test_calculate_backoff_third_attempt(self):
        """Test backoff calculation for third attempt."""
        result = calculate_backoff(3, base=1.0, factor=2.0)
        assert result == 4.0

    def test_calculate_backoff_custom_base(self):
        """Test backoff calculation with custom base."""
        result = calculate_backoff(2, base=0.5, factor=2.0)
        assert result == 1.0

    def test_calculate_backoff_custom_factor(self):
        """Test backoff calculation with custom factor."""
        result = calculate_backoff(3, base=1.0, factor=3.0)
        assert result == 9.0


class TestPromptTemplate:
    """Test PromptTemplate class."""

    def test_render_simple_template(self):
        """Test rendering a simple template."""
        template = PromptTemplate("Hello {{ name }}!")
        result = template.render(name="World")
        assert result == "Hello World!"

    def test_render_with_multiple_variables(self):
        """Test rendering template with multiple variables."""
        template = PromptTemplate("{{ greeting }} {{ name }}, you are {{ age }} years old.")
        result = template.render(greeting="Hi", name="Alice", age=30)
        assert "Hi Alice" in result
        assert "30" in result

    def test_render_with_conditional(self):
        """Test rendering template with Jinja2 conditional."""
        template = PromptTemplate(
            "Hello {{ name }}!{% if age %} You are {{ age }} years old.{% endif %}"
        )
        result = template.render(name="Bob", age=25)
        assert "Hello Bob!" in result
        assert "25 years old" in result

    def test_render_with_loop(self):
        """Test rendering template with Jinja2 loop."""
        template = PromptTemplate(
            "Items:\n{% for item in items %}- {{ item }}\n{% endfor %}"
        )
        result = template.render(items=["apple", "banana", "cherry"])
        assert "apple" in result
        assert "banana" in result
        assert "cherry" in result

    def test_render_missing_variable_raises_error(self):
        """Test that missing variable raises PromptTemplateError."""
        template = PromptTemplate("Hello {{ name }}!")
        with pytest.raises(PromptTemplateError):
            template.render()

    def test_from_file(self, tmp_path):
        """Test loading template from file."""
        template_file = tmp_path / "template.jinja2"
        template_file.write_text("Hello {{ name }}!", encoding="utf-8")

        template = PromptTemplate.from_file(template_file)
        result = template.render(name="World")
        assert result == "Hello World!"

    def test_from_file_not_found(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(PromptTemplateError, match="not found"):
            PromptTemplate.from_file("/nonexistent/path/template.jinja2")

    def test_invalid_template_raises_error(self):
        """Test that invalid Jinja2 syntax raises error."""
        with pytest.raises(PromptTemplateError):
            PromptTemplate("{% for item in items %}")  # Unclosed for loop


class TestRenderPrompt:
    """Test render_prompt convenience function."""

    def test_render_prompt_simple(self):
        """Test rendering a simple prompt."""
        result = render_prompt("Hello {{ name }}!", name="Alice")
        assert result == "Hello Alice!"

    def test_render_prompt_with_multiple_vars(self):
        """Test rendering prompt with multiple variables."""
        result = render_prompt(
            "{{ greeting }} {{ name }}",
            greeting="Hi",
            name="Bob"
        )
        assert result == "Hi Bob"


class TestLoadPrompt:
    """Test load_prompt convenience function."""

    def test_load_prompt(self, tmp_path):
        """Test loading and rendering prompt from file."""
        template_file = tmp_path / "greeting.jinja2"
        template_file.write_text("Welcome, {{ user }}!", encoding="utf-8")

        result = load_prompt(template_file, user="Charlie")
        assert result == "Welcome, Charlie!"


class TestCreateChatMessages:
    """Test create_chat_messages function."""

    def test_create_chat_messages_all_roles(self):
        """Test creating messages with all roles."""
        messages = create_chat_messages(
            system_prompt="You are helpful.",
            user_prompt="Hello!",
            assistant_prompt="Hi there!"
        )

        assert len(messages) == 3
        assert messages[0] == {"role": "system", "content": "You are helpful."}
        assert messages[1] == {"role": "user", "content": "Hello!"}
        assert messages[2] == {"role": "assistant", "content": "Hi there!"}

    def test_create_chat_messages_partial(self):
        """Test creating messages with only some roles."""
        messages = create_chat_messages(
            system_prompt="System message",
            user_prompt="User message"
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_create_chat_messages_empty(self):
        """Test creating empty message list."""
        messages = create_chat_messages()
        assert messages == []


class TestRenderChatMessages:
    """Test render_chat_messages function."""

    def test_render_chat_messages_with_templates(self):
        """Test rendering chat messages from templates."""
        messages = render_chat_messages(
            system_template="You are {{ role }}.",
            user_template="Process {{ data }}",
            role="assistant",
            data="input"
        )

        assert len(messages) == 2
        assert messages[0]["content"] == "You are assistant."
        assert messages[1]["content"] == "Process input"

    def test_render_chat_messages_partial_templates(self):
        """Test rendering with only some templates."""
        messages = render_chat_messages(
            user_template="Hello {{ name }}!",
            name="World"
        )

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello World!"


class TestAsyncRetry:
    """Test async retry decorators."""

    @pytest.mark.asyncio
    async def test_with_async_retry_succeeds(self):
        """Test async retry with successful function."""
        call_count = 0

        @with_async_retry
        async def async_success():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await async_success()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_with_async_retry_retries(self):
        """Test async retry with temporary failures."""
        call_count = 0

        @with_async_retry(max_retries=3, backoff_base=0.01)
        async def async_retry_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary error")
            return "success"

        result = await async_retry_func()
        assert result == "success"
        assert call_count == 2
