"""Jinja2-based prompt templates for LLM workflows.

Provides a simple wrapper around Jinja2 for rendering dynamic prompts
with variable substitution, control flow, and filters.

Usage:
    from llm_framework.workflow.tools.prompt_template import PromptTemplate

    template = PromptTemplate('''
    Hello {{ name }}!
    {% if age %}You are {{ age }} years old.{% endif %}
    ''')

    prompt = template.render(name="Alice", age=30)
    print(prompt)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, Template, TemplateError, StrictUndefined

logger = logging.getLogger(__name__)


class PromptTemplateError(Exception):
    """Raised when template rendering or loading fails."""


class PromptTemplate:
    """Jinja2-based prompt template with strict variable checking.

    Uses Jinja2's StrictUndefined to catch missing variables early
    and prevent silent failures in prompt rendering.

    Args:
        template: Jinja2 template string.

    Examples:
        >>> template = PromptTemplate("Hello {{ name }}!")
        >>> template.render(name="World")
        'Hello World!'

        >>> template = PromptTemplate('''
        ... {% for item in items %}
        ... - {{ item }}
        ... {% endfor %}
        ... ''')
        >>> template.render(items=["apple", "banana"])
        '- apple\\n- banana\\n'
    """

    def __init__(self, template: str):
        self._template_str = template
        self._env = Environment(
            autoescape=False,
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        try:
            self._template: Template = self._env.from_string(template)
        except TemplateError as exc:
            raise PromptTemplateError(
                f"Failed to parse template: {exc}"
            ) from exc

        logger.debug("Initialized PromptTemplate")

    def render(self, **kwargs: Any) -> str:
        """Render the template with provided variables.

        Args:
            **kwargs: Variables to substitute in the template.

        Returns:
            Rendered template string.

        Raises:
            PromptTemplateError: If rendering fails (e.g., missing variables).

        Examples:
            >>> template = PromptTemplate("User: {{ username }}, Role: {{ role }}")
            >>> template.render(username="alice", role="admin")
            'User: alice, Role: admin'
        """
        try:
            rendered = self._template.render(**kwargs)
            logger.debug(f"Rendered template with {len(kwargs)} variables")
            return rendered
        except TemplateError as exc:
            raise PromptTemplateError(
                f"Failed to render template: {exc}\n"
                f"Template: {self._template_str[:200]}...\n"
                f"Variables: {list(kwargs.keys())}"
            ) from exc

    @classmethod
    def from_file(cls, path: str | Path) -> PromptTemplate:
        """Load a template from a file.

        Args:
            path: Path to the template file.

        Returns:
            PromptTemplate instance loaded from the file.

        Raises:
            PromptTemplateError: If the file cannot be read or parsed.

        Examples:
            >>> template = PromptTemplate.from_file("prompts/greeting.jinja2")
            >>> template.render(name="Bob")
            'Hello Bob!'
        """
        path_obj = Path(path)

        if not path_obj.exists():
            raise PromptTemplateError(
                f"Template file not found: {path_obj}"
            )

        try:
            template_str = path_obj.read_text(encoding="utf-8")
            logger.debug(f"Loaded template from {path_obj}")
            return cls(template_str)
        except (OSError, UnicodeDecodeError) as exc:
            raise PromptTemplateError(
                f"Failed to read template file {path_obj}: {exc}"
            ) from exc

    def get_variables(self) -> set[str]:
        """Get the set of undefined variables in the template.

        Returns:
            Set of variable names that need to be provided during rendering.

        Examples:
            >>> template = PromptTemplate("Hello {{ name }}, age {{ age }}")
            >>> template.get_variables()
            {'name', 'age'}
        """
        return self._template.environment.parse(self._template_str).find_all(
            lambda node: hasattr(node, 'name')
        )


def render_prompt(template_str: str, **kwargs: Any) -> str:
    """Convenience function to render a template string directly.

    Args:
        template_str: Jinja2 template string.
        **kwargs: Variables to substitute in the template.

    Returns:
        Rendered template string.

    Raises:
        PromptTemplateError: If parsing or rendering fails.

    Examples:
        >>> render_prompt("Hello {{ name }}!", name="World")
        'Hello World!'
    """
    template = PromptTemplate(template_str)
    return template.render(**kwargs)


def load_prompt(path: str | Path, **kwargs: Any) -> str:
    """Convenience function to load and render a template file.

    Args:
        path: Path to the template file.
        **kwargs: Variables to substitute in the template.

    Returns:
        Rendered template string.

    Raises:
        PromptTemplateError: If loading, parsing, or rendering fails.

    Examples:
        >>> prompt = load_prompt("prompts/system.jinja2", role="assistant")
    """
    template = PromptTemplate.from_file(path)
    return template.render(**kwargs)


# ---------------------------------------------------------------------------
# Common prompt utilities
# ---------------------------------------------------------------------------


def create_chat_messages(
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    assistant_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Create a chat message list from prompt strings.

    Convenience helper for building message arrays for LLM clients.

    Args:
        system_prompt: Optional system message.
        user_prompt: Optional user message.
        assistant_prompt: Optional assistant message (for few-shot examples).

    Returns:
        List of message dicts with role and content fields.

    Examples:
        >>> create_chat_messages(
        ...     system_prompt="You are a helpful assistant.",
        ...     user_prompt="What is 2+2?"
        ... )
        [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'What is 2+2?'}
        ]
    """
    messages: list[dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})

    if assistant_prompt:
        messages.append({"role": "assistant", "content": assistant_prompt})

    return messages


def render_chat_messages(
    system_template: str | None = None,
    user_template: str | None = None,
    assistant_template: str | None = None,
    **kwargs: Any,
) -> list[dict[str, str]]:
    """Render templates and create chat messages in one step.

    Args:
        system_template: Optional Jinja2 system message template.
        user_template: Optional Jinja2 user message template.
        assistant_template: Optional Jinja2 assistant message template.
        **kwargs: Variables for template rendering.

    Returns:
        List of rendered message dicts.

    Examples:
        >>> render_chat_messages(
        ...     system_template="You are {{ role }}.",
        ...     user_template="Process this: {{ data }}",
        ...     role="assistant",
        ...     data="example"
        ... )
        [
            {'role': 'system', 'content': 'You are assistant.'},
            {'role': 'user', 'content': 'Process this: example'}
        ]
    """
    messages: list[dict[str, str]] = []

    if system_template:
        rendered = render_prompt(system_template, **kwargs)
        messages.append({"role": "system", "content": rendered})

    if user_template:
        rendered = render_prompt(user_template, **kwargs)
        messages.append({"role": "user", "content": rendered})

    if assistant_template:
        rendered = render_prompt(assistant_template, **kwargs)
        messages.append({"role": "assistant", "content": rendered})

    return messages
