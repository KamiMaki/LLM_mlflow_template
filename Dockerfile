# Multi-stage Dockerfile for LLM Project Template
FROM python:3.12-slim as builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock* ./
COPY app/ ./app/
COPY llm_service/ ./llm_service/

RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv sync --no-dev

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN groupadd -r llmuser && useradd -r -g llmuser llmuser

COPY --from=builder /opt/venv /opt/venv

COPY app/ ./app/
COPY llm_service/ ./llm_service/
COPY config/ ./config/
COPY prompts/ ./prompts/
RUN chown -R llmuser:llmuser /app

USER llmuser

EXPOSE 8000

CMD ["python", "-m", "app.main", "env=prod"]
