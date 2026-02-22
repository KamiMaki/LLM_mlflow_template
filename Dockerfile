# Multi-stage Dockerfile for LLM Framework
# Stage 1: Builder - Install dependencies using uv
FROM python:3.12-slim as builder

# Set working directory
WORKDIR /app

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files for dependency installation
COPY pyproject.toml uv.lock* ./
COPY src/ ./src/

# Create venv and install production dependencies only
RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv sync --group prod --no-dev

# Stage 2: Final production image
FROM python:3.12-slim

# Set environment variables
ENV LLM_ENV=prod \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN groupadd -r llmuser && useradd -r -g llmuser llmuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Set proper permissions for non-root user
RUN chown -R llmuser:llmuser /app

# Switch to non-root user
USER llmuser

# Expose port 8000 for API service
EXPOSE 8000

# Default command - can be overridden
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
