# Multi-stage build for Huxley with optional local model support
from python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY poetry.lock* ./
COPY README.md ./
COPY src ./src

# Install Python dependencies
RUN pip install --upgrade pip setuptools hatchling && \
    pip install . && \
    pip install anthropic cohere google-generativeai

# Development stage (includes dev tools)
from base as development

RUN pip install -e ".[dev]"

# Production stage (minimal, API-focused)
from base as production

# Create non-root user for security
RUN useradd -m -u 1000 huxley && \
    chown -R huxley:huxley /app

USER huxley

# Expose API port
EXPOSE 8000

# Start with the onboarding WebUI
CMD ["python", "-m", "huxley.docker.onboarding"]

# Onboarding stage (interactive setup)
from base as onboarding

RUN pip install flask flask-cors python-dotenv

# Copy entrypoint script
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Copy WebUI assets (static files and templates)
COPY src/huxley/docker/static /app/static
COPY src/huxley/docker/templates /app/templates

# Create non-root user
RUN useradd -m -u 1000 huxley && \
    chown -R huxley:huxley /app

USER huxley

# Expose WebUI and API ports
EXPOSE 3000 8000

# Start the onboarding WebUI
CMD ["python", "-m", "huxley.docker.onboarding"]

# Start with entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]
