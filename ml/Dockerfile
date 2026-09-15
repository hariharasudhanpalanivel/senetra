# syntax=docker/dockerfile:1.7

# ---- build: resolve the locked environment with uv -------------------------------------------
FROM ghcr.io/astral-sh/uv:0.8.17-python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_CONCURRENT_DOWNLOADS=4 \
    UV_HTTP_TIMEOUT=120 \
    UV_HTTP_RETRIES=5
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ---- runtime: slim image, non-root user -----------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 senetra
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY configs ./configs
ENV PATH="/app/.venv/bin:${PATH}" \
    SENETRA_ML_HOME=/app \
    SENETRA_DB_PATH=/data/senetra.db \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet \
    PYTHONUNBUFFERED=1
RUN mkdir -p /app/reports /mlflow && chown -R senetra:senetra /app/reports /mlflow
USER senetra
EXPOSE 5000 8080
ENTRYPOINT ["senetra-ml"]
CMD ["--help"]
