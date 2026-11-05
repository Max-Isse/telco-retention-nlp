# syntax=docker/dockerfile:1
# Serving image: core dependencies only (no torch), the package installed non-editable, non-root user (D-025).
# Models are not baked in: mount the artifacts directory read-only, e.g.
#   docker run -p 8000:8000 -v "$PWD/artifacts:/app/artifacts:ro" telco-retention
FROM python:3.11.11-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
WORKDIR /src
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-editable

FROM python:3.11.11-slim-bookworm
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home app
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY configs ./configs
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    TELCO_ROOT=/app \
    TELCO_ARTIFACTS=/app/artifacts
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["uvicorn", "--factory", "telco_retention.api:create_app", "--host", "0.0.0.0", "--port", "8000"]
