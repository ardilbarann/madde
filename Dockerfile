FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY migrations ./migrations
RUN uv sync --frozen --no-dev
EXPOSE 8000
CMD ["sh", "-c", "uv run --no-sync python -m madde migrate && uv run --no-sync uvicorn madde.main:app --host 0.0.0.0 --port 8000"]
