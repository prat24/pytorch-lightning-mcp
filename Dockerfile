# syntax=docker/dockerfile:1.7

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_NO_CACHE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -Ls https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.cargo/bin:${PATH}"

COPY pyproject.toml uv.lock* ./

RUN uv sync --all-extras --no-dev

COPY src ./src

CMD ["uv", "run", "python", "-m", "lightning_mcp.server"]
