FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir uv==0.8.22

COPY pyproject.toml README.md ./
COPY src ./src

RUN uv sync --no-dev --no-editable

ENV PATH="/app/.venv/bin:${PATH}"

CMD ["ctok", "--help"]
