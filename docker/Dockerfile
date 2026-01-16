FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install -e .

CMD ["ctok", "--help"]
