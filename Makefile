.PHONY: lint format test typecheck bench

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

test:
	uv run pytest

typecheck:
	uv run mypy src

bench:
	uv run python -m ctok_extras.cli.main bench --help
