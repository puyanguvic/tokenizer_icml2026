.PHONY: lint format test typecheck bench

lint:
	ruff check src tests

format:
	ruff format src tests

test:
	pytest

typecheck:
	mypy src

bench:
	python -m ctok.cli.main bench --help
