.PHONY: check format lint type-check

check: lint type-check

format:
	poetry run ruff format ./nue

lint:
	poetry run ruff check ./nue

type-check:
	poetry run pyright ./nue

