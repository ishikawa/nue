.PHONY: check format lint type-check

check: lint type-check test

format:
	uv run ruff format ./nue

lint:
	uv run ruff check ./nue

type-check:
	uv run pyright ./nue

test:
	uv run python -m pytest ./nue

# Training
train-local:
	uv run python -m nue.cli train \
		--batch-size 4 \
		--max-warmup-steps 20 \
		--log-interval 10 \
		--save-interval 10 \
		--override-data-size 3% \
		--log-validation-max-tokens 2048 \
		--measure-time

train-local-mlx:
	uv run python -m nue.cli train \
		--backend mlx \
		--batch-size 8 \
		--max-warmup-steps 20 \
		--log-interval 10 \
		--save-interval 10 \
		--override-data-size 3% \
		--log-validation-max-tokens 4096 \
		--measure-time

train:
	uv run python -m nue.cli train --batch-size 256
