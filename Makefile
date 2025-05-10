.PHONY: check format lint type-check

check: lint type-check

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
		--lr 3e-4 \
		--max-warmup-steps 20 \
		--log-interval 10 \
		--save-interval 10 \
		--override-data-size 3% \
		--log-validation-max-tokens 2048

train:
	uv run python -m nue.cli train --batch-size 256
