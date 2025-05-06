.PHONY: check format lint type-check

check: lint type-check

format:
	poetry run ruff format ./nue

lint:
	poetry run ruff check ./nue

type-check:
	poetry run pyright ./nue

# Training
train-local:
	poetry run nue train \
		--batch-size 8 \
		--lr 3e-4 \
		--max-warmup-steps 100 \
		--log-interval 10 \
		--save-interval 10 \
		--override-data-size 3% \
		--log-validation-max-tokens 2048

train:
	poetry run nue train --batch-size 256
