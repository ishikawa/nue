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
	poetry run nue train --batch-size 8 --log-interval 10 --save-interval 10 --override-data-size 3%

train:
	poetry run nue train --batch-size 320 --log-interval 10 --save-interval 10 --override-data-size 3% --model-dir gs://nue-models