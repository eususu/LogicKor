.PHONY: init format check requirements

init:
	python -m pip install -q -U poetry ruff isort
	python -m poetry install

format:
	isort --profile black -l 119 .
	ruff format .

check:
	ruff check

requirements:
	poetry export -f requirements.txt --output requirements.txt --without-hashes
	poetry export -f requirements.txt --output requirements-dev.txt --without-hashes --with dev


test:
	python3 -m eval google/gemma-2-2b-it -lm ../vrl/dpo_result -f

score:
	python3 cli.py score
