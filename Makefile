.PHONY: install fetch inspect preprocess test lint

install:
	python -m pip install -e ".[dev]"

fetch:
	openmobility fetch --max-records 5000

inspect:
	openmobility inspect

preprocess:
	openmobility preprocess

test:
	pytest

lint:
	ruff check .
