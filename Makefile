.PHONY: reproduce test lint sensitive check

reproduce:
	python scripts/run_pipeline.py

test:
	python -m pytest -q

lint:
	python -m ruff check .

sensitive:
	python scripts/check_sensitive.py

check: lint reproduce test sensitive
