.PHONY: install test lint typecheck demo user-value clean

PYTHON ?= python
VENV   ?= .venv

install:  ## Create venv and install package + dev extras in editable mode.
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -e ".[dev]"

test:  ## Run the full pytest suite.
	$(VENV)/bin/python -m pytest -q

lint:  ## Run ruff lint over src and tests.
	$(VENV)/bin/python -m ruff check src tests

typecheck:  ## Run mypy in strict mode.
	$(VENV)/bin/python -m mypy src/reify

demo:  ## Run the no-LLM wedge demo (fresh-clone smoke test).
	./scripts/demo.sh

user-value:  ## Run the user-value release gate.
	$(VENV)/bin/reify eval user-value

clean:  ## Remove build/test caches (does not delete the venv).
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build *.egg-info
