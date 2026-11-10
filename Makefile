# Common commands. `make help` lists them.

VENV      := .venv
PY        := $(VENV)/bin/python
# The environment the arm's tooling runs in. It is used as it is: nothing is installed into it from here.
RIG_PY    ?= $(HOME)/miniconda3/envs/lerobot/bin/python

.PHONY: help venv test test-rig-env lint format check simulate figures clean

help:            ## list the commands
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

venv:            ## build the development environment from the local package cache (no network)
	uv venv --python 3.12 $(VENV)
	uv pip install --offline --python $(PY) numpy scipy matplotlib pytest pyyaml

test:            ## run the hardware-free tests
	$(PY) -m pytest

test-rig-env:    ## run the same tests with the rig's interpreter (no pytest there, so unittest)
	PYTHONPATH=src $(RIG_PY) -m unittest discover -s tests -q

lint:            ## check style (needs ruff: `uv pip install --python .venv/bin/python ruff`)
	$(VENV)/bin/ruff check . && $(VENV)/bin/ruff format --check .

format:          ## apply the formatter
	$(VENV)/bin/ruff check --fix . && $(VENV)/bin/ruff format .

figures:         ## draw the paper's figures whose data are there (a missing input is skipped, not an error)
	$(PY) scripts/make_figures.py

check: test test-rig-env  ## everything that must pass before a commit

clean:           ## remove caches
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
