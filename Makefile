.PHONY: check fmt lint types test test-integration forecast

UV ?= uv

check: ## The quality gate: format, lint, types, unit tests (offline)
	$(UV) run ruff format --check .
	$(UV) run ruff check .
	$(UV) run mypy
	$(UV) run pytest -q

fmt:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

lint:
	$(UV) run ruff check .

types:
	$(UV) run mypy

test:
	$(UV) run pytest -q

test-integration: ## Tests that need a local Ollama or live credentials
	$(UV) run pytest -q -m integration

forecast: ## Dry run on MiniBench (live listing with a token, committed samples without)
	$(UV) run augur forecast --tournament minibench
