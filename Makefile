.PHONY: check fmt lint types test test-integration forecast results eval backtest

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

FB_SET ?= data/forecastbench/2026-05-24
FB_LEDGER ?= data/ledgers/m4-forecastbench-qwen.sqlite

results: ## Re-render RESULTS.md's generated blocks from the committed ledgers and snapshots
	$(UV) run augur report

eval: ## Offline and free: scan every committed ledger, re-score the backtest, check RESULTS.md
	$(UV) run augur ledger check data/ledgers/*.sqlite
	$(UV) run augur score $(FB_LEDGER) --mode backtest
	$(UV) run augur report --check

backtest: ## ForecastBench, no retrieval, local model; hours on a CPU (ARGS adds options)
	$(UV) run augur backtest forecastbench --questions $(FB_SET)-llm.subset.json \
		--resolutions $(FB_SET)_resolution_set.subset.json --ledger $(FB_LEDGER) $(ARGS)
	$(UV) run augur resolve --ledger $(FB_LEDGER) \
		--forecastbench $(FB_SET)_resolution_set.subset.json
