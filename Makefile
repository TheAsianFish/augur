.PHONY: check fmt lint types test test-integration forecast results eval backtest backtest-q2

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

Q2_SET ?= data/forecastbench/2026-06-21
Q2_ARGS = --questions $(Q2_SET)-llm.subset.json --resolutions $(Q2_SET)_resolution_set.subset.json
Q2_TOURNAMENT = forecastbench:2026-06-21-llm.json
Q2_QUERIES = data/backtest/m6-c2-queries.json

backtest-q2: ## Q2 and the research loop on C2's fixed sample (free hosted model; about an hour each arm)
	$(UV) run augur backtest forecastbench $(Q2_ARGS) --sample 64 --plan-out $(Q2_QUERIES) \
		--ledger data/ledgers/m6-c2-plan.sqlite
	$(UV) run augur backtest forecastbench $(Q2_ARGS) --sample 64 --retrieval open \
		--queries $(Q2_QUERIES) --ledger data/ledgers/m6-c2-open.sqlite
	$(UV) run augur backtest forecastbench $(Q2_ARGS) --sample 64 --retrieval now \
		--queries $(Q2_QUERIES) --ledger data/ledgers/m6-c2-now.sqlite
	$(UV) run augur backtest forecastbench $(Q2_ARGS) --sample 40 --retrieval open --loop 3 \
		--queries $(Q2_QUERIES) --ledger data/ledgers/m6-c2-loop.sqlite
	$(UV) run augur judge $(Q2_ARGS) --ledger data/ledgers/m6-c2-open.sqlite \
		--tournament "$(Q2_TOURNAMENT)+onepass-asof-open"
	$(UV) run augur judge $(Q2_ARGS) --ledger data/ledgers/m6-c2-now.sqlite \
		--tournament "$(Q2_TOURNAMENT)+onepass-asof-now"
	$(UV) run augur report
