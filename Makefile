.PHONY: check fmt lint types test test-integration forecast results eval backtest backtest-q2 backtest-q3 power backtest-q4 calibration q6

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
	$(UV) run augur power --check
	$(UV) run augur calibrate --check
	$(UV) run augur q6 --check
	$(UV) run augur fbcompare --check
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

Q3_U = openrouter:nvidia/nemotron-3-ultra-550b-a55b:free
Q3_N = openrouter:nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
Q3_OPEN = data/ledgers/m6-c2-open.sqlite

backtest-q3: ## Q3's arms on the first 40 of C2's sample (free hosted models; about 2 hours)
	AUGUR_FORECAST_MODELS=$(Q3_U) AUGUR_RESEARCH_MODELS=$(Q3_U) AUGUR_FORECASTERS=6 \
		$(UV) run augur backtest forecastbench $(Q2_ARGS) --sample 40 --research-from $(Q3_OPEN) \
		--first-member 3 --label draws --ledger data/ledgers/m7-c2-draws.sqlite
	AUGUR_FORECAST_MODELS=$(Q3_N) AUGUR_RESEARCH_MODELS=$(Q3_N) AUGUR_FORECASTERS=3 \
		$(UV) run augur backtest forecastbench $(Q2_ARGS) --sample 40 --research-from $(Q3_OPEN) \
		--label model-nano --ledger data/ledgers/m7-c2-nano.sqlite
	AUGUR_FORECAST_MODELS=$(Q3_U) AUGUR_RESEARCH_MODELS=$(Q3_U) AUGUR_FORECASTERS=3 \
		$(UV) run augur backtest forecastbench $(Q2_ARGS) --sample 40 --retrieval open --passes 2 \
		--queries $(Q2_QUERIES) --ledger data/ledgers/m7-c2-twopass.sqlite
	$(MAKE) power results

power: ## The planted-effect power simulation from the Q3 arms (data/power/m7-power.json)
	$(UV) run augur power

Q4_SETS = 2026-07-05 2026-07-19 2026-08-02 2026-08-16 2026-08-30 2026-09-13 2026-09-27 2026-10-11
Q4_LEDGER = data/ledgers/m8-c4.sqlite

backtest-q4: ## Q4's corpus C4: eight ForecastBench sets, no retrieval, one draw (792 free calls)
	for s in $(Q4_SETS); do \
		AUGUR_FORECAST_MODELS=$(Q3_U) AUGUR_RESEARCH_MODELS=$(Q3_U) AUGUR_FORECASTERS=1 \
		$(UV) run augur backtest forecastbench --questions data/forecastbench/$$s-llm.subset.json \
		--resolutions data/forecastbench/$${s}_resolution_set.subset.json --ledger $(Q4_LEDGER) \
		|| exit 1; \
	done
	for s in $(Q4_SETS); do \
		$(UV) run augur resolve --ledger $(Q4_LEDGER) \
		--forecastbench data/forecastbench/$${s}_resolution_set.subset.json || exit 1; \
	done
	$(UV) run python scripts/resolution_times.py data/calibration/m8-resolution-times.json
	$(MAKE) calibration results

calibration: ## Q4: the calibrators, walk-forward and learning curve (data/calibration/)
	$(UV) run augur calibrate

q6: ## Q6: disagreement against error, shrink and skip policies, the live half's counts (data/q6/)
	$(UV) run augur q6
	$(UV) run augur report
