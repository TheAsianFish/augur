# Contributing

Read `PLAN.md` first: it is the source of truth for scope, milestones and the
predictions each experiment is judged against.

## Setup

```sh
uv sync                 # Python 3.12, from uv.lock
cp .env.example .env    # optional; nothing is required for tests
set -a; . ./.env; set +a  # Augur reads the environment only; it never loads .env itself
ollama pull qwen2.5:7b  # optional; local model for dry runs and backtests
```

Python 3.12 is pinned (`.python-version`). The optional `baseline` extra
installs Metaculus's template bot and does not support 3.14.

## Commands

| Command | What it does |
|---------|--------------|
| `make check` | The quality gate: `ruff format --check`, `ruff check`, `mypy --strict`, unit tests. Offline, under two minutes. This is all CI runs on a pull request. |
| `make test` | Unit tests only. |
| `make test-integration` | Tests that need a local Ollama or live credentials; each skips with its reason when the requirement is missing. The OpenRouter ones call `:free` models only. |
| `make eval` | (M3) Replays the committed backtest ledgers and recordings and re-scores them. Offline, free. |
| `make backtest ARGS=...` | (M4) Runs a new backtest (local model by default). Manual; never in per-PR CI. |
| `make results` | (M3) Renders RESULTS.md from the ledger and result files. A test fails if the committed file differs. |
| `augur forecast --tournament minibench` | Dry run: the live listing with `METACULUS_TOKEN`, the committed samples in `data/samples/` without it. Add `--submit` to post. Probes the model chains first (`--no-probe` to skip). `--reforecast` re-forecasts questions already forecast, in the bot-testing-area only. |
| `augur models list` | Each configured route with its listed price, listing date and whether it is in the frontier tier. No model calls. |
| `augur models probe [--route E:M ...] [--burst N]` | One short call per route, recorded in the ledger; `--burst N` then measures each free route's real rate limit (stops at the first refusal; free routes only). Prints the OpenRouter key's credit and daily free allowance. |
| `augur models snapshot` | Refreshes the committed copy of OpenRouter's model list (`augur/llm/openrouter_models.json`), used when the live list cannot be fetched. |
| `augur ledger check PATH...` | Scans ledgers for leakage and for forecasts submitted below the floor; exits 1 on any violation. |

## Configuration

Everything is an environment variable; nothing secret is ever written to a
file, a log line, the ledger or a commit.

| Variable | Purpose |
|----------|---------|
| `METACULUS_TOKEN` | Metaculus API. |
| `ASKNEWS_CLIENT_ID`, `ASKNEWS_CLIENT_SECRET` (or `ASKNEWS_SECRET`) | News search. |
| `AUGUR_FORECAST_MODELS`, `AUGUR_RESEARCH_MODELS` | Ordered model chains, `endpoint:model,...` (split at the first colon, so `openrouter:google/gemma-4-31b-it:free` and `ollama:qwen2.5:7b` both parse). The forecast chain answers; the research chain plans queries and writes the brief, and defaults to the forecast chain. |
| `OPENROUTER_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `GITHUB_TOKEN` | Keys for the `openrouter`, `groq`, `gemini` and `github` presets. Only the variables an endpoint names are read. |
| `AUGUR_ENDPOINT_<NAME>_URL`, `_KEY_ENV`, `_BILLING`, `_RPM`, `_API` | Any other OpenAI-compatible host: base URL, the name of the variable holding its key, `free`, `openrouter` or `unpriced` (the default: its models are not called until priced), a requests-per-minute ceiling, and `openai` or `ollama`. |
| `AUGUR_PROVIDER`, `AUGUR_MODEL` | Without the chains: one endpoint and model for both roles (default `ollama`, `qwen2.5:7b`), or `AUGUR_PROVIDER=replay`. |
| `OLLAMA_HOST` | Where the local model lives. |
| `AUGUR_PROBE` | `0` skips the opening probe. |
| `AUGUR_RECORD_PATH`, `AUGUR_REPLAY_PATH` | Record every model call to a JSONL file; replay one (with `AUGUR_PROVIDER=replay`). |
| `AUGUR_FORECASTERS` | Forecaster draws per question (default 5), combined by the median. |
| `AUGUR_LEDGER` | Path of the SQLite ledger (default `./augur.db`). |
| `AUGUR_MAX_QUESTIONS_PER_RUN`, `AUGUR_MAX_COST_PER_QUESTION`, `AUGUR_MAX_COST_PER_RUN` | Caps. |
| `AUGUR_LOG_LEVEL` | Structured JSON logs to stderr. |

## Conventions

- **Types.** `mypy --strict` with no ignores that lack a reason. Pydantic
  models at every boundary (API payloads, model outputs, ledger rows).
- **Time.** Every datetime is timezone-aware UTC. Code never calls
  `datetime.now()` directly: it takes a `Clock`, which backtests pin.
- **Sources.** A source is written to the ledger before a model sees it, with
  the evidence for its date. Research code takes an `as_of` instant and must
  not return anything it cannot date at or before it.
- **Ledger.** Append-only. No `UPDATE`, no `DELETE`; a correction is a new
  row. Schema changes are numbered migrations.
- **Async.** All network I/O is `async`; every provider call goes through its
  rate limiter; no unbounded `gather`. The ledger is the exception: local
  SQLite writes take microseconds, so it is synchronous.
- **Tests.** Unit tests are offline, deterministic and fast: HTTP is mocked,
  models are the `replay` provider. Mocks never produce a reported number.
- **Logging.** `structlog`, JSON, with the run id and question id bound. A
  test asserts that tokens and secrets are redacted.
- **Errors.** Typed exceptions with a message that says what to do next; one
  failing forecaster or searcher degrades the question, it does not abort the
  run.

## Rules that fail the build

1. **Leakage.** `tests/test_leakage.py` scans every committed backtest ledger.
   A source dated after its question's cutoff, an undated source in a
   backtest, or a model released after the question opened fails the suite.
2. **Results drift.** RESULTS.md must equal a fresh render.
3. **Pre-stated predictions.** A hypothesis and its decision rule are
   committed before the run that tests it. Do not edit a prediction after
   seeing the result; record the result as held or failed.

## Numbers

Every number in the README or RESULTS.md comes from a command in this
repository, with the dataset, model, date, cost and raw output recorded. If
something cannot be measured in the current environment, it goes under
"To measure" in RESULTS.md with the exact command.

## Commits and pull requests

One logical step per commit, each passing `make check`, with an imperative
subject ("Add release-date gate to backtest loader") and a body only when the
reason is not obvious. One milestone per pull request; tick its box in
PLAN.md only when its definition of done is met.

## Tournament conduct

One bot account. Forecast and comment only through the documented API, within
its rate limits. Comments must reflect the reasoning that produced the
forecast. Never run `--submit` from a branch that has not passed `make check`. Outside the
bot-testing-area, `--submit` refuses unless the first forecasting model is a
hosted model in the frontier tier (`augur/llm/registry.py`), and a question
whose forecasts would come from below the tier is skipped; do not widen the
tier to get a run through. Never add a paid balance to any key; `--burst`
runs only on free models.
