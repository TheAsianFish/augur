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
| `make eval` | Scans every committed ledger for leakage, re-scores the ForecastBench backtest, re-runs the power simulation, M14's synthetic re-read table, the calibration snapshot, the Q6 snapshot, the ForecastBench same-question snapshot and M12's C4 snapshot, and checks RESULTS.md against a fresh render. Offline, free. |
| `make install-check` | Builds the sdist and the wheel, installs each into a fresh Python 3.12 venv outside the repository and runs the installed `augur` from an empty directory with a scrubbed environment and no network: `--help`, `models list` from the bundled snapshot, a dry run over the bundled sample questions, the leakage scan, scoring and a ledger round trip on a committed backtest ledger, `twine check`; with a local Ollama serving `qwen2.5:7b`, also a full dry run (`AUGUR_INSTALL_CHECK_MODEL=0` skips it). Prints PASS or FAIL. A few minutes; needs PyPI to install dependencies. |
| `make docker` | Builds the image `augur-forecast:VERSION` (`Dockerfile`: `python:3.12-slim`, installed from `uv.lock`, non-root, ledger on the `/data` volume, `ENTRYPOINT augur`, default a MiniBench dry run). Secrets only at `docker run -e`; RELEASE.md has the commands. |
| Release workflow (`.github/workflows/release.yml`) | On a `v*` tag: `make check`, the tag against the package version, `make install-check`, `uv build`, `twine check`, then publishes to PyPI by trusted publishing from the `pypi` environment. `gh workflow run release.yml -f dry_run=true` runs the same checks and never publishes. RELEASE.md, "Releasing 0.1.0". |
| `make backtest ARGS=...` | The ForecastBench backtest (no retrieval, the configured model, local by default), then its resolution. Resumable; hours on a CPU. Manual; never in per-PR CI. |
| `make backtest-q2` | Q2's arms and the research loop on C2's fixed 64-question sample: the shared query plan, sources as of open, as of now, the loop, then the leakage judge on both arms and `augur report`. Free hosted model; resumable; about an hour an arm. Manual. |
| `make results` | Re-renders RESULTS.md's generated blocks (between `<!-- augur:begin NAME -->` markers) from the committed ledgers and snapshots. A test fails if the committed file differs. |
| `augur forecast --tournament minibench` | Dry run: the live listing with `METACULUS_TOKEN`, the committed samples in `data/samples/` without it. Add `--submit` to post. Probes the model chains first (`--no-probe` to skip). `--reforecast` re-forecasts questions already forecast, in the bot-testing-area only. The markets searcher and the exact-price blend are on by default since M5 (`--no-markets`, `--no-blend`); since M8 the live calibrator (`augur/live_calibrator.json`) maps binary forecasts from the model it was fitted on, before the blend (`--no-calibrate`), except on the sources it is withheld from (since M9, Metaculus, so live forecasts go out raw). |
| `augur forecast ... --shadow-ledger S` | A shadow tick (M11): skips every question ledger S already holds an Augur aggregate for, before any model call (the probe included), ignores what the account posted, and refuses `--submit` and `--reforecast`. S is the ledger rebuilt from the `shadow/` tree of the `ledger` branch; the run writes to its own fresh ledger. Since M12 a shadow tick also writes the statistical arm (`stat`) and the registered blend (`blend`) as reference rows of each question, before delivery (`--stat-arm`/`--no-stat-arm`; refused beside `--submit`). |
| `augur baseline --ledger L [--shadow-ledger S] [--max-minutes M]` | The template on L's questions, plus S's still-open questions with no template row (a tick a timeout cut off), starting none after M minutes. |
| `scripts/shadow_fetch.sh DIR`, `scripts/push_shadow.sh DB TXT NAME` | Copy the `ledger` branch's `shadow/` tree into DIR (empty if none; exits 1 if the remote is unreachable); push a tick's export as `shadow/<day>/NAME.jsonl`, the second push of a tick replacing its first. `AUGUR_LEDGER_REMOTE` names another remote (a bare repository under `act`). |
| `augur models list` | Each configured route with its listed price, listing date and whether it is in the frontier tier. No model calls. |
| `augur models probe [--route E:M ...] [--burst N]` | One short call per route, recorded in the ledger; `--burst N` then measures each free route's real rate limit (stops at the first refusal; free routes only). Prints the OpenRouter key's credit and daily free allowance. |
| `augur models snapshot` | Refreshes the committed copy of OpenRouter's model list (`augur/llm/openrouter_models.json`), used when the live list cannot be fetched. |
| `augur ledger check PATH...` | Scans ledgers for leakage and for forecasts submitted below the floor; exits 1 on any violation. |
| `augur resolve [--ledger L]` | Appends resolutions for the ledger's live and dry-run questions, plus the community aggregate and Metaculus's own scores where the API shows them (a bot token sees none on scored tournaments: `scripts/resolution_visibility.py`). `--forecastbench SET` resolves a backtest; `--manual POST VALUE --evidence URL` records one by hand; `--manual-file F --standings S` records a round's from a CSV (`post_id,value,evidence`, an evidence URL on every line): every line is checked first, and nothing is written if a line is bad, names a post outside the snapshot `S`, or the file would leave any of the round's questions the ledger holds an Augur or template forecast for without a resolution. |
| `augur score LEDGER...` | Brier, log and baseline scores, and paired head-to-head and Brier differences against every reference (shadow template, community, market, always 50%, base rate) with clustered bootstrap intervals and the detectable effect. `--cluster week` for live questions; an `unrestricted` run is scored only with `--mode unrestricted`. |
| `augur baseline [--ledger L] [--post ID ...]` | Runs Metaculus's real template bot unpublished on questions Augur forecast, with Augur's first forecast route, and records it as a `template` reference. Needs `uv sync --extra baseline`. |
| `augur floor --round SLUG --ledger L [--standings S] [--candidate augur\|blend] [--json OUT]` | PLAN.md's floor rule for the free model on one MiniBench round (M11): Augur's shadow aggregate (or, with `--candidate blend`, M12's registered blend, which also prints P39: the blend against Augur alone) against the shadow template on the posts in the round's snapshot (default `data/standings/SLUG.json`), the counts (forecast by each side, scored, annulled, uncovered, unresolved), mean head-to-head with its 90% bootstrap interval (10,000 resamples, seed 20261119), its SD, S for `metac-nemotron-3-ultra+asknews`, the implied peer score, and the verdict: admitted, not admitted or not yet decidable, with its reason. Offline; every verdict exits 0. |
| `augur floor --pooled --round R1 --round R2 ... --ledger L [--standings S ...] [--json OUT]` | M14's pooled Q1 analysis (PLAN.md, `m14-registration`): each round's coverage counted as the floor rule counts it, then the mean head-to-head of Augur's aggregate against the shadow template over every pooled question with its 90% bootstrap interval clustered by opening day (10,000 resamples, seed 20261119), n, the cluster count (the question-level interval beside it below eight), the measured spread, the n a +2 or +5 lead needs, and the blend as the secondary quantity. Never prints a floor verdict; only a reading on 2027-01-11 is the registered result (earlier: a progress reading; later: a late reading), and a round outside the registration gets a note. Offline. |
| `augur rereads [--check]` | M14, synthetic: how often the one-round floor rule admits a candidate with no lead when re-read after each of one to four rounds (cumulatively, and each round alone), seeded, written to `data/power/m14-rereads.json`; `--check` re-runs and compares, as `make eval` does. A few seconds. |
| `python3 scripts/chain_spacing.py RUNS.json --since T [--target 30]` | The gaps between the Tournament workflow's consecutive scheduled or dispatched starts on `main`, from a saved runs list (`curl -s "https://api.github.com/repos/OWNER/REPO/actions/workflows/tournament.yml/runs?per_page=100"`; public, no token): median, largest, and how many fall within 10 minutes of the chain's target. Standard library only. |
| `augur standings --round SLUG [--out F] [--ledger L ...]` | A MiniBench round's leaderboard: rank 1, Augur, the best template bot, the best house bot, the community aggregates, a listed open-source bot; each question Augur did not forecast explained from the ledgers. `--from F` renders a saved snapshot offline. |
| `augur report [--check]` | What `make results` runs. |
| `augur backtest forecastbench --questions Q --resolutions R` | What `make backtest` runs. |
| `augur backtest dataset-closes [--ledger L]` | M15: each `yfinance` question of the dataset corpus (`data/forecastbench/dataset`) gets the ticker's daily closes from Yahoo's keyless chart endpoint for the 800 days before its freeze instant, kept only if dated at or before it, written as one dated source and admitted through the leakage gate (default ledger `data/ledgers/m15-dataset-closes.sqlite`). No model call; resumable; about 6 minutes. |
| `python -I scripts/extract_dataset_questions.py RAW OUT --fetched DAY` | M15's extract: the single-series dataset questions of every 2026 set from 2026-01-04 to 2026-09-27 with their resolved outcomes and what was left out, from the published question and resolution sets saved as `RAW/<set>-llm.json` and `RAW/<set>-res.json`. Standard library only. |
| `augur backtest prices --questions Q --resolutions R --ledger L` | Records each ForecastBench market question's exact price as of its freeze instant (Manifold's last bet, Polymarket's last history point, the Metaculus aggregate where the token sees it) as a dated market source. No model call; free. |
| `augur backtest forecastbench ... --retrieval open\|now [--loop N] [--queries F] [--sample N]` | Research each question as of its open instant (a backtest) or as of this moment (Q2's leaky arm, an `unrestricted` run never scored as a backtest), one pass or the bounded research loop; `--plan-out F` only plans the queries every arm shares. Searchers: Wikipedia revisions, Wayback captures and (M13) `gdelt` news. |
| `augur judge --ledger L --tournament T --questions Q --resolutions R` | Rates every source an arm's forecasters were shown on El Lahib et al.'s 0 to 4 leakage rubric, with the resolution, against the question's open instant. Resumable. |
| `make backtest-q2-news` | M13's arm G on C2's 64-question sample: O's recorded sources plus at most three dated news sources a question (`gdelt`: GDELT discovery bounded at the cutoff, text only from a Wayback capture at or before it), a new brief and three draws on the free model, then the leakage judge and `augur report`. Builds the local GKG index first if it is missing. Resumable; GDELT answers are cached in `data/backtest/gdelt-cache/`. Manual. |
| `augur backtest gkg-index --start S --end E` | Downloads GDELT's 15-minute GKG 2.1 files in (S, E], each checked against the master list's MD5, into a local index of (batch time, URL, page title) (`data/backtest/gdelt-gkg-index.tsv.gz`, not committed; its manifest is). The `gdelt` searcher answers from it once GDELT's DOC API has refused a request. About 15 GB downloaded for 30 days. |
| `augur backtest forecastbench ... --retrieval open --searchers gdelt --add-to L` | Arm G: start from the sources arm O recorded in ledger L, add what the searchers find (at most three); a question that gains nothing keeps O's forecast and makes no model call. |
| `make backtest-q3` | Q3's arms on the first 40 of C2's sample: six more draws and a second model on M6's recorded one-pass briefs, and the two-pass researcher; then `augur power` and `make results`. Free hosted models; resumable; about two hours. Manual. |
| `augur backtest forecastbench ... --research-from L [--research-arm A] [--first-member K] [--label X]` | Forecast from the briefs and sources arm A recorded in ledger L (each rebuilt request must hash to the recorded one), numbering the new draws from K (seed = member): head count and model mix without re-running research. |
| `augur backtest forecastbench ... --retrieval open --passes 2` | Two research passes: the second writes the sub-questions that would most change the forecast, searches them as of the cutoff and rewrites the brief. |
| `make backtest-q4` | Q4's corpus C4: eight ForecastBench sets (2026-07-05 to 2026-10-11), no retrieval, one draw on the free model (792 free calls), their resolutions, when each market's outcome became public by its own API (`scripts/resolution_times.py`; Metaculus needs `METACULUS_TOKEN`), then `make calibration results`. Resumable; about 45 minutes with the sets run in parallel. Manual. |
| `augur calibrate [--check]` | Q4: raw, fixed extremizing, base rate, Platt and isotonic on C4, each question calibrated only on outcomes known before it opened, dated by the later of ForecastBench's resolution date and the market's own resolution time (walk-forward and learning curve). Also the source-aware maps (per-source and partially pooled Platt) by source. Writes `data/calibration/m8-calibration.json`, its SVG reliability diagram and `augur/live_calibrator.json` (the calibrator M7's rule picks, fitted on all of C4, with `withheld_from` naming the sources it is not applied to: Metaculus until a map passes there); `--check` re-runs and compares, as `make eval` does. |
| `augur statarm [--check]` | M12: the statistical arm (readable price, else the source's walk-forward base rate) and both blends against Augur's raw forecast on C4's walk-forward instances, by source, with no model call. Writes `data/statarm/m12-c4.json` and the live base rates `augur/stat_base_rates.json`; `--check` re-runs and compares and re-derives the coverage file's reached titles, as `make eval` does. `--coverage ROUND` runs the series rule on a round's question list (needs `METACULUS_TOKEN`; closes from Yahoo Finance's keyless chart endpoint as of each question's open). |
| `augur q6 [--check]` | Q6 (M9): AUROC of each disagreement measure (draws, models, research passes) for large errors, and the shrink and skip policies against constant shrinkage and the source-aware calibrator, fitted leave-one-out on draws already in the ledgers; the pre-registered live half's counts, recomputed from member rows. Writes `data/q6/m9-q6.json`; `--check` re-runs and compares, as `make eval` does. |
| `augur fbcompare [--check]` | M15's dataset row as well (`data/forecastbench/m15-dataset.json`: the no-model arms against the baselines and named entries on the dataset instances, from the extract, the closes ledger and `data/forecastbench/forecast-sets-dataset`, cut by `extract_forecast_sets.py --dataset`). M10's same-question row: every ForecastBench entrant's published forecasts (the committed extract in `data/forecastbench/forecast-sets/`, cut from the processed forecast sets by `python -I scripts/extract_forecast_sets.py`) paired with Augur's arms C1, C2, O and C4 on the instances each scored, imputed forecasts excluded: Brier differences with intervals over markets, Augur's rank per entrant group, and P30 to P33. Writes `data/forecastbench/m10-same-question.json`; `--check` re-runs and compares, as `make eval` does. No model call; about a minute. |
| `augur power [--check]` | The planted-effect power simulation over the Q3 arms' per-question differences, written to `data/power/m7-power.json` (seeded; `--check` re-runs and compares, as `make eval` does). |
| `augur backtest forecastbench ... --prices L` | The replay evaluation's arm (d): the recorded price is the brief's only source; questions without one are skipped. |
| `augur ledger import PATH... --out DB` | Rebuilds one SQLite ledger from JSONL exports (files or a directory, such as a checkout of the `ledger` branch); the leakage triggers re-check every row. |
| `augur ledger export DB OUT` | Writes a ledger as JSONL, one row per line, byte-stable; the scheduled workflow pushes one per run to the `ledger` branch. |
| `augur forecast ... --coverage PATH` | Also writes the run's coverage report as JSON: each question forecast in time, missed, skipped or failed, and why (also in the ledger's `question_outcomes`). |
| `.github/workflows/standings.yml` | Daily: `augur standings` for the current MiniBench round, the two before it (`scripts/standings_rounds.py`) and, with `AUGUR_SCHEDULE=all`, the FutureEval project; each snapshot goes to the `ledger` branch (`scripts/push_standings.sh`). |
| `python3 scripts/poll.py` | The scheduled workflow's first job: standard library only, lists open questions this account has not forecast and sets `has_work`. Needs `METACULUS_TOKEN` and `AUGUR_SCHEDULE`. |

## Configuration

Everything is an environment variable; nothing secret is ever written to a
file, a log line, the ledger or a commit.

| Variable | Purpose |
|----------|---------|
| `METACULUS_TOKEN` | Metaculus API. |
| `ASKNEWS_CLIENT_ID`, `ASKNEWS_CLIENT_SECRET` (or `ASKNEWS_SECRET`) | News search. |
| `AUGUR_FORECAST_MODELS`, `AUGUR_RESEARCH_MODELS` | Ordered model chains, `endpoint:model,...` (split at the first colon, so `openrouter:google/gemma-4-31b-it:free` and `ollama:qwen2.5:7b` both parse). The forecast chain answers; the research chain plans queries and writes the brief, and defaults to the forecast chain. A route may end in `@minimal`, `@low`, `@medium`, `@high` or `@xhigh` (`openrouter:openai/gpt-5.4@high`) to ask for that reasoning effort; it is sent to the host and recorded with every call. |
| `OPENROUTER_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `GITHUB_TOKEN` | Keys for the `openrouter`, `groq`, `gemini` and `github` presets. Only the variables an endpoint names are read. |
| `AUGUR_ENDPOINT_<NAME>_URL`, `_KEY_ENV`, `_BILLING`, `_RPM`, `_API` | Any other OpenAI-compatible host: base URL, the name of the variable holding its key, `free`, `openrouter` or `unpriced` (the default: its models are not called until priced), a requests-per-minute ceiling, and `openai` or `ollama`. |
| `AUGUR_PROVIDER`, `AUGUR_MODEL` | Without the chains: one endpoint and model for both roles (default `ollama`, `qwen2.5:7b`), or `AUGUR_PROVIDER=replay`. |
| `OLLAMA_HOST` | Where the local model lives. |
| `AUGUR_PROBE` | `0` skips the opening probe. |
| `AUGUR_RECORD_PATH`, `AUGUR_REPLAY_PATH` | Record every model call to a JSONL file; replay one (with `AUGUR_PROVIDER=replay`). |
| `AUGUR_FORECASTERS` | Forecaster draws per question (default 5), combined by the median (binary), the per-option median normalised (multiple choice) or the mean of their CDFs (numeric, discrete). |
| `AUGUR_RESEARCH_PASSES` | `1` (default) or `2`: a second research agent reads the first brief, searches the sub-questions that would most change the forecast, and the brief is rewritten (M7). |
| `AUGUR_FORECASTER_VARIANTS` | Comma list of prompt stances, cycled over the members: `base` (default), `outside`, `inside`, `sceptic`. |
| `AUGUR_MEMBER_MODELS` | Optional per-member model chains, `;`-separated, each in the chain syntax above; member m takes chain m mod k (a mixed-model fleet). Every route still passes the floor and the release gate. |
| `AUGUR_QUESTION_CONCURRENCY` | Questions forecast at once (default 1, at most 8); every call still goes through its endpoint's token bucket and the per-question and per-run caps. |
| `AUGUR_LEDGER` | Path of the SQLite ledger (default `./augur.db`). |
| `AUGUR_MAX_QUESTIONS_PER_RUN`, `AUGUR_MAX_COST_PER_QUESTION`, `AUGUR_MAX_COST_PER_RUN` | Caps. The run prints its worst-case cost estimate before the first question. |
| `AUGUR_CLOSE_MARGIN_MINUTES`, `AUGUR_MIN_FORECAST_MINUTES`, `AUGUR_RUN_DEADLINE_MINUTES` | Deadlines (defaults 5, 4, 30): work on a question stops this long before it closes; a question with less time than the minimum before that is missed, not started; the run starts no question after its own deadline. |
| `AUGUR_SHADOW`, `AUGUR_SHADOW_MODELS`, `AUGUR_SHADOW_MAX_QUESTIONS`, `AUGUR_SHADOW_DAILY_CALLS` (repository variables) | M11's shadow schedule: `true` runs the shadow job on scheduled and chained ticks instead of the forecast job (it never submits); the route both contestants use (default `openrouter:nvidia/nemotron-3-ultra-550b-a55b:free`); questions a tick takes (default 4); the free calls a day the poll budgets before it polls FutureEval (default 1000). Needs `AUGUR_SCHEDULE` set; never `AUGUR_SUBMIT`. |
| `AUGUR_SCHEDULE`, `AUGUR_SUBMIT` (repository variables) | The scheduled workflow's switches: `off` (or unset), `minibench` or `all`; and `true` to post from scheduled runs on `main` (otherwise scheduled runs only poll; dry runs are manual dispatches). |
| `AUGUR_FUTUREEVAL` (repository variable) | The FutureEval project the `all` schedule polls and the standings job reads: an id or slug, default `33121` (Fall 2026); set the Spring 2027 id when it opens. |
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
