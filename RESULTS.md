# Results

Every number here comes from a command in this repository, with the model,
date, cost and raw output recorded. Nothing scored yet: the first scored
number (ForecastBench, no retrieval) and the first live score arrive in M4.

## M2: model access, measured limits and the floor on who answered (2026-11-18)

**Setup.** OpenRouter through the generic OpenAI-compatible provider, with
the key in `OPENROUTER_API_KEY`; `:free` models only, so every call below
cost $0. The key reports a $1 credit limit, $0 used throughout, and 1,000
free-model requests a day (`augur models probe` prints this from
OpenRouter's `GET /key`). I did not call any billed model: whether that $1
is Metaculus's sponsorship or not is not visible from the key, so no frontier
model was probed. Prices and listing dates are from OpenRouter's model list
fetched at 11:05 UTC (478 models, 14 of them `:free`), committed as
`augur/llm/openrouter_models.json`.

### What the free tier really allows

`augur models probe --burst 30` sends one short request per route, then
sends requests back to back to each free route until the first refusal
(three sessions, ledger `data/ledgers/m2-probe.sqlite`, printed output in
`data/runs/m2-probe-*.txt`).

| Session (UTC) | Route | Probe | Burst: accepted / sent, seconds | First refusal |
|---------------|-------|-------|----------------------------------|---------------|
| 11:10 | `google/gemma-4-31b-it:free` | 429 in 265 ms, "temporarily rate-limited upstream" (Google AI Studio) | not run | |
| 11:10 | `nvidia/nemotron-3-ultra-550b-a55b:free` | answered, 1,139 ms | 23 / 24, 24.9 s | 429 `free-models-per-min`, `X-RateLimit-Limit: 20`, reset at the next minute boundary |
| 11:10 | `google/gemma-4-26b-a4b-it:free` | answered, 2,154 ms | 0 / 1, 0.4 s | 429 upstream, no headers |
| 11:11 | `nvidia/nemotron-3-ultra-550b-a55b:free` | answered, 613 ms | 20 / 21, 25.7 s | 429 `free-models-per-min`, reset 11:12:00 |
| 11:11 | `inclusionai/ling-3.2-flash:free` | answered, 1,705 ms | 2 / 3, 3.1 s (right after the burst above) | 429 `free-models-per-min`, **same reset, 11:12:00** |
| 11:11 | `thinkingmachines/inkling-2-small:free` | 403 "only available on agentic harnesses" | not run | |
| 11:18 | `nvidia/nemotron-3-ultra-550b-a55b:free` | answered, 1,816 ms | 18 / 19, 17.4 s | 429 `free-models-per-min`, reset 11:19:00 |
| 11:18 | `inclusionai/ling-3.2-flash:free` | answered, 1,447 ms | 4 / 5, 9.1 s (right after) | 429 `free-models-per-min`, **same reset, 11:19:00** |
| 11:18 | `thinkingmachines/inkling-2-small:free` | 403, now classed `unavailable` | not run | |

What this measured, against what the documentation and headers say:

- **The per-minute limit is per key, across all free models, in fixed
  clock-minute windows.** The second free model was refused with the same
  `free-models-per-min` error and the same reset time as the first, in both
  sessions that tried it. The header says 20; the key's accepted requests in
  the minute before its last refusal were 25, 24 and 24 (probe calls
  included), so the limit is enforced loosely above 20. Augur's client-side
  ceiling for the `openrouter` endpoint stays at 20 a minute.
- **A second, independent limit: upstream capacity.** Google's free Gemma
  models returned 429 "temporarily rate-limited upstream" with no rate-limit
  headers on 8 of the 10 attempts the ledgers record across the probes and
  the forecast runs below (a forecast-run attempt includes up to three short
  retries), including the first request of a session. That is
  shared capacity, not this key's quota, so it cannot be planned around;
  only a fallback handles it.
- **Not every 403 is the key.** Inkling's 403 is a gate on the model. The
  first session recorded it as `auth` ("check OPENROUTER_API_KEY"), which
  would have sent a maintainer to the wrong place; a 403 whose body does not
  mention the key or credentials is now `unavailable`
  (`test_compat_classifies_refusals`).
- **Tokens per minute: not measured.** OpenRouter sends no token-limit
  header, and the bursts use about ten tokens a request. Groq, Gemini and
  GitHub Models were not probed (no keys here); see To measure.

The first two sessions' `limit_tests.started_at` holds each burst's end
time rather than its start (fixed before the third session); the times in
the table come from the probe rows.

### A bot-testing-area forecast on a free hosted model

`augur forecast --tournament bot-testing-area --submit --reforecast` with
`AUGUR_FORECAST_MODELS=openrouter:nvidia/nemotron-3-ultra-550b-a55b:free,openrouter:inclusionai/ling-3.2-flash:free,openrouter:google/gemma-4-31b-it:free`
and `AUGUR_RESEARCH_MODELS=openrouter:google/gemma-4-26b-a4b-it:free,openrouter:inclusionai/ling-3.2-flash:free,openrouter:nvidia/nemotron-3-ultra-550b-a55b:free`
(question 43327, the only binary question in the practice area, forecast
in M1, hence `--reforecast`). Ledger `data/ledgers/m2-live-bot-testing-area.sqlite`,
recordings `data/recordings/m2-live-bot-testing-area.jsonl`, output
`data/runs/m2-live-bot-testing-area-{1,2}.txt`.

| Run | Probe | Research | Forecasters | Median | Posted | Wall time |
|-----|-------|----------|-------------|--------|--------|-----------|
| 1 (11:12) | nemotron, ling answered; both Gemmas 429 upstream | planner on ling returned **no text** (its 200 tokens went on reasoning), so the title became the query; brief on gemma-26b, 2 sources | 5 nemotron calls, **2 without a probability line** (stopped at 600 tokens), 1 with only a number | 1% of 3 | forecast 201, private comment 201 | 41 s |
| 2 (11:14), after the fix below | same pattern | planner and brief each **fell back** from gemma-26b (429 upstream) to ling: 2 `model_switches` rows; 4 sources | 5 of 5 parsed, 643 to 1,472 output tokens each (all 5 above the old 600 cap, reasoning included) | 1% of 5 | forecast 201, private comment 201 | 2 min 32 s |

The fix between the runs (commit "Give reasoning models room to answer"):
token budgets 200 to 600 (planner), 500 to 1,000 (brief), 600 to 1,500
(forecasters); an empty answer now moves that one call to the next model
without disabling the route; and a draw whose answer is only a number is
explained in the comment by the model's own reasoning trace, labelled as
such. For scale: M1's live run of the same question on `qwen2.5:7b` on
CPU took 11 min 40 s.

Not a result: one question is not a score, and these are below-tier models
that the floor keeps out of scored tournaments. The runs show the path
(probe, chains, fallback, ledger, private comment) works against the real
APIs.

### The floor, live and in the ledger

- `augur forecast --tournament minibench --submit` with the free chain
  above: refused before any model call, exit 2
  (`data/runs/m2-submit-floor-minibench.txt`).
- `test_a_fallback_below_the_tier_never_submits`: on the fake Metaculus
  server, a scored MiniBench run whose frontier model runs out of credit
  mid-run skips the question rather than letting the free model behind it
  forecast; nothing is posted and the refusal is a `model_switches` row. The
  same chain in the bot-testing-area falls back and posts.
- `augur ledger check` on all four committed ledgers: 0 violations, now
  including `below_floor` (a submitted forecast whose members came from a
  model outside the tier).

**The tier did not match OpenRouter's names.** Matched against the
committed model list, M1's rule (bare name, as written) found 4 of the 11
tier names: no Claude model at all, because OpenRouter writes
`anthropic/claude-opus-5.5` where the tier had `claude-opus-5-5`. With dots
and dashes read alike, 9 of 11 match (`gemini-3-pro-preview` and `grok-4`
are not listed). M1's floor would have refused every Claude model on the
sponsored key. The tier also predates the newest listings
(`openai/gpt-6.2-astra`, listed 2026-11-17, is below it); reviewing it is
the maintainer's decision, and it was not widened here.

## M1: one question end to end (2026-11-17)

**Setup.** `qwen2.5:7b` (Q4_K_M, released 2024-09-19) through Ollama 0.40.16
on a 4-vCPU CPU-only runner, 5 forecaster draws (temperature 0.7, seeds 0-4)
combined by the median, one researcher (3 planned queries, Wikipedia
revisions as of the cutoff; AskNews skipped: no credentials). Cost $0 (local
model). The sponsored proxy could not be used (see To measure).

| Run | Command | Question | Sources (admitted / rejected) | Members | Median | Sent | Wall time |
|-----|---------|----------|-------------------------------|---------|--------|------|-----------|
| Dry run, committed sample | `env -u METACULUS_TOKEN augur forecast --tournament minibench` | [43327](https://www.metaculus.com/questions/43327/) (EU exit before 2027) | 3 / 0 | 5, 5, 10, 5, 5% | 5% | nothing (printed) | 11 min 25 s |
| Live, bot-testing-area | `augur forecast --tournament bot-testing-area --submit` | same question, live listing; 5 non-binary questions skipped | 3 / 0 | 5, 5, 5, 7, 5% | 5% | forecast HTTP 201, private comment HTTP 201 (comment id 1336215) | 11 min 40 s |
| Live, second run | same command again | listed 6, skipped 6 (43327: "already forecast by this account") | | | | nothing | seconds, 0 model calls |

What each run left in its ledger (`data/ledgers/`, scanned by
`tests/test_leakage.py` and `augur ledger check`: 0 violations each):

| Ledger | runs | questions | sources | gate decisions | model calls (planner, brief, forecasters) | call-source links | forecasts | submissions |
|--------|------|-----------|---------|----------------|-------------------------------------------|-------------------|-----------|-------------|
| `m1-dryrun-samples.sqlite` | 1 | 1 | 3 | 3 | 7 (1, 1, 5) | 18 | 6 (5 members, 1 median) | 2 `dry_run` |
| `m1-live-bot-testing-area.sqlite` | 2 | 1 | 3 | 3 | 7 (1, 1, 5) | 18 | 6 | 2 `submitted` (201) |

Tokens, live run: planner 487 in / 32 out, brief 2,440 / 285, five
forecasters 4,225 / 1,499. Each draw waits for the one before it (Ollama runs
one generation at a time on CPU), which is why a question takes about 11
minutes.

**Raw output.** The printed run output is in `data/runs/`, every model call
(request and response) in `data/recordings/`. The live run's text file
predates the change that prints the posted comment; the comment as sent is
the `payload_json` of the `comment` row in `submissions`.

**What the runs showed.**

- The first dry run (before commit `1422e20`) had every draw say the question
  closed in "14 days" when it closed in 44: the 7B model miscounted the dates
  in the prompt. The prompt now states the time left; the runs above say
  "44 days".
- The live API returns `my_forecasts` times as Unix seconds, not ISO text.
  Found from the submission above, fixed in `b00863f`; the skip had worked
  anyway because it keys on `forecast_values`, but a withdrawn forecast would
  have been read as standing.
- Both committed ledgers record `code_version` `0.1.0.dev0+cc5994aaa0af`,
  which is not a commit of this repository (a `GITHUB_SHA` from the host job).
  The ledger is append-only, so the rows stay; runs from `code_version`'s fix
  onward record this checkout's commit.

Not a result: one question is not a score. These runs show the pipeline
works against the real API; they say nothing yet about accuracy.

## To measure

| What | Why not yet | Command |
|------|-------------|---------|
| A frontier model answering, and a binary forecast on a live MiniBench round | No billed model was called: the key's $1 limit may not be the sponsorship, and nothing paid is spent without the maintainer's say-so. No MiniBench question was open at 11:05 UTC on 2026-11-18 either | `AUGUR_FORECAST_MODELS=openrouter:openai/gpt-5.1,openrouter:anthropic/claude-sonnet-5.5 AUGUR_RESEARCH_MODELS=openrouter:nvidia/nemotron-3-ultra-550b-a55b:free augur models probe`, then the same variables with `augur forecast --tournament minibench --submit` (about $0.05 a question: run 2's forecaster tokens, 3,200 in and 4,629 out, at `gpt-5.1`'s listed $1.25 and $10 per million; at most $0.08 if all five use their 1,500 tokens; needs `METACULUS_TOKEN`) |
| Groq, Gemini and GitHub Models free tiers | No `GROQ_API_KEY`, `GEMINI_API_KEY` or `GITHUB_TOKEN` here; the presets are configured, not exercised | `AUGUR_FORECAST_MODELS=groq:<model> augur models probe --burst 30` (and `gemini:`, `github:`) |
| Tokens-per-minute limits | OpenRouter sends no token-limit header and the bursts are tiny | the same burst on Groq, which sends `x-ratelimit-limit-tokens` |
| AskNews sources | No `ASKNEWS_CLIENT_ID`/`ASKNEWS_CLIENT_SECRET` in the build environment; the searcher is tested only against recorded payloads | `make test-integration` with both set, then `augur forecast --tournament minibench` |
