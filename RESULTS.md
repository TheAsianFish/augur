# Results

Every number here comes from a command in this repository, with the model,
date, cost and raw output recorded. Nothing scored yet: the first scored
number (ForecastBench, no retrieval) and the first live score arrive in M4.
M3 measured what a live round needs: the question windows, how late
GitHub's cron fires, and whether every question type is accepted.

## M3: every question type live, deadlines, and the schedule (2026-11-18)

**Setup.** Hosted runs used OpenRouter `:free` models only (nemotron-3-ultra
first in the forecast chain, ling-3.2-flash first for research), so every
call cost $0; no billed model was called (the key still reports a $1 limit,
$0 used). Measurements of Metaculus and GitHub used their public APIs,
read-only. Raw output for everything below is in `data/runs/m3-*`.

### Every question type, accepted by the real API

Two runs of `augur forecast --tournament bot-testing-area --submit` against
the practice area, which held one question of each kind on 2026-11-18
(ledgers `data/ledgers/m3-live-bot-testing-area{,-2}.sqlite`, recordings
`data/recordings/m3-live-bot-testing-area{,-2}.jsonl`, coverage JSON beside
the printed output in `data/runs/`).

| Question (post) | Type | Run 1 (18:28, 1,500-token cap) | Run 2 (18:40, `--reforecast`, 2,500 for non-binary) |
|-----------------|------|-------------------------------|------------------------------------------------------|
| 43323 Trump net approval, 31 Dec | numeric, 200 buckets, open bounds | 5 of 5 draws; forecast 201, comment 201 | 5 of 5; 201, 201 |
| 43321 officials out before 2027 | discrete, 16 buckets, closed bounds | 4 of 5; 201, 201 | 5 of 5; 201, 201 |
| 45517 new forecasters per day | discrete, 71 buckets, step 0.01 | 4 of 5; 201, 201 | 4 of 5; 201, 201 |
| 46224 US outlays FY 2028 | numeric, 200 buckets | 3 of 5; 201, 201 | 4 of 5; 201, 201 |
| 46223 2028 presidential winner | multiple choice, 3 options | 3 of 5; 201, 201 | 5 of 5; 201, 201 |
| 43327 EU exit before 2027 | binary | skipped: already forecast by this account | 5 of 5; 201, 201 |
| Coverage line | | 6 listed; 5 forecast in time (5 submitted), 1 skipped, 0 failed; 11 min 2 s | 6 listed; 6 forecast in time (6 submitted); 15 min 35 s |

Every forecast the bot built was accepted by the server: 11 of 11, every
CDF and every option set. What the runs found:

- **Draws lost to the token cap.** Run 1 kept 19 of 25 draws. Five had used
  exactly the 1,500-token cap and stopped before their last percentile or
  option line (the ledger's `output_tokens` equals the request's
  `max_tokens`); the sixth found every free model in the chain rate-limited.
  Non-binary answers end in six lines rather than one, so run 2 gave them
  2,500: 28 of 30 draws kept, none at the cap, both losses rate limits.
- **The same on a binary question, live.** A dry run on MiniBench question
  [46620](https://www.metaculus.com/questions/46620/) (open 18:43 to 21:43
  UTC, found by the workflow under `act`, below) failed at 1,500 tokens: 3 of
  5 binary draws ran out before their probability line, so the question was
  reported failed rather than posted with two members. At 4,000 tokens for
  every type, the same question gave 5 of 5 draws (one used 3,632 tokens),
  median 35%, nothing posted (`data/runs/m3-dryrun-minibench{,-2}.txt`).
  At GPT-5.4's listed $15 per million output tokens, five draws at 4,000
  cost at most about $0.30, inside the $0.50 per-question cap.
- **The floor on MiniBench.** `augur forecast --tournament minibench
  --submit` with the free chain refused before any model call, exit 2
  (`data/runs/m3-submit-floor-minibench.txt`).
- All four new ledgers scan clean: `augur ledger check` 0 violations each.

**The CDF rules, tested before the server saw one.** `tests/test_distributions.py`
generates arbitrary scalings (linear and log, 2 to 200 buckets, every bound
combination) and arbitrary percentile sets with Hypothesis (400 examples per
property, derandomized) and asserts every CDF passes Metaculus's rules as
read from its server code (`questions/serializers/common.py`, 2026-11-18):
n+1 values, every step at least 0.01/n and at most 0.2 × 200/n, the open or
closed ends. The fake server checks every posted body against its own
transliteration of those rules (`tests/fakes/metaculus.py::_continuous_errors`),
which does not import Augur's validator.

**Never twice.** `test_every_question_type_is_accepted_and_a_second_run_forecasts_none_twice`
runs the five-question practice area on the fake server twice: the first run
posts five forecasts and five comments, the second lists the same five and
posts nothing ("already forecast by this account" for each). Live, run 1
skipped 43327 for the same reason.

### MiniBench windows, measured

`python3 scripts/question_windows.py --interval 2` read every question of the
current round and of each earlier round reachable by slug (2026-11-18 18:23
UTC, 36 requests; `data/runs/m3-question-windows.{txt,json}`): 954 questions
in 16 rounds from 2026-04-13.

| Measure | Value |
|---------|-------|
| Types | binary 503, numeric 216, discrete 165, multiple choice 70: **47% are not binary** |
| Window (scheduled close − open) | 3.00 h for all 729 questions since 2026-06-15; 1.50 h for the 225 before |
| Gap between consecutive opens in a round | median 0.67 h, p90 3.47 h, max 115.6 h |
| Opens by UTC hour | every hour of the day; fewest 20 (22:00), most 58 (00:00) |
| Share of a round open by day 2 | 22% to 100% (median round 92%) |
| Current round (from 2026-11-16) | 59 questions by 18:23 on 2026-11-18, all 3.00 h |

So a forecast has a three-hour window that can open at any hour, often less
than an hour after the previous one.

### How late GitHub's cron fires, measured on public bot repositories

This repository is private and no session here can push to it, so its own
schedule cannot be measured yet. `python3 scripts/cron_delay.py`
(unauthenticated GitHub API, 2026-11-18 19:38 UTC; `data/runs/m3-cron-delay.{txt,json}`)
measured the same kind of workflow on public forks of Metaculus's template
bot instead. It listed 439 forks, fetched 94 recently pushed ones, and found
55 with recent scheduled runs: 99 workflow and cron groups, 2,813 runs. Each
run is matched to the latest cron tick before it.

| Group | Runs | What happened |
|-------|------|---------------|
| Crons every 20 min or less (47 repos, 56 workflows) | 1,971 | 5.0% of ticks produced a run; median 5.5 h between runs |
| Crons every 30 to 60 min (9 repos, 11 workflows) | 460 | 12.7% of ticks produced a run; median 6.0 h between runs |
| Crons every 1 to 12 h (7 repos, 10 workflows) | 238 | 53.2% of ticks |
| Daily and twice-daily crons (7 repos, 22 workflows) | 144 | every tick ran, but late: delay p50 374, p90 553, max 591 min |
| The two workflows with near-complete coverage | 174 | delay p50 9.1, p90 59.5, p99 107.8, max 175.7 min; 1 tick of 175 dropped |

On these repositories GitHub's scheduler is not a clock. Short crons are
throttled to a run every five to six hours, and long ones fire hours late.
The repositories also recorded 16,938 runs from other triggers between their
scheduled runs: bot makers already work around the scheduler. Caveats: public
forks only, newest push first, at most 100 scheduled runs per repository; a
delay longer than a short cron's interval looks like a dropped tick; private
repositories like this one, and paid accounts, are not in the sample.

### The cadence, and the fallback

`python3 scripts/cadence.py` (`data/runs/m3-cadence.txt`, seed 20261118, 200
replications) replays the 954 measured open times against each candidate
cron. Delays and drops are drawn from the two on-time workflows above. Each
run that finds work is billed an 8-minute forecast job, and each poll 1 minute
(GitHub rounds every job up to a minute). A question counts as caught if it
is forecast with at least 15 minutes left.

| Cadence | Caught, all windows | Caught, 3 h windows | Billed min per 30 days (poll + forecast) |
|---------|---------------------|---------------------|-------------------------------------------|
| every 30 min | 99.42% | 100.00% | 2,164 (1,432 + 732) |
| every 60 min | 96.02% | 99.93% | 1,307 |
| **every 90 min (chosen)** | 91.58% | **99.25%** | **979 (477 + 502)** |
| every 90 min + a line 45 min later | 97.43% | 100.00% | 1,607 |
| every 120 min | 86.97% | 96.97% | 791 |
| throttled, as on most public forks (any cadence) | 37.4% | 43.2% | |

Chosen while the repository is private: every 90 minutes on two offset lines
(`0 0,3,…,21 * * *` and `30 1,4,…,22 * * *`). Every window since June is
3 hours, and this is the cheapest cadence that catches 99% of those, for
about 979 billed minutes a month. That is half of the shared 2,000-minute
allowance; every 30 minutes would exceed it. The estimate charges all 954
questions an 8-minute job, including the earlier 1.5-hour rounds.

**Fallback trigger.** The measurement says the cron cannot be the only
trigger: on a throttled repository no cron catches more than 43% of 3-hour
windows. The fallback is `workflow_dispatch`, which a job may send with its
own token and which is not subject to the scheduler. With the repository
variable `AUGUR_CHAIN_MINUTES` set, each scheduled or chained run on `main`
waits that long and dispatches the next run. It dispatches nothing while
another run is queued, so there is never more than one chain, and the cron
restarts the chain if it breaks. The chain's waiting bills minutes on a
private repository, so it is off until the repository is public (RELEASE.md).
That this repository's own runs fire on time is unmeasured until its first
scheduled runs land on the `ledger` branch (M4).

### The workflow, run under `act`

`act` 0.2.98 on `catthehacker/ubuntu:act-latest` ran
`.github/workflows/tournament.yml` five ways (`data/runs/m3-act-dry-run.txt`):

1. No secrets: the poll reports no token and the forecast job is skipped.
2. The real token on the practice area: "bot-testing-area: 6 open, 0 not yet
   forecast", job skipped.
3. A fake token: the poll's 403 counts as work, so the forecast job installs,
   runs, fails as it should, and the job is marked failed.
4. Scheduled with `AUGUR_SCHEDULE` unset: "schedule is off", no request.
5. Scheduled with `AUGUR_SCHEDULE=minibench` and the real token: the poll
   found MiniBench question 46620, opened minutes earlier. The forecast job
   ran the bot on the live listing and wrote the coverage report. No model
   variable was set, so it reported "skipped: no forecasting model answered
   the probe". The ledger push then failed before writing anything, because
   this machine has no GitHub credentials.

`scripts/push_ledger.sh` was tested against a local bare remote: it creates
the orphan `ledger` branch, pushes two runs in a row, and retries after a
refused push.

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
| A frontier model answering, and forecasts with private comments on a live MiniBench round | No billed model was called: the key reports a $1 limit that may not be the sponsorship, and nothing paid is spent without the maintainer's say-so. The floor refused the free chain (exit 2, `data/runs/m3-submit-floor-minibench.txt`) | `AUGUR_FORECAST_MODELS=openrouter:openai/gpt-5.4@high,openrouter:anthropic/claude-opus-5 AUGUR_RESEARCH_MODELS=openrouter:nvidia/nemotron-3-ultra-550b-a55b:free augur models probe`, then the same variables with `augur forecast --tournament minibench --submit --coverage coverage.json` while a round is open (from about 2026-11-30). Worst case about $0.30 to $0.35 a question at GPT-5.4's listed $2.50/$15 per million: five 4,000-token draws ($0.30) plus their prompts of about 3,000 tokens each; needs `METACULUS_TOKEN` |
| This repository's own cron delay and coverage | Private, and no push credentials here | After merge, with `AUGUR_SCHEDULE=minibench`: `python3 scripts/cron_delay.py` cannot see private runs, so read the run times from the `ledger` branch's `*.txt` files against the two cron lines (M4) |
| The ledger-branch push and the dispatch chain on GitHub | Need the repository's own token | `gh workflow run tournament.yml --ref main -f tournament=bot-testing-area -f push_ledger=true`, then check the `ledger` branch; set `AUGUR_CHAIN_MINUTES=60` once public and check that consecutive runs are about 60 min apart |
| Groq, Gemini and GitHub Models free tiers | No `GROQ_API_KEY`, `GEMINI_API_KEY` or `GITHUB_TOKEN` here; the presets are configured, not exercised | `AUGUR_FORECAST_MODELS=groq:<model> augur models probe --burst 30` (and `gemini:`, `github:`) |
| Tokens-per-minute limits | OpenRouter sends no token-limit header and the bursts are tiny | the same burst on Groq, which sends `x-ratelimit-limit-tokens` |
| AskNews sources | No `ASKNEWS_CLIENT_ID`/`ASKNEWS_CLIENT_SECRET` in the build environment; the searcher is tested only against recorded payloads | `make test-integration` with both set, then `augur forecast --tournament minibench` |
