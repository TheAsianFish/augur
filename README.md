# Augur

**Status: 0.1.0, release candidate; M1 to M16 built.** Fall 2026 was the shakedown season: the bot
posted no scored forecast, because the submit floor admits only frontier-tier models and
no billed model was called (the key's spend was never approved), and the [postmortem](docs/postmortem-fall-2026.md) says
what the backtests found instead. The 2026-11-30 MiniBench round runs as
an unattended shadow round against Metaculus's template bot on the same free
model, under a rule registered before it opened that decides whether the
free model may submit; the candidate it judges is one blend registered in
advance, one-third Augur and two-thirds a forecast with no model in it.
Because one round cannot settle whether the pipeline beats the template, a
pooled analysis over the season's shadow questions was registered before
the round as well, with one fixed date (2027-01-11). Spring 2027 is the
season to place in.

A forecast about a real event is scored only after the event happens, often
months later, so a team building an AI forecaster cannot tell in time whether
a change helped, and most forecasters that look accurate in backtests quietly
read the answer in news written after the question closed.

Augur is a forecasting bot for Metaculus's FutureEval bot tournament and the
two-week MiniBench rounds, where it is scored on the same questions as every
other bot. Around it is a ledger that keeps every question, prompt, source
(with proof of its date) and forecast, and a leakage gate that refuses any
source dated after the question's cutoff, enforced three times: in code, in
the ledger's own triggers, and by a test that scans every committed ledger.

## Demo

```text
$ export AUGUR_FORECAST_MODELS=openrouter:nvidia/nemotron-3-ultra-550b-a55b:free,openrouter:inclusionai/ling-3.2-flash:free,openrouter:google/gemma-4-31b-it:free
$ export AUGUR_RESEARCH_MODELS=openrouter:inclusionai/ling-3.2-flash:free,openrouter:google/gemma-4-26b-a4b-it:free,openrouter:nvidia/nemotron-3-ultra-550b-a55b:free
$ augur forecast --tournament bot-testing-area --submit --reforecast
run run_53eeaa89f3784e2a8bbf (live) on metaculus:bot-testing-area
probe:
  openrouter:nvidia/nemotron-3-ultra-550b-a55b:free answered      939 ms  $0/$0 per M  listed 2026-06-04  below tier  [research+forecast]
  openrouter:google/gemma-4-31b-it:free             rate_limited 1524 ms  $0/$0 per M  listed 2026-04-02  below tier  [forecast]
  ...
estimated worst case $0.0000 a question, $0.0000 for 6 (caps $0.50 a question, $5.00 a run)

== What will be Donald Trump's net approval on December 31, 2026?
   https://www.metaculus.com/questions/43323/ (numeric)
   sources: 5 admitted, 0 rejected; members: median -22.0% (80% between -32.0 and -12.0) | ...
   combined: median -22.1% (80% between -32.5 and -11.7)
   submitted median -22.1% (80% between -32.5 and -11.7); posted this private comment:
   | **Forecast: median -22.1% (80% between -32.5 and -11.7)**, the average of the CDFs of 5 independent forecasts by ...

== Which party's nominee will win the 2028 US presidential election?
   https://www.metaculus.com/questions/46223/ (multiple_choice)
   combined: Democratic Party 48%, Republican Party 50%, Other 2%
   submitted Democratic Party 48%, Republican Party 50%, Other 2%; posted this private comment: ...

coverage: 6 listed; 6 forecast in time (6 submitted), 0 missed, 0 skipped, 0 failed; estimated cost $0.0000
```

A real run on free OpenRouter models (trimmed; the full output is
`data/runs/m3-live-bot-testing-area-2.txt`): one question of every type in
the practice area, soonest-closing first, each CDF and option set built to
Metaculus's validity rules and accepted. These models are below the
frontier tier, so the same command against MiniBench refuses before any
model call.

To record the demo as a GIF (no GIF is committed yet): install
[`vhs`](https://github.com/charmbracelet/vhs) and run `vhs docs/demo.tape`,
which types the quickstart's dry run below and writes `docs/demo.gif`; or
`asciinema rec docs/demo.cast -c "uv run augur forecast --tournament minibench"`.

## Quickstart

A dry run on a local model, no account and no key needed (checked from a
fresh clone with an empty environment: `data/runs/m10-fresh-clone-quickstart.txt`,
about 14 minutes on a 4-core CPU):

```sh
git clone https://github.com/TheAsianFish/augur && cd augur
uv sync                                                  # Python 3.12, from uv.lock
ollama pull qwen2.5:7b
uv run augur forecast --tournament minibench            # the committed sample question; prints what it would post
uv run augur ledger check augur.db                      # leakage and floor scan; exit 1 on a violation
```

Without the repository: `pip install augur-forecast` (once 0.1.0 is
published; RELEASE.md), or the container, which keeps its ledger on a
volume and takes secrets only at `docker run`:

```sh
make docker                                              # builds augur-forecast:0.1.0
docker run --rm --network host -v augur-data:/data augur-forecast:0.1.0 forecast --tournament minibench
```

Hosted models are configuration: an ordered chain per role, `endpoint:model`.

```sh
export OPENROUTER_API_KEY=...                            # Metaculus's sponsored key, or any key for :free models
export AUGUR_FORECAST_MODELS=openrouter:openai/gpt-5.1,openrouter:anthropic/claude-sonnet-5.5
export AUGUR_RESEARCH_MODELS=openrouter:nvidia/nemotron-3-ultra-550b-a55b:free
uv run augur models list                                 # prices, listing dates, tier; no calls
uv run augur models probe                                # one call per model: who answers right now
export METACULUS_TOKEN=...                               # the bot account's token
uv run augur forecast --tournament bot-testing-area --submit
```

After questions resolve:

```sh
uv run augur resolve --ledger augur.db                  # append resolutions; hidden ones are counted, not guessed
uv run augur score augur.db --cluster week               # scores, and paired intervals against every reference
uv run augur standings --round minibench                 # the round's leaderboard beside Augur
make eval                                                # offline: re-score the committed backtest, check RESULTS.md
```

### The shadow round (M11)

Whether Augur's pipeline adds anything over Metaculus's template bot on the
same free model can be measured without posting anything. A shadow tick
forecasts every open question unpublished, runs the real template bot on
the same questions with the same route, and pushes the tick's ledger to
the `ledger` branch after each; the next tick skips what that branch
already holds. `AUGUR_SHADOW=true` puts the scheduled workflow in this mode,
and no path in it can pass `--submit`.

```sh
uv sync --extra baseline                                 # the template bot (forecasting-tools)
uv run augur forecast --tournament minibench --shadow-ledger shadow.sqlite --ledger run.sqlite
uv run augur baseline --ledger run.sqlite --shadow-ledger shadow.sqlite
uv run augur ledger import ledger-branch/shadow --out shadow.sqlite        # every tick, pooled
uv run augur resolve --ledger shadow.sqlite --manual-file round.csv --standings data/standings/ROUND.json
uv run augur floor --round ROUND --ledger shadow.sqlite  # admitted, not admitted or not yet decidable
```

`augur floor` applies the rule registered in PLAN.md before the 2026-11-30
round opened (RESULTS.md carries it verbatim): the free model may submit
to MiniBench only if the 90% interval of its head-to-head against the
shadow template, and of the peer score that implies, both sit above 0.

### A forecast with no model in it (M12)

On the questions where the free model is worst, something simpler already
knows more: a market price, or for a stock or index close, the series'
own history. The statistical arm (`augur/statarm.py`) makes no model call.
On a binary question it is the question's own price where the markets
searcher reads one, otherwise the base rate of its source. On a numeric
question about a quoted series it is the last close times the empirical
distribution of the series' past changes over the same horizon. Each close
comes from Yahoo Finance's keyless chart endpoint, is dated 00:00 UTC of the
next day and is written to the ledger as a dated source, so the leakage gate
covers it. Anywhere else the arm abstains and says why. Every shadow tick
writes the arm (`stat`) and the registered candidate (`blend`: one-third
Augur, two-thirds arm, only where the arm has a price or a series) beside
Augur's own forecast, and `augur floor --candidate blend` scores the blend.

```sh
uv run augur statarm                    # the arm and both blends on C4, no model call
uv run augur floor --round ROUND --ledger shadow.sqlite --candidate blend
```

### One pooled analysis, registered before the round (M14)

One round of about 60 questions can only tell a lead of about 10 points
from 0 at 80% power, so a single round's floor verdict says little about
whether the pipeline helps (Q1). M14 fixed, before the 2026-11-30 round
opened, the one pooled analysis the season's shadow questions will get:
Augur's head-to-head against the same-model template over both remaining
MiniBench rounds and the FutureEval shadow questions, bootstrapped by the
day questions opened, read once on 2027-01-11. `augur floor --pooled`
prints it and cannot print a floor verdict; only the one-round rule can
admit a route. A seeded synthetic table shows why: re-reading the
one-round rule after each of four rounds admits a candidate with no lead
in 11.8% of simulated seasons, not 5%.

```sh
uv run augur floor --pooled --round R1 --round R2 --round fall-futureeval-2026 --ledger shadow.sqlite
uv run augur rereads --check            # the synthetic re-read table, as make eval runs it
```

### A newer free reader on the same questions (M16)

M7 and M13 both point at the reader, not the evidence, as this pipeline's
limit, so M16 asks whether the newer free model (`nemotron-3.5-super-vl`)
reads better than the current route (`nemotron-3-ultra`) on identical
questions. Both forecast the 292 market questions of the two ForecastBench
sets posed after the newer model's listing, one draw, no retrieval, before
the outcomes were read, under a route rule committed before any call. A
reported live arm (`AUGUR_SHADOW_READER=true`) adds three draws on the
newer model to each shadow tick from the exact forecaster request Augur's
draws saw, served last and only within the day's free calls.

```sh
make backtest-reader                    # both routes on both sets (free, resumable, about two hours)
make resolve-reader                     # each set's resolution file as it appears, then augur readers
uv run augur readers --check            # the paired comparison and the rule, as make eval runs it
```

### ForecastBench's dataset half, with no model (M15)

Half of every ForecastBench set asks whether a series (a stock's close, a
FRED or DBnomics series, an ACLED count, a Wikipedia ranking) will be
higher on each resolution date than on the due date, and the leaderboard's
leaders pull furthest ahead there. M15 forecasts 4,398 of these instances
with no model call: the source's base rate, the series' own past outcomes,
and for stocks the share of the ticker's past same-horizon moves that were
up, from closes dated at or before each question's freeze instant and
written to a backtest ledger first. Each is scored beside ForecastBench's
own baselines and named entries on exactly the instances they forecast.
The base rate beats a coin and the Naive Forecaster; nothing beats a coin
on stocks, the leaders included; and the leaders' edge sits in Wikipedia
rankings and, for two of the three, in daily temperatures, where a
walk-forward rate built over spring points the wrong way in autumn.

```sh
uv run augur backtest dataset-closes    # the stocks' closes as of each freeze instant (Yahoo, keyless)
uv run augur fbcompare --check          # M10's and M15's same-instance rows, as make eval runs them
```

Presets: `openrouter`, `groq`, `gemini`, `github` (each reads its own key
variable) and `ollama`; any other OpenAI-compatible host is
`AUGUR_ENDPOINT_<NAME>_URL` plus `_KEY_ENV`. `--submit` to MiniBench or
FutureEval refuses unless the first forecasting model is a hosted model in
the named frontier tier (`augur/llm/registry.py`), and a question whose
forecasts would have to fall back below that tier is skipped, not posted: a
weak forecast costs peer points that skipping does not.

## Architecture

```mermaid
flowchart LR
  MQ[Metaculus API<br/>or committed samples] --> RUN[runner<br/>skip, cap, budget]
  CRON[scheduled workflow<br/>stdlib poll first] --> RUN
  RUN --> PROBE[probe<br/>one call per model]
  PROBE --> ROUTER{router<br/>ordered chain per role,<br/>fallback, floor}
  CAT[OpenRouter model list<br/>prices, listing dates] --> ROUTER
  RUN --> RES[researcher<br/>plan queries]
  RES --> S[searchers as_of cutoff<br/>Wikipedia revisions, Wayback captures,<br/>GDELT news via Wayback (backtests),<br/>AskNews, market prices]
  S --> LED[(ledger<br/>SQLite, append-only)]
  LED --> GATE{leakage gate<br/>proven date <= cutoff}
  GATE -->|admitted only| BRIEF[one-pass brief, a second pass<br/>on sub-questions, or the loop]
  BRIEF --> FC[forecaster fleet<br/>5 draws by default; stances<br/>and models per member]
  FC --> MED[combine by type<br/>median by default; trimmed,<br/>log-odds, extremized]
  MED --> CAL[calibrator<br/>Platt map, only for the<br/>model it was fitted on,<br/>withheld from Metaculus questions]
  CAL --> VALID{Metaculus's<br/>validity rules}
  VALID --> CHECK{who answered<br/>in the tier?}
  CHECK -->|yes, or practice area| OUT[forecast + private comment<br/>posted or printed]
  ROUTER -.-> RES & FC
  RES & FC & PROBE & ROUTER --> LED
  OUT --> LED
  LED --> BR[ledger branch<br/>one JSONL export per run]
  BR -->|augur ledger import| LED
  TPL[shadow template bot<br/>real TemplateBot, unpublished] --> LED
  RESV[augur resolve<br/>Metaculus, ForecastBench, manual] --> LED
  LED --> SCORE[augur score<br/>Metaculus scores, paired<br/>clustered bootstrap, MDE]
  LB[leaderboard API] --> STAND[augur standings]
  BT[augur backtest<br/>clock pinned to open,<br/>or as of now: unrestricted] --> RES
  LED --> JUDGE[augur judge<br/>0-4 leakage rubric]
  JUDGE --> LED
  LED --> Q4[augur calibrate<br/>walk-forward: fit only on<br/>outcomes known at open]
  Q4 -.->|live map| CAL
  SCORE & STAND & Q4 --> REPORT[augur report<br/>RESULTS.md blocks, drift test]
```

| Module | Job |
|--------|-----|
| `augur/metaculus.py` | Typed `httpx` client: list, detail (`my_forecasts`), a forecast of any type, private comment; token bucket and `Retry-After`. |
| `augur/predictions.py`, `augur/distributions.py` | What a forecast is per type, how members combine, the payload; percentiles to the CDF Metaculus accepts, its validity rules mirrored from the server, property-tested. |
| `augur/llm/` | `endpoints` (presets and `AUGUR_ENDPOINT_*`, `endpoint:model` routes), `openai_compat` (one provider for every OpenAI-compatible host, sorting refusals into rate-limited, out-of-credit, auth, unavailable, empty), `router` (the chain per role, cool-downs, the floor), `catalog` (OpenRouter's model list: prices, listing dates), `probe` (opening probe, burst limit measurement), `registry` (frontier tier, local release dates), `ollama`, `replay`. |
| `augur/research/` | `Source` record with `date_evidence`; `wikipedia` (the revision current at `as_of`), `wayback` (the newest capture at or before `as_of` of each URL the question names, its own timestamp compared, never the query's day bound), `asknews` (bounded by `end_timestamp`), `markets` (prices as of the cutoff), `gdelt` (M13, backtests only: GDELT discovery bounded at the cutoff, an article seen after it refused whatever GDELT returned, its text only from a Wayback capture at or before the cutoff; a local index of GDELT's GKG files answers when the API refuses). |
| `augur/leakage.py` | The gate, and `scan_ledger`, which re-checks every stored row. |
| `augur/ledger.py` | Numbered migrations; every table refuses `UPDATE` and `DELETE`. |
| `augur/agents/` | Researcher (one pass; a second pass that searches the sub-questions that would most change the answer; or a bounded loop that writes a belief and what is still unknown and picks the next query), forecaster draws with per-member stances and model chains, and the recorder every model call passes through: it walks the role's chain, and times, costs, budgets and ledgers every attempt and switch. |
| `augur/fleet.py`, `augur/aggregate.py` | The fleet plan from settings (research passes, member stances and models) and `bounded_map`, which runs questions in parallel with a fixed bound; aggregation rules for every question type (median, trimmed mean, mean of log-odds, extremized log-odds). |
| `augur/redraw.py`, `augur/q3.py`, `augur/power.py` | Q3's ablation: new draws or another model on an arm's recorded briefs (each rebuilt request checked against the recorded hash), the depth, head-count, aggregation and model-mix tables, and the planted-effect power simulation. |
| `augur/calibrate.py`, `augur/q4.py`, `augur/livecal.py`, `augur/diagram.py` | Q4: calibrators (raw, fixed extremizing, base rate, Platt, isotonic) behind one fit-then-map interface; the walk-forward harness that calibrates each question only on outcomes resolved before it opened and refuses any fit that would see a later one; reliability bins, calibration error and sharpness; the live map, applied only to forecasts from the model it was fitted on; the reliability diagram as a byte-stable SVG. |
| `augur/sourcecal.py` | Calibrators that see a question's source: Platt per source, and a partially pooled map (one slope, a per-source intercept shrunk toward the shared one) on a small penalised Newton solver. |
| `augur/q6.py` | Q6: disagreement measures (draws, models, research passes), their AUROC for large errors, leave-one-out controls (constant shrinkage, the source-aware map) and the shrink and skip policies against them; the live half recomputed from the ledgers' member rows. |
| `augur/comment.py` | The private comment, built only from the members' reasoning and the cited sources. |
| `augur/score.py`, `augur/scorecard.py` | Metaculus's baseline, head-to-head and peer scores for every type (matched to its own `score_math.py`), Brier and log score; scoring every resolved forecast against the template, the community, a market price, 50% and the base rate with a paired, clustered bootstrap and the minimum detectable effect. |
| `augur/resolve.py`, `augur/baseline.py`, `augur/standings.py` | Resolutions (hidden ones counted, not guessed), the shadow template run unpublished on Augur's model, and each round's leaderboard beside Augur. |
| `augur/forecastbench.py`, `augur/resolved.py`, `augur/backtest.py` | Resolved questions (ForecastBench sets; past Metaculus tournaments where the token sees resolutions) with eligibility filters, and the bot's own researcher and forecaster run with the clock pinned to each question's open instant, every route's release date checked at call time; Q2's leaky arm as an `unrestricted` run. |
| `augur/judge.py`, `augur/q2.py` | The leakage judge (El Lahib et al.'s 0 to 4 rubric; its check against hand labels is under To measure) and Q2's scoring: arms, paired differences, edge over the price. |
| `augur/report.py`, `augur/importer.py` | RESULTS.md's generated blocks (a test fails on drift); one ledger rebuilt from the `ledger` branch's exports. |
| `augur/fbcompare.py` | The ForecastBench same-question row: every entrant's published forecasts on exactly the instances Augur's backtests scored, paired differences on one seeded resample sequence (`paired_many`), ranks per group. |
| `augur/runner.py`, `augur/cli.py` | `augur forecast` (soonest-closing first, a deadline per question, a coverage report), `augur models list|probe|snapshot`, `augur ledger check|export`. |
| `.github/workflows/standings.yml`, `scripts/push_standings.sh` | Every day, the current MiniBench round and the two before it (and, with `AUGUR_SCHEDULE=all`, the FutureEval season) re-read and kept on the `ledger` branch, so each round's final standing is recorded after it finalizes. |
| `.github/workflows/release.yml`, `scripts/check_install.sh`, `Dockerfile` | Release: `make check` and an install of the built wheel and sdist into fresh venvs outside the repository on every run; PyPI by trusted publishing only from a `v*` tag; the container for running the bot outside Actions. |
| `.github/workflows/tournament.yml`, `scripts/` | The schedule: `poll.py` (standard library) decides whether to install anything; the forecast job posts from `main` only and `push_ledger.sh` commits each run's export to the `ledger` branch. `question_windows.py`, `cron_delay.py` and `cadence.py` measured how often it must run. |

### The hard part

A backtest is only worth anything if no forecast could have read the answer.
In Augur a source enters a prompt only after it has been written to the
ledger and the gate has admitted it on **evidence** for its date (a wiki
revision's save time, a news provider's publish date from a query bounded at
the cutoff), never on a date the page states about itself. The rule is held
three times, so a bug in one layer is caught by the next:

1. `admit_source` in `augur/leakage.py` decides at run time.
2. SQLite triggers in `augur/migrations/0001_initial.sql` abort an admitted
   use of a source dated after its question's cutoff, an admitted undated
   source in a backtest, and a model call that cites a source the gate did
   not admit.
3. `tests/test_leakage.py` scans every ledger committed under `data/ledgers/`
   and plants one violation of each kind to prove the scan finds it.

M6 measures what the rule is worth (Q2): the same questions researched as
of their open instant and as of today, and a judge that reads every source
the forecasters saw for information from after the question opened. The
"as of today" arm is recorded in a mode of its own (`unrestricted`): its
cutoff is the moment it ran, so the gate and the triggers still hold, and
nothing in it can be scored as a backtest.

M2 adds a second rule held the same way: **a scored forecast comes only from
a frontier-tier model**, even when the models in front of it fail mid-run.
The router never walks a scored question's chain past the tier (it skips
the question and writes the refusal to the ledger); the runner checks the
routes that actually answered before posting; and `augur ledger check`
flags any submitted forecast whose members came from outside the tier.

M3 adds a third: **every posted forecast is one the server accepts, inside
its three-hour window.** A numeric answer is six percentiles; the server
wants 201 CDF points with a minimum and maximum step and exact or open
ends. `augur/distributions.py` builds to those rules, Hypothesis checks it
over arbitrary questions and answers, and the fake server re-checks every
body with its own transliteration of the server code. Each question has a
deadline from its close; work still running there is cancelled and charged,
and what finished is posted.

M8 holds the same rule for what the bot learns from its own record: **a
calibrator may only see outcomes that were known when the question it
corrects was opened.** Fitting a calibration map on all resolved questions
and scoring it on the same ones would leak the answers back in through the
map. `augur/q4.py` calibrates each question only on outcomes known before
the day it opened. An outcome is dated on evidence, as a source is: the
later of the dataset's resolution date and the market's own resolution
time, because the dataset's date proved to be the close for 239 of the 536
markets. `check_time_order` refuses
any fit that would see a later outcome (a test proves a leaky history rule
is refused), and the snapshot records the latest resolution every fit saw
so `make eval` can re-check it.

M9 turns the same suspicion on a result that would look good. The model's
log-odds are worth a third of their face value, so any rule that pulls
forecasts toward 50% beats raw. A rule that shrinks *by disagreement*
therefore counts only if it beats, on the same questions, a constant shrink
of the same strength and the source-aware calibrator; a rule that skips
disagreed questions must beat skipping as many at random. Measured that
way, every apparent gain disappeared into its control.

## Design decisions

| Decision | Chosen | Alternative | Why |
|----------|--------|-------------|-----|
| Metaculus access | Own `httpx` client | `forecasting-tools`' `ForecastBot` | Every request must land in the ledger and be explainable. |
| Model hosts | One OpenAI-compatible provider over plain `httpx`; a host is a base URL and a key variable | One class per host, or the `openai` SDK | Credits moved from a Metaculus proxy to an OpenRouter key in a week; the next move is a config change. One HTTP stack to mock and record. |
| Fallback | Ordered chain per role; rate limit cools a route, a spent key or dead model disables it for the run, an empty answer moves only that call | Retry the same model; a single model | Free tiers fail often (Google's free Gemma models answered upstream 429s through most of M2's and M3's runs, RESULTS.md M2); a question is open for 3 hours. |
| Prices | OpenRouter's model list; a billed model with no price is not called | M1's rule: unknown models priced above frontier rates | M1's rule would have charged every free OpenRouter model at $15/$75 per million, so the caps could stop a run that cost nothing; refusing an unpriced model is safe both ways. |
| Model dates | OpenRouter's listing date as the release date | Hand-written dates for every hosted model | A model cannot be listed before it exists, so the listing date only excludes more backtest questions. |
| Limits | Measured by `augur models probe --burst`, recorded in the ledger | Documented numbers | The free limit turned out to be per key across models, enforced loosely above the advertised 20 a minute, plus a separate upstream limit with no headers. |
| Ollama | Its native API | Its OpenAI-compatible `/v1` | `/v1` cannot set the context window per request and silently drops the start of long prompts (M1). |
| Source dates | Evidence only (revision, capture, bounded publish date) | Page metadata | Audits show self-reported dates leak. |
| Ledger | SQLite, append-only by trigger | DuckDB, PostgreSQL | Tens of thousands of rows, one writer, ships with Python. |
| Aggregation | Binary: median of 5 draws. Multiple choice: per-option median, renormalised. Numeric and discrete: the mean of the members' CDFs | Mean; median of each percentile | The median is robust to one wild draw. A mixture of valid CDFs is valid by construction and keeps the members' disagreement as spread instead of inventing a confident consensus. M7 measured the alternatives on nine draws: trimmed mean and mean of log-odds tie with the median, extremizing loses 0.027 Brier, so the median stays. |
| Percentiles to a CDF | Linear between six stated percentiles in Metaculus's unscaled space, exponential tails, isotonic repair of a non-monotone answer, then the least uniform mixing that clears the server's minimum step | Asking the model for 201 numbers; a parametric fit | The server's rules are hard constraints (minimum and maximum step, exact or open ends); building to them, and checking against an independent transliteration in the fake server, means no live forecast is rejected. |
| Deadlines | Each question stops at its close less 5 minutes (never past the run's own deadline); draws still running are cancelled and charged the worst case; the finished ones are posted | One timeout for the whole run | A MiniBench question is open for 3 hours; a forecast posted after close scores nothing, and a cancelled request may still be billed. |
| Schedule | Cron every 90 min on two offset lines, plus a self-dispatch chain (`AUGUR_CHAIN_MINUTES`) once public | Cron every 20 to 30 min, as many bots use | Measured: GitHub dropped about 95% of short-interval ticks on public bot repositories, and every 30 min would bill more than the shared allowance while private. A `workflow_dispatch` from the job's own token is not subject to the scheduler. |
| Market prices | The market's own API as of the cutoff (Manifold's last bet, Polymarket's last history point, Kalshi's last candle), filtered client-side too; dated by that trade or point | ForecastBench's published freeze value; today's price | A price is a source like any other and must pass the leakage gate. ForecastBench's own value sometimes matches a price from hours after the freeze (M5). |
| Using a price | Shown to the model as a source; blended (⅓ Augur, ⅔ price in log-odds, a weight fixed in advance) only with the question's own community prediction | Blending any similar market; fitting the weight | A searched market may ask something different. A fitted weight on 101 questions would be tuned to them. Both turned on only because rules stated before the run held. |
| Live record | Each scheduled run pushes its ledger as JSONL to a `ledger` branch, one new file per run | Actions artifacts | Artifacts expire; M5's replay evaluation needs every live source forever. |
| Submit floor | Live tournaments need a frontier-tier hosted model | Submit whatever model answers | An expired key must never post small-model forecasts. |
| Comments | Private, rendered from the members' own reasoning | Boilerplate | Prize rules require the reasoning that produced the forecast. |
| Template baseline | Metaculus's real `TemplateBot` (optional `baseline` extra), unpublished, every role on Augur's model | Reimplementing its prompt | "Beats the template" must not mean "beats my copy of it", and the model must be the same so the difference is the pipeline. |
| Comparisons | Paired per question, bootstrap over clusters (question, opening week), 10,000 resamples, seeded; the detectable effect printed beside every interval | Unpaired means, or a t-test | Pairing removes question difficulty; questions that move together are not independent; the detectable effect keeps "no difference" apart from "could not tell". |
| Checking the scores | Against Metaculus's own `score_math.py` run on fixed cases (outputs committed) | Against the API's per-question scores | The API hides resolutions and scores from a bot token (0 of 80 resolved posts). The code check found an open-tail bug that hand-computed cases had not. |
| Judging a disagreement policy | Against constant shrinkage, the source-aware map and (for skip) random skipping, all fitted leave-one-out on the same questions | Against raw | Raw is overconfident everywhere, so any shrink beats it: a constant shrink did (−0.007 [−0.014, −0.000]), and skipping the most-disagreed quarter gained +7.3 points against raw scoring but +0.8 [−3.0, +5.1] against random skips. |
| Calibration | A Platt map fitted on the free model's resolved backtest history, applied only when every member answered from that model, before the exact-price blend, and withheld from Metaculus questions until a map is shown to be no worse than raw there | One map for every model; isotonic; no calibration; a per-source map | Calibration belongs to a forecaster: the free model's log-odds are worth about a third of what it states (slope 0.32), which says nothing about a frontier model. Platt and isotonic both passed the rule on 671 walk-forward questions; Platt by more, with two parameters instead of a step per block. |
| Changing a default | Only if the paired interval against the default lies below 0, or a cheaper variant's upper bound is under +0.01 Brier (fixed before M7's run) | Adopting whatever scores best on the backtest | At 40 questions most differences are noise; a rule written first keeps a lucky draw out of the live bot. |
| Release | PyPI by trusted publishing from a `v*` tag, behind a `pypi` environment; the publish job is the only one with `id-token: write` and never runs on a manual dispatch | An API token in a repository secret | No long-lived credential to leak; a dry run (`workflow_dispatch`) exercises every step but the upload. |
| README numbers | A test fails if the README or the postmortem states a number RESULTS.md does not | Trusting the copy | The README is what an interviewer reads; six of its numbers had drifted from RESULTS.md before the test existed. |
| RESULTS.md | Prose plus blocks rendered from the committed ledgers; a test fails if the file differs from a fresh render | Hand-copied tables | A number in the report cannot drift from the data it came from. |

## Results

Details, commands and raw output are in [RESULTS.md](RESULTS.md).

| Run | Outcome |
|-----|---------|
| M16, the newer free reader against the current route on the same ForecastBench questions (2026-11-08 and 2026-11-22 sets, one draw, no retrieval, $0; progress reading of 2026-11-26) | On the 15 questions resolved so far the newer route is +0.033 Brier worse with an interval from −0.073 to +0.147: not decidable, and a 0.01 difference would need about 3,800 questions. It is twice as fast (median 19.3 s a call against 40.1 s). Both trail the freeze price on the same questions (0.279 against 0.431 and 0.464). The rule binds on 2027-01-11 |
| M15, ForecastBench's dataset questions with no model (4,398 instances, seven sets 2026-07-05 to 2026-09-27, $0; a backtest row in raw Brier, not a Brier Index) | The source's walk-forward base rate scores 0.2200 against 0.25 for always 50% and 0.2795 for ForecastBench's Naive Forecaster; the series arm on stocks adds nothing (+0.003), and no forecaster, the leaders included, beats 0.25 on stocks. The leaders score 0.155 and 0.129 and the no-model entry stat-v1 0.1739 on their instances, ahead of the base rate by 0.047 to 0.103: on Wikipedia rankings and, for hybrid ensemble 1 and stat-v1, above all on daily temperatures (their largest single source of gain), where a rate built over spring points the wrong way in autumn. Three of six predictions held; the base rate's gain over 50% missed its registered margin by 0.00001, so the follow-up blend does not open |
| M14, the pooled Q1 analysis, registered 2026-11-25 before the 2026-11-30 round (P47 to P51) | Nothing to pool yet: the shadow schedule has not run on GitHub (no `ledger` branch; the Tournament workflow's 35 ticks since 2026-11-19 were all cron, median gap 220.4 minutes, against a question's three-hour window). At the assumed spread of 32 points, a +5 lead needs 254 scored questions and +2 needs 1,583. Synthetic (seeded, no real data): re-reading the one-round floor rule after each of four 60-question rounds admits a no-lead candidate in 11.8% of seasons, and reading each round alone in 19.5%, against a nominal 5% (P50 and P51 held). P47 to P49 are read on 2027-01-11 |
| M13, dated news in the leak-free backtest (C2's 64 questions, the same free model, three draws, $0): arm O's sources plus at most three news sources a question (GDELT discovery bounded at the cutoff, text only from a Wayback capture at or before it) | 54 of 64 questions gained news seen in the 30 days before open (39 counting real articles only; one wire-digest page matched 26), and the judge rated none of the 99 news uses as leaking. Adding it did not help: G − O +0.006 [−0.010, +0.022] Brier, which rules out a Hindcast-sized gain (−0.055), so the registered verdict is *not the limit*. Four of six predictions failed (coverage was better than predicted, and there was no small gain). Discovery ran on a GKG title index, because GDELT's API refused this machine |
| M12, the statistical arm and the registered blend on C4 (671 walk-forward instances, $0, no model call); all four predictions held | Where the bot could read a price (480 instances), the primary candidate beats the model's raw forecast by −0.110 [−0.137, −0.084] Brier, and the price alone is still better than the blend (+0.023 [+0.008, +0.038]). On the 190 Metaculus-written instances, where there is no price, the base rate alone is worse than the raw model (+0.063 [+0.017, +0.107]), so the candidate leaves Augur's forecast unchanged there. On the 2026-11-16 round's 60 questions the series rule reaches 7 and forecasts 5; the live comparison is the 2026-11-30 round (To measure) |
| M10, the season so far (Q5): every MiniBench leaderboard read 2026-11-23, Fall FutureEval re-read 2026-11-25 | Augur has no entry in any round (0 scored forecasts; the floor refused the free models). The template on Augur's free model, `metac-nemotron-3-ultra+asknews`, averaged −7.66 [−14.51, −0.73] in MiniBench 2026-11-02, −2.55 in 2026-10-19 and −12.19 so far in Fall FutureEval, where rank 1 (Panshul42) averages 20.91 on 51 scored questions. Final ranks are re-read daily by the `Standings` workflow |
| M10, ForecastBench same-question row: Augur's backtest arms against every entrant's published forecasts on the same question instances (C1, C2, O, C4; $0, no model call) | Without retrieval the free model is an ordinary zero-shot LLM: on C2 it ranks 13 of 23 among ForecastBench's zero-shot runs shown no price (median difference +0.002 Brier), on C4 59 of 78, or 34 of 53 without ForecastBench's web-search runs (each entrant on only the sets it entered, and ForecastBench's runs renamed in July count as two entrants). Shown the price, 17 of 22 of those models beat it with an interval above 0. Researched as of open (64 questions), every one of the 16 external entries beats it, by a median of +0.083. A backtest placing, not a leaderboard rank |
| M9, does the fleet know when it does not know (Q6): draws already in the ledgers, 135 and 39 ForecastBench market questions, `nemotron-3-ultra:free` (and `nemotron-3-nano:free` for the model gap); every policy against constant shrinkage and the source-aware calibrator, fitted leave-one-out ($0) | No disagreement measure predicts a large error: draw spread AUROC 0.507 [0.405, 0.608] on 135 questions; research passes 0.53, the two-model gap 0.33 on 39. A plain constant shrink beats raw (−0.007 [−0.014, −0.000]); scaling it by disagreement adds −0.005, inside the interval. Skipping the most-disagreed quarter gains +7.3 points against forecasting everything but +0.8 [−3.0, +5.1] against skipping at random. Not resolvable at this n (an AUROC of 0.6 needs about 250 questions); the pre-registered label proved to mark hedged forecasts; the live test is registered by question-id parity and has 0 resolutions |
| M8, calibration learned from resolved history (Q4): 926 ForecastBench market questions over nine sets, `nemotron-3-ultra:free`, no retrieval; each question calibrated only on questions resolved before it opened ($0, 802 free calls) | By source first: the gain is all on Polymarket and Manifold (−0.041 and −0.051 Brier against raw); on the 190 Metaculus questions, the nearest to what the bot forecasts live, pooled Platt is 0.035 *worse* than raw [0.013, 0.054]. Source-aware maps (per-source, or partially pooled with a shrunk per-source intercept) cut that to +0.014 and +0.019 but are not shown to be no worse than raw (upper bounds +0.032, +0.043), so the live map is withheld from Metaculus questions and live forecasts go out raw. Overall, on 671 walk-forward questions, Platt is 0.022 better than raw [0.007, 0.037] and cuts calibration error 0.152 → 0.069; extremizing by 1.5 hurts (+0.015). Outcomes are dated by the markets' own APIs (ForecastBench's dates were close dates for 239 of the 536 markets) |
| M7, what carries the score (Q3): the first 40 of the same sample, `nemotron-3-ultra:free`, one factor at a time against the default, paired on the same questions ($0) | A second research pass on sub-questions: −0.0003 Brier [−0.025, +0.022] for 1.8 more calls a question. One, three, five or nine forecasters: within 0.002 of each other. Trimmed mean and mean of log-odds: within 0.0014 of the median. A smaller second model mixed in: +0.006 [−0.015, +0.026]. Extremizing by 1.5: 0.027 worse [0.009, 0.048], the only resolved effect. The live default is unchanged |
| M7, how many questions a difference needs: a planted improvement in resampled per-question differences, 1,000 studies per n | 80% power on 0.01 Brier at about 450 questions (research-change spread) to 1,340 (retrieval against none); 5 head-to-head points at about 270; FutureSearch's 0.010 lead at n = 1,810 is consistent with either. At 40 questions this backtest resolves only differences of about 0.03 Brier or more |
| M6, what leakage is worth (Q2): ForecastBench 2026-06-21, a fixed sample of 64 resolved questions, the same free model in every role, three draws, one shared query plan, Wikipedia revisions and Wayback captures ($0) | Brier: no retrieval 0.205; sources as of each question's open date 0.189, no better than none (−0.016 [−0.050, +0.017]); the same searches as of today 0.063, 0.142 better [0.074, 0.217], about the gap El Lahib et al. published (0.244 to 0.108). On the 47 priced questions, read as of open the bot trails the market price by 0.093 [0.013, 0.176]; read as of today it draws level with it (−0.046 [−0.117, +0.022]). A leaky backtest would have erased a real gap to the market |
| M6, a bounded research loop (3 searches, a written belief and open question after each) against the one-pass brief, the first 40 of the sample | 0.026 Brier worse [−0.005, +0.060], with 6 model calls a question instead of 4; not adopted (the replacement rule needs an interval below 0) |
| M5, the price alone against the bot: ForecastBench 2026-06-21, 101 resolved Polymarket and Manifold questions, the free model that answers live (`nemotron-3-ultra:free`), three draws, each price read as of the freeze ($0) | Brier: the market price alone 0.106; Augur without it 0.240; Augur shown the price 0.143, better than without by 0.097 [0.054, 0.144] but still 0.037 [0.003, 0.073] worse than the price it was shown; a fixed blend (⅓ Augur in log-odds) 0.129. On these questions the price alone beats every version of the bot, so its score is what the price gives it |
| M5, how often a live question has a market: 59 questions of the 2026-11-16 MiniBench round, judged by hand | 2 of 59 had a market about the same event; 0 showed the community prediction to the bot's token |
| M4, ForecastBench 2026-05-24, 139 resolved market questions, `qwen2.5:7b` with no retrieval, one draw, local CPU ($0) | Brier 0.2395 [0.1929, 0.2895]: no better than always 50% (0.2500) or the base rate (0.2406); the market price it never saw scores 0.1311, 0.108 better [0.058, 0.161]. Without sources a small model is a constant; not comparable to published ForecastBench systems |
| M4, scoring against Metaculus's own `score_math.py` (pinned commit) | Spot baseline and spot peer scores match to 1e-9 on 8 cases of every type, open tails included, after fixing one bug the check found (an out-of-range outcome scored 34.7 points too high) |
| M4, what a bot token sees of resolved questions | 0 of 80 resolved posts (two MiniBench rounds, Fall and Spring 2026) show a resolution, an aggregate or a score |
| M4, MiniBench 2026-11-02 standings (435 entries, 59 questions) | Rank 1 averaged 22.77 spot peer points a question, the best template bot 3.66, the community aggregates (of the bots themselves in MiniBench) 5.06 to 7.56, the median bot 50.0 summed over 19 questions; Augur has no entry (the floor has kept it out until a frontier model answers) |
| M4, shadow template on 3 bot-testing-area questions (same free model as Augur, unpublished, $0) | All three recorded; close to Augur's own forecasts on each; none resolved yet |
| M3, every question type in the bot-testing-area (free OpenRouter models, $0) | 11 of 11 forecasts accepted by the live API over two runs on the practice area's six questions (2 numeric, 2 discrete, 1 multiple choice, 1 binary; HTTP 201 each), each with a private comment |
| M3, MiniBench windows over 16 rounds | 954 questions, 47% not binary; every window since June exactly 3.00 h; median 0.67 h between opens |
| M3, GitHub's cron on 55 public bot repositories (2,813 scheduled runs) | Crons every 20 min or less ran on 5.0% of ticks (median 5.5 h between runs); the two on-time workflows: delay p50 9 min, p99 108 min |
| M3, cadence simulated on the measured windows and delays | Every 90 min catches 99.25% of 3-hour windows for about 979 billed min a month while private; throttled as most forks are, no cron catches over 43%, hence the dispatch-chain fallback |
| M2, OpenRouter's free-model limit, three burst sessions | Per key across all free models, fixed clock minutes; 24 to 25 accepted against an advertised 20 |
| Leakage and floor scan of every committed ledger | 0 violations |

Q1 (does Augur beat the same-model template?) is not yet resolvable: no
live question with both forecasts has resolved. Q5 (where it stands) has no
reading for Fall 2026: Augur has no scored forecast in any round, so it has
no rank and no prize; the [postmortem](docs/postmortem-fall-2026.md) says
why and what the season's backtests found. The decision rule is
committed in PLAN.md; at an assumed per-question spread of 32 points (from
the Spring 2026 winner's interval, not yet measured on Augur) a 5-point
difference needs about 300 resolved questions.

## Known limits

- Wikipedia's search ranks today's index, so *which* pages are chosen can
  reflect later events even though each page's *text* is the revision as of
  the cutoff. Old revisions are also rendered with today's templates. M6's
  judge audits what the forecasters were shown for the residual.
- M6 is one free model on one ForecastBench set. The Wayback archive refused
  or timed out on many connections from the build machine, so most backtest
  sources are Wikipedia revisions. Past Metaculus tournaments cannot be a
  backtest corpus with a bot token (it sees no resolutions).
- No frontier model has answered yet: the only key here has a $1 limit that
  may not be the sponsorship, so only free models were called. Until a
  frontier model answers, the floor keeps the bot out of scored
  tournaments, so no MiniBench forecast has been posted. The tier was
  reviewed in M3 (GPT-5.4 and Claude Opus 5 admitted on cited results).
- Groq, Gemini and GitHub Models are configured but not exercised (no keys).
  AskNews is wired but unverified without keys.
- Date questions are listed and skipped (MiniBench had none in 16 rounds).
- The scheduled workflow has run under `act` and against the live listing,
  not yet on GitHub: the repository is private and the ledger-branch push is
  proven only against a local bare remote.
- A bot token sees no resolution on scored tournaments, so live questions
  are resolved by hand (`augur resolve --manual ... --evidence URL`) until
  Metaculus shows them; scoring itself is checked against Metaculus's code.
- The ForecastBench number uses one forecaster draw (CPU time), market
  questions only, and no retrieval; it is not comparable to published
  ForecastBench results with search and frontier models.
- The M5 replay is one free model on ForecastBench market questions, where
  every question is itself a market: the best case for prices. Live, most
  MiniBench questions have no market (2 of 59), and a bot token sees no
  community prediction, so the blend will rarely fire. Research depth waits
  on resolved live questions (none yet).

- The calibrator is fitted on one free model with no retrieval. Live
  forecasts are researched, and scored ones come from a frontier model
  under the floor, so the live map applies only where its own model
  answers, and not on Metaculus questions, where no map has yet been shown
  to be no worse than raw; a frontier model's forecasts stay raw until it
  has resolved history (the release gate keeps it off C4's earlier sets).
- Spring 2027's FutureEval project id is not known yet; the schedule polls
  Fall 2026 (`33121`) until `AUGUR_FUTUREEVAL` is set. That variable steers
  the scheduled poll and dispatches only: `augur forecast --tournament
  futureeval` and Q6's live count still mean `33121`, so pass the Spring id
  itself (`--tournament <id>`) and add it to `q6.LIVE_PROJECT_IDS`.
- The ForecastBench same-question row compares a backtest with live
  entries: the entrants forecast in real time with their own tools, so it
  places the model and harness, it does not rank Augur on ForecastBench.
- No demo GIF is committed; `docs/demo.tape` records one with `vhs`.
- Q6's backtest is 39 to 135 questions of one free model on market
  questions; an AUROC of 0.6 needs about 250 questions to clear 0.5. Its live half waits on resolved shadow-round or
  scored questions.

## Roadmap

Spring 2027: the maintainer approves a floor route (the proposed rule and
the cost of a round on each admitted model are in RESULTS.md), sets
`AUGUR_FUTUREEVAL` to the Spring project and makes the repository public so
the dispatch chain runs; the shadow template runs beside every round, and
Q1, Q5 and Q6's live half are read from the ledger branch as rounds resolve.
The first experiment on a frontier model is the agentic research loop,
the largest outside effect Augur has not shown on its own questions.

Milestones: M1 (done) one question end to end; M2 (done) one OpenAI-compatible provider with probed model chains, and the
repository public at its merge; M3 (done) every question type, a deadline
per question, the scheduled workflow and the durable ledger; M4 (done) scoring checked against Metaculus's code, the shadow template, standings and the
first scored number; M5 (done) market
prices as of the cutoff and the replay evaluation against the price; M6 (done) the leakage-proof
backtest, what leakage is worth, and a first research loop; M7 (done) the fleet and its ablation, with the power curve; M8 (done) calibration from resolved history; M9 (done)
"does it know when it does not know", against controls for overconfidence; M10 (done) release (package, container, release workflow), the
season's standing, the ForecastBench same-question row and the postmortem; M11 (done) an unattended
shadow round against the same-model template, and the rule that decides whether the free model may
be submitted; M12 (done) a forecast with no model in it and one registered blend; M13 (done) dated news in the
backtest, and the same-model agent-against-template pairs in FutureEval; M14 (done) a pooled analysis of the shadow
rounds, registered before the first one opens; M15 (done) ForecastBench's dataset questions with no model, against the
leaderboard's own baselines on the same instances; M16 (done) a newer free model on the same questions, under a rule
for choosing the default route (its registered reading is 2027-01-11). Details in
[PLAN.md](PLAN.md).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

MIT
