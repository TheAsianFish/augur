# Results

Every number here comes from a command in this repository, with the model,
date, cost and raw output recorded. Blocks between `augur:begin` and
`augur:end` comment markers are rendered by `augur report` (`make
results`) from the committed ledgers and snapshots, and a test fails if the
committed file differs from a fresh render. M5 sets the bot against the
market price on resolved ForecastBench questions (the price alone wins); M4
has the first scored number and the scoring it rests on; the first live
score waits on a scored round (see Q1 and Q5 below).

## M5: market prices, and the bot that ships against them (2026-11-19)

**Question.** How much of a forecast's score could a copier get by reading
the market, and does Augur add anything to it? The question a quant asks
first: "how does yours compare to just reading Polymarket?" The arms,
corpora, predictions P1 to P6 and the decision rules were committed to
PLAN.md (`5c2e32e`) before any run.

**Method.** `augur backtest prices` reads each ForecastBench market
question's price from its market's public API as of the freeze instant
(Manifold's last bet, Polymarket's last hourly price-history point) and
records it as a dated market source; the leakage scan covers it like any
other source. Arm (b) is the bot's own forecaster with no sources
(`augur backtest forecastbench`), arm (d) the same forecaster with the price
as the brief's only source (`--prices`), arm (a) the price itself, arm (c)
the fixed blend. C1 reuses M4's `qwen2.5:7b` run (one draw) and adds prices;
C2 is new: 138 resolved market questions frozen 2026-06-11, three draws on
`nvidia/nemotron-3-ultra-550b-a55b:free`, the free model that answered most
live forecasts in M3, listed on OpenRouter 2026-06-04 (before the freeze, so
the release gate admits it). Arm (b): 419 calls (4 failed; three draws
parsed on 135 questions, two on 3), 420,556 output tokens, 2.2 h of model
time; arm (d): 303 calls (1 failed; three draws on 100 questions, two on 1),
307,086 output tokens, 1.6 h; $0,
2026-11-19 (`data/runs/m5-replay-c2-b.txt`, `-d.txt`; one question was
re-run after its first attempt lost two draws to the token cap). The table
is rendered from the ledgers by `make results` and re-scored by `make eval`.

<!-- augur:begin replay -->
| Corpus | Model (arm b) | Draws | Questions | Exact price as of the freeze | Within 0.01 of ForecastBench's freeze value |
|---|---|---|---|---|---|
| C1 | `qwen2.5:7b` | 1 | 139 | 105 (infer 0/6, manifold 44/44, metaculus 0/28, polymarket 61/61) | 84/105 (80%) |
| C2 | `nvidia/nemotron-3-ultra-550b-a55b:free` | 3 | 138 | 101 (infer 0/5, manifold 45/45, metaculus 0/32, polymarket 56/56) | 91/101 (90%) |

| Arm | C1 | C2 |
|---|---|---|
| (a) price only | 0.1362 [0.1034, 0.1705], log -0.418 (n=105) | 0.1062 [0.0742, 0.1417], log -0.354 (n=101) |
| (b) Augur, no price | 0.2760 [0.2170, 0.3360], log -0.860 (n=105) | 0.2401 [0.1746, 0.3094], log -0.860 (n=101) |
| (c) blend, ⅓ Augur in log-odds | 0.1462 [0.1137, 0.1807], log -0.453 (n=105) | 0.1291 [0.0904, 0.1721], log -0.420 (n=101) |
| (d) Augur, price shown | not run | 0.1432 [0.0978, 0.1940], log -0.463 (n=101) |
| (b) on every question, priced or not | 0.2395 [0.1922, 0.2893] (n=139) | 0.2176 [0.1663, 0.2708] (n=138) |

| Paired Brier difference [95% CI] (MDE) | C1 | C2 |
|---|---|---|
| (b) − (a): Augur minus the price | +0.1398 [+0.0736, +0.2052] (0.0937) | +0.1338 [+0.0726, +0.1980] (0.0900) |
| (c) − (b): blend minus Augur | -0.1298 [-0.1786, -0.0822] (0.0689) | -0.1110 [-0.1567, -0.0685] (0.0634) |
| (c) − (a): blend minus the price | +0.0101 [-0.0130, +0.0318] (0.0316) | +0.0229 [-0.0013, +0.0471] (0.0350) |
| (d) − (b): showing the price |  | -0.0969 [-0.1437, -0.0536] (0.0645) |
| (d) − (a): Augur shown the price, minus the price |  | +0.0370 [+0.0030, +0.0729] (0.0498) |

Settings arms on C2, re-aggregated from (b)'s own draws (135 questions with three parsed draws; no new calls):

| Paired Brier difference against the median of 3 [95% CI] (MDE) | C2 |
|---|---|
| one draw - median of 3 | +0.0223 [+0.0040, +0.0440] (0.0283) |
| mean - median | -0.0017 [-0.0072, +0.0039] (0.0080) |
| mean log-odds - median | -0.0002 [-0.0052, +0.0050] (0.0073) |
| extremized x1.5 - median | +0.0076 [-0.0021, +0.0182] (0.0145) |

C3, live and dry-run questions in the committed ledgers with a resolution: 0. Research depth and the replay of live research wait on these.

Predictions stated in PLAN.md before the run:

- P1 (price alone beats Augur, interval above 0): C1 **held**, C2 **held**.
- P2 (blend beats Augur, interval below 0): C1 **held**, C2 **held**.
- P3 (blend worse than the price on C1; within ±0.01 on C2, interval including 0): C1 **held**, C2 **failed**.
- P4 (showing the price improves Brier by at least 0.03, interval below 0; the model still under-uses it): first part **held**, second part **held**.
- P5, first two parts (freeze value matched on ≥90% of priced C2 questions; an exact price for ≥90% of Polymarket and Manifold, 0% of Metaculus and INFER): 91/101 matched **held**; 101/101 priced **held**; 0 of 37 Metaculus and INFER priced **held**. The third part (MiniBench) is below.
  As first measured, with the seven-day staleness cut the price run applied before Manifold was exempted (a change made after seeing the data): 75/85 matched **failed**; 85/101 priced **failed**. So P5's first two parts failed as first measured and hold only after that change.
- P6 (one draw within 0.01 of three, interval including 0; mean of log-odds within 0.005 of the median; extremizing worse, interval including 0): **failed**, **held**, **held**.

Method: arms (a)–(d) are paired on the questions with an exact price (C1 has no arm (d)); every probability is clipped to [0.01, 0.99] before scoring; log is the mean natural log of the probability given to the outcome (higher is better). Intervals: paired bootstrap over questions, 10,000 resamples, seed 20261119; MDE is the true difference this n would detect with 80% power at the 5% level.
<!-- augur:end replay -->

**What it says.**

- **The price alone beats the bot, by a lot.** On C2's 101 priced
  questions the price scores 0.106 and Augur without it 0.240, 0.134 worse
  [0.073, 0.198]; C1 says the same with a 7B local model. Both are
  ForecastBench market questions, the best case for prices; most MiniBench
  questions have no market at all (below).
- **Showing the price helps, and the bot still under-uses it.** With the
  price as its only source Augur scores 0.143: 0.097 better than without
  [0.054, 0.144] (P4 held), yet still 0.037 worse than the price it was shown
  [0.003, 0.073]. The survey's r = +0.34 for "checking similar questions or
  markets" is a correlation across bots; this is the same direction measured
  as a paired difference on one bot.
- **The fixed blend recovers most of the gap without a second call.** One
  third Augur in log-odds: 0.129, 0.111 better than Augur alone (P2 held),
  0.023 worse than the price alone (P3's C2 half failed: the interval
  [−0.001, +0.047] includes 0 but the point estimate is outside ±0.01).
  On C1 the blend is 0.010 worse than the price (P3's C1 half held).
- **Settings.** One draw is 0.022 worse than the median of three [0.004,
  0.044] (P6's first part failed: three draws matter more than predicted).
  "One draw" is each question's first draw, a choice the plan did not
  state; the second or third draw alone is within 0.004 of the median, so
  this part of P6 rests on which draw is taken;
  mean, mean of log-odds and median are within 0.002 of each other;
  extremizing at 1.5 is 0.008 worse, interval including 0 (held).
- **Decision rules.** Both pre-stated conditions are met on C2: the
  interval of (d) − (b) and of (c) − (b) lie below 0. So `augur forecast`
  now runs the markets searcher by default, and blends a binary forecast
  with the question's own community prediction when the API shows it to
  the bot (`--no-markets`, `--no-blend` turn them off). In practice the
  blend will rarely fire on scored tournaments: a bot token sees no
  community prediction there (M4, and 0 of 59 in the coverage run below).

### How often a live question has a market (P5)

`scripts/market_coverage.py` ran the markets searcher on the 59 questions
of the 2026-11-16 MiniBench round as of each question's open time, then I
judged each top hit by hand (`data/runs/m5-market-coverage-labels.json`).
The first run (`m5-market-coverage-1.json`) found a priced hit for 5
questions, three of them matched only on a shared date ("November 26,
2026"); after dates stopped counting as shared words the second run
(`m5-market-coverage.json`) found 4. In both, the top hit was about the same
event for **2 of 59** (3%): the Alaska Senate race and La Jolla's peak water
level. P5's third part (under 25%) held. No question showed its community
prediction to the bot's token.

P5's first two parts also held on C2 (above), with one disclosed change:
the first price run cut prices older than seven days and found 85 of 101.
A Manifold market maker's price stands until the next bet, so the last bet
is the price however old; I checked one (`1y80gk7t9k`: last bet 12 days
before the freeze, equal to ForecastBench's value to 16 digits), removed the
cut for Manifold, and re-ran: 101 of 101. The
change was made after seeing the data, so the block above reports P5 both
ways: as first measured its first two parts failed (85 of 101 priced, 75 of
85 matched), and they hold only after the change. The seven-day window still bounds
the Polymarket and Kalshi history reads.

**ForecastBench's freeze value is not always an as-of price.** Where
Augur's as-of Polymarket price and ForecastBench's value differ by more
than 0.01 (9 questions on C2, 19 on C1), `scripts/freeze_value_timing.py`
looks for their value in the hourly history three days either side. On C2
the nearest exact match is after the freeze for 5 of 9 (1 to 6 hours); on
C1 it is most often a day or more before. Polymarket resamples history
per query, so this is evidence, not proof, but it means M4's "market"
reference row (built from ForecastBench's values) may read a few hours past
its cutoff, and the price arms here use Augur's own as-of reads instead
(`data/runs/m5-freeze-value-timing.txt`, `-c1.txt`).

**Beside the frontier, not against it.** AIA Forecaster's ablation on
ForecastBench (arXiv 2511.07678, self-reported) runs the same experiment
on a frontier model with agentic search (FB-Market, n = 76, Table 4): the
market price alone 0.096, the model shown the price without search 0.103,
no search 0.116, search 0.085, both 0.075. Its shown-the-price row is the
outside twin of arm (d): a frontier model shown the price without search
also failed to beat the price (0.103 against 0.096), as Augur's arm (d)
does here (0.143 against 0.106). On liquid markets the market beats
it (0.1258 against 0.1106) and a fixed ⅓ : ⅔ blend scores 0.106. C2's
direction matches that last row (the price beats the bot), but the gap is
far larger here (0.240 against 0.106) because arm (b) has no search and a
free model; the question sets differ, so none of these numbers is
comparable to C2's, and nothing here places Augur against AIA.

**Unfair or missing comparisons, named.** C2 is one free model, not the
frontier model the floor requires live; the replay of live research (C3,
and research depth with it) waits on resolved live questions; ForecastBench
market questions are a best case for price-only, since each question is a
market.

## M4: scoring, baselines and the first scored number (2026-11-19)

**Setup.** Scoring is Metaculus's (`augur/score.py`): the baseline score,
head-to-head and peer scores for every type, Brier and log score; paired
comparisons with a bootstrap that resamples questions (or opening weeks),
10,000 resamples, a fixed seed, and the minimum detectable effect (MDE) at
80% power and 5% two-sided. Everything below cost $0: the backtest ran on a
local model, hosted calls used OpenRouter `:free` models, and the
Metaculus and GitHub reads are free API calls.

### Scoring, checked against Metaculus's own code

`scripts/metaculus_score_vectors.py` runs Metaculus's `scoring/score_math.py`
(BSD 2-Clause, pinned at commit `8d7ac79`, its Django imports stubbed) on
eight fixed cases and commits the outputs; `tests/test_score_vectors.py`
requires Augur's spot baseline and spot peer scores to match them to 1e-9.

| Case | Forecasters | Matched |
|------|-------------|---------|
| Binary, resolved Yes and resolved No | 3 each | yes |
| Multiple choice, 3 options (one uniform) | 3 | yes |
| Numeric, closed bounds, inside the range | 3 | yes |
| Numeric, open bounds, inside, below and above the range | 3 each | yes |
| Discrete, 11 outcomes, closed bounds | 3 | yes |

The check found one bug before any number was reported: Augur split the
0.05 baseline mass between a question's two open tails, while Metaculus
gives each open tail 0.05. An out-of-range outcome on such a question
would have scored 34.7 points too high (100 ln 2 / 2). Hand-computed cases
for every type are in `tests/test_score.py`.

**Why not against the API's scores.** The plan was to compare with
Metaculus's score for a resolved question of each type. With this
account's token the API hides that: `scripts/resolution_visibility.py`
(output `data/runs/m4-resolution-visibility.txt`, 2026-11-19) read 20
resolved posts from each of four tournaments (two MiniBench rounds, Fall
2026, Spring 2026), and **0 of 80** showed a `resolution`, a community
aggregate or a score for this account; the account is a bot (`Augur-bot-1`).
So `augur resolve` counts such questions as *hidden* instead of guessing,
and the comparison with Metaculus's own per-question score is under To
measure. On open bot-testing-area questions the community aggregate *is*
shown: `augur resolve` kept it for all 6 questions in the M3 ledger
(`data/runs/m4-resolve-live.txt`).

### The tokenless ForecastBench number (no retrieval, `qwen2.5:7b`)

The bot's own forecaster prompt and median, no sources, the clock pinned to
each question's freeze date, on ForecastBench market questions that
resolved to 0 or 1 and were posed after the model's release (2024-09-19).
The market's price at the freeze date is a reference row only: the model
never saw it. Keeping only questions already resolved to 0 or 1 by the
resolution set favours markets that resolved early (often "by a date"
markets settled before it), which is a property of the sample, not of the
model.

<!-- augur:begin forecastbench -->
Question set `2026-05-24-llm.json` (ForecastBench, CC BY-SA 4.0): 500 questions; removed 250 not a market question (dataset questions are M6's); 111 not resolved to 0 or 1 in the resolution set; **139 scored** (139 binary, 40.3% resolved Yes).

| Forecaster | Brier [95% CI] | Baseline score | Augur minus it: Brier [95% CI] | MDE (Brier) | Augur over it: head-to-head [95% CI] | MDE (points) |
|---|---|---|---|---|---|---|
| **Augur**, qwen2.5:7b, no retrieval | 0.2395 [0.1929, 0.2895] | -7.9 [-30.9, 12.9] | | | | |
| Market price at the freeze date (not shown to Augur) | 0.1311 | 41.4 | +0.1084 [+0.0582, +0.1612] | 0.0740 | -34.2 [-50.9, -19.0] | 22.8 |
| Always 50% | 0.2500 | 0.0 | -0.0105 [-0.0571, +0.0395] | 0.0691 | -5.5 [-21.4, +9.0] | 22.0 |
| Base rate (40%) | 0.2406 | 2.7 | -0.0011 [-0.0432, +0.0427] | 0.0614 | -7.4 [-22.0, +5.8] | 20.2 |

Method: 1 backtest run(s) in `data/ledgers/m4-forecastbench-qwen.sqlite` (code 0.1.0.dev0+8e787a7faf8c); 1 draw(s) per question, median; clock pinned to each question's freeze date (2026-05-14 00:00 UTC); no sources. 139 model calls (0 failed), 32,185 output tokens, 3.1 h of model time on a 4-core CPU, cost $0.00. Outcomes: forecast 139. Intervals: paired bootstrap over questions, 10,000 resamples, seed 20261119; MDE is the true difference this n would detect with 80% power at the 5% level. The base rate is the scored set's own, known only after the fact.
<!-- augur:end forecastbench -->

**What it says.** Without sources, `qwen2.5:7b` in Augur's forecaster
prompt is no better than a constant: Brier 0.2395 against 0.2500 for always
50% and 0.2406 for the set's own base rate, and both paired intervals span
zero. The market's price at the freeze date, which the model never saw,
scores 0.1311; Augur is behind it by 0.108 Brier [0.058, 0.161], well past
the detectable effect at n = 139. So the run measured the floor a pipeline
starts from, not a forecaster: a small model's prior about events after its
training cut-off carries almost no information, and the price of a liquid
market carries most of what is knowable. That is the gap retrieval (M5's
market prices and news, M6's backtest) and a frontier model have to close.
For scale only, El Lahib et al. report 0.244 with no retrieval for
gpt-oss-120b on their own 93 questions (arXiv 2602.00758): the same
neighbourhood, on different questions and a different model, so not a
comparison. Nothing was predicted for this number before the run.

**One draw, not five.** The plan said "the M1 pipeline", five draws and the
median. On this 4-core CPU `qwen2.5:7b` decodes about the same number of
tokens a second whether one request runs or five run at once
(`scripts/ollama_throughput.py`, `data/runs/m4-qwen-throughput.txt`), so
five draws cost five times the model time in the block above (3.1 h),
about 15 hours. The run used `AUGUR_FORECASTERS=1`; the five-draw run is under To
measure. The outside evidence on what the median adds is small: AIA
Forecaster's ablation went from 0.1182 (one draw) to 0.1138 (median of
several), arXiv 2511.07678.

**Beside the frontier, not against it.** AIA Forecaster reports Brier
0.1076 against superforecasters' 0.1110 on ForecastBench's FB-7-21 set
(self-reported; agentic search and a frontier model). This run is **not
comparable** to that: a different question set (one 2026 set, market
questions only), no search, a 7B local model. ForecastBench's public
`forecast_sets` hold only the 2024-07-21 set, so no outside forecaster can
be scored on these same questions yet; that row is M10's.

### Live: Augur against the shadow template and the community

`augur baseline` runs Metaculus's real `TemplateBot` (forecasting-tools
0.3.7, `FallTemplateBot2026`) unpublished on questions Augur forecast, with
Augur's first forecast route for every template role, and records its
forecast as a `template` reference. First run, 2026-11-19
(`data/runs/m4-shadow-template.txt`, ledger `m4-shadow-template.sqlite`), on
three bot-testing-area questions Augur forecast in M3 with the same free
model (`nvidia/nemotron-3-ultra-550b-a55b:free`):

| Question | Augur (M3, posted) | Shadow template (M4, unpublished) |
|----------|--------------------|-----------------------------------|
| 43327 EU exit before 2027 (binary) | 1% | 1.0% |
| 46223 2028 presidential winner (multiple choice) | Democratic 48%, Republican 50%, Other 2% | Democratic 47.4%, Republican 50.2%, Other 2.4% |
| 43323 Trump net approval, 31 Dec (numeric) | median −22.1, 80% in [−32.5, −11.7] | 40th to 60th percentile −18.1 to −13.0, 80% in [−32.1, −2.0] |

The template had no news search (no AskNews keys here) while Augur's M3 run
read Wikipedia, and it forecast about 10 hours after Augur did (05:07 on
2026-11-19 against 18:46 the day before), so this pair is not yet like for
like; none of the three questions has resolved. The same caution applies to
the community aggregate `augur resolve` keeps: it is the latest one when
resolve runs, usually after Augur's forecast, so a head-to-head against it
favours the community. Each reference row records its `as_of` for that
reason; a live comparison should run the template at Augur's forecast time. The first try on `google/gemma-4-31b-it:free` got
HTTP 429 on all 15 template calls (an upstream shared pool).

<!-- augur:begin live-scores -->
Live questions in the committed ledgers: 6; posted by Augur: 6 (bot-testing-area 6). Reference forecasts recorded: community on 6, market on 0, template on 3. With a resolution in the ledger: 0; annulled 0; scored **0**.

Nothing is scored yet, so no interval and no detectable effect can be computed from data. Until there is a measured spread, the detectable effect at the sizes this season can reach, for a per-question spread of 32 points (PLAN.md, from the Spring 2026 winner's interval; a paired difference against the same-model template is likely tighter, and M4's first resolved round replaces this with the measured one):

| Resolved questions | Detectable mean difference (points per question, 80% power, 5% two-sided) |
|---|---|
| 30 | 16.4 |
| 60 | 11.6 |
| 120 | 8.2 |
| 300 | 5.2 |
<!-- augur:end live-scores -->

**Q1 (does Augur beat the same-model template?): not yet resolvable.** No
live question with both forecasts has resolved, and Augur has posted no
MiniBench forecast: the submit floor refuses until a frontier-tier model
answers (M3). The decision rule was committed to PLAN.md before any round
resolved: target +5 points per question; held if the paired interval is
above 0; failed if below 0, or if the point estimate is negative after both
remaining Fall rounds. At the 32-point spread above, a 5-point difference
needs about 300 resolved questions to be detected four times in five; Fall
2026 can give at most about 120.

### Standings

`augur standings` reads each round's primary leaderboard
(`/api/leaderboards/project/{id}/`) and keeps a snapshot
(`data/standings/*.json`); `augur report` renders from the snapshots.

<!-- augur:begin standings -->
#### MiniBench - 2026-11-02 (`minibench-2026-11-02`)

Finalized: no. Leaderboard entries: 435. Most questions any entry covers: 59. Score type: spot_peer_tournament. Fetched 2026-11-19T05:00:05.413788Z from https://www.metaculus.com/api/leaderboards/project/33135/.

| Row | Rank | Name | Kind | Questions | Avg spot peer score / question [interval] | Summed score |
|---|---|---|---|---|---|---|
| Rank 1 | 1 | takeoff.watch | bot | 59 | 22.77 [17.36, 29.05] | 1343.4 |
| Best template bot | 107 | metac-claude-opus-5-5-high+asknews | Metaculus template bot | 59 | 3.66 [-0.97, 7.94] | 216.0 |
| Best Metaculus house bot (not the template) | 3 | metac-autocast-experimental | Metaculus house bot (not the template) | 59 | 16.34 [10.92, 21.58] | 964.2 |
| Community prediction | 80 | community prediction (unweighted) | community prediction | 59 | 5.06 [4.08, 6.12] | 298.6 |
| Community prediction | 63 | community prediction (recency_weighted) | community prediction | 59 | 6.17 [4.77, 7.69] | 364.2 |
| Community prediction | 48 | community prediction (reputation_weighted) | community prediction | 59 | 7.56 [6.13, 9.07] | 446.2 |

- Augur (Augur-bot-1) is not on this round's leaderboard: it has no scored forecast here.
- Best listed open-source bot: no listed open-source bot in this round.
- Average = summed score / questions covered; the interval is Metaculus's interval on the sum divided by the same count (their method is not verified here). Rank is Metaculus's, by summed score; ties share a rank. Template variants (`+asknews[...]`, `+asknews(...)`) are Metaculus's experiments on the template. Open-source bots are matched against a short sourced list: nostreambot (https://www.lesswrong.com/posts/wZBbDqzfBjYG58CxK (Spring 2026 results, MIT)); Panshul42 (https://www.metaculus.com/notebooks/39140/ (Q2 2025 winner, open source)).

Median bot by summed score (the line Q5 predicts Augur clears): 50.0 over 19 questions, 2.63 a question (rank 177; the middle of 431 bot entries, many of which covered only part of the round, which a summed score punishes).

Questions in the round: 60; forecast and submitted by Augur: 0; not forecast: 60 no run saw it (from the committed ledgers; `augur standings --from data/standings/minibench-2026-11-02.json --ledger ...` lists each one).

#### MiniBench (`minibench`)

Finalized: no. Leaderboard entries: 0. Most questions any entry covers: 0. Score type: spot_peer_tournament. Fetched 2026-11-19T05:00:08.935168Z from https://www.metaculus.com/api/leaderboards/project/33169/.

The leaderboard has no entries yet (round starts 2026-11-16T00:00:00Z, closes 2026-12-04T00:00:00Z); there is nothing to compare.

Questions in the round: 60; forecast and submitted by Augur: 0; not forecast: 59 no run saw it; 1 forecast in a dry run, not submitted (from the committed ledgers; `augur standings --from data/standings/minibench.json --ledger ...` lists each one).
<!-- augur:end standings -->

**Unfair comparisons, named.** (1) Augur has no entry: there is nothing of
its own to compare yet. (2) The averages divide Metaculus's summed score
and its interval by the questions covered; Metaculus's interval method was
not checked. (3) In MiniBench the community prediction is an aggregate of
the bots themselves (`include_bots_in_aggregates` is true on these
questions), not of human forecasters, so it is not the human Community
Prediction of the Frontier table (14.8 among humans in Spring 2026).
(4) The template bots search AskNews and run on paid frontier models; the
house bots (`metac-agent-*`, `metac-autocast*`) are Metaculus's own agentic
systems, not the template. (5) Open-source bots are recognised only from a
short sourced list; none of them entered this round.

**Q5's first reading: not placed.** The prediction was "above the median
bot and above the best same-model template bot; below the top 10." With no
scored forecast in any round, Q5 has no reading; the line it would have to
clear is now measured (the median bot above), and the best template bot of
the 2026-11-02 round averaged 3.66 a question against rank 1's 22.77. One
round of about 60 questions gives an interval of about ±8 points on an
average (PLAN.md), enough to place the bot above or below the median, not
among the leaders.

### Scheduled runs on `main`

Not observed: the repository is still private and this environment has no
GitHub credentials, so the `ledger` branch could not be read. The commands
are under To measure; `augur ledger import` and `augur standings --ledger`
are what turn that branch into the coverage check (forecast, skipped,
missed, and why).

### Ledgers

<!-- augur:begin ledgers -->
| Ledger | Runs (modes) | Questions | Model calls | Forecasts (aggregate) | Posted | Resolutions | References |
|---|---|---|---|---|---|---|---|
| `m1-dryrun-samples.sqlite` | 1 (dry_run 1) | 1 | 7 | 1 | 0 | 0 | 0 |
| `m1-live-bot-testing-area.sqlite` | 2 (live 2) | 1 | 7 | 1 | 1 | 0 | 0 |
| `m2-live-bot-testing-area.sqlite` | 2 (live 2) | 1 | 16 | 2 | 2 | 0 | 0 |
| `m2-probe.sqlite` | 0 (-) | 0 | 0 | 0 | 0 | 0 | 0 |
| `m3-dryrun-minibench-2.sqlite` | 1 (dry_run 1) | 1 | 6 | 1 | 0 | 0 | 0 |
| `m3-dryrun-minibench.sqlite` | 1 (dry_run 1) | 1 | 9 | 0 | 0 | 0 | 0 |
| `m3-live-bot-testing-area-2.sqlite` | 1 (live 1) | 6 | 53 | 6 | 6 | 0 | 6 |
| `m3-live-bot-testing-area.sqlite` | 1 (live 1) | 5 | 43 | 5 | 5 | 0 | 0 |
| `m4-forecastbench-qwen.sqlite` | 1 (backtest 1) | 139 | 139 | 139 | 0 | 139 | 139 |
| `m4-shadow-template.sqlite` | 0 (-) | 0 | 0 | 0 | 0 | 0 | 3 |
| `m5-replay-c1-prices.sqlite` | 1 (backtest 1) | 139 | 0 | 0 | 0 | 0 | 0 |
| `m5-replay-c2-prices.sqlite` | 2 (backtest 2) | 138 | 303 | 101 | 0 | 0 | 0 |
| `m5-replay-c2.sqlite` | 4 (backtest 4) | 138 | 419 | 138 | 0 | 138 | 138 |
<!-- augur:end ledgers -->

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
  `max_tokens`); in the sixth, the first two models ran out of tokens with
  no answer text and the third returned a 429. Non-binary answers end in six
  lines rather than one, so run 2 gave them 2,500: 28 of 30 draws kept. No
  kept or recorded call ended at exactly the cap, but in both lost draws the
  first two models again ran out of tokens (reported as an error, so the
  ledger has no token count) before the third returned a 429: the cap still
  contributed to both losses.
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
| Window (scheduled close − open) | 3.00 h for all 729 questions from the 2026-06-05 change on (716 of them in rounds from 2026-06-15); 1.50 h for the 225 before |
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
scheduled runs land on the `ledger` branch (M4). Until the chain is on, the
cron is the only trigger, and the 99.25% rests on the delay tail of the only
two on-time workflows found (174 delays); at 90 minutes no question gets a
second tick in time, so a repository throttled like most forks would catch
about 43%. The M4 check of this repository's own run times decides between
those two numbers.

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
| This repository's own cron delay and coverage, and the first scheduled runs on `main` | Private, and no GitHub credentials here, so the `ledger` branch cannot be read | After merge, with `AUGUR_SCHEDULE=minibench`: `git fetch origin ledger && git worktree add /tmp/ledger FETCH_HEAD && augur ledger import /tmp/ledger/ledger --out /tmp/live.sqlite && augur standings --round minibench --ledger /tmp/live.sqlite` (forecast, skipped, missed and why per question); the run times are the `*.txt` files' timestamps against the two cron lines; billed minutes: `gh api repos/TheAsianFish/augur/actions/runs --jq '.workflow_runs[] \| select(.event=="schedule") \| [.created_at, .run_started_at, .updated_at]'` |
| Augur against the same-model shadow template and the community prediction on resolved live questions (Q1) | No live MiniBench forecast yet (the floor), and a bot token sees no resolution on scored tournaments | After each forecast run: `uv sync --extra baseline && augur baseline --ledger L` (same model as Augur, AskNews keys set so the template searches as it does on the leaderboard); after resolution: `augur resolve --ledger L` (or `--manual POST VALUE --evidence URL` read from the question page), then `augur score L data/ledgers/m4-shadow-template.sqlite --mode live --cluster week` |
| Augur's score beside Metaculus's own for one resolved question of each type | The API hides resolutions and scores from this bot account (0 of 80) | `augur resolve` records `my_forecasts.score_data` whenever the API returns it, and `augur score` prints each `check` line; with a personal (non-bot) token, `python3 scripts/resolution_visibility.py` first shows whether that account sees them |
| The ForecastBench number with five draws, as the pipeline runs live | About 15 hours of CPU on 4 cores (five times the measured 3.1 h) | `AUGUR_FORECASTERS=5 make backtest FB_LEDGER=data/ledgers/m4-forecastbench-qwen-5.sqlite`, then `make results` after adding the ledger to the report |
| The ledger-branch push and the dispatch chain on GitHub | Need the repository's own token | `gh workflow run tournament.yml --ref main -f tournament=bot-testing-area -f push_ledger=true`, then check the `ledger` branch; set `AUGUR_CHAIN_MINUTES=60` once public and check that consecutive runs are about 60 min apart |
| The replay evaluation (arms b and d on C2) on the frontier model the floor requires live | Nothing paid without the maintainer's say-so; C2 ran on the free model that answered live in M3 | `AUGUR_FORECAST_MODELS=openrouter:openai/gpt-5.4@high AUGUR_FORECASTERS=3 augur backtest forecastbench --questions data/forecastbench/2026-06-21-llm.subset.json --resolutions data/forecastbench/2026-06-21_resolution_set.subset.json --ledger data/ledgers/m5-replay-c2-gpt54.sqlite`, then the same with `--prices data/ledgers/m5-replay-c2-prices.sqlite` (into a copy of that ledger). C2's free run used 430,222 input and 727,642 output tokens over 717 draws: about $12 at GPT-5.4's listed $2.50/$15 per million, more at high effort. The listing date (2026-03) is before the 2026-06-11 freeze, so the gate admits it |
| Research depth, and the similar-market (search) half of the markets searcher, on replayed live research (C3) | No live question with an Augur forecast has resolved (C3 = 0); C1 and C2 have no retrieval, and their price is exact by construction | Once live questions resolve: `augur resolve --ledger L` (or `--manual`), then the replay arms on the ledger's recorded sources (M7's harness) |
| Groq, Gemini and GitHub Models free tiers | No `GROQ_API_KEY`, `GEMINI_API_KEY` or `GITHUB_TOKEN` here; the presets are configured, not exercised | `AUGUR_FORECAST_MODELS=groq:<model> augur models probe --burst 30` (and `gemini:`, `github:`) |
| Tokens-per-minute limits | OpenRouter sends no token-limit header and the bursts are tiny | the same burst on Groq, which sends `x-ratelimit-limit-tokens` |
| AskNews sources | No `ASKNEWS_CLIENT_ID`/`ASKNEWS_CLIENT_SECRET` in the build environment; the searcher is tested only against recorded payloads | `make test-integration` with both set, then `augur forecast --tournament minibench` |
