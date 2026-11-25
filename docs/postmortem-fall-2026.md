# Augur, Fall 2026: a season postmortem

Written 2026-11-23, at the release, as promised on Metaculus's credits form:
what carried the score and what leakage was worth. The season is not over
(Fall FutureEval and the last MiniBench rounds are still open), so the final standings will be the `Standings`
workflow's last readings on the `ledger` branch. Every number below is in
[RESULTS.md](../RESULTS.md) with its command, model, date and cost; a test
fails the build if this file states one that RESULTS.md does not.

## The result: no score

Augur has no scored forecast in any round. The bot works end to end: it
posted every question type in Metaculus's practice area (11 of 11 accepted,
with private comments) and a fresh clone runs a dry run on a local model.
What kept it off the leaderboard is a rule I wrote on purpose: `--submit`
to a scored tournament refuses unless a frontier-tier model answers, and the
only key this season reported a $1 limit, so only free models were called
and the floor refused (exit 2).

Was that the right call? The outside evidence says skipping was cheaper
than posting. Metaculus runs its own template bot on the free model Augur
used, and it lost points everywhere it played: −7.66 [−14.51, −0.73] a
question in MiniBench 2026-11-02, −2.55 in 2026-10-19 and −12.19 so far in
Fall FutureEval (read 2026-11-25). That is not proof that Augur would have lost too, but
it is the bar a free-model pipeline had to clear, and nothing in the
backtests says Augur clears it.

## What carried the score

On resolved questions, in order of size:

1. **The market price.** On 101 ForecastBench market questions the price
   alone scored a Brier of 0.106. Augur without it scored 0.240; shown the
   price, 0.143, still 0.037 [0.003, 0.073] behind the price it was shown.
   Its own forecast mixed one-third into the price in log-odds scored 0.129.
   ForecastBench's own entrants tell the same story on all 138 of that set's
   market questions: 17 of 22 zero-shot models shown the price beat Augur's no-retrieval arm
   with an interval above 0. On a market question the price is the forecast,
   and live it rarely exists: 2 of 59 MiniBench questions had one.
2. **The model and its harness.** Without retrieval, the free model is an
   ordinary zero-shot LLM: level with ForecastBench's median zero-shot run
   on the same 138 questions (Augur ranks 13 of 23). Researched as of each
   question's open date, it still trails every one of the 16 external entries,
   by a median of +0.0834 Brier. On MiniBench, Metaculus's agentic house bots on frontier models
   with live search beat their own template on the same model by 7.7 to
   16.2 points a question. Augur's research
   is template-shaped (one plan, one brief), and the bounded loop I tried on
   the free model was 0.026 Brier worse than one pass [−0.005, +0.060].
3. **Calibration, by source.** A Platt map fitted only on outcomes known
   before each question opened improved Brier by 0.022 [0.007, 0.037] on
   671 questions, all of it on Polymarket and Manifold. On the 190 Metaculus
   questions, the kind the bot forecasts live, the same map was 0.035
   worse than raw, so it is withheld there and live forecasts go out raw.
4. **Fleet settings: little.** On M7's 40 questions, two research passes,
   one to nine forecasters, trimmed or log-odds means and a second model
   all sat within noise of the default; extremizing by 1.5 cost 0.027
   [0.009, 0.048]. On C2's 138 questions, one draw was worse than the
   median of three (+0.0223 [+0.0040, +0.0440]), so a few draws matter
   and more than three did not measurably help. Does
   the fleet's disagreement predict its errors? Not at this n: the draws'
   spread scored an AUROC of 0.507 [0.405, 0.608].

## What leakage was worth

The same forecaster, the same 64 questions and the same planned searches,
read once as the pages stood when each question opened and once as they
stand today: Brier 0.189 against 0.063, a difference (today minus open)
of −0.1259 [−0.1930, −0.0652] from reading the future. Against no retrieval at all (0.205), today's pages
are worth 0.142 [0.074, 0.217], close to the gap El Lahib et al. published
on Google results (0.136), under a different protocol (their leaky arm
keeps only documents a judge rated as leaking). Read as of open, research was worth nothing measurable
over no retrieval (−0.016 [−0.050, +0.017]); read as of today, the bot
appears to draw level with the market (−0.046 [−0.117, +0.022]) when as
of open it trailed by 0.093 [0.013, 0.176]. A judge reading every source
the forecasters saw flagged major leakage on 40 of the 64 questions as
of today and on 1 of 64 as of open. A leaky backtest would have told me the
bot was as good as the market. It is not.

## What I would do differently

- **Run the $0 shadow round from the first live milestone.** Augur and the
  real template on the same free model, both unpublished, scored
  head-to-head after resolution, needs no spend and no floor. I planned it
  after the scored round instead of before it, and it lands on 2026-11-30.
- **Measure the harness before the knobs.** The ablations answered
  questions (head count, aggregation) that the outside evidence already
  ranked small; the agentic harness, the largest outside effect, was
  measured last and only on a free model.
- **Pre-register the label, not just the rule.** Q6's large-error label
  turned out to mark hedged forecasts rather than wrong ones; the rule held,
  the label did not.

## Spring 2027

Approve a floor route (Claude Haiku 5.5's worst case is $0.92 a
60-question round at listed prices), set `AUGUR_FUTUREEVAL` to the Spring
project, make the repository public so the dispatch chain runs, and run the
shadow template beside every round. The first live number Augur owes is
Q1: Augur against the same-model template on resolved questions.
