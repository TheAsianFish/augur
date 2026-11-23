# Changelog

## 0.1.0 (unreleased; the date is set when the `v0.1.0` tag is pushed)

The first release on PyPI, as `augur-forecast` (the import and the CLI are
`augur`). Milestones M1 to M9 of PLAN.md; measured results are in RESULTS.md.

- **Ledger and leakage gate.** An append-only SQLite ledger with numbered
  migrations: every source is written, with the evidence for its date, before
  a model sees it. `augur ledger check` and the test suite fail on a source
  dated after its question's cutoff, an undated source in a backtest, or a
  model released after the question opened. JSONL export and import.
- **Every Metaculus question type.** Binary, multiple choice, numeric and
  discrete forecasts, posted with a private comment that carries the
  reasoning; `augur forecast` is a dry run unless `--submit` is passed with a
  token, and without a token it forecasts the bundled sample questions.
- **Model chains.** Ordered forecast and research chains over OpenRouter,
  Groq, Gemini, GitHub Models, Ollama or any OpenAI-compatible host, with
  reasoning effort per route, rate limits and cost caps, an opening probe, and
  the submit floor: outside the bot-testing area only a frontier-tier hosted
  model may answer. Record and replay of every model call.
- **Research.** Wikipedia revisions and Wayback captures as of a cutoff,
  AskNews when configured, and market prices (Metaculus community, Manifold,
  Polymarket, Kalshi) with the exact-price blend.
- **Backtests and the price of leakage.** ForecastBench backtests with the
  clock pinned and the release-date gate; research as of the question's open
  instant against as of now, the bounded research loop, and an LLM judge that
  rates every source on a 0 to 4 leakage rubric.
- **The fleet.** Several forecasters per question with prompt stances and
  mixed models, one or two research passes, combined by the median (binary),
  normalised per-option median (multiple choice) or mean CDF (numeric).
- **Calibration.** Walk-forward Platt, isotonic, extremizing and base-rate
  calibrators fitted only on outcomes known before each question opened,
  source-aware maps, and a live calibrator applied before the blend except on
  the sources it is withheld from. Disagreement measures and their shrink and
  skip policies (Q6).
- **Scoring and standings.** Brier, log and baseline scores with clustered
  bootstrap intervals and paired differences against the template bot, the
  community, markets, always 50% and the base rate; resolution fetching;
  MiniBench round standings; the planted-effect power simulation.
- **Operations.** A scheduled GitHub Actions workflow with a standard-library
  poll and a kill switch; a release workflow with trusted publishing; a
  clean-environment install check; a Docker image.
