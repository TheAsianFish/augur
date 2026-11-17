# Augur

**Status: in progress** (planning complete, nothing built yet; see [PLAN.md](PLAN.md)).

A forecast about a real event is scored only after the event happens, often
months later, so a team building an AI forecaster cannot tell in time whether
a change helped, and most forecasters that look accurate in backtests quietly
read the answer in news written after the question closed.

Augur is a fleet of research, forecasting and calibration agents that enters
Metaculus's FutureEval bot tournament and the two-week MiniBench rounds, where
it is scored on the same questions as every other bot. Around it is a ledger
that keeps every prompt, source (with proof of its date) and forecast, and a
backtest that fails the build if any source postdates its question or any
model is newer than the question it forecasts. Changes are judged on that
backtest first and on the public leaderboard second.

## Results

None yet. Every number will be produced by a command in this repository and
recorded in `RESULTS.md`.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

MIT
