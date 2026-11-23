#!/usr/bin/env bash
# Export a run's ledger to JSONL and commit it, beside the run's printed output,
# to the `ledger` branch: ledger/<UTC date>/<UTC timestamp>-<run id>.{jsonl,txt}, plus
# each tournament's coverage report (<...>.coverage-<tournament>.json).
# One new file per run, so concurrent pushes never touch the same content.
#
#   scripts/push_ledger.sh [LEDGER.sqlite] [OUTPUT.txt]
set -euo pipefail
# shellcheck source=scripts/ledger_branch.sh
. "$(dirname "$0")/ledger_branch.sh"

db="${1:-run.sqlite}"
txt="${2:-run.txt}"
if [ ! -f "$db" ]; then
  echo "no ledger at $db (the run stopped before writing one); nothing to push"
  exit 0
fi

day="$(date -u +%Y-%m-%d)"
name="$(date -u +%Y%m%dT%H%M%SZ)-${GITHUB_RUN_ID:-local}"
rel="ledger/$day/$name"
ledger_branch_checkout

mkdir -p "$LEDGER_TREE/ledger/$day"
uv run --no-sync python -m augur.export "$db" "$LEDGER_TREE/$rel.jsonl"
if [ -f "$txt" ]; then cp "$txt" "$LEDGER_TREE/$rel.txt"; fi
for coverage in coverage-*.json; do
  if [ -f "$coverage" ]; then cp "$coverage" "$LEDGER_TREE/$rel.${coverage%.json}.json"; fi
done
git -C "$LEDGER_TREE" add ledger
ledger_branch_push "Add ledger export for run $name" "$rel.jsonl"
