#!/usr/bin/env bash
# Export a shadow tick's ledger to the `ledger` branch as shadow/<UTC day>/<NAME>.jsonl,
# with the tick's printed output beside it (M11). The scheduled job calls this with
# the same NAME after Augur's forecasts, again after the template's, and (M16) after
# the reader's when it runs, so a timeout in a later step loses nothing earlier. Each
# export is a superset of the one before and replaces it; no other run writes that path.
#
#   scripts/push_shadow.sh LEDGER.sqlite OUTPUT.txt NAME
set -euo pipefail
# shellcheck source=scripts/ledger_branch.sh
. "$(dirname "$0")/ledger_branch.sh"

db="$1"
txt="$2"
name="$3"
case "$name" in
  [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]T*) ;;
  *) echo "NAME must start with its UTC timestamp (YYYYMMDDTHHMMSSZ), got '$name'" >&2; exit 1 ;;
esac
case "$name" in
  *[!A-Za-z0-9._-]*) echo "NAME may hold letters, digits, dot, dash and underscore only" >&2; exit 1 ;;
esac
if [ ! -f "$db" ]; then
  echo "no ledger at $db (the tick stopped before writing one); nothing to push"
  exit 0
fi

day="${name:0:4}-${name:4:2}-${name:6:2}"
rel="shadow/$day/$name"
ledger_branch_checkout

mkdir -p "$LEDGER_TREE/shadow/$day"
${AUGUR_PYTHON:-uv run --no-sync python} -m augur.export "$db" "$LEDGER_TREE/$rel.jsonl"
if [ -f "$txt" ]; then cp "$txt" "$LEDGER_TREE/$rel.txt"; fi
for coverage in coverage-*.json; do
  if [ -f "$coverage" ]; then cp "$coverage" "$LEDGER_TREE/$rel.${coverage%.json}.json"; fi
done
git -C "$LEDGER_TREE" add shadow
if git -C "$LEDGER_TREE" diff --cached --quiet; then
  echo "the ledger branch already holds $rel.jsonl as it is; nothing to push"
  exit 0
fi
ledger_branch_push "Add shadow export for tick $name" "$rel.jsonl"
