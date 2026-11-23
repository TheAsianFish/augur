#!/usr/bin/env bash
# Commit leaderboard snapshots (`augur standings --out`) to the `ledger` branch:
# standings/<UTC date>/<UTC timestamp>-<file name>, one new file per snapshot, so
# every round's leaderboard is kept as it stood each day, including its final reading.
#
#   scripts/push_standings.sh DIR      # every DIR/*.json, plus DIR/standings.txt if present
set -euo pipefail
# shellcheck source=scripts/ledger_branch.sh
. "$(dirname "$0")/ledger_branch.sh"

dir="${1:-standings}"
shopt -s nullglob
files=("$dir"/*.json)
if [ "${#files[@]}" -eq 0 ]; then
  echo "no snapshots in $dir; nothing to push"
  exit 0
fi

day="$(date -u +%Y-%m-%d)"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
ledger_branch_checkout
mkdir -p "$LEDGER_TREE/standings/$day"
for f in "${files[@]}"; do
  cp "$f" "$LEDGER_TREE/standings/$day/$stamp-$(basename "$f")"
done
if [ -f "$dir/standings.txt" ]; then
  cp "$dir/standings.txt" "$LEDGER_TREE/standings/$day/$stamp-standings.txt"
fi
git -C "$LEDGER_TREE" add standings
ledger_branch_push "Add leaderboard snapshots of $stamp" "${#files[@]} snapshot(s)"
