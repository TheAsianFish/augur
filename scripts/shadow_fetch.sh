#!/usr/bin/env bash
# Copy the shadow tree (shadow/) of the `ledger` branch into DIR, for the poll and
# for `augur ledger import` (M11). No ledger branch, or no shadow tree on it yet,
# leaves DIR empty: nothing is held. Any other failure exits non-zero, because a
# tick that cannot see the shadow ledger would forecast every question again.
#
#   scripts/shadow_fetch.sh DIR
set -euo pipefail

dir="$1"
remote="${AUGUR_LEDGER_REMOTE:-origin}"
mkdir -p "$dir"
found=0
git ls-remote --exit-code --heads "$remote" ledger > /dev/null || found=$?
if [ "$found" -eq 2 ]; then
  echo "no ledger branch yet: the shadow ledger is empty"
  exit 0
elif [ "$found" -ne 0 ]; then
  echo "cannot reach $remote to read the shadow ledger (git exit $found)" >&2
  exit 1
fi
git fetch --quiet --depth=1 "$remote" ledger
if ! git cat-file -e FETCH_HEAD:shadow 2> /dev/null; then
  echo "the ledger branch has no shadow tree yet: the shadow ledger is empty"
  exit 0
fi
git archive FETCH_HEAD shadow | tar -x -C "$dir"
echo "shadow ledger: $(find "$dir" -name '*.jsonl' | wc -l) export(s) in $dir/shadow"
