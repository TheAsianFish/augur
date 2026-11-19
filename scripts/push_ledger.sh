#!/usr/bin/env bash
# Export a run's ledger to JSONL and commit it, beside the run's printed output,
# to the `ledger` branch: ledger/<UTC date>/<UTC timestamp>-<run id>.{jsonl,txt}, plus
# each tournament's coverage report (<...>.coverage-<tournament>.json).
# One new file per run, so concurrent pushes never touch the same content.
#
#   scripts/push_ledger.sh [LEDGER.sqlite] [OUTPUT.txt]
set -euo pipefail

db="${1:-run.sqlite}"
txt="${2:-run.txt}"
if [ ! -f "$db" ]; then
  echo "no ledger at $db (the run stopped before writing one); nothing to push"
  exit 0
fi

day="$(date -u +%Y-%m-%d)"
name="$(date -u +%Y%m%dT%H%M%SZ)-${GITHUB_RUN_ID:-local}"
rel="ledger/$day/$name"
tree="$(mktemp -d)/ledger"
git_bot=(git -C "$tree" -c user.name='github-actions[bot]'
  -c user.email='41898282+github-actions[bot]@users.noreply.github.com')

found=0
git ls-remote --exit-code --heads origin ledger > /dev/null || found=$?
if [ "$found" -eq 0 ]; then
  git fetch --quiet --depth=1 origin ledger
  git worktree add --quiet --detach "$tree" FETCH_HEAD
elif [ "$found" -ne 2 ]; then
  echo "cannot reach origin to look for the ledger branch (git exit $found)" >&2
  exit 1
else
  echo "no ledger branch on origin yet; starting one"
  git worktree add --quiet --detach "$tree"
  git -C "$tree" checkout --quiet --orphan ledger-new
  git -C "$tree" rm -rq --cached .
  git -C "$tree" clean -fdxq
  printf '%s\n' "# Augur ledger" "" \
    "One JSONL export of the SQLite ledger per scheduled run (\`python -m augur.export\`)," \
    "with the run's printed output beside it." > "$tree/README.md"
  git -C "$tree" add README.md
fi

mkdir -p "$tree/ledger/$day"
uv run --no-sync python -m augur.export "$db" "$tree/$rel.jsonl"
if [ -f "$txt" ]; then cp "$txt" "$tree/$rel.txt"; fi
for coverage in coverage-*.json; do
  if [ -f "$coverage" ]; then cp "$coverage" "$tree/$rel.${coverage%.json}.json"; fi
done
git -C "$tree" add ledger
"${git_bot[@]}" commit --quiet -m "Add ledger export for run $name"

for attempt in 1 2 3 4; do
  if git -C "$tree" push --quiet origin HEAD:refs/heads/ledger; then
    echo "pushed $rel.jsonl to the ledger branch"
    exit 0
  fi
  if [ "$attempt" -eq 4 ]; then break; fi
  echo "push refused (attempt $attempt); rebasing on the ledger branch and retrying"
  sleep $((attempt * 5))
  git -C "$tree" fetch --quiet --depth=50 origin ledger
  "${git_bot[@]}" rebase --quiet FETCH_HEAD
done
echo "could not push the ledger export after 3 retries; $tree holds the commit" >&2
exit 1
