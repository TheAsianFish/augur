# Sourced by push_ledger.sh and push_standings.sh: check out the `ledger`
# branch (or start it) in a temporary worktree, then commit and push with
# retries. Every caller adds new files only, so concurrent pushes rebase cleanly.

git_bot() {
  git -C "$LEDGER_TREE" -c user.name='github-actions[bot]' \
    -c user.email='41898282+github-actions[bot]@users.noreply.github.com' "$@"
}

# Sets LEDGER_TREE to a worktree of origin's ledger branch (a new orphan branch
# if origin has none yet). Exits 1 if origin cannot be reached.
ledger_branch_checkout() {
  LEDGER_TREE="$(mktemp -d)/ledger"
  local found=0
  git ls-remote --exit-code --heads origin ledger > /dev/null || found=$?
  if [ "$found" -eq 0 ]; then
    git fetch --quiet --depth=1 origin ledger
    git worktree add --quiet --detach "$LEDGER_TREE" FETCH_HEAD
  elif [ "$found" -ne 2 ]; then
    echo "cannot reach origin to look for the ledger branch (git exit $found)" >&2
    exit 1
  else
    echo "no ledger branch on origin yet; starting one"
    git worktree add --quiet --detach "$LEDGER_TREE"
    git -C "$LEDGER_TREE" checkout --quiet --orphan ledger-new
    git -C "$LEDGER_TREE" rm -rq --cached .
    git -C "$LEDGER_TREE" clean -fdxq
    printf '%s\n' "# Augur ledger" "" \
      "One JSONL export of the SQLite ledger per scheduled run (\`python -m augur.export\`)," \
      "with the run's printed output beside it, and the daily leaderboard snapshots" \
      "under \`standings/\`." > "$LEDGER_TREE/README.md"
    git -C "$LEDGER_TREE" add README.md
  fi
}

# Commits everything staged in LEDGER_TREE with message $1 and pushes it,
# rebasing on the remote branch and retrying when another run pushed first.
ledger_branch_push() {
  local message="$1" what="$2"
  git_bot commit --quiet -m "$message"
  local attempt
  for attempt in 1 2 3 4; do
    if git -C "$LEDGER_TREE" push --quiet origin HEAD:refs/heads/ledger; then
      echo "pushed $what to the ledger branch"
      return 0
    fi
    if [ "$attempt" -eq 4 ]; then break; fi
    echo "push refused (attempt $attempt); rebasing on the ledger branch and retrying"
    sleep $((attempt * ${LEDGER_RETRY_SECONDS:-5}))
    git -C "$LEDGER_TREE" fetch --quiet --depth=50 origin ledger
    git_bot rebase --quiet FETCH_HEAD
  done
  echo "could not push after 3 retries; $LEDGER_TREE holds the commit" >&2
  exit 1
}
