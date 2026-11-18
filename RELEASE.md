# Release notes

## Going public (when M2 merges)

The repository becomes public at the M2 merge: Metaculus lists open-source
bots and gives them about twice the sponsored credits, and a public
repository runs the scheduled tournament workflow (M3) without spending the
private Actions-minute allowance. Nothing is published to PyPI at this step.

### Audit (run on `dev/m2`, 2026-11-18)

| Check | How | Result |
|-------|-----|--------|
| No secret in the history | Every commit's patch (`git log --all -p --binary`, 11 commits) searched for the exact values of `METACULUS_TOKEN` and `OPENROUTER_API_KEY`, and for key shapes (`sk-or-v1-…`, `sk-ant-…`, `sk-…`, `ghp_…`, `AKIA…`, a literal `Token <hex>` or `Bearer <token>`) | 0 matches |
| No secret in the committed data | The same exact-value search over every file in `data/` and `augur/llm/openrouter_models.json` (ledgers, recordings, printed runs) | 0 matches |
| Secrets never logged | `tests/test_logging.py`, `test_no_token_reaches_the_logs_during_a_live_run`, and the provider tests that assert a key echoed by a server is redacted | pass in `make check` |
| No private notes tracked | `git ls-files` has no `CLAUDE.md`, `AGENTS.md`, `.agent/`, `.claude/` or `.env` | none tracked |
| Licence | `LICENSE` | MIT |
| README first screen | What the bot is, the problem, the status line, a real run | yes |
| Commit metadata | Author names and email addresses become public with the history | check that the address in the commits is the one you want public; changing it means rewriting history, so decide before the switch |

Re-run the history check before switching (prints only true or false):

```sh
python3 - <<'PY'
import os, subprocess
log = subprocess.run(["git", "log", "--all", "-p", "--binary"], capture_output=True).stdout
for name in ("METACULUS_TOKEN", "OPENROUTER_API_KEY", "ASKNEWS_CLIENT_SECRET"):
    value = os.environ.get(name, "").encode()
    print(name, "in history:", bool(value) and value in log)
PY
```

### Steps for the maintainer

1. Merge the M2 pull request with CI green.
2. Run the check above with the secrets in your environment; every line
   must say `False`.
3. GitHub: Settings → General → Danger Zone → Change visibility → Public.
4. Repository secrets (Settings → Secrets and variables → Actions):
   `METACULUS_TOKEN`, `OPENROUTER_API_KEY` (the key Metaculus issues; never
   top it up with a paid balance), and `ASKNEWS_CLIENT_ID` /
   `ASKNEWS_CLIENT_SECRET` when AskNews access arrives.
5. Metaculus: link the bot to your account (Settings → My Forecasting Bots)
   and give the public repository URL where the form asks whether the bot is
   open source, so the credit request counts it as one.
6. Before the first scored run, decide the frontier tier
   (`augur/llm/registry.py`, `FRONTIER_TIER`) against what the key can call:
   `augur models list` shows each configured model's price, listing date and
   whether it is in the tier. The tier was written before OpenRouter listed
   the newest models (for example `openai/gpt-6.2-astra`, listed 2026-11-17);
   changing it is a decision about what may post to a scored tournament, not
   a way to get a run through.
7. Then, with a frontier model first in `AUGUR_FORECAST_MODELS`:
   `augur models probe`, and `augur forecast --tournament minibench --submit`
   during an open round (M3 adds the schedule that does this unattended).
