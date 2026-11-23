# Release notes

## Going public (when M2 merges)

The repository becomes public at the M2 merge: Metaculus lists open-source
bots and gives them about twice the sponsored credits, and a public
repository runs the scheduled tournament workflow (M3) without spending the
private Actions-minute allowance. Nothing is published to PyPI at this step.

### Audit (run on `dev/m2`, 2026-11-18)

| Check | How | Result |
|-------|-----|--------|
| No secret in the history | Every commit's patch (`git log --all -p --binary`, every commit on `dev/m2`, re-run at review) searched for the exact values of `METACULUS_TOKEN` and `OPENROUTER_API_KEY`, and for key shapes (`sk-or-v1-…`, `sk-ant-…`, `sk-…`, `ghp_…`, `AKIA…`, a literal `Token <hex>` or `Bearer <token>`) | 0 matches |
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

## Turning on the schedule (when M3 merges)

The scheduled workflow (`.github/workflows/tournament.yml`) does nothing
until the variables below are set: unset `AUGUR_SCHEDULE` is the kill switch.

1. Merge the M3 pull request with CI green. Only `main` ever posts.
2. Repository variables (Settings → Secrets and variables → Actions →
   Variables):
   - `AUGUR_FORECAST_MODELS`: a frontier-tier model first, e.g.
     `openrouter:openai/gpt-5.4@high,openrouter:anthropic/claude-opus-5`
     (the tier is `frontier-2026-11b`; `augur models list` shows it).
   - `AUGUR_RESEARCH_MODELS`: a cheap or free chain, e.g.
     `openrouter:nvidia/nemotron-3-ultra-550b-a55b:free,openrouter:inclusionai/ling-3.2-flash:free`.
   - `AUGUR_MAX_COST_PER_RUN`: `1.00` to start (a question costs at most
     about $0.35 at GPT-5.4's listed price).
   - `AUGUR_SCHEDULE=minibench`. While `AUGUR_SUBMIT` is unset, scheduled
     runs only poll: a dry run posts nothing, so every tick would forecast
     the same questions again. For a dry run, dispatch one by hand:
     `gh workflow run tournament.yml --ref main -f tournament=minibench`.
3. Check one manual run first:
   `gh workflow run tournament.yml --ref main -f tournament=bot-testing-area -f submit=true`;
   it posts only in the practice area, and its export appears on the
   `ledger` branch.
4. Then set `AUGUR_SUBMIT=true` before the round opens (about 2026-11-30).
5. Once the repository is public, set `AUGUR_CHAIN_MINUTES=60`: GitHub's
   scheduler was measured to drop most ticks (RESULTS.md M3), and the chain
   dispatches each next run itself. While private, its waiting would bill
   about 43,000 minutes a month (a 60-minute job, 24 times a day), so leave
   it unset (the workflow also refuses to chain on a private repository).
6. Kill switch: set `AUGUR_SCHEDULE=off`.

## Releasing 0.1.0

The package is `augur-forecast` on PyPI (the name `augur` is taken; the
import and the CLI stay `augur`). It is published only by
`.github/workflows/release.yml`, from a pushed `v*` tag, by trusted
publishing: PyPI trusts this repository's workflow and environment, so no
PyPI token exists anywhere, in a secret or otherwise.

### Once, before the first release (the maintainer)

1. PyPI: log in, Account settings → Publishing → Add a new pending
   publisher (the project does not exist yet), GitHub tab:
   - PyPI project name: `augur-forecast`
   - Owner: `TheAsianFish`
   - Repository name: `augur`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
2. GitHub: Settings → Environments → New environment `pypi`. Under
   Deployment protection rules add yourself as a required reviewer, and
   under Deployment branches and tags allow only tags matching `v*`. The
   publish job waits for your approval; nothing else can use the
   environment.
3. Nothing else: no repository secret is needed for a release.

### Dry run

```sh
gh workflow run release.yml -f dry_run=true
gh run watch
```

A dispatch runs the `build` job only: `make check`, `make install-check`
(sdist and wheel each installed into a fresh Python 3.12 venv outside the
checkout, the installed `augur` run offline with a scrubbed environment),
`uv build`, `twine check`, and uploads `dist/` as an artifact you can
download and inspect. The `publish` job is skipped: its condition is a pushed
`v*` tag, so a dispatch cannot publish whatever `dry_run` says
(`tests/test_release_workflow.py` pins this). The local equivalents are
`make check install-check`; a run of the workflow under `act` is in
`data/runs/m10-release-dry-run.txt`, and of the install check in
`data/runs/m10-install-check.txt`.

### Tagging

On `main`, with CI green and `version` in `pyproject.toml` equal to
`augur.__version__` (a test checks it):

```sh
git switch main && git pull
git tag v0.1.0
git push origin v0.1.0
```

Set the date of the 0.1.0 entry in CHANGELOG.md in the same commit that
the tag points at, or in the next one.

### What the workflow checks

1. `build`: `uv sync --locked`, `make check`, that the tag is exactly
   `v` + the project version (a mismatch fails before anything is built),
   `make install-check`, `uv build`, `twine check dist/*`, and the
   artifact upload.
2. `publish` (tag only, after `build`): waits for the `pypi` environment's
   reviewer, downloads the artifact and publishes it with
   `pypa/gh-action-pypi-publish`. Only this job holds `id-token: write`; the
   rest of the workflow can only read the repository.

A failed publish can be re-run from the Actions page. A version on PyPI
cannot be replaced: a broken release is fixed by 0.1.1, never by moving the
tag.

### Verifying the release

In a clean venv, away from the checkout:

```sh
cd "$(mktemp -d)"
uv venv --python 3.12 && . .venv/bin/activate
uv pip install augur-forecast==0.1.0      # or: pip install augur-forecast==0.1.0
augur --help
augur models list
augur forecast --tournament minibench --max-questions 0   # lists the bundled samples
```

Without `METACULUS_TOKEN` the last command uses the sample questions that
ship in the wheel; with a local model (`ollama pull qwen2.5:7b`) drop
`--max-questions 0` for a full dry run.

### Running the bot with Docker

The `Dockerfile` builds a `python:3.12-slim` image with Augur installed from
`uv.lock`, running as a non-root user, with the ledger on the `/data` volume
(`AUGUR_LEDGER=/data/augur.db`). No secret is in the image: pass them at run
time with `-e NAME` (the value is taken from your shell) or `--env-file`.

```sh
make docker            # docker build -t augur-forecast:0.1.0 .

# A dry run on the bundled samples (nothing posted; needs a model, e.g. a local Ollama:
# add --network host, or -e OLLAMA_HOST=http://<host>:11434)
docker run --rm -v augur-data:/data augur-forecast:0.1.0

# The live listing, still a dry run (no --submit)
docker run --rm -e METACULUS_TOKEN -e OPENROUTER_API_KEY \
    -e AUGUR_FORECAST_MODELS -e AUGUR_RESEARCH_MODELS \
    -v augur-data:/data augur-forecast:0.1.0 forecast --tournament minibench
```

Add `--submit` only from an image built from a commit that passed
`make check`, with a frontier-tier model first in `AUGUR_FORECAST_MODELS`
(the floor refuses otherwise). On a server, a cron line that forecasts every
three hours, keeps the secrets in a root-readable file and appends to a log:

```cron
0 */3 * * * docker run --rm --env-file /etc/augur.env -v augur-data:/data augur-forecast:0.1.0 forecast --tournament minibench --submit >> /var/log/augur.log 2>&1
```

`/etc/augur.env` holds `METACULUS_TOKEN=…`, `OPENROUTER_API_KEY=…`,
`AUGUR_FORECAST_MODELS=…` and the caps (`AUGUR_MAX_COST_PER_RUN`); make it
`chmod 600`. The ledger survives in the `augur-data` volume; export it with
`docker run --rm -v augur-data:/data augur-forecast:0.1.0 ledger export /data/augur.db /data/augur.jsonl`.
Do not run the container and the scheduled workflow against the same
tournament with `--submit`: one bot account, one schedule.
