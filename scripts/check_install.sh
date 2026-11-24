#!/usr/bin/env bash
# Clean-environment install check (make install-check).
#
# Builds the sdist and the wheel, installs each into a fresh Python 3.12 venv
# outside the repository, and runs the installed `augur` from an empty working
# directory, so nothing in the checkout can be imported or read by accident.
# Every `augur` command runs under `env -i` with only PATH, HOME and the
# variables named here: no token or key from the caller's environment reaches
# it, and HTTP(S) goes to a dead proxy, so the checks are offline. The one
# optional step that calls a model uses a local Ollama (localhost only); it is
# skipped with its reason when Ollama or qwen2.5:7b is missing, or when
# AUGUR_INSTALL_CHECK_MODEL=0.
#
# Prints PASS or FAIL per check and overall; exits 1 on any failure.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/augur-install-check.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
UV="$(command -v uv)" || { echo "install check: FAIL (uv is not on PATH)"; exit 1; }
OLLAMA_URL="${OLLAMA_HOST:-http://localhost:11434}"
RUN_MODEL="${AUGUR_INSTALL_CHECK_MODEL:-1}"
UV_CACHE="$("$UV" cache dir)"
UV_PYTHONS="$("$UV" python dir)"

passed=0
failed=0
failures=()

mask() {
    # No absolute paths of this machine in the output.
    sed -e $'s/\x1b\\[[0-9;]*m//g' -e "s#$WORK#<tmp>#g" -e "s#$REPO#<repo>#g" -e "s#$HOME#~#g"
}

# A scrubbed environment: nothing from the caller but PATH and the uv caches.
clean_env() {
    env -i PATH="/usr/local/bin:/usr/bin:/bin" HOME="$WORK/home" LANG=C.UTF-8 \
        UV_CACHE_DIR="$UV_CACHE" UV_PYTHON_INSTALL_DIR="$UV_PYTHONS" "$@"
}

# The same, offline: every HTTP(S) request goes to a port nothing listens on.
offline_env() {
    clean_env HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 \
        NO_PROXY=localhost,127.0.0.1 AUGUR_LEDGER="$WORK/run/augur.db" AUGUR_PROBE=0 "$@"
}

# check NAME EXPECTED_REGEX COMMAND...: runs in $WORK/run, passes if the command
# exits 0 and its output matches the regex (an empty regex matches anything).
check() {
    local name="$1" expect="$2"
    shift 2
    local out="$WORK/out.txt"
    (cd "$WORK/run" && "$@") >"$out" 2>&1
    local code=$?
    echo "--- $name"
    echo "\$ $*" | mask
    grep -v '"level": "info"' "$out" | tail -n 25 | mask
    if [[ $code -eq 0 ]] && { [[ -z "$expect" ]] || grep -Eq "$expect" "$out"; }; then
        echo "PASS  $name"
        passed=$((passed + 1))
    else
        echo "FAIL  $name (exit $code${expect:+, expected /$expect/})"
        failed=$((failed + 1))
        failures+=("$name")
    fi
}

mkdir -p "$WORK/home" "$WORK/run" "$WORK/dist"
echo "augur install check: $(date -u +%Y-%m-%dT%H:%M:%SZ), $("$UV" --version)"
echo "commit: $(git -C "$REPO" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"

# 1. Build both distributions from the checkout.
check "build sdist and wheel" "Successfully built .*\.whl" \
    clean_env "$UV" build "$REPO" --out-dir "$WORK/dist"
WHEEL="$(ls "$WORK"/dist/*.whl 2>/dev/null | head -n 1)"
SDIST="$(ls "$WORK"/dist/*.tar.gz 2>/dev/null | head -n 1)"
if [[ -z "$WHEEL" || -z "$SDIST" ]]; then
    echo "install check: FAIL (no wheel or sdist was built)"
    exit 1
fi
for f in "$WHEEL" "$SDIST"; do
    printf '%-45s %8s bytes\n' "$(basename "$f")" "$(stat -c %s "$f")"
done
check "wheel carries the runtime data files" "augur/samples/minibench/post_" \
    python3 -c 'import sys, zipfile
names = set(zipfile.ZipFile(sys.argv[1]).namelist())
need = ["augur/py.typed", "augur/live_calibrator.json", "augur/stat_base_rates.json",
        "augur/llm/openrouter_models.json",
        "augur/migrations/0001_initial.sql"]
missing = [n for n in need if n not in names]
samples = sorted(n for n in names if n.startswith("augur/samples/minibench/post_"))
print("\n".join(need + samples))
sys.exit(1 if missing or not samples else 0)' "$WHEEL"
check "sdist leaves out the ledgers, runs and CI" "^sdist members: " \
    python3 -c 'import sys, tarfile
names = tarfile.open(sys.argv[1]).getnames()
bad = [n for n in names if "/data/ledgers/" in n or "/data/runs/" in n or "/.github/" in n]
print("sdist members:", len(names), "unwanted:", bad)
sys.exit(1 if bad else 0)' "$SDIST"
check "twine check" "PASSED" clean_env "$UV" tool run twine check "$WHEEL" "$SDIST"

# 2. The wheel alone, in a fresh venv outside the repository.
VENV="$WORK/wheel-venv"
check "fresh Python 3.12 venv (wheel)" "" clean_env "$UV" venv --python 3.12 "$VENV"
check "install the wheel" "augur-forecast==" \
    clean_env "$UV" pip install --python "$VENV/bin/python" "$WHEEL"
AUGUR="$VENV/bin/augur"
check "augur imports from the venv, not the checkout" "site-packages/augur/__init__.py" \
    offline_env "$VENV/bin/python" -c \
    'import augur; print(augur.__version__, augur.__file__)'
check "augur --help" "usage: augur" offline_env "$AUGUR" --help
check "augur models list (offline, bundled snapshot)" "model list: snapshot" \
    offline_env "$AUGUR" models list
check "dry run lists the bundled sample questions (no model call)" \
    "coverage: [1-9][0-9]* listed.* skipped" \
    offline_env "$AUGUR" forecast --tournament minibench --max-questions 0 --no-markets
check "ledger written by the run passes the leakage scan" "" \
    offline_env "$AUGUR" ledger check "$WORK/run/augur.db"

# A committed backtest ledger, copied in as input data: scan, score, round-trip.
cp "$REPO/data/ledgers/m4-forecastbench-qwen.sqlite" "$WORK/run/m4.sqlite"
check "ledger check on a committed backtest ledger" "" \
    offline_env "$AUGUR" ledger check "$WORK/run/m4.sqlite"
check "score the committed backtest" "Brier" \
    offline_env "$AUGUR" score "$WORK/run/m4.sqlite" --mode backtest
check "ledger export, import, and the rebuilt ledger scores the same" "scores identical" \
    offline_env bash -c '"$1" ledger export "$2/m4.sqlite" "$2/m4.jsonl" &&
        "$1" ledger import "$2/m4.jsonl" --out "$2/rebuilt.sqlite" &&
        "$1" ledger check "$2/rebuilt.sqlite" &&
        "$1" score "$2/m4.sqlite" --mode backtest >"$2/a.txt" &&
        "$1" score "$2/rebuilt.sqlite" --mode backtest >"$2/b.txt" &&
        diff "$2/a.txt" "$2/b.txt" && echo "scores identical"' _ "$AUGUR" "$WORK/run"

# 3. Optional: a full dry run with a local model (no network beyond localhost).
if [[ "$RUN_MODEL" == "0" ]]; then
    echo "SKIP  dry run with a local model (AUGUR_INSTALL_CHECK_MODEL=0)"
elif ! curl -sf -m 3 "$OLLAMA_URL/api/tags" 2>/dev/null | grep -q '"qwen2.5:7b"'; then
    echo "SKIP  dry run with a local model (no Ollama with qwen2.5:7b at $OLLAMA_URL)"
else
    check "dry run on the bundled samples with local qwen2.5:7b, 3 draws" \
        "coverage: 1 listed; 1 forecast in time \(0 submitted\)" \
        offline_env OLLAMA_HOST="$OLLAMA_URL" AUGUR_FORECASTERS=3 \
        "$AUGUR" forecast --tournament minibench --no-markets
fi

# 4. The sdist, built into a wheel by the installer, in a second fresh venv.
SVENV="$WORK/sdist-venv"
check "fresh Python 3.12 venv (sdist)" "" clean_env "$UV" venv --python 3.12 "$SVENV"
check "install the sdist" "augur-forecast==" \
    clean_env "$UV" pip install --python "$SVENV/bin/python" "$SDIST"
check "augur --help (from the sdist)" "usage: augur" offline_env "$SVENV/bin/augur" --help
check "dry run lists the sample questions (from the sdist)" "coverage: [1-9][0-9]* listed" \
    offline_env "$SVENV/bin/augur" forecast --tournament minibench --max-questions 0 --no-markets

echo
if [[ $failed -eq 0 ]]; then
    echo "install check: PASS ($passed checks)"
else
    echo "install check: FAIL ($failed of $((passed + failed)) checks: ${failures[*]})"
    exit 1
fi
