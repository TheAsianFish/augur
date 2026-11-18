-- Augur ledger, schema 2: model access (M2).
-- Which host answered each call, the probe that opens every run, measured
-- rate limits, and every switch from one model to the next. Append-only like
-- schema 1.

ALTER TABLE llm_calls ADD COLUMN endpoint_host TEXT;
ALTER TABLE llm_calls ADD COLUMN served_model TEXT;
ALTER TABLE llm_calls ADD COLUMN release_basis TEXT;

-- One row per route per probe: did the model answer a one-line request, how
-- fast, at what listed price, and whether it may answer a scored forecast.
-- ``run_id`` is NULL for a standalone ``augur models probe``.
CREATE TABLE model_probes (
    probe_id        TEXT PRIMARY KEY,
    run_id          TEXT REFERENCES runs (run_id),
    session_id      TEXT NOT NULL,
    route           TEXT NOT NULL,
    endpoint        TEXT NOT NULL,
    endpoint_host   TEXT NOT NULL,
    model           TEXT NOT NULL,
    roles           TEXT NOT NULL,
    outcome         TEXT NOT NULL CHECK (outcome IN (
        'answered', 'empty', 'rate_limited', 'out_of_credit', 'auth', 'unavailable',
        'unpriced', 'error')),
    http_status     INTEGER,
    latency_ms      INTEGER,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cost_usd        REAL NOT NULL DEFAULT 0,
    usd_per_million_input  REAL,
    usd_per_million_output REAL,
    price_basis     TEXT NOT NULL,
    listed_on       TEXT,
    release_basis   TEXT NOT NULL,
    in_tier         INTEGER NOT NULL CHECK (in_tier IN (0, 1)),
    served_model    TEXT,
    limit_headers_json TEXT NOT NULL,
    detail          TEXT NOT NULL,
    probed_at       TEXT NOT NULL
);

-- A measured rate limit: requests sent back to back until the first refusal
-- (or the cap), and what the refusal looked like.
CREATE TABLE limit_tests (
    test_id            TEXT PRIMARY KEY,
    session_id         TEXT NOT NULL,
    route              TEXT NOT NULL,
    endpoint_host      TEXT NOT NULL,
    requests_cap       INTEGER NOT NULL,
    requests_sent      INTEGER NOT NULL,
    requests_ok        INTEGER NOT NULL,
    input_tokens       INTEGER NOT NULL,
    output_tokens      INTEGER NOT NULL,
    seconds            REAL NOT NULL,
    first_refusal_at   INTEGER,
    refusal_status     INTEGER,
    refusal_kind       TEXT,
    refusal_retry_after REAL,
    refusal_excerpt    TEXT,
    limit_headers_json TEXT NOT NULL,
    started_at         TEXT NOT NULL
);

-- Every time a call moved off a route: to the next one (``switched``), to
-- nothing because the next one is below the scored run's floor
-- (``refused_by_floor``), or to nothing because the chain ran out
-- (``exhausted``).
CREATE TABLE model_switches (
    switch_id     TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs (run_id),
    question_id   INTEGER,
    call_role     TEXT NOT NULL,
    member        INTEGER,
    from_route    TEXT,
    to_route      TEXT,
    reason        TEXT NOT NULL,
    outcome       TEXT NOT NULL CHECK (outcome IN ('switched', 'refused_by_floor', 'exhausted')),
    detail        TEXT NOT NULL,
    recorded_at   TEXT NOT NULL
);
