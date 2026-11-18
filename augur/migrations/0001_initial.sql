-- Augur ledger, schema 1. Append-only: every table refuses UPDATE and DELETE.
-- Timestamps are fixed-width UTC ISO 8601 text (YYYY-MM-DDTHH:MM:SS.ffffffZ),
-- so string comparison is time comparison.

CREATE TABLE runs (
    run_id        TEXT PRIMARY KEY,
    started_at    TEXT NOT NULL,
    mode          TEXT NOT NULL CHECK (mode IN ('live', 'dry_run', 'backtest')),
    tournament    TEXT NOT NULL,
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL,
    settings_json TEXT NOT NULL,
    code_version  TEXT NOT NULL
);

CREATE TABLE questions (
    run_id        TEXT NOT NULL REFERENCES runs (run_id),
    question_id   INTEGER NOT NULL,
    post_id       INTEGER NOT NULL,
    type          TEXT NOT NULL,
    title         TEXT NOT NULL,
    open_time     TEXT NOT NULL,
    close_time    TEXT NOT NULL,
    cutoff        TEXT NOT NULL,
    payload_json  TEXT NOT NULL,
    recorded_at   TEXT NOT NULL,
    PRIMARY KEY (run_id, question_id)
);

CREATE TABLE sources (
    source_id       TEXT PRIMARY KEY,
    searcher        TEXT NOT NULL,
    query           TEXT NOT NULL,
    url             TEXT NOT NULL,
    title           TEXT NOT NULL,
    excerpt         TEXT NOT NULL,
    content_sha256  TEXT NOT NULL,
    published_at    TEXT,
    retrieved_at    TEXT NOT NULL,
    as_of           TEXT NOT NULL,
    date_evidence   TEXT NOT NULL
        CHECK (date_evidence IN ('revision_timestamp', 'archive_capture',
                                 'provider_published_bounded', 'none')),
    evidence_detail TEXT NOT NULL,
    recorded_at     TEXT NOT NULL
);

CREATE TABLE source_uses (
    use_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    question_id  INTEGER NOT NULL,
    source_id    TEXT NOT NULL REFERENCES sources (source_id),
    admitted     INTEGER NOT NULL CHECK (admitted IN (0, 1)),
    reason       TEXT NOT NULL,
    recorded_at  TEXT NOT NULL,
    FOREIGN KEY (run_id, question_id) REFERENCES questions (run_id, question_id)
);

CREATE TABLE llm_calls (
    call_id            TEXT PRIMARY KEY,
    run_id             TEXT NOT NULL,
    question_id        INTEGER NOT NULL,
    role               TEXT NOT NULL,
    member             INTEGER,
    provider           TEXT NOT NULL,
    model              TEXT NOT NULL,
    model_release_date TEXT,
    request_hash       TEXT NOT NULL,
    request_json       TEXT NOT NULL,
    response_text      TEXT,
    input_tokens       INTEGER,
    output_tokens      INTEGER,
    cost_usd           REAL NOT NULL DEFAULT 0,
    latency_ms         INTEGER NOT NULL,
    error              TEXT,
    started_at         TEXT NOT NULL,
    FOREIGN KEY (run_id, question_id) REFERENCES questions (run_id, question_id)
);

CREATE TABLE call_sources (
    call_id    TEXT NOT NULL REFERENCES llm_calls (call_id),
    source_id  TEXT NOT NULL REFERENCES sources (source_id),
    PRIMARY KEY (call_id, source_id)
);

CREATE TABLE forecasts (
    forecast_id  TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL,
    question_id  INTEGER NOT NULL,
    kind         TEXT NOT NULL CHECK (kind IN ('member', 'aggregate')),
    member       INTEGER,
    method       TEXT NOT NULL,
    probability  REAL NOT NULL CHECK (probability > 0 AND probability < 1),
    reasoning    TEXT NOT NULL,
    call_id      TEXT REFERENCES llm_calls (call_id),
    recorded_at  TEXT NOT NULL,
    FOREIGN KEY (run_id, question_id) REFERENCES questions (run_id, question_id)
);

CREATE TABLE submissions (
    submission_id    TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL,
    question_id      INTEGER NOT NULL,
    post_id          INTEGER NOT NULL,
    kind             TEXT NOT NULL CHECK (kind IN ('forecast', 'comment')),
    payload_json     TEXT NOT NULL,
    status           TEXT NOT NULL CHECK (status IN ('dry_run', 'submitted', 'failed')),
    http_status      INTEGER,
    response_excerpt TEXT,
    recorded_at      TEXT NOT NULL,
    FOREIGN KEY (run_id, question_id) REFERENCES questions (run_id, question_id)
);

-- The leakage gate, enforced by the store itself: an admitted use of a
-- source dated after its question's cutoff aborts the insert, and so does an
-- admitted undated or unproven source in a backtest.
CREATE TRIGGER source_uses_no_late_source
BEFORE INSERT ON source_uses
WHEN NEW.admitted = 1 AND EXISTS (
    SELECT 1 FROM sources s, questions q
    WHERE s.source_id = NEW.source_id
      AND q.run_id = NEW.run_id AND q.question_id = NEW.question_id
      AND s.published_at IS NOT NULL AND s.published_at > q.cutoff
)
BEGIN
    SELECT RAISE(ABORT, 'leakage: source dated after the question cutoff');
END;

CREATE TRIGGER source_uses_no_undated_in_backtest
BEFORE INSERT ON source_uses
WHEN NEW.admitted = 1 AND EXISTS (
    SELECT 1 FROM sources s, runs r
    WHERE s.source_id = NEW.source_id AND r.run_id = NEW.run_id
      AND r.mode = 'backtest'
      AND (s.published_at IS NULL OR s.date_evidence = 'none')
)
BEGIN
    SELECT RAISE(ABORT, 'leakage: undated or unproven source in a backtest');
END;

CREATE TRIGGER call_sources_admitted_only
BEFORE INSERT ON call_sources
WHEN NOT EXISTS (
    SELECT 1 FROM llm_calls c, source_uses u
    WHERE c.call_id = NEW.call_id
      AND u.run_id = c.run_id AND u.question_id = c.question_id
      AND u.source_id = NEW.source_id AND u.admitted = 1
)
BEGIN
    SELECT RAISE(ABORT, 'leakage: a model call cites a source the gate did not admit');
END;
