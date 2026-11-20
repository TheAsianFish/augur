-- Augur ledger, schema 7: the leakage experiment (M6).
--
-- Q2 measures what leakage is worth by running the same questions with
-- retrieval as of today on purpose. Such a run is not a backtest: its cutoff
-- is the moment it ran, so the gate and the triggers hold it to that instant
-- and nothing it reads is ever counted as a backtest result. It gets its own
-- mode, ``unrestricted``. SQLite cannot change a CHECK in place, so ``runs``
-- is rebuilt with every row copied unchanged; its append-only triggers are
-- recreated when the ledger opens.
--
-- ``leak_judgments`` holds the judge's 0 to 4 rating of each admitted source
-- (El Lahib et al.'s rubric): how much post-cutoff information about the
-- question's resolution the source carries.

DROP TRIGGER IF EXISTS runs_no_update;
DROP TRIGGER IF EXISTS runs_no_delete;

CREATE TABLE runs_v7 (
    run_id        TEXT PRIMARY KEY,
    started_at    TEXT NOT NULL,
    mode          TEXT NOT NULL
        CHECK (mode IN ('live', 'dry_run', 'backtest', 'unrestricted')),
    tournament    TEXT NOT NULL,
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL,
    settings_json TEXT NOT NULL,
    code_version  TEXT NOT NULL
);

INSERT INTO runs_v7 SELECT * FROM runs;

DROP TABLE runs;

ALTER TABLE runs_v7 RENAME TO runs;

-- An unrestricted run needs proof of date like a backtest: only the instant differs.
DROP TRIGGER IF EXISTS source_uses_no_undated_in_backtest;

CREATE TRIGGER source_uses_no_undated_in_backtest
BEFORE INSERT ON source_uses
WHEN NEW.admitted = 1 AND EXISTS (
    SELECT 1 FROM sources s, runs r
    WHERE s.source_id = NEW.source_id AND r.run_id = NEW.run_id
      AND r.mode IN ('backtest', 'unrestricted')
      AND (s.published_at IS NULL OR s.date_evidence = 'none')
)
BEGIN
    SELECT RAISE(ABORT, 'leakage: undated or unproven source in a backtest');
END;

CREATE TABLE leak_judgments (
    judgment_id  TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL,
    question_id  INTEGER NOT NULL,
    source_id    TEXT NOT NULL REFERENCES sources (source_id),
    call_id      TEXT NOT NULL REFERENCES llm_calls (call_id),
    judge_model  TEXT NOT NULL,
    score        INTEGER NOT NULL CHECK (score BETWEEN 0 AND 4),
    rationale    TEXT NOT NULL,
    recorded_at  TEXT NOT NULL,
    FOREIGN KEY (run_id, question_id) REFERENCES questions (run_id, question_id)
);
