-- Augur ledger, schema 4: every question type, and what happened to each
-- question a run saw (M3).
--
-- A multiple-choice or numeric forecast is not one probability, so the
-- forecasts table is rebuilt with ``probability`` nullable and the full
-- prediction in ``values_json``. Every existing row is copied unchanged (all
-- binary); the append-only guards are reinstalled after the migration.

CREATE TABLE forecasts_v4 (
    forecast_id   TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    question_id   INTEGER NOT NULL,
    kind          TEXT NOT NULL CHECK (kind IN ('member', 'aggregate')),
    member        INTEGER,
    method        TEXT NOT NULL,
    question_type TEXT NOT NULL DEFAULT 'binary'
        CHECK (question_type IN ('binary', 'multiple_choice', 'numeric', 'discrete')),
    probability   REAL CHECK (probability IS NULL OR (probability > 0 AND probability < 1)),
    values_json   TEXT,
    reasoning     TEXT NOT NULL,
    call_id       TEXT REFERENCES llm_calls (call_id),
    recorded_at   TEXT NOT NULL,
    CHECK ((question_type = 'binary') = (probability IS NOT NULL)),
    CHECK (question_type = 'binary' OR values_json IS NOT NULL),
    FOREIGN KEY (run_id, question_id) REFERENCES questions (run_id, question_id)
);

INSERT INTO forecasts_v4 (
    forecast_id, run_id, question_id, kind, member, method, question_type,
    probability, values_json, reasoning, call_id, recorded_at
)
SELECT
    forecast_id, run_id, question_id, kind, member, method, 'binary',
    probability, json_object('probability', probability), reasoning, call_id, recorded_at
FROM forecasts;

DROP TABLE forecasts;

ALTER TABLE forecasts_v4 RENAME TO forecasts;

-- One row per question a run listed: forecast in time, missed (closed, or too
-- close to its close to forecast), skipped (already forecast, over a cap,
-- unsupported type, below the floor) or failed, with the reason. This is the
-- run's coverage report, kept so a missed question can be explained later.
CREATE TABLE question_outcomes (
    outcome_id   TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES runs (run_id),
    question_id  INTEGER NOT NULL,
    post_id      INTEGER NOT NULL,
    question_type TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('forecast', 'missed', 'skipped', 'failed')),
    reason       TEXT NOT NULL,
    submitted    INTEGER NOT NULL CHECK (submitted IN (0, 1)),
    closes_at    TEXT,
    minutes_left REAL,
    cost_usd     REAL NOT NULL DEFAULT 0,
    seconds      REAL,
    recorded_at  TEXT NOT NULL
);
