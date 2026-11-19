-- Augur ledger, schema 5: what happened, and who else forecast it (M4).
--
-- A resolution is a new row, never an edit: a question re-resolved or
-- annulled later gets another row, and the latest one counts. The value is
-- text in one form for every type: 'yes' or 'no' (binary), the option label
-- (multiple choice), the number (numeric, discrete), or 'below_lower_bound'
-- and 'above_upper_bound' for an out-of-range numeric outcome.

CREATE TABLE resolutions (
    resolution_id TEXT PRIMARY KEY,
    question_id   INTEGER NOT NULL,
    post_id       INTEGER NOT NULL,
    question_type TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('resolved', 'annulled', 'ambiguous')),
    value         TEXT,
    resolved_at   TEXT,
    source        TEXT NOT NULL CHECK (source IN ('metaculus_api', 'forecastbench', 'manual')),
    evidence      TEXT NOT NULL,
    recorded_at   TEXT NOT NULL,
    CHECK ((status = 'resolved') = (value IS NOT NULL))
);

-- Forecasts on the same questions by someone other than the bot: the
-- Metaculus community prediction, the shadow template bot, a market price.
-- ``values_json`` has the shape ``forecasts.values_json`` has.
CREATE TABLE reference_forecasts (
    reference_id  TEXT PRIMARY KEY,
    question_id   INTEGER NOT NULL,
    post_id       INTEGER NOT NULL,
    question_type TEXT NOT NULL,
    forecaster    TEXT NOT NULL CHECK (forecaster IN ('community', 'template', 'market')),
    detail        TEXT NOT NULL,
    probability   REAL CHECK (probability IS NULL OR (probability >= 0 AND probability <= 1)),
    values_json   TEXT NOT NULL,
    as_of         TEXT NOT NULL,
    source        TEXT NOT NULL,
    recorded_at   TEXT NOT NULL
);

-- Metaculus's own score for a forecast, as its API reported it, so Augur's
-- scoring can be checked against it.
CREATE TABLE metaculus_scores (
    score_id     TEXT PRIMARY KEY,
    question_id  INTEGER NOT NULL,
    post_id      INTEGER NOT NULL,
    forecaster   TEXT NOT NULL CHECK (forecaster IN ('augur', 'community')),
    score_type   TEXT NOT NULL,
    score        REAL NOT NULL,
    recorded_at  TEXT NOT NULL
);
