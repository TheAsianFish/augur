-- Augur ledger, schema 8: the statistical arm's reference rows (M12).
--
-- A shadow tick writes two more forecasts beside Augur's on each question:
-- ``stat``, the model-free arm (a price, a base rate or a series), and
-- ``blend``, the registered primary candidate (one-third Augur, two-thirds
-- the arm where it has a price or a series). Neither is ever posted; ``augur
-- floor --candidate blend`` scores the blend against the template. SQLite
-- cannot change a CHECK in place, so ``reference_forecasts`` is rebuilt with
-- every row copied unchanged; its append-only triggers are recreated when the
-- ledger opens.

DROP TRIGGER IF EXISTS reference_forecasts_no_update;
DROP TRIGGER IF EXISTS reference_forecasts_no_delete;

CREATE TABLE reference_forecasts_v8 (
    reference_id  TEXT PRIMARY KEY,
    question_id   INTEGER NOT NULL,
    post_id       INTEGER NOT NULL,
    question_type TEXT NOT NULL,
    forecaster    TEXT NOT NULL
        CHECK (forecaster IN ('community', 'template', 'market', 'stat', 'blend')),
    detail        TEXT NOT NULL,
    probability   REAL CHECK (probability IS NULL OR (probability >= 0 AND probability <= 1)),
    values_json   TEXT NOT NULL,
    as_of         TEXT NOT NULL,
    source        TEXT NOT NULL,
    recorded_at   TEXT NOT NULL
);

INSERT INTO reference_forecasts_v8 SELECT * FROM reference_forecasts;

DROP TABLE reference_forecasts;

ALTER TABLE reference_forecasts_v8 RENAME TO reference_forecasts;
