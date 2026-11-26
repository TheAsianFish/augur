-- Augur ledger, schema 9: the newer free model's live arm (M16).
--
-- A shadow tick may write two more reference rows on a question Augur and the
-- template both forecast: ``reader``, three draws of the newer free route on
-- the forecaster request Augur's member 0 saw, and ``reader-base``, the same
-- combination of Augur's own members 0 to 2 from that tick (the registered
-- comparator). Neither is ever posted or the floor's candidate. SQLite cannot
-- change a CHECK in place, so ``reference_forecasts`` is rebuilt with every
-- row copied unchanged; its append-only triggers are recreated when the
-- ledger opens.

DROP TRIGGER IF EXISTS reference_forecasts_no_update;
DROP TRIGGER IF EXISTS reference_forecasts_no_delete;

CREATE TABLE reference_forecasts_v9 (
    reference_id  TEXT PRIMARY KEY,
    question_id   INTEGER NOT NULL,
    post_id       INTEGER NOT NULL,
    question_type TEXT NOT NULL,
    forecaster    TEXT NOT NULL
        CHECK (forecaster IN (
            'community', 'template', 'market', 'stat', 'blend', 'reader', 'reader-base'
        )),
    detail        TEXT NOT NULL,
    probability   REAL CHECK (probability IS NULL OR (probability >= 0 AND probability <= 1)),
    values_json   TEXT NOT NULL,
    as_of         TEXT NOT NULL,
    source        TEXT NOT NULL,
    recorded_at   TEXT NOT NULL
);

INSERT INTO reference_forecasts_v9 SELECT * FROM reference_forecasts;

DROP TABLE reference_forecasts;

ALTER TABLE reference_forecasts_v9 RENAME TO reference_forecasts;
