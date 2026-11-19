-- Augur ledger, schema 6: market prices as sources (M5).
--
-- A price is a source like any other: it is written before a model sees it,
-- the gate admits it only if its date is at or before the cutoff, and its
-- date is the timestamp of the trade, history point or aggregate entry that
-- set it. ``sources.date_evidence`` gains the three kinds of proof that
-- carries. SQLite cannot change a CHECK in place, so ``sources`` is rebuilt
-- with every row copied unchanged; its append-only triggers are recreated
-- when the ledger opens.

DROP TRIGGER IF EXISTS sources_no_update;
DROP TRIGGER IF EXISTS sources_no_delete;

CREATE TABLE sources_v6 (
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
                                 'provider_published_bounded', 'market_trade',
                                 'price_history', 'aggregate_history', 'none')),
    evidence_detail TEXT NOT NULL,
    recorded_at     TEXT NOT NULL
);

INSERT INTO sources_v6 SELECT * FROM sources;

DROP TABLE sources;

ALTER TABLE sources_v6 RENAME TO sources;

-- The structured price behind each market source: what the price-only and
-- blend arms of the replay evaluation read. One row per source, written in
-- the same step as the source. A price must be dated at or before the
-- instant it was asked for, and its date must be the source's date.
CREATE TABLE market_quotes (
    source_id    TEXT PRIMARY KEY REFERENCES sources (source_id),
    platform     TEXT NOT NULL
        CHECK (platform IN ('polymarket', 'manifold', 'kalshi', 'metaculus')),
    market_id    TEXT NOT NULL,
    title        TEXT NOT NULL,
    url          TEXT NOT NULL,
    outcome      TEXT NOT NULL,
    probability  REAL NOT NULL CHECK (probability >= 0 AND probability <= 1),
    price_time   TEXT NOT NULL,
    as_of        TEXT NOT NULL,
    match        TEXT NOT NULL CHECK (match IN ('exact', 'search')),
    volume       REAL,
    closes_at    TEXT,
    recorded_at  TEXT NOT NULL,
    CHECK (price_time <= as_of)
);
