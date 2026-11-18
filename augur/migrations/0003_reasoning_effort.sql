-- Augur ledger, schema 3: the reasoning effort each call asked for.
-- ``reasoning_effort`` is what the route requested (NULL: none);
-- ``reasoning_effort_applied`` is 1 when the host was sent it and 0 when the
-- host cannot take it (Ollama), NULL when none was requested. Existing rows
-- keep NULL. Append-only as before: the guards forbid UPDATE and DELETE, not
-- new columns.

ALTER TABLE llm_calls ADD COLUMN reasoning_effort TEXT;
ALTER TABLE llm_calls ADD COLUMN reasoning_effort_applied INTEGER;
ALTER TABLE model_probes ADD COLUMN reasoning_effort TEXT;
ALTER TABLE model_probes ADD COLUMN reasoning_effort_applied INTEGER;
