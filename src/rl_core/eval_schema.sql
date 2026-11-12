-- eval.db, owned by EvalStore. A project's metric columns are added to
-- dev_metrics by its first row; test_metrics is a key/value table made on
-- first write.

-- One row per scored checkpoint; epoch is the learner update count.
CREATE TABLE IF NOT EXISTS dev_metrics (
    epoch      INTEGER PRIMARY KEY,
    checkpoint TEXT NOT NULL,
    scored_at  TEXT NOT NULL
);
