-- PostgreSQL schema for MNIST E2E

-- Data Snapshots for reproducibility & intergrity
CREATE TABLE IF NOT EXISTS data_snapshots (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source TEXT NOT NULL,                
    rows_train INT NOT NULL,
    rows_test INT NOT NULL,
    train_sha256 TEXT NOT NULL,          
    test_sha256 TEXT NOT NULL            
);

-- Image Storage: each row contains the image (label + raw pixel bytes)
-- Each image is 28x28 grayscale -> 784 bytes (uint8).
CREATE TABLE IF NOT EXISTS mnist_images_train (
    id BIGSERIAL PRIMARY KEY,
    label SMALLINT NOT NULL CHECK (label BETWEEN 0 AND 9),
    pixels BYTEA NOT NULL                
);
CREATE TABLE IF NOT EXISTS mnist_images_test (
    id BIGSERIAL PRIMARY KEY,
    label SMALLINT NOT NULL CHECK (label BETWEEN 0 AND 9),
    pixels BYTEA NOT NULL
);

-- 3) (Future steps) Model registry + prod pointer + service metrics
-- Keeping them here now so you don't have to alter schema later.

-- CREATE TABLE IF NOT EXISTS model_artifacts (
--     id BIGSERIAL PRIMARY KEY,
--     version TEXT UNIQUE NOT NULL,        -- e.g., 'model_v1'
--     artifact BYTEA NOT NULL,             -- e.g., zipped SavedModel or Torch .pt
--     signature JSONB NOT NULL,            -- input/output spec
--     eval_summary JSONB NOT NULL,         -- metrics: accuracy, robustness, latency
--     requirements TEXT NOT NULL,          -- pinned deps
--     training_seed INT,
--     code_ref TEXT,                       -- git SHA / tag for training code
--     created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
-- );

-- CREATE TABLE IF NOT EXISTS prod_pointer (
--     id BOOL PRIMARY KEY DEFAULT TRUE,    -- single-row table (constant key)
--     version TEXT NOT NULL,
--     updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
-- );

-- INSERT INTO prod_pointer (id, version)
--     VALUES (TRUE, 'model_v0')
-- ON CONFLICT (id) DO NOTHING;

-- CREATE TABLE IF NOT EXISTS service_metrics_daily (
--     day DATE PRIMARY KEY,
--     p95_latency_ms DOUBLE PRECISION,
--     error_rate_5xx DOUBLE PRECISION,
--     req_count BIGINT,
--     model_version TEXT
-- );
