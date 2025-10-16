-- PostgreSQL schema for MNIST E2E

-- Data Snapshots for reproducibility & integrity
CREATE TABLE IF NOT EXISTS data_snapshots (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source TEXT NOT NULL,
    rows_train INT NOT NULL,
    rows_test INT NOT NULL,
    train_sha256 TEXT NOT NULL,
    test_sha256 TEXT NOT NULL,
    CONSTRAINT uq_snapshot_content UNIQUE (train_sha256, test_sha256)
);

-- Image Storage: each row contains the image (label + raw pixel bytes)
-- Each image is 28x28 grayscale -> 784 bytes (uint8).
CREATE TABLE IF NOT EXISTS mnist_images_train (
    id BIGSERIAL PRIMARY KEY,
    snapshot_id BIGINT NOT NULL REFERENCES data_snapshots (id) ON DELETE CASCADE,
    label SMALLINT NOT NULL CHECK (label BETWEEN 0 AND 9),
    pixels BYTEA NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mnist_train_snapshot ON mnist_images_train (snapshot_id);
CREATE INDEX IF NOT EXISTS idx_mnist_train_label ON mnist_images_train (label);

CREATE TABLE IF NOT EXISTS mnist_images_test (
    id BIGSERIAL PRIMARY KEY,
    snapshot_id BIGINT NOT NULL REFERENCES data_snapshots (id) ON DELETE CASCADE,
    label SMALLINT NOT NULL CHECK (label BETWEEN 0 AND 9),
    pixels BYTEA NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mnist_test_snapshot ON mnist_images_test (snapshot_id);
CREATE INDEX IF NOT EXISTS idx_mnist_test_label ON mnist_images_test (label);

-- Info on how the train data was split into {train, val}
CREATE TABLE IF NOT EXISTS split_configurations (
    id BIGSERIAL PRIMARY KEY,
    snapshot_id BIGINT NOT NULL REFERENCES data_snapshots (id) ON DELETE CASCADE,
    name TEXT NOT NULL, -- e.g., 'default', 'seed_42', 'aug_v1'
    params JSONB NOT NULL, -- val_frac, seed, transforms, scaling, etc.
    stats JSONB NOT NULL, -- per-label counts, distributions, etc.
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_split_config UNIQUE (snapshot_id, name)
);

CREATE INDEX IF NOT EXISTS idx_split_config_snapshot ON split_configurations (snapshot_id);

-- Deterministic assignment of TRAIN rows into {train, val}
-- One row per (snapshot_id, image_id)
CREATE TABLE IF NOT EXISTS train_val_splits (
    split_configuration_id BIGINT NOT NULL REFERENCES split_configurations (id) ON DELETE CASCADE,
    image_id BIGINT NOT NULL, -- references mnist_images_train.id
    split TEXT NOT NULL CHECK (split IN ('train', 'val')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (
        split_configuration_id,
        image_id
    )
);

CREATE INDEX IF NOT EXISTS idx_tvs_split ON train_val_splits (split_configuration_id, split);
CREATE INDEX IF NOT EXISTS idx_tvs_image ON train_val_splits (image_id);

-- Info on how a model was trained for reproduciblity
-- Model version -> Feature Version -> Snapshot Version
CREATE TABLE IF NOT EXISTS model_artifacts (
    id BIGSERIAL PRIMARY KEY,
    split_configuration_id BIGINT NOT NULL REFERENCES split_configurations (id) ON DELETE RESTRICT,
    training_hash TEXT NOT NULL, -- SHA256 of training_params JSON (incl. model def, seed, etc.)
    params JSONB NOT NULL, -- full training params used
    metrics JSONB NOT NULL, -- val_acc, val_loss, optional per-class
    path TEXT NOT NULL, -- artifacts/snapshot_<id>/<ts>_<hash>/
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_model_key UNIQUE (
        split_configuration_id,
        training_hash
    )
);

CREATE INDEX IF NOT EXISTS idx_model_split_config ON model_artifacts (split_configuration_id);
CREATE INDEX IF NOT EXISTS idx_model_created_at ON model_artifacts (created_at DESC);