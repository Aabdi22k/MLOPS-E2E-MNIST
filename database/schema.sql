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
    snapshot_id BIGINT NOT NULL
        REFERENCES data_snapshots(id) ON DELETE CASCADE,
    label SMALLINT NOT NULL CHECK (label BETWEEN 0 AND 9),
    pixels BYTEA NOT NULL                 
);

CREATE INDEX IF NOT EXISTS idx_mnist_train_snapshot ON mnist_images_train(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_mnist_train_label    ON mnist_images_train(label);

CREATE TABLE IF NOT EXISTS mnist_images_test (
    id BIGSERIAL PRIMARY KEY,
    snapshot_id BIGINT NOT NULL
        REFERENCES data_snapshots(id) ON DELETE CASCADE,
    label SMALLINT NOT NULL CHECK (label BETWEEN 0 AND 9),
    pixels BYTEA NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mnist_test_snapshot ON mnist_images_test(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_mnist_test_label    ON mnist_images_test(label);

-- Deterministic assignment of TRAIN rows into {train, val}
-- One row per (snapshot_id, image_id)
CREATE TABLE IF NOT EXISTS train_val_splits (
    snapshot_id BIGINT NOT NULL,
    image_id    BIGINT NOT NULL,          -- FK to mnist_images_train.id
    split       TEXT   NOT NULL CHECK (split IN ('train','val')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (snapshot_id, image_id),
    CONSTRAINT fk_tvs_snapshot
        FOREIGN KEY (snapshot_id) REFERENCES data_snapshots(id) ON DELETE CASCADE,
    CONSTRAINT fk_tvs_image
        FOREIGN KEY (image_id)    REFERENCES mnist_images_train(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tvs_snapshot_split ON train_val_splits (snapshot_id, split);
CREATE INDEX IF NOT EXISTS idx_tvs_image          ON train_val_splits (image_id);

-- How the train data was split into {train, val}
-- params example: {"random_seed":42,"val_frac":0.10,"feature_version":"v1"}
-- stats  example: {"counts":{"train":54000,"val":6000},"per_class":{"0":{"train":...,"val":...},...}}
CREATE TABLE IF NOT EXISTS transform_manifests (
    snapshot_id BIGINT PRIMARY KEY
        REFERENCES data_snapshots(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    params      JSONB NOT NULL,
    stats       JSONB NOT NULL
);