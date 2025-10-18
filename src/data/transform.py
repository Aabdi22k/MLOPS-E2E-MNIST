import datetime as dt
import json
import os

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from src.common.db import executemany, latest_snapshot_id, return_all, return_scalar
from src.common.utils import ensure_env_loaded, get_env_float, get_env_int, get_env_str

ensure_env_loaded()


# Load ids & labels for this snapshot's training dataset
def load_train_ids_and_labels(snapshot_id):
    sql = """
        SELECT id, label
        FROM mnist_images_train
        WHERE snapshot_id = %s
        ORDER BY id
    """
    rows = return_all(sql, [snapshot_id])
    if not rows:
        raise RuntimeError(f"No training rows for snapshot_id={snapshot_id}")
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    labels = np.array([r[1] for r in rows], dtype=np.int64)

    return ids, labels


# Insert or update split configurations table with the params
# and stats of the train & val split for this snapshot
def upsert_split_configurations(snapshot_id, name, params, stats):
    sql = """
        INSERT INTO split_configurations (snapshot_id, name, params, stats)
        VALUES (%s, %s, %s::jsonb, %s::jsonb)
        ON CONFLICT (snapshot_id, name)
        DO UPDATE SET params = EXCLUDED.params, stats = EXCLUDED.stats, created_at = NOW()
        RETURNING id
    """
    row = return_scalar(sql, (snapshot_id, name, json.dumps(params), json.dumps(stats)))
    return int(row) if row else None


# mapping into train_val_splits table which images are part of
# training or validation {train, val} for this snapshot
def upsert_split_rows(split_configuration_id, ids_train, ids_val):
    sql = """
        INSERT INTO train_val_splits (split_configuration_id, image_id, split)
        VALUES (%s, %s, %s)
        ON CONFLICT (split_configuration_id, image_id)
        DO UPDATE SET split = EXCLUDED.split, created_at = NOW()
    """
    params = [(split_configuration_id, int(i), "train") for i in ids_train] + [
        (split_configuration_id, int(i), "val") for i in ids_val
    ]
    executemany(sql, params)


# Copmute counts of train rows, val rows, and both along with
# counts of train and val rows per label (0 - 9) for manifest logging
def compute_stats(ids_all, labels_all, ids_train, ids_val):
    id_to_label = dict(zip(ids_all.tolist(), labels_all.tolist()))

    labels_train = np.array([id_to_label[i] for i in ids_train], dtype=np.int64)
    labels_val = np.array([id_to_label[i] for i in ids_val], dtype=np.int64)

    stats = {
        "counts": {
            "total_all": int(len(ids_all)),
            "total_train": int(len(ids_train)),
            "total_val": int(len(ids_val)),
        },
        "per_label": {
            str(d): {
                "train": int(np.sum(labels_train == d)),
                "val": int(np.sum(labels_val == d)),
            }
            for d in range(10)
        },
    }
    return stats


def main():
    print("[TRANSFORM] start")

    VAL_FRACTION = get_env_float("VAL_FRACTION", 0.1)
    TRANSFORM_RANDOM_SEED = get_env_int("TRANSFORM_RANDOM_SEED", 42)
    SPLIT_NAME = get_env_str("SPLIT_NAME", "default")

    # Get the latest snapshot to transform
    snapshot_id = latest_snapshot_id()

    # Pull image ids and labels for this snapshot from DB
    ids, labels = load_train_ids_and_labels(snapshot_id)

    # Make deterministic stratified train/val split
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=VAL_FRACTION, random_state=TRANSFORM_RANDOM_SEED
    )
    ((idx_train, idx_val),) = sss.split(ids.reshape(-1, 1), labels)
    ids_train = ids[idx_train]
    ids_val = ids[idx_val]

    # Get and log train/val split stats and params to DB
    stats = compute_stats(ids, labels, ids_train, ids_val)
    params = {
        "random_seed": TRANSFORM_RANDOM_SEED,
        "val_frac": VAL_FRACTION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    split_configuration_id = upsert_split_configurations(
        snapshot_id, SPLIT_NAME, params, stats
    )

    # Log exact train/val split to DB
    upsert_split_rows(split_configuration_id, ids_train, ids_val)

    print(
        json.dumps(
            {
                "snapshot_id": snapshot_id,
                "split_configuration_id": split_configuration_id,
                "split_name": SPLIT_NAME,
                "params": params,
                "stats": stats,
            },
            indent=2,
        )
    )

    print("[TRANSFORM] done")


if __name__ == "__main__":
    main()
