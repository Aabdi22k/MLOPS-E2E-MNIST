import json
from wsgiref.handlers import CGIHandler

from pytest import param
import tensorflow as tf

from src.common.db import get_snapshot_by_split_config, return_row, return_scalar
from src.common.utils import historyto_serializable, sha256_json, utc_compact
from src.train.artifact import (
    make_artifact_dir,
    save_keras_model,
    write_artifact_bundle,
)
from src.train.build import build_cnn, build_tf_datasets, create_model_signature
from src.train.config import CFG
from src.train.evaluate import evaluate_on_val_dataset
from src.train.load import load_training_data


def get_existing_model_artifact(split_configuration_id, training_hash):
    sql = """
        SELECT id
        FROM model_artifacts
        WHERE split_configuration_id=%s AND training_hash=%s
        LIMIT 1
    """
    existing = return_row(sql, (split_configuration_id, training_hash))
    return int(existing[0]) if existing else None


def upsert_model_artifact(split_configuration_id, training_hash, params, metrics, path):
    sql = """
        INSERT INTO model_artifacts (split_configuration_id, training_hash, params, metrics, path)
        VALUES (%s, %s, %s::jsonb, %s::jsonb, %s)
        ON CONFLICT (split_configuration_id, training_hash)
        DO NOTHING
        RETURNING id
    """
    row = return_row(
        sql,
        (
            split_configuration_id,
            training_hash,
            json.dumps(params),
            json.dumps(metrics),
            path,
        ),
    )
    return int(row[0]) if row else None


def main():

    scid, (Xtr, ytr), (Xva, yva) = load_training_data(CFG.snapshot_id)

    sid = get_snapshot_by_split_config(scid)

    ds_tr, ds_va = build_tf_datasets(
        Xtr, ytr, Xva, yva, CFG.shuffle_buffer, CFG.seed, CFG.batch_size
    )

    model = build_cnn(
        CFG.conv_filters,
        CFG.seed,
        CFG.input_shape,
        CFG.dropout_conv,
        CFG.dense_units,
        CFG.dropout_dense,
        CFG.num_classes,
        CFG.learning_rate,
    )
    sig = create_model_signature(
        CFG.model_name, CFG.input_shape, CFG.num_classes, CFG.learning_rate
    )

    training_params = {
        "epochs": CFG.epochs,
        "batch_size": CFG.batch_size,
        "seed": CFG.seed,
        "learning_rate": CFG.learning_rate,
        "early_stop_patience": CFG.early_stop_patience,
        "early_stop_monitor": CFG.early_stop_monitor,
        "early_stop_min_delta": CFG.early_stop_min_delta,
        "limit": None if CFG.limit in (None, 0) else int(CFG.limit),     
    }

    training_hash = sha256_json(training_params)

    existing = get_existing_model_artifact(scid, training_hash)
    if existing and not CFG.force:
        summary = {
            "message": "Existing model artifact found",
            "snapshot_id": sid,
            "split_configuration_id": scid,
            "model_artifact_id": existing,
            "split_configuration_id": scid,
            "training_hash": training_hash,
            "training_parameters": training_params,
            "model_signature": sig,
        }
        print(json.dumps(summary, indent=2))
        return

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=CFG.early_stop_monitor,
            patience=CFG.early_stop_patience,
            min_delta=CFG.early_stop_min_delta,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        ds_tr, validation_data=ds_va, epochs=CFG.epochs, verbose=1, callbacks=callbacks
    )

    hist = historyto_serializable(history)

    metrics = evaluate_on_val_dataset(model, ds_va)

    artifact_dir = make_artifact_dir(CFG.artifacts_dir, sid, training_hash)
    _ = save_keras_model(model, artifact_dir)
    write_artifact_bundle(artifact_dir, training_params, metrics, hist, sig)

    artifact_id = upsert_model_artifact(
        scid, training_hash, training_params, metrics, artifact_dir.as_posix()
    )

    summary = {
        "status": "trained",
        "model_artifact_id": artifact_id,
        "split_configuration_id": scid,
        "snapshot_id": sid,
        "training_hash": training_hash,
        "artifact_path": artifact_dir.as_posix(),
        "metrics": {
            "val_accuracy": metrics.get("val_accuracy"),
            "val_loss": metrics.get("val_loss"),
            "n_val": metrics.get("n_val"),
        },
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
