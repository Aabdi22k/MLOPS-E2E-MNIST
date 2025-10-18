# src/train/evaluate.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.common.utils import to_serializable


def argmax_predictions(probs: np.ndarray) -> np.ndarray:
    """
    Accepts model probabilities of shape (N, C) and returns integer labels (N,).
    """
    if probs.ndim != 2:
        raise ValueError(f"Expected probs shape (N, C); got {probs.shape}")
    return probs.argmax(axis=1).astype(np.int64)


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 10,
) -> np.ndarray:
    """
    Fast confusion matrix without sklearn. Returns (num_classes, num_classes)
    where rows = true, cols = pred.
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    # vectorized bincount trick
    idx = y_true * num_classes + y_pred
    counts = np.bincount(idx, minlength=num_classes * num_classes)
    cm[:] = counts.reshape(num_classes, num_classes)
    return cm


def per_class_accuracy(cm: np.ndarray) -> Dict[str, float]:
    """
    Compute accuracy per true class from a confusion matrix.
    """
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("Confusion matrix must be square")
    totals = cm.sum(axis=1)
    correct = np.diag(cm)
    # avoid divide-by-zero
    accs = {}
    for i in range(cm.shape[0]):
        acc = float(correct[i]) / float(totals[i]) if totals[i] > 0 else 0.0
        accs[str(i)] = acc
    return accs


def evaluate_on_val_dataset(
    model,  # Keras model
    ds_val,  # tf.data.Dataset or tuple (X_val, y_val)
    num_classes: int = 10,
) -> Dict[str, Any]:
    """
    Evaluate model on validation data and return a JSON-safe metrics dict:
    {
      "val_loss": ...,
      "val_accuracy": ...,
      "confusion_matrix": [[...], ...],
      "per_class_accuracy": {"0": ..., "1": ..., ...}
    }
    """
    # Import TensorFlow lazily to keep non-train code paths light
    import tensorflow as tf

    # Support either a tf.data dataset or raw arrays
    if isinstance(ds_val, tf.data.Dataset):
        val_loss, val_accuracy = model.evaluate(ds_val, verbose=0)
        # Collect arrays for confusion matrix
        y_true_list, y_pred_list = [], []
        for Xb, yb in ds_val:
            probs = model.predict(Xb, verbose=0)
            y_pred_list.append(np.asarray(probs).argmax(axis=1))
            y_true_list.append(np.asarray(yb))
        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
    else:
        # Expect (X_val, y_val) tuple
        X_val, y_val = ds_val
        probs = model.predict(X_val, verbose=0)
        y_pred = argmax_predictions(np.asarray(probs))
        y_true = np.asarray(y_val)
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    per_cls = per_class_accuracy(cm)

    return {
        "val_loss": to_serializable(val_loss),
        "val_accuracy": to_serializable(val_accuracy),
        "confusion_matrix": to_serializable(cm),
        "per_class_accuracy": {k: to_serializable(v) for k, v in per_cls.items()},
        "n_val": int(len(y_true)),
    }
