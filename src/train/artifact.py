# src/train/artifacts.py
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from src.common.utils import ensure_dir, save_json, save_text, to_jsonable, utc_compact


def make_artifact_dir(
    artifacts_dir: str,
    snapshot_id: int,
    training_hash: str,
) -> Path:
    """
    Create directory like:
      artifacts/snapshot_<id>/<UTC_TIMESTAMP>_<training_hash>/
    Returns the created Path.
    """
    root = Path(artifacts_dir)
    run_dir = root / f"snapshot_{snapshot_id}" / f"{utc_compact()}_{training_hash}"
    ensure_dir(run_dir)
    return run_dir


def save_keras_model(model, run_dir: Path) -> Path:
    """
    Save a compact Keras model file + model config JSON.
    Returns the model path.
    """
    # Prefer single-file H5 for portability; SavedModel is larger/noisier for this use-case.
    model_path = run_dir / "model.keras"
    model.save(model_path)  # includes weights & topology
    # Also persist explicit model config for readability
    cfg_path = run_dir / "model_config.json"
    try:
        cfg_json = model.to_json()
        save_text(cfg_json, cfg_path)
    except Exception:
        # Some custom models may not support to_json; skip gracefully
        pass
    return model_path


def write_artifact_bundle(
    run_dir: Path,
    training_params,
    metrics: Dict[str, Any],
    history: Optional[Dict[str, Any]] = None,
    model_signature: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save params/metrics/history/signature JSON files under the run directory.
    """
    save_json(to_jsonable(training_params), run_dir / "params.json")
    save_json(to_jsonable(metrics), run_dir / "metrics.json")
    if history:
        save_json(to_jsonable(history), run_dir / "train_history.json")
    if model_signature:
        save_json(to_jsonable(model_signature), run_dir / "model_signature.json")
