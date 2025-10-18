# src/train/config.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from tracemalloc import Snapshot
from typing import Any, Dict, Optional

from click import Option

from src.common.utils import (
    ensure_env_loaded,
    get_env_bool,
    get_env_float,
    get_env_int,
    get_env_str,
)

ensure_env_loaded()


@dataclass(frozen=True)
class TrainConfig:
    """
    Class to hold all configurations needed for training the CNN
    """

    epochs: int
    batch_size: int
    seed: int
    learning_rate: float
    early_stop_patience: int
    artifacts_dir: str
    limit: Optional[int]
    force: bool
    model_name: str  # "baseline_cnn_v1"
    conv_filters: tuple[int, int]  # (32, 64)
    dense_units: int  # 128
    dropout_conv: float  # 0.25
    dropout_dense: float  # 0.5
    optimizer: str  # "adam"
    shuffle_buffer: int  # 10000
    early_stop_monitor: str  # "val_loss"
    early_stop_min_delta: float  # 0.0
    train_split_name: Optional[str]
    input_shape: tuple[int, int, int]
    num_classes: int
    snapshot_id: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config() -> TrainConfig:
    return TrainConfig(
        epochs=get_env_int("TRAIN_EPOCHS", 10),
        batch_size=get_env_int("TRAIN_BATCH_SIZE", 128),
        seed=get_env_int("TRAIN_RANDOM_SEED", 42),
        learning_rate=get_env_float("LEARNING_RATE", 0.001),
        early_stop_patience=get_env_int("EARLY_STOP_PATIENCE", 3),
        artifacts_dir=get_env_str("ARTIFACTS_DIR", "artifacts"),
        limit=(lambda x: x if x > 0 else None)(get_env_int("TRAIN_LIMIT", 0)),
        force=get_env_bool("TRAIN_FORCE", False),
        model_name=get_env_str("MODEL_NAME", "baseline_cnn_v1"),
        conv_filters=(get_env_int("CONV_F1", 32), get_env_int("CONV_F2", 64)),
        dense_units=get_env_int("DENSE_UNITS", 128),
        dropout_conv=get_env_float("DROPOUT_CONV", 0.25),
        dropout_dense=get_env_float("DROPOUT_DENSE", 0.5),
        optimizer=get_env_str("OPTIMIZER", "adam"),
        shuffle_buffer=get_env_int("SHUFFLE_BUFFER", 10000),
        early_stop_monitor=get_env_str("EARLY_STOP_MONITOR", "val_loss"),
        early_stop_min_delta=get_env_float("EARLY_STOP_MIN_DELTA", 0.0),
        train_split_name=(
            s if (s := get_env_str("TRAIN_SPLIT_NAME", "").strip()) else None
        ),
        input_shape=(
            get_env_int("INPUT_HEIGHT", 28),
            get_env_int("INPUT_WIDTH", 28),
            get_env_int("INPUT_CHANNELS", 1),
        ),
        num_classes=get_env_int("NUM_CLASSES", 10),
        snapshot_id=sid if (sid := get_env_int("SNAPSHOT_ID", 0) != 0) else None,
    )


CFG = load_config()
