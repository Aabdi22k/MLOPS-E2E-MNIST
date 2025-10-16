# src/train/config.py
from typing import Any, Dict
from __future__ import annotations
from dataclasses import dataclass, asdict
from src.common.utils import get_env_bool, get_env_float, get_env_int, get_env_str, ensure_env_loaded

ensure_env_loaded()

@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    seed: int
    learning_rate: float
    early_stop_patience: int
    artifacts_dir: str
    limit: int | None
    force: bool
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
    )

CFG = load_config()
