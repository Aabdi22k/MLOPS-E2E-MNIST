# Miscellaneous Helper Functions
import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone

import numpy as np
from dotenv import load_dotenv

# load .env exactly once for the whole process
_load_done = False


def ensure_env_loaded() -> None:
    global _load_done
    if not _load_done:
        load_dotenv()
        _load_done = True


# Create an integrity hash for a byte sequence
def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


# Converts an array of unit8(pixel values) into raw bytes
# to store in PostgreSQL Database
def numpy_uint8_to_bytes(arr: np.ndarray) -> bytes:
    assert arr.dtype == np.uint8, "Array must be uint8 (0-255)"
    return arr.tobytes(order="C")


def get_env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    return v.strip() if v is not None else default


def get_env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if not v or not v.strip():
        return default
    try:
        return int(v)
    except ValueError:
        return default


def get_env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if not v or not v.strip():
        return default
    try:
        return float(v)
    except ValueError:
        return default


def get_env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_to_float32_01(X_uint8: np.ndarray) -> np.ndarray:
    return X_uint8.astype(np.float32) / 255.0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_json(d) -> str:
    payload = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def to_serializable(x):
    """
    Convert numpy scalars/arrays to JSON-safe Python types.
    """
    if isinstance(x, (np.floating, np.float32, np.float64)):  # type: ignore
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):  # type: ignore
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def historyto_serializable(history_obj):
    """
    Convert Keras History to JSON-safe dict.
    """
    if history_obj is None:
        return {}
    hist = getattr(history_obj, "history", {})
    return {k: to_serializable(np.asarray(v)) for k, v in hist.items()}


def ensure_dir(p) -> None:
    p.mkdir(parents=True, exist_ok=False)


def to_jsonable(obj):
    """
    Make objects JSON-safe (dataclasses, numpy scalars/arrays handled upstream).
    """
    if is_dataclass(obj):
        return asdict(obj)  # type: ignore
    return obj


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def save_text(text, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
