# Miscellaneous Helper Functions
import os
import hashlib
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
    return (X_uint8.astype(np.float32) / 255.0)
