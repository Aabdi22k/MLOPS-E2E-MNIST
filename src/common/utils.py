# Miscellaneous Helper Functions

import hashlib
import numpy as np

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
