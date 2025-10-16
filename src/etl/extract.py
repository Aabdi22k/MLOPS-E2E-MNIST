import os
import json
import numpy as np
from dotenv import load_dotenv
from sklearn.datasets import fetch_openml
from src.common.db import with_cursor, executemany, return_scalar
from sklearn.model_selection import train_test_split
from src.common.utils import sha256_bytes, numpy_uint8_to_bytes

load_dotenv()

def fetch_mnist_uint8():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)

    X = mnist['data']
    X_uint8 = np.clip(np.rint(X), 0, 255).astype(np.uint8)

    y_uint8 = mnist['target'].astype(int).astype(np.uint8)

    return X_uint8, y_uint8

def compute_hash(X, y):
    return sha256_bytes(X.tobytes(order='C') + y.tobytes(order='C'))

def find_snapshot_by_hashes(train_hash, test_hash):
    sql = """
        SELECT id
        FROM data_snapshots
        WHERE train_sha256 = %s AND test_sha256 = %s
        ORDER BY created_at DESC
        LIMIT 1
    """
    row = return_scalar(sql, (train_hash, test_hash))
    return int(row) if row else None

def insert_rows(snapshot_id, table_name, X, y, batch_size=5000):
    n = X.shape[0]
    sql = f'INSERT INTO {table_name} (snapshot_id, label, pixels) VALUES (%s, %s, %s)'

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = [(int(snapshot_id), int(lbl), numpy_uint8_to_bytes(row)) for row, lbl in zip(X[start:end], y[start:end])]
        executemany(sql, batch)
        print(f"[extract] inserted {end}/{n} into {table_name}...")

def main():
    print('[EXTRACT] start')

    DATA_SOURCE = os.getenv('DATA_SOURCE', "openml: mnist_784 v1")
    TEST_SIZE = int(os.getenv('TEST_SIZE', '10000'))
    RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))

    # Get the Images & Values from mnist as uint8 types
    X, y = fetch_mnist_uint8()

    # Split the data into the train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

    # Compute the train and test hash for integrity
    train_hash = compute_hash(X_train, y_train)
    test_hash = compute_hash(X_test, y_test)

    rows_train = int(len(X_train))
    rows_test = int(len(X_test))

    existing = find_snapshot_by_hashes(train_hash, test_hash)

    if not existing:
        # Create the snapshot of data for reproduciblity and get id
        snapshot_sql = """
            INSERT INTO data_snapshots (source, rows_train, rows_test, train_sha256, test_sha256) 
            VALUES (%s, %s, %s, %s, %s) 
            RETURNING id
        """
        snapshot_id = return_scalar(snapshot_sql, (DATA_SOURCE, rows_train, rows_test, train_hash, test_hash))
        
        # Insert the train and test data into the database
        insert_rows(snapshot_id, "mnist_images_train", X_train, y_train)
        insert_rows(snapshot_id, "mnist_images_test", X_test, y_test)
    
    # Print a summary of what data was extracted
    summary = {
        "message": "Dataset unchanged; reusing existing snapshot" if existing else "Created new snapshot",
        "snapshot_id": existing if existing else snapshot_id,
        "rows_train": rows_train,
        "rows_test": rows_test,
        "train_sha256": train_hash,
        "test_sha256": test_hash,
        "features": 784,
        "label_classes": list(range(10)),
        "source": DATA_SOURCE,
    }
    print(json.dumps(summary, indent=2))
    print("[EXTRACT] done.")

if __name__ == "__main__":
    main()
