import json
import numpy as np
from sklearn.datasets import fetch_openml
from src.common.db import execute, executemany
from sklearn.model_selection import train_test_split
from src.common.utils import sha256_bytes, numpy_uint8_to_bytes

def fetch_mnist_uint8():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)

    X = mnist['data']
    X_uint8 = np.clip(np.rint(X), 0, 255).astype(np.uint8)

    y_uint8 = mnist['target'].astype(int).astype(np.uint8)

    return X_uint8, y_uint8

def compute_hash(X, y):
    return sha256_bytes(X.tobytes(order='C') + y.tobytes(order='C'))

def insert_rows(table_name, X, y, batch_size=5000):
    n = X.shape[0]
    sql = f'INSERT INTO {table_name} (label, pixels) VALUES (%s, %s)'

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = [(int(lbl), numpy_uint8_to_bytes(row)) for row, lbl in zip(X[start:end], y[start:end])]
        executemany(sql, batch)
        print(f"[extract] inserted {end}/{n} into {table_name}...")

def main():
    # Get the Images & Values from mnist as uint8 types
    X, y = fetch_mnist_uint8()

    # Split the data into the train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42, stratify=y)

    # Insert the train and test data into the database
    insert_rows("mnist_images_train", X_train, y_train)
    insert_rows("mnist_images_test", X_test, y_test)

    # Computer the train and test hash for integrity
    train_hash = compute_hash(X_train, y_train)
    test_hash = compute_hash(X_test, y_test)

    # Create the snapshot of data for reproduciblity
    snapshot_source = "openml: mnist_784 v1"
    snapshot_rows_train = int(len(X_train))
    snapshot_rows_test = int(len(X_test))
    snapshot_sql = f'INSERT INTO data_snapshots (source, rows_train, rows_test, train_sha256, test_sha256) VALUES (%s, %s, %s, %s, %s)'
    execute(snapshot_sql, (snapshot_source, snapshot_rows_train, snapshot_rows_test, train_hash, test_hash))

    # Print a summary of what data was extracted
    summary = {
        "rows_train": snapshot_rows_train,
        "rows_test": snapshot_rows_test,
        "train_sha256": train_hash,
        "test_sha256": test_hash,
        "features": 784,
        "label_classes": list(range(10)),
        "source": snapshot_source,
    }
    print(json.dumps(summary, indent=2))
    print("[extract] done.")

if __name__ == "__main__":
    main()
