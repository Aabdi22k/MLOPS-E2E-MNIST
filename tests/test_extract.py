import os

import psycopg2
import psycopg2.extras
import pytest
from dotenv import load_dotenv

load_dotenv()


# Connect to running PostgreSQL database
def get_conn():
    return psycopg2.connect(
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        host=os.getenv("PGHOST"),
        port=int(os.getenv("PGPORT", "5432")),
    )


@pytest.fixture(scope="module")
def conn():
    c = get_conn()
    yield c
    c.close()


def scalar(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        return row[0] if row else None


def latest_snapshot(conn):
    sql = """
      SELECT id, rows_train, rows_test, train_sha256, test_sha256
      FROM data_snapshots
      ORDER BY created_at DESC
      LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
        assert row is not None, "No snapshot found; run src/etl/extract.py first."
        return {
            "id": row[0],
            "rows_train": row[1],
            "rows_test": row[2],
            "train_sha256": row[3],
            "test_sha256": row[4],
        }


def test_counts_match_snapshot(conn):
    snap = latest_snapshot(conn)
    cnt_train = scalar(
        conn,
        "SELECT COUNT(*) FROM mnist_images_train WHERE snapshot_id = %s",
        (snap["id"],),
    )
    cnt_test = scalar(
        conn,
        "SELECT COUNT(*) FROM mnist_images_test  WHERE snapshot_id = %s",
        (snap["id"],),
    )
    assert (
        cnt_train == snap["rows_train"]
    ), f"train count {cnt_train} != snapshot {snap['rows_train']}"
    assert (
        cnt_test == snap["rows_test"]
    ), f"test count {cnt_test} != snapshot {snap['rows_test']}"


def test_label_range(conn):
    snap = latest_snapshot(conn)
    min_train = scalar(
        conn,
        "SELECT MIN(label) FROM mnist_images_train WHERE snapshot_id = %s",
        (snap["id"],),
    )
    max_train = scalar(
        conn,
        "SELECT MAX(label) FROM mnist_images_train WHERE snapshot_id = %s",
        (snap["id"],),
    )
    min_test = scalar(
        conn,
        "SELECT MIN(label) FROM mnist_images_test  WHERE snapshot_id = %s",
        (snap["id"],),
    )
    max_test = scalar(
        conn,
        "SELECT MAX(label) FROM mnist_images_test  WHERE snapshot_id = %s",
        (snap["id"],),
    )
    assert (
        min_train >= 0 and max_train <= 9
    ), f"train labels out of range: [{min_train},{max_train}]"
    assert (
        min_test >= 0 and max_test <= 9
    ), f"test labels out of range: [{min_test},{max_test}]"


def test_no_nulls(conn):
    snap = latest_snapshot(conn)
    n1 = scalar(
        conn,
        "SELECT COUNT(*) FROM mnist_images_train WHERE snapshot_id = %s AND (label IS NULL OR pixels IS NULL);",
        (snap["id"],),
    )
    n2 = scalar(
        conn,
        "SELECT COUNT(*) FROM mnist_images_test  WHERE snapshot_id = %s AND (label IS NULL OR pixels IS NULL);",
        (snap["id"],),
    )
    assert n1 == 0, f"mnist_images_train has {n1} NULL label/pixels rows"
    assert n2 == 0, f"mnist_images_test has {n2} NULL label/pixels rows"


def test_pixels_length_784(conn):
    snap = latest_snapshot(conn)
    # Sample up to 1000 rows for speed; octet_length(pixels) is byte length (should be 784)
    bad_train = scalar(
        conn,
        "SELECT COUNT(*) FROM (SELECT 1 FROM mnist_images_train WHERE snapshot_id = %s AND octet_length(pixels) <> 784 LIMIT 1000) t;",
        (snap["id"],),
    )
    bad_test = scalar(
        conn,
        "SELECT COUNT(*) FROM (SELECT 1 FROM mnist_images_test WHERE snapshot_id = %s AND octet_length(pixels) <> 784 LIMIT 1000) t;",
        (snap["id"],),
    )
    assert bad_train == 0, "Found train rows with bytes != 784 (first 1000 sampled)"
    assert bad_test == 0, "Found test rows with bytes != 784 (first 1000 sampled)"


def test_label_distribution_reasonable(conn):
    snap = latest_snapshot(conn)
    sql = """
      SELECT label, COUNT(*) AS c
      FROM mnist_images_train
      WHERE snapshot_id = %s
      GROUP BY label
      ORDER BY label
    """
    with conn.cursor() as cur:
        cur.execute(sql, (snap["id"],))
        rows = cur.fetchall()
    counts = {int(label): int(c) for (label, c) in rows}
    for d in range(10):
        assert d in counts, f"Missing label {d} in train set"
        assert counts[d] >= 4000, f"Label {d} has too few examples: {counts[d]} (<4000)"
