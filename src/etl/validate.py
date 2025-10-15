import sys
import psycopg2
from src.common.db import get_conn, scalar

def check_counts(conn):
    expected = scalar(conn, "SELECT rows_train FROM data_snapshots ORDER BY created_at DESC LIMIT 1;")
    actual = scalar(conn, "SELECT COUNT(*) FROM mnist_images_train;")
    return actual == expected, f"Row count {actual} vs {expected}"

def check_labels(conn):
    bad = scalar(conn, "SELECT COUNT(*) FROM mnist_images_train WHERE label NOT BETWEEN 0 AND 9;")
    return bad == 0, f"{bad} invalid labels" if bad else "Labels valid"

def check_nulls(conn):
    bad = scalar(conn, "SELECT COUNT(*) FROM mnist_images_train WHERE label IS NULL OR pixels IS NULL;")
    return bad == 0, f"{bad} NULL values" if bad else "No NULLs"

def check_pixel_length(conn):
    bad = scalar(conn, "SELECT COUNT(*) FROM mnist_images_train WHERE octet_length(pixels) != 784;")
    return bad == 0, f"{bad} invalid pixel arrays" if bad else "Pixels OK"

def check_distribution(conn):
    result = scalar(conn, "SELECT MIN(c) FROM (SELECT COUNT(*) c FROM mnist_images_train GROUP BY label) s;")
    return result >= 4000, f"Min label count {result}"

def main():
    checks = [check_counts, check_labels, check_nulls, check_pixel_length, check_distribution]
    with get_conn() as conn:
        all_passed = True
        for func in checks:
            ok, msg = func(conn)
            if ok:
                print(f"Passed: {msg}")
            else:
                print(f"Failed: {msg}")
                all_passed = False
        sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
