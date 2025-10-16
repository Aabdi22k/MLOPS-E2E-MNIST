import os
import json
import math
import psycopg2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def _conn():
    return psycopg2.connect(
        dbname=os.getenv('PGDATABASE'),
        user=os.getenv('PGUSER'),
        password=os.getenv('PGPASSWORD'),
        host=os.getenv('PGHOST'),
        port=int(os.getenv('PGPORT', '5432'))
    )

def _latest_snapshot_id(cur):
    cur.execute("SELECT id FROM data_snapshots ORDER BY created_at DESC LIMIT 1;")
    row = cur.fetchone()
    assert row, "No snapshot; run extract first."
    return int(row[0])

def test_split_coverage_and_exclusivity():
    with _conn() as conn:
        with conn.cursor() as cur:
            snap = _latest_snapshot_id(cur)

            # expected counts from snapshot
            cur.execute("SELECT rows_train FROM data_snapshots WHERE id=%s;", (snap,))
            expected_train = int(cur.fetchone()[0])

            # counts in split table (should match rows_train)
            cur.execute("""
                SELECT split, COUNT(*) 
                FROM train_val_splits 
                WHERE snapshot_id=%s 
                GROUP BY split;
            """, (snap,))
            counts = dict(cur.fetchall())
            total = sum(counts.values())
            assert total == expected_train, f"Split rows={total}, expected rows_train={expected_train}"

            # exclusivity: no dup image_ids within snapshot
            cur.execute("""
                SELECT image_id, COUNT(*) 
                FROM train_val_splits 
                WHERE snapshot_id=%s 
                GROUP BY image_id 
                HAVING COUNT(*)<>1
            """, (snap,))
            dupes = cur.fetchall()
            assert not dupes, f"Some image_ids assigned !=1 time: {dupes[:5]}"

            # ensure all image_ids exist in mnist_images_train
            cur.execute("""
                SELECT COUNT(*) 
                FROM train_val_splits s
                LEFT JOIN mnist_images_train t
                ON t.id = s.image_id AND t.snapshot_id = s.snapshot_id
                WHERE s.snapshot_id=%s AND t.id IS NULL
            """, (snap,))
            missing = int(cur.fetchone()[0])
            assert missing == 0, f"{missing} split rows reference missing train images"

def test_ratio_and_per_label_balance():
    with _conn() as conn:
        with conn.cursor() as cur:
            snap = _latest_snapshot_id(cur)

            # join labels for this snapshot
            cur.execute("""
                SELECT t.label, s.split
                FROM mnist_images_train t
                JOIN train_val_splits s
                  ON s.image_id = t.id AND s.snapshot_id = t.snapshot_id
                WHERE t.snapshot_id=%s;
            """, (snap,))
            rows = cur.fetchall()
            assert rows, "No joined rows; run transform."

            # overall ratio ~ 0.10 val
            total = len(rows)
            val = sum(1 for (_, split) in rows if split == 'val')
            frac_val = val / total
            assert 0.09 <= frac_val <= 0.11, f"Val frac={frac_val:.4f} not ~0.10"

            # per-label ratio sanity
            from collections import defaultdict
            per = defaultdict(lambda: {"train":0, "val":0})
            for label, split in rows:
                per[int(label)][split] += 1
            for d in range(10):
                c = per[d]["train"] + per[d]["val"]
                assert c > 0, f"Label {d} missing"
                f = per[d]["val"] / c
                assert 0.08 <= f <= 0.12, f"Label {d} val frac={f:.4f} off 10%"

def test_manifest_present_and_sane():
    with _conn() as conn:
        with conn.cursor() as cur:
            snap = _latest_snapshot_id(cur)
            cur.execute("SELECT params::text, stats::text FROM transform_manifests WHERE snapshot_id=%s;", (snap,))
            row = cur.fetchone()
            assert row, "No transform_manifest for latest snapshot."
            params = json.loads(row[0])
            stats = json.loads(row[1])

            # basic keys present
            assert "random_seed" in params and "val_frac" in params
            assert "counts" in stats and "per_label" in stats or "per_class" in stats

            # counts sanity
            counts = stats["counts"]
            assert counts["total_all"] == counts["total_train"] + counts["total_val"]
            assert 0.09 <= counts["total_val"]/counts["total_all"] <= 0.11
