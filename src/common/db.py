# Database Helper Functions

import os
from functools import wraps

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()


# Connect to running PostgreSQL database
def get_conn():
    return psycopg2.connect(
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        host=os.getenv("PGHOST"),
        port=int(os.getenv("PGPORT")),  # type: ignore
    )


# Decorator function to open a DB connection
def with_conn(fn=None, *, commit=True):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # psycopg2 connection context manager auto-commits on success
            # and rolls back on exception.
            with get_conn() as conn:
                return f(conn, *args, **kwargs)

        return wrapper

    return decorator(fn) if fn else decorator


# Decorator Function to open a DB connection + cursor
def with_cursor(fn=None, *, commit=True):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            with get_conn() as conn:
                with conn.cursor() as cur:
                    return f(cur, *args, **kwargs)

        return wrapper

    return decorator(fn) if fn else decorator


# Execute a SQL command
@with_cursor
def execute(cur, sql, params=None):
    cur.execute(sql, params)


# Execture a batch of SQL commands at once
@with_cursor
def executemany(cur, sql, seq_of_params=None):
    psycopg2.extras.execute_batch(cur, sql, seq_of_params, page_size=1000)


# Execture a SQL command and return all rows
@with_cursor
def return_all(cur, sql, params=None):
    cur.execute(sql, params)
    return cur.fetchall()


# Execute a SQL command and return a single row
@with_cursor
def return_row(cur, sql, params=None):
    cur.execute(sql, params)
    row = cur.fetchone()
    return row if row else None


# Execute a SQL command and return a single value
@with_cursor
def return_scalar(cur, sql, params=None):
    cur.execute(sql, params)
    row = cur.fetchone()
    return row[0] if row else None


# Returns the id of the latest data snapshot
def latest_snapshot_id():
    sql = """ 
        SELECT id
        FROM data_snapshots
        ORDER BY created_at DESC
        LIMIT 1
    """
    sid = return_scalar(sql)  # type: ignore
    if not sid:
        raise RuntimeError("No data_snapshots found. Run extract first.")
    return int(sid)


def get_snapshot_by_split_config(split_configuration_id: int) -> int:
    sql = "SELECT snapshot_id FROM split_configurations WHERE id=%s"
    row = return_scalar(sql, [split_configuration_id])
    if row is None:
        raise RuntimeError(
            f"No split_configurations row for id={split_configuration_id}"
        )
    return int(row)
