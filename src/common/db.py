# Database Helper Functions

import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

# Connect to running PostgreSQL database
def get_conn():
    return psycopg2.connect(
        dbname=os.getenv('PGDATABASE'),
        user=os.getenv('PGUSER'),
        password=os.getenv('PGPASSWORD'),
        host=os.getenv('PGHOST'),
        port=int(os.getenv('PGPORT'))
    )

# Execute 1 sql command with params
def execute(sql, params=None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
    
# Execture a bunch of sql command with all their params
def executemany(sql, seq_of_params):
    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, seq_of_params, page_size=1000)

def scalar(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()[0]
