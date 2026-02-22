import os
import psycopg2

DB_URL = os.environ["DATABASE_URL"]

def get_conn():
    return psycopg2.connect(DB_URL)

def init_db():
    sql = """
    CREATE TABLE IF NOT EXISTS daily_snapshots (
        ticker TEXT,
        asof_date DATE,
        close NUMERIC
    );
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
