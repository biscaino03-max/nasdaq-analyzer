import os
import psycopg2

DB_URL = os.environ["DATABASE_URL"]

def get_conn():
    conn = psycopg2.connect(DB_URL, connect_timeout=15)
    conn.autocommit = True
    return conn

def init_db():
    sql = """
    CREATE TABLE IF NOT EXISTS daily_snapshots (
        ticker TEXT NOT NULL,
        asof_date DATE NOT NULL,
        close NUMERIC,
        ret_1d NUMERIC,
        ret_1w NUMERIC,
        ret_2w NUMERIC,
        ret_3m NUMERIC,
        ret_6m NUMERIC,
        ret_1y NUMERIC,
        analyst_summary TEXT,
        target_low NUMERIC,
        target_mean NUMERIC,
        target_high NUMERIC,
        source_note TEXT,
        PRIMARY KEY (ticker, asof_date)
    );
    CREATE INDEX IF NOT EXISTS idx_daily_snapshots_ticker_date
    ON daily_snapshots (ticker, asof_date DESC);
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
