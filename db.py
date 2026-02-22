import os
import socket
from urllib.parse import urlparse, parse_qs

import psycopg2

DB_URL = os.environ["DATABASE_URL"]

def get_conn():
    u = urlparse(DB_URL)

    # user/pass
    user = u.username
    password = u.password

    # host/port/db
    host = u.hostname
    port = u.port or 5432
    dbname = (u.path or "").lstrip("/") or "postgres"

    # força IPv4 (resolve hostname -> IPv4)
    host_ipv4 = socket.gethostbyname(host)

    # sslmode (recomendado: require)
    qs = parse_qs(u.query or "")
    sslmode = (qs.get("sslmode", ["require"])[0]) or "require"

    return psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host_ipv4,
        port=port,
        sslmode=sslmode,
        connect_timeout=10,
    )

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
