import datetime as dt
import yfinance as yf
from db import init_db, get_conn

TICKERS = ["AMCI","VMAR","VITL","UAL","MSFT","DIS","GPCR","NVDA"]

def main():

    init_db()

    today = dt.date.today()

    for ticker in TICKERS:

        tk = yf.Ticker(ticker)

        hist = tk.history(period="5d")

        if hist.empty:
            continue

        close = float(hist["Close"].iloc[-1])

        sql = """
        INSERT INTO daily_snapshots
        (ticker, asof_date, close)
        VALUES (%s,%s,%s)
        """

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql,(ticker,today,close))

if __name__ == "__main__":
    main()
