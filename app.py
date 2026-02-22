import pandas as pd
import streamlit as st
from db import init_db, get_conn

WATCHLIST = ["AMCI","VMAR","VITL","UAL","MSFT","DIS","GPCR","NVDA"]

st.set_page_config(page_title="Nasdaq Analyzer", layout="wide")
st.title("Nasdaq Analyzer")

init_db()

def fmt_pct(x):
    if x is None or pd.isna(x):
        return ""
    return f"{x*100:.2f}%"

@st.cache_data(ttl=60)
def load_latest():
    sql = """
    SELECT DISTINCT ON (ticker) *
    FROM daily_snapshots
    ORDER BY ticker, asof_date DESC;
    """
    with get_conn() as conn:
        return pd.read_sql(sql, conn)

df = load_latest()

tab1, tab2 = st.tabs(["Nasdaq – Recomendações", "Investidos"])

with tab2:
    st.subheader("Investidos")
    if df.empty:
        st.warning("Ainda sem dados.")
    else:
        st.dataframe(df)

with tab1:
    st.subheader("Nasdaq – Recomendações")
    st.dataframe(df)
