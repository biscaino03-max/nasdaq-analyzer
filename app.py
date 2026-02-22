import re
import pandas as pd
import streamlit as st
import yfinance as yf

DEFAULT_TICKERS = ["MSFT","NVDA","UAL","DIS"]

WINDOWS = {
    "1D": 1,
    "1W": 5,
    "2W": 10,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
}

st.set_page_config(page_title="Nasdaq Analyzer", layout="wide")

st.title("Nasdaq Analyzer (ao vivo)")

# Lista dinâmica
if "tickers" not in st.session_state:
    st.session_state.tickers = DEFAULT_TICKERS.copy()

def normalize_ticker(t):
    return re.sub(r"[^A-Z0-9\.]", "", t.upper())

# Sidebar
with st.sidebar:

    st.header("Adicionar ação")

    t = st.text_input("Ticker")

    if st.button("Adicionar"):
        t = normalize_ticker(t)

        if t and t not in st.session_state.tickers:
            st.session_state.tickers.append(t)

    if st.button("Resetar"):
        st.session_state.tickers = DEFAULT_TICKERS.copy()

tickers = st.session_state.tickers

st.write("Tickers:", tickers)

# Cálculo retornos

def pct(closes,n):

    if len(closes)<=n:
        return None

    return (closes.iloc[-1]/closes.iloc[-n]-1)


@st.cache_data(ttl=300)
def get_data(t):

    tk=yf.Ticker(t)

    h=tk.history(period="13mo")

    if len(h)==0:
        return None

    c=h["Close"]

    return {
        "Ticker":t,
        "Preço":round(c.iloc[-1],2),
        "1D":pct(c,1),
        "1W":pct(c,5),
        "2W":pct(c,10),
        "3M":pct(c,63),
        "6M":pct(c,126),
        "1Y":pct(c,252)
    }

rows=[]

for t in tickers:

    d=get_data(t)

    if d:
        rows.append(d)

df=pd.DataFrame(rows)

for c in ["1D","1W","2W","3M","6M","1Y"]:

    if c in df:
        df[c]=df[c].apply(lambda x: "" if x is None else f"{x*100:.2f}%")

st.dataframe(df,use_container_width=True)

st.caption("Dados ao vivo - Yahoo Finance")
