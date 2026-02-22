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

# ---------------- LISTA DINÂMICA ----------------

if "tickers" not in st.session_state:
    st.session_state.tickers = DEFAULT_TICKERS.copy()


def normalize_ticker(t):
    t = (t or "").strip().upper()
    return re.sub(r"[^A-Z0-9\.\-]", "", t)


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("Adicionar ação")

    new_t = st.text_input("Ticker")

    if st.button("Adicionar"):

        t = normalize_ticker(new_t)

        if t and t not in st.session_state.tickers:

            st.session_state.tickers.append(t)

    if st.button("Resetar lista"):

        st.session_state.tickers = DEFAULT_TICKERS.copy()


    st.divider()

    if st.button("Atualizar dados"):

        st.cache_data.clear()


tickers = st.session_state.tickers


# ---------------- APAGAR TICKER ----------------

st.subheader("Tickers atuais")

cols = st.columns(len(tickers))

for i,t in enumerate(tickers):

    with cols[i]:

        st.write(t)

        if st.button("❌", key=f"del{i}"):

            st.session_state.tickers.remove(t)

            st.rerun()


# ---------------- CALCULOS ----------------


def pct(closes,n):

    if len(closes)<=n:
        return None

    return (closes.iloc[-1]/closes.iloc[-n]-1)


def pct_fmt(x):

    if x is None:
        return ""

    return f"{x*100:.2f}%"


@st.cache_data(ttl=300)
def get_data(t):

    tk=yf.Ticker(t)

    h=tk.history(period="13mo")

    if len(h)==0:
        return None

    c=h["Close"]

    info={}

    try:
        info=tk.info
    except:
        pass


    return {

        "Ticker":t,

        "Preço":round(c.iloc[-1],2),

        "1D":pct(c,1),

        "1W":pct(c,5),

        "2W":pct(c,10),

        "3M":pct(c,63),

        "6M":pct(c,126),

        "1Y":pct(c,252),

        "Analistas":info.get("recommendationKey",""),

        "Target Min":info.get("targetLowPrice",""),

        "Target Médio":info.get("targetMeanPrice",""),

        "Target Máx":info.get("targetHighPrice","")

    }



# ---------------- TABELA ----------------

st.subheader("Tabela ao vivo")

rows=[]

progress=st.progress(0)

for i,t in enumerate(tickers):

    d=get_data(t)

    if d:

        rows.append(d)

    progress.progress((i+1)/len(tickers))

progress.empty()


df=pd.DataFrame(rows)

for c in ["1D","1W","2W","3M","6M","1Y"]:

    if c in df:

        df[c]=df[c].apply(pct_fmt)


st.dataframe(df,use_container_width=True)

st.caption("Dados ao vivo - Yahoo Finance")
