import re
import pandas as pd
import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup

# ---------- CONFIG ----------

INVESTIDOS_DEFAULT = ["AMCI","VMAR","VITL","UAL","MSFT","DIS","GPCR","NVDA"]
EM_ANALISE_DEFAULT = []

WINDOWS = {
    "1D":1,
    "1W":5,
    "2W":10,
    "3M":63,
    "6M":126,
    "1Y":252
}

ANALYST_LABELS = {
    "strong_buy": "🟢 Compra forte",
    "buy": "🔵 Compra",
    "hold": "🟡 Manter",
    "sell": "🔴 Venda",
    "strong_sell": "🟥 Venda forte",
    None: "—",
    "none": "—",
}

TTL_SECONDS = 300

st.set_page_config(page_title="Nasdaq Analyzer",layout="wide")
st.title("Nasdaq Analyzer (ao vivo)")


# ---------- STATE ----------

if "tickers_investidos" not in st.session_state:
    st.session_state.tickers_investidos=INVESTIDOS_DEFAULT.copy()

if "tickers_em_analise" not in st.session_state:
    st.session_state.tickers_em_analise=EM_ANALISE_DEFAULT.copy()



# ---------- HELPERS ----------

def normalize_ticker(t):
    t=(t or "").upper().strip()
    t=re.sub(r"[^A-Z0-9\.\-]","",t)
    return t


def pct_change(closes,n):

    if len(closes)<=n:
        return None

    return float((closes.iloc[-1]/closes.iloc[-1-n])-1)


def pretty_label(key):

    if key in ANALYST_LABELS:
        return ANALYST_LABELS[key]

    return key



def bg_return(v):

    if v is None:
        return ""

    if v>0:
        return "background-color:rgba(0,200,0,0.18)"

    if v<0:
        return "background-color:rgba(255,0,0,0.18)"

    return ""


# ---------- STATUS ----------

def yahoo_test():

    try:
        t=yf.Ticker("MSFT")
        d=t.history(period="5d")
        return not d.empty
    except:
        return False



def stockanalysis_targets(ticker):

    try:

        url=f"https://stockanalysis.com/stocks/{ticker.lower()}/forecast/"

        r=requests.get(url,timeout=10)

        if r.status_code!=200:
            return None


        soup=BeautifulSoup(r.text,"html.parser")

        txt=soup.get_text(" ")

        nums=re.findall(r"\d+\.\d+",txt)

        if len(nums)<3:
            return None


        return float(nums[0]),float(nums[1]),float(nums[2])

    except:
        return None



def stockanalysis_test():

    try:
        t=stockanalysis_targets("MSFT")
        return t is not None
    except:
        return False



# ---------- FETCH ----------

@st.cache_data(ttl=TTL_SECONDS)
def fetch_one(ticker):

    tk=yf.Ticker(ticker)

    hist=tk.history(period="13mo")

    closes=hist["Close"].dropna()

    price=float(closes.iloc[-1])


    row={

        "Ticker":ticker,

        "1D":pct_change(closes,1),
        "1W":pct_change(closes,5),
        "2W":pct_change(closes,10),
        "3M":pct_change(closes,63),
        "6M":pct_change(closes,126),
        "1Y":pct_change(closes,252),

        "Analistas":"—",

        "Preço":price,

        "Target Min":None,
        "Target Médio":None,
        "Target Máx":None,

        "Fonte":"—"
    }


    # Yahoo principal

    try:

        info=tk.info

        key=info.get("recommendationKey")

        row["Analistas"]=pretty_label(key)

        row["Target Min"]=info.get("targetLowPrice")
        row["Target Médio"]=info.get("targetMeanPrice")
        row["Target Máx"]=info.get("targetHighPrice")

        row["Fonte"]="Yahoo"

    except:

        pass



    # fallback StockAnalysis

    if row["Target Médio"] is None:

        t=stockanalysis_targets(ticker)

        if t:

            row["Target Min"]=t[0]
            row["Target Médio"]=t[1]
            row["Target Máx"]=t[2]

            row["Fonte"]="StockAnalysis"


    return row



# ---------- DATAFRAME ----------

def build_df(tickers):

    rows=[]

    for t in tickers:
        rows.append(fetch_one(t))

    return pd.DataFrame(rows)



# ---------- TABLE ----------

def show_table(df):

    if df.empty:
        st.warning("Sem dados")
        return


    df=df[[

        "Ticker",

        "1D",
        "1W",
        "2W",
        "3M",
        "6M",
        "1Y",

        "Analistas",

        "Preço",

        "Target Min",
        "Target Médio",
        "Target Máx",

        "Fonte"
    ]]


    styler=df.style


    for c in ["1D","1W","2W","3M","6M","1Y"]:
        styler=styler.applymap(bg_return,subset=[c])


    styler=styler.format({

        "Preço":"{:.2f}",

        "Target Min":"{:.2f}",
        "Target Médio":"{:.2f}",
        "Target Máx":"{:.2f}",

        "1D":"{:.2%}",
        "1W":"{:.2%}",
        "2W":"{:.2%}",
        "3M":"{:.2%}",
        "6M":"{:.2%}",
        "1Y":"{:.2%}"
    })


    st.dataframe(
        styler,
        use_container_width=True,
        height=520
    )



# ---------- MANAGER ----------

def ticker_manager(title,key,default):

    st.subheader(title)

    t=st.text_input("Ticker",key=f"add{key}")

    if st.button("Adicionar",key=f"badd{key}"):

        t=normalize_ticker(t)

        if t:
            st.session_state[key].append(t)
            st.rerun()


    if st.button("Resetar",key=f"reset{key}"):

        st.session_state[key]=default.copy()
        st.rerun()


    if st.button("Atualizar dados",key=f"update{key}"):

        st.cache_data.clear()
        st.rerun()


    df=build_df(st.session_state[key])

    show_table(df)



# ---------- TABS ----------

tab1,tab2,tab3=st.tabs(["Investidos","Em análise","Diagnóstico"])


with tab1:

    ticker_manager(
        "Investidos",
        "tickers_investidos",
        INVESTIDOS_DEFAULT
    )


with tab2:

    ticker_manager(
        "Em análise",
        "tickers_em_analise",
        EM_ANALISE_DEFAULT
    )


with tab3:

    st.subheader("Status das Fontes")

    st.write("Yahoo:", "✅" if yahoo_test() else "❌")

    st.write("StockAnalysis:", "✅" if stockanalysis_test() else "❌")


    st.divider()

    st.subheader("Teste Yahoo")

    t=st.text_input("Ticker teste Yahoo","NVDA")

    if st.button("Testar Yahoo"):

        tk=yf.Ticker(t)

        st.write(tk.info)



    st.divider()

    st.subheader("Teste Targets")

    t=st.text_input("Ticker teste Targets","MSFT")

    if st.button("Testar Targets"):

        st.write(stockanalysis_targets(t))
