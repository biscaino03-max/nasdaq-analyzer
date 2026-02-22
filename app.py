import re
import pandas as pd
import streamlit as st
import yfinance as yf

DEFAULT_TICKERS = ["AMCI","VMAR","VITL","UAL","MSFT","DIS","GPCR","NVDA"]

WINDOWS = {
    "1D": 1,
    "1W": 5,
    "2W": 10,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
}

st.set_page_config(page_title="Nasdaq Analyzer (ao vivo)", layout="wide")
st.title("Nasdaq Analyzer (ao vivo — sem banco)")

# ---------- estado (lista dinâmica) ----------
if "tickers" not in st.session_state:
    st.session_state.tickers = DEFAULT_TICKERS.copy()

def normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    t = re.sub(r"[^A-Z0-9\.\-]", "", t)  # permite BRK.B, RDS-A etc
    return t

# ---------- UI: adicionar / remover ----------
with st.sidebar:
    st.header("Tickers")
    st.caption("Adicione/remova ações para analisar (sem salvar).")

    new_t = st.text_input("Adicionar ticker", placeholder="Ex: AAPL, TSLA, NVDA")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Adicionar"):
            t = normalize_ticker(new_t)
            if not t:
                st.warning("Digite um ticker válido.")
            elif t in st.session_state.tickers:
                st.info("Esse ticker já está na lista.")
            else:
                st.session_state.tickers.append(t)
                st.success(f"Adicionado: {t}")

    with col_b:
        if st.button("Resetar lista"):
            st.session_state.tickers = DEFAULT_TICKERS.copy()
            st.success("Lista resetada.")

    st.divider()
    to_remove = st.multiselect("Remover tickers", options=st.session_state.tickers)
    if st.button("Remover selecionados"):
        st.session_state.tickers = [t for t in st.session_state.tickers if t not in to_remove]
        st.success("Removidos.")

    st.divider()
    if st.button("Atualizar agora (limpar cache)"):
        st.cache_data.clear()
        st.info("Cache limpo. Recarregue os dados.")

tickers = st.session_state.tickers

# ---------- cálculo ----------
def pct_change(closes: pd.Series, n: int):
    if closes is None or closes.empty or len(closes) <= n:
        return None
    last = closes.iloc[-1]
    prev = closes.iloc[-1-n]
    if pd.isna(last) or pd.isna(prev) or prev == 0:
        return None
    return float((last / prev) - 1.0)

def fmt_pct(x):
    if x is None or pd.isna(x):
        return ""
    return f"{x*100:.2f}%"

@st.cache_data(ttl=300)
def fetch_one(ticker: str):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="13mo", interval="1d")
    if hist is None or hist.empty:
        return {"Ticker": ticker, "Erro": "Sem dados"}

    closes = hist["Close"].dropna()
    if closes.empty:
        return {"Ticker": ticker, "Erro": "Sem dados"}

    last_close = float(closes.iloc[-1])

    row = {
        "Ticker": ticker,
        "Close": last_close,
        **{k: pct_change(closes, n) for k, n in WINDOWS.items()},
        "Analyst": None,
        "Target Low": None,
        "Target Mean": None,
        "Target High": None,
        "Erro": "",
    }

    # recommendation/targets podem não existir para todas
    try:
        info = tk.info or {}
        row["Analyst"] = info.get("recommendationKey")
        row["Target Low"] = info.get("targetLowPrice")
        row["Target Mean"] = info.get("targetMeanPrice")
        row["Target High"] = info.get("targetHighPrice")
    except Exception:
        pass

    return row

st.subheader("Investidos (ao vivo)")
st.write("Tickers na lista:", ", ".join(tickers) if tickers else "(vazio)")

rows = []
progress = st.progress(0)
for i, t in enumerate(tickers):
    rows.append(fetch_one(t))
    progress.progress(int(((i + 1) / max(len(tickers), 1)) * 100))
progress.empty()

df = pd.DataFrame(rows)

# formatar retornos
for k in WINDOWS.keys():
    if k in df.columns:
        df[k] = df[k].map(fmt_pct)

# ordenar: melhores 1Y primeiro (quando existir)
if "1Y" in df.columns:
    # 1Y está string %, então só não ordena por enquanto.
    pass

st.dataframe(df, use_container_width=True)

st.caption("Sem banco: os dados são buscados ao abrir (cache ~5 min).")
