import re
import pandas as pd
import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup

# ---------- CONFIG ----------

INVESTIDOS_DEFAULT = ["AMCI", "VMAR", "VITL", "UAL", "MSFT", "DIS", "GPCR", "NVDA"]
EM_ANALISE_DEFAULT = []

WINDOWS = {
    "1D": 1,
    "1W": 5,
    "2W": 10,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
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
TTL_LONG_SECONDS = 6 * 60 * 60  # 6h

st.set_page_config(page_title="Nasdaq Analyzer", layout="wide")
st.title("Nasdaq Analyzer (ao vivo)")

# ---------- STATE ----------

if "tickers_investidos" not in st.session_state:
    st.session_state.tickers_investidos = INVESTIDOS_DEFAULT.copy()

if "tickers_em_analise" not in st.session_state:
    st.session_state.tickers_em_analise = EM_ANALISE_DEFAULT.copy()


# ---------- HELPERS ----------

def normalize_ticker(t: str) -> str:
    t = (t or "").upper().strip()
    t = re.sub(r"[^A-Z0-9\.\-]", "", t)
    return t


def pct_change(closes: pd.Series, n: int):
    if closes is None or closes.empty or len(closes) <= n:
        return None
    last = closes.iloc[-1]
    prev = closes.iloc[-1 - n]
    if pd.isna(last) or pd.isna(prev) or prev == 0:
        return None
    return float((last / prev) - 1.0)


def pretty_label(key):
    if key in ANALYST_LABELS:
        return ANALYST_LABELS[key]
    return "—" if key is None else str(key)


def bg_return(v):
    if v is None or pd.isna(v):
        return ""
    try:
        v = float(v)
    except Exception:
        return ""
    if v > 0:
        return "background-color:rgba(0,200,0,0.18)"
    if v < 0:
        return "background-color:rgba(255,0,0,0.18)"
    return "background-color:rgba(120,120,120,0.10)"


def _http_get(url: str, timeout=12):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120 Safari/537.36"
    }
    return requests.get(url, headers=headers, timeout=timeout)


# ---------- STATUS ----------

def yahoo_test():
    try:
        t = yf.Ticker("MSFT")
        d = t.history(period="5d")
        return d is not None and not d.empty
    except Exception:
        return False


# ---------- STOCKANALYSIS: TARGETS ----------

@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def stockanalysis_targets(ticker: str):
    """
    Extrai Low/Average/High na ordem correta do /forecast/ do StockAnalysis.
    Retorna (low, avg, high) ou None.
    """
    try:
        url = f"https://stockanalysis.com/stocks/{ticker.lower()}/forecast/"
        r = _http_get(url, timeout=12)
        if r.status_code != 200:
            return None

        text = r.text

        m = re.search(
            r"Target\s+Low\s+Average\s+Median\s+High.*?Price\$(\d+(?:\.\d+)?)\$(\d+(?:\.\d+)?)\$(\d+(?:\.\d+)?)\$(\d+(?:\.\d+)?)",
            text,
            flags=re.DOTALL | re.IGNORECASE
        )
        if not m:
            return None

        low = float(m.group(1))
        avg = float(m.group(2))
        high = float(m.group(4))

        # Garantir coerência
        low, avg, high = sorted([low, avg, high])
        return low, avg, high
    except Exception:
        return None


def stockanalysis_test():
    try:
        return stockanalysis_targets("MSFT") is not None
    except Exception:
        return False


# ---------- STOCKANALYSIS: TOP 5 STRONG BUY (NASDAQ-100) ----------

@st.cache_data(ttl=TTL_LONG_SECONDS, show_spinner=False)
def stockanalysis_nasdaq100_set() -> set:
    try:
        url = "https://stockanalysis.com/list/nasdaq-100-stocks/"
        r = _http_get(url, timeout=12)
        if r.status_code != 200:
            return set()

        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table")
        if not table:
            return set()

        out = set()
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue
            tk = normalize_ticker(tds[0].get_text(strip=True))
            if tk:
                out.add(tk)
        return out
    except Exception:
        return set()


@st.cache_data(ttl=TTL_LONG_SECONDS, show_spinner=False)
def stockanalysis_top5_strong_buy_nasdaq100() -> list:
    try:
        url = "https://stockanalysis.com/analysts/top-stocks/"
        r = _http_get(url, timeout=12)
        if r.status_code != 200:
            return []

        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table")
        if not table:
            return []

        nasdaq100 = stockanalysis_nasdaq100_set()
        tickers = []

        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue
            tk = normalize_ticker(tds[0].get_text(strip=True))
            if not tk:
                continue
            if nasdaq100 and tk not in nasdaq100:
                continue
            if tk not in tickers:
                tickers.append(tk)
            if len(tickers) >= 5:
                break
        return tickers
    except Exception:
        return []


# ---------- FETCH (YAHOO + FALLBACK) ----------

@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def fetch_one(ticker: str):
    ticker = normalize_ticker(ticker)

    row = {
        "Ticker": ticker,
        "1D": None, "1W": None, "2W": None, "3M": None, "6M": None, "1Y": None,
        "Analistas": "—",
        "Preço": None,
        "Target Min": None, "Target Médio": None, "Target Máx": None,
        "Fonte": "—",
        "Erro": "",
    }

    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="13mo", interval="1d")

        if hist is None or hist.empty:
            row["Erro"] = "Sem histórico (Yahoo)"
            return row

        closes = hist["Close"].dropna()
        if closes.empty:
            row["Erro"] = "Sem closes (Yahoo)"
            return row

        last_close = float(closes.iloc[-1])
        row["Preço"] = last_close

        row["1D"] = pct_change(closes, WINDOWS["1D"])
        row["1W"] = pct_change(closes, WINDOWS["1W"])
        row["2W"] = pct_change(closes, WINDOWS["2W"])
        row["3M"] = pct_change(closes, WINDOWS["3M"])
        row["6M"] = pct_change(closes, WINDOWS["6M"])
        row["1Y"] = pct_change(closes, WINDOWS["1Y"])

        # Yahoo: analistas + targets
        try:
            info = tk.info or {}
            key = info.get("recommendationKey")
            row["Analistas"] = pretty_label(key)

            row["Target Min"] = info.get("targetLowPrice")
            row["Target Médio"] = info.get("targetMeanPrice")
            row["Target Máx"] = info.get("targetHighPrice")

            row["Fonte"] = "Yahoo"
        except Exception:
            pass

    except Exception as e:
        row["Erro"] = f"Erro Yahoo: {e}"
        return row

    # fallback StockAnalysis
    if row["Target Médio"] is None:
        t = stockanalysis_targets(ticker)
        if t:
            low, avg, high = t
            row["Target Min"] = low
            row["Target Médio"] = avg
            row["Target Máx"] = high
            row["Fonte"] = "StockAnalysis"

    # segurança final (ordena)
    try:
        a, b, c = row.get("Target Min"), row.get("Target Médio"), row.get("Target Máx")
        if a is not None and b is not None and c is not None:
            low, mid, high = sorted([float(a), float(b), float(c)])
            row["Target Min"], row["Target Médio"], row["Target Máx"] = low, mid, high
    except Exception:
        pass

    return row


def build_df(tickers: list) -> pd.DataFrame:
    tickers = [normalize_ticker(t) for t in (tickers or []) if normalize_ticker(t)]
    if not tickers:
        return pd.DataFrame()

    rows = []
    prog = st.progress(0)
    for i, t in enumerate(tickers):
        rows.append(fetch_one(t))
        prog.progress(int(((i + 1) / len(tickers)) * 100))
    prog.empty()

    return pd.DataFrame(rows)


# ---------- TABLE ----------

def show_table(df: pd.DataFrame, height=520):
    if df is None or df.empty:
        st.warning("Sem dados.")
        return

    cols = [
        "Ticker",
        "1D", "1W", "2W", "3M", "6M", "1Y",
        "Analistas",
        "Preço",
        "Target Min", "Target Médio", "Target Máx",
        "Fonte",
    ]
    df = df[[c for c in cols if c in df.columns]].copy()

    styler = df.style
    for c in ["1D", "1W", "2W", "3M", "6M", "1Y"]:
        if c in df.columns:
            styler = styler.applymap(bg_return, subset=[c])

    fmt = {}
    for c in ["1D", "1W", "2W", "3M", "6M", "1Y"]:
        if c in df.columns:
            fmt[c] = lambda x: "" if (x is None or pd.isna(x)) else f"{x*100:.2f}%"
    if "Preço" in df.columns:
        fmt["Preço"] = lambda x: "" if (x is None or pd.isna(x)) else f"{float(x):.2f}"
    for c in ["Target Min", "Target Médio", "Target Máx"]:
        if c in df.columns:
            fmt[c] = lambda x: "" if (x is None or pd.isna(x)) else f"{float(x):.2f}"

    styler = styler.format(fmt)
    st.dataframe(styler, use_container_width=True, height=height)


# ---------- MANAGER (EDITÁVEL) ----------

def ticker_manager(title: str, key_state: str, default_list: list[str]):
    st.subheader(title)

    tickers = st.session_state[key_state]

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        new_t = st.text_input("Ticker", key=f"{key_state}_new_input", placeholder="Ex: AAPL, TSLA, NVDA")
    with c2:
        if st.button("Adicionar", key=f"{key_state}_add_btn"):
            t = normalize_ticker(new_t)
            if not t:
                st.warning("Digite um ticker válido.")
            elif t in st.session_state[key_state]:
                st.info("Esse ticker já está na lista.")
            else:
                st.session_state[key_state].append(t)
                st.rerun()
    with c3:
        if st.button("Resetar", key=f"{key_state}_reset_btn"):
            st.session_state[key_state] = default_list.copy()
            st.rerun()

    if st.button("Atualizar dados (manual)", key=f"{key_state}_refresh_btn"):
        st.cache_data.clear()
        st.rerun()

    if tickers:
        st.markdown("**Tickers atuais (clique ❌ para remover):**")
        cols = st.columns(min(len(tickers), 8))
        for i, t in enumerate(tickers):
            col = cols[i % len(cols)]
            with col:
                st.write(t)
                if st.button("❌", key=f"{key_state}_del_{t}_{i}"):
                    try:
                        st.session_state[key_state].remove(t)
                    except ValueError:
                        pass
                    st.rerun()
    else:
        st.info("Lista vazia.")

    df = build_df(st.session_state[key_state])
    show_table(df)


# ---------- TABS ----------

tab1, tab2, tab3 = st.tabs(["Investidos", "Em análise", "Diagnóstico"])

with tab1:
    ticker_manager("Investidos (editável)", "tickers_investidos", INVESTIDOS_DEFAULT)

with tab2:
    # Top 5 fixo
    st.subheader("Top 5 mais recomendadas (fixas) — Nasdaq-100")
    top5 = stockanalysis_top5_strong_buy_nasdaq100()
    if not top5:
        st.warning("Não consegui carregar o Top 5 agora. Tente novamente em alguns minutos.")
    else:
        df_top = build_df(top5)
        show_table(df_top, height=320)
        st.caption("Fonte Top 5: StockAnalysis (Top Strong Buy) filtrado para Nasdaq-100.")

    st.divider()

    # Lista editável do usuário (continua editável!)
    ticker_manager("Em análise (editável)", "tickers_em_analise", EM_ANALISE_DEFAULT)

with tab3:
    st.subheader("Status das Fontes")
    st.write("Yahoo:", "✅" if yahoo_test() else "❌")
    st.write("StockAnalysis (Targets):", "✅" if stockanalysis_test() else "❌")
    st.write("StockAnalysis (Top 5):", "✅" if bool(stockanalysis_top5_strong_buy_nasdaq100()) else "❌")

    st.divider()
    st.subheader("Teste Targets (StockAnalysis)")
    t1 = st.text_input("Ticker teste Targets", "MSFT", key="diag_t_targets")
    if st.button("Testar Targets", key="diag_btn_targets"):
        st.write(stockanalysis_targets(t1))

    st.divider()
    st.subheader("Teste Yahoo (info completo)")
    t2 = st.text_input("Ticker teste Yahoo", "NVDA", key="diag_t_yahoo")
    if st.button("Testar Yahoo", key="diag_btn_yahoo"):
        tk = yf.Ticker(t2)
        st.write(tk.info)
