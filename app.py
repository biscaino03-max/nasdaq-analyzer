import re
import pandas as pd
import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup

# ----------------- CONFIG -----------------
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

TTL_SECONDS = 300          # 5 minutos (tabela)
TTL_LONG_SECONDS = 6 * 3600  # 6 horas (Top 5 / Nasdaq-100)

st.set_page_config(page_title="Nasdaq Analyzer (ao vivo)", layout="wide")
st.title("Nasdaq Analyzer (ao vivo)")

# ----------------- STATE -----------------
if "tickers_investidos" not in st.session_state:
    st.session_state.tickers_investidos = INVESTIDOS_DEFAULT.copy()

if "tickers_em_analise" not in st.session_state:
    st.session_state.tickers_em_analise = EM_ANALISE_DEFAULT.copy()


def normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
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


def pretty_analyst_label(key):
    if not key:
        return ANALYST_LABELS[None]
    return ANALYST_LABELS.get(key, key)


def is_strong_buy_label(label: str) -> bool:
    return isinstance(label, str) and label.strip().startswith("🟢")


def _http_get(url: str, timeout=12):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120 Safari/537.36"
        )
    }
    return requests.get(url, headers=headers, timeout=timeout)


def _as_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def validate_and_fix_targets(low, mean, high):
    """
    Garante Low <= Mean <= High quando existirem.
    Retorna: (low2, mean2, high2, ok_bool|None, msg)
    """
    lowf, meanf, highf = _as_float(low), _as_float(mean), _as_float(high)
    if lowf is None or meanf is None or highf is None:
        return lowf, meanf, highf, None, "targets incompletos"

    ok = (lowf <= meanf <= highf)
    if ok:
        return lowf, meanf, highf, True, ""

    lo2, mi2, hi2 = sorted([lowf, meanf, highf])
    return lo2, mi2, hi2, False, "targets incoerentes (corrigidos)"


# ----------------- SOURCE: STOCKANALYSIS (targets fallback #2) -----------------
@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def stockanalysis_targets(ticker: str):
    """
    Pega targets no StockAnalysis (/forecast/) quando Yahoo não tiver.
    Retorna (low, mean, high) ou None.
    """
    try:
        url = f"https://stockanalysis.com/stocks/{ticker.lower()}/forecast/"
        r = _http_get(url, timeout=12)
        if r.status_code != 200:
            return None

        # Procurar pelos três valores no bloco do forecast.
        # (Mantemos uma extração tolerante para mudanças de layout)
        text = r.text

        # estratégia 1: tenta pegar "Target Low", "Average", "High" no HTML por regex
        m = re.search(
            r"Target\s+Low.*?\$(\d+(?:\.\d+)?).*?Average.*?\$(\d+(?:\.\d+)?).*?High.*?\$(\d+(?:\.\d+)?)",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
        if not m:
            return None

        low = float(m.group(1))
        mean = float(m.group(2))
        high = float(m.group(3))

        low, mean, high, _, _ = validate_and_fix_targets(low, mean, high)
        return (low, mean, high)
    except Exception:
        return None


# ----------------- SOURCE: MARKETBEAT (targets fallback #1) -----------------
@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def marketbeat_targets_and_rating(ticker: str):
    """
    MarketBeat pode retornar 403 em alguns ambientes.
    Retorna dict:
      {"targets": (low,mean,high) ou None, "rating": label ou None, "status": int, "note": str}
    """
    url = f"https://www.marketbeat.com/stocks/NASDAQ/{ticker}/price-target/"
    try:
        r = _http_get(url, timeout=12)
        if r.status_code != 200:
            return {"targets": None, "rating": None, "status": r.status_code, "note": f"HTTP {r.status_code}"}

        soup = BeautifulSoup(r.text, "html.parser")
        txt = soup.get_text(" ", strip=True)

        def find_num(patterns):
            for pat in patterns:
                mm = re.search(pat, txt, flags=re.IGNORECASE)
                if mm:
                    try:
                        return float(mm.group(1))
                    except Exception:
                        pass
            return None

        low = find_num([
            r"Low\s+Price\s+Target[^$]*\$\s*(\d+(?:\.\d+)?)",
            r"Low\s+Target[^$]*\$\s*(\d+(?:\.\d+)?)",
        ])
        mean = find_num([
            r"Average\s+Price\s+Target[^$]*\$\s*(\d+(?:\.\d+)?)",
            r"Average\s+Target[^$]*\$\s*(\d+(?:\.\d+)?)",
            r"Consensus\s+Price\s+Target[^$]*\$\s*(\d+(?:\.\d+)?)",
        ])
        high = find_num([
            r"High\s+Price\s+Target[^$]*\$\s*(\d+(?:\.\d+)?)",
            r"High\s+Target[^$]*\$\s*(\d+(?:\.\d+)?)",
        ])

        targets = None
        if low is not None or mean is not None or high is not None:
            low, mean, high, _, _ = validate_and_fix_targets(low, mean, high)
            targets = (low, mean, high)

        rating = None
        mr = re.search(r"(Strong\s+Buy|Moderate\s+Buy|Buy|Hold|Sell|Strong\s+Sell)", txt, flags=re.IGNORECASE)
        if mr:
            rr = mr.group(1).strip().lower()
            if "strong" in rr and "buy" in rr:
                rating = ANALYST_LABELS["strong_buy"]
            elif "moderate" in rr and "buy" in rr:
                rating = ANALYST_LABELS["buy"]
            elif rr == "buy":
                rating = ANALYST_LABELS["buy"]
            elif "hold" in rr:
                rating = ANALYST_LABELS["hold"]
            elif "strong" in rr and "sell" in rr:
                rating = ANALYST_LABELS["strong_sell"]
            elif "sell" in rr:
                rating = ANALYST_LABELS["sell"]

        return {"targets": targets, "rating": rating, "status": 200, "note": ""}
    except Exception as e:
        return {"targets": None, "rating": None, "status": 0, "note": f"erro: {e}"}


# ----------------- TOP 5 AUTOMÁTICO (REAL) -----------------
@st.cache_data(ttl=TTL_LONG_SECONDS, show_spinner=False)
def stockanalysis_nasdaq100_set() -> set:
    """
    Lista de tickers do Nasdaq-100 via StockAnalysis.
    """
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
        # Primeira coluna geralmente é o ticker; pegar o texto do link/td.
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
    """
    Pega Top Analyst Stocks do StockAnalysis e filtra para Nasdaq-100.
    Retorna só tickers (5).
    """
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
            # tenta achar o link do ticker (mais robusto do que pegar td[0] bruto)
            a = tr.find("a", href=True)
            if not a:
                continue

            # Ex.: /stocks/msft/ ou /stocks/nvda/
            href = a.get("href", "")
            m = re.search(r"/stocks/([a-z0-9\.\-]+)/", href, flags=re.IGNORECASE)
            if not m:
                continue

            tk = normalize_ticker(m.group(1))
            if not tk:
                continue

            # filtra para nasdaq-100
            if nasdaq100 and tk not in nasdaq100:
                continue

            if tk not in tickers:
                tickers.append(tk)

            if len(tickers) >= 5:
                break

        return tickers
    except Exception:
        return []


# ----------------- CACHE (TTL 5min) -----------------
@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def fetch_one(ticker: str):
    ticker = normalize_ticker(ticker)
    tk = yf.Ticker(ticker)

    hist = tk.history(period="13mo", interval="1d")
    if hist is None or hist.empty:
        return {"Ticker": ticker, "Erro": "Sem dados (Yahoo)"}

    closes = hist["Close"].dropna()
    if closes.empty:
        return {"Ticker": ticker, "Erro": "Sem closes (Yahoo)"}

    last_close = float(closes.iloc[-1])

    row = {
        "Ticker": ticker,
        "1D": pct_change(closes, WINDOWS["1D"]),
        "1W": pct_change(closes, WINDOWS["1W"]),
        "2W": pct_change(closes, WINDOWS["2W"]),
        "3M": pct_change(closes, WINDOWS["3M"]),
        "6M": pct_change(closes, WINDOWS["6M"]),
        "1Y": pct_change(closes, WINDOWS["1Y"]),
        "Analistas": "—",
        "Preço": last_close,  # preço entre Analistas e Target Min
        "Target Min": None,
        "Target Médio": None,
        "Target Máx": None,
        "Fonte": "—",
        "Alerta": "",
        "Erro": "",
    }

    # -------- 1) Yahoo (PRIORIDADE) --------
    try:
        info = tk.info or {}
        rec = info.get("recommendationKey")
        row["Analistas"] = pretty_analyst_label(rec)

        row["Target Min"] = info.get("targetLowPrice")
        row["Target Médio"] = info.get("targetMeanPrice")
        row["Target Máx"] = info.get("targetHighPrice")

        if row["Target Médio"] is not None:
            row["Fonte"] = "Yahoo"
    except Exception:
        pass

    # Valida se Yahoo trouxe incoerente
    low, mean, high, ok, msg = validate_and_fix_targets(row["Target Min"], row["Target Médio"], row["Target Máx"])
    if ok is False:
        row["Target Min"], row["Target Médio"], row["Target Máx"] = low, mean, high
        row["Alerta"] = "⚠️ " + msg

    # -------- 2) MarketBeat (fallback #1) --------
    if row["Target Médio"] is None:
        mb = marketbeat_targets_and_rating(ticker)
        if mb.get("targets"):
            low, mean, high = mb["targets"]
            row["Target Min"], row["Target Médio"], row["Target Máx"] = low, mean, high
            row["Fonte"] = "MarketBeat"

            # se Yahoo não trouxe analista, usa rating MB (se existir)
            if row["Analistas"] == "—" and mb.get("rating"):
                row["Analistas"] = mb["rating"]

            _, _, _, ok2, msg2 = validate_and_fix_targets(low, mean, high)
            if ok2 is False and not row["Alerta"]:
                row["Alerta"] = "⚠️ " + msg2
        else:
            # se for 403, só registra para diagnóstico/entendimento
            if mb.get("status") == 403:
                row["Alerta"] = (row["Alerta"] + " | " if row["Alerta"] else "") + "MarketBeat 403"

    # -------- 3) StockAnalysis (fallback #2) --------
    if row["Target Médio"] is None:
        sa = stockanalysis_targets(ticker)
        if sa:
            low, mean, high = sa
            row["Target Min"], row["Target Médio"], row["Target Máx"] = low, mean, high
            row["Fonte"] = "StockAnalysis"

            _, _, _, ok3, msg3 = validate_and_fix_targets(low, mean, high)
            if ok3 is False and not row["Alerta"]:
                row["Alerta"] = "⚠️ " + msg3

    return row


def build_df(tickers: list[str]) -> pd.DataFrame:
    tickers = [normalize_ticker(t) for t in (tickers or []) if normalize_ticker(t)]
    if not tickers:
        return pd.DataFrame()

    rows = []
    progress = st.progress(0)
    for i, t in enumerate(tickers):
        rows.append(fetch_one(t))
        progress.progress(int(((i + 1) / len(tickers)) * 100))
    progress.empty()

    return pd.DataFrame(rows)


# ----------------- STYLES (CORES) -----------------
def _bg_for_return(v):
    if v is None or pd.isna(v):
        return ""
    try:
        v = float(v)
    except Exception:
        return ""
    if v > 0:
        return "background-color: rgba(0, 200, 0, 0.18);"
    if v < 0:
        return "background-color: rgba(255, 0, 0, 0.18);"
    return "background-color: rgba(120, 120, 120, 0.10);"


def show_table_colored(df_raw: pd.DataFrame, height=560):
    st.subheader("Tabela ao vivo")

    if df_raw is None or df_raw.empty:
        st.warning("Lista vazia. Adicione um ticker.")
        return

    df = df_raw.copy()

    final_cols = [
        "Ticker",
        "1D", "1W", "2W", "3M", "6M", "1Y",
        "Analistas",
        "Preço",
        "Target Min", "Target Médio", "Target Máx",
        "Fonte",
        "Alerta",
        "Erro",
    ]
    df = df[[c for c in final_cols if c in df.columns]]

    styler = df.style
    for c in ["1D", "1W", "2W", "3M", "6M", "1Y"]:
        if c in df.columns:
            styler = styler.applymap(_bg_for_return, subset=[c])

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
    st.caption("Dados ao vivo via Yahoo (yfinance). Targets: Yahoo → MarketBeat → StockAnalysis. Cache TTL 5 min + botão manual.")


# ----------------- MANAGER (IGUAL PARA AS DUAS ABAS) -----------------
def ticker_manager(title: str, key_state: str, default_list: list[str]):
    st.header(title)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        new_t = st.text_input("Ticker", key=f"{key_state}_new_input", placeholder="Ex: AAPL, TSLA, NVDA")
    with col2:
        if st.button("Adicionar", key=f"{key_state}_add_btn"):
            t = normalize_ticker(new_t)
            if not t:
                st.warning("Digite um ticker válido.")
            elif t in st.session_state[key_state]:
                st.info("Esse ticker já está na lista.")
            else:
                st.session_state[key_state].append(t)
                st.rerun()
    with col3:
        if st.button("Resetar", key=f"{key_state}_reset_btn"):
            st.session_state[key_state] = default_list.copy()
            st.rerun()

    if st.button("Atualizar dados (manual)", key=f"{key_state}_refresh_btn"):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Auto (TTL cache): {TTL_SECONDS//60} min. Manual: botão acima.")

    tickers = st.session_state[key_state]
    if tickers:
        st.markdown("**Tickers atuais (clique ❌ para remover):**")
        cols = st.columns(min(len(tickers), 10))
        for i, t in enumerate(tickers):
            with cols[i % len(cols)]:
                st.write(t)
                if st.button("❌", key=f"{key_state}_del_{t}_{i}"):
                    try:
                        st.session_state[key_state].remove(t)
                    except ValueError:
                        pass
                    st.rerun()
    else:
        st.info("Lista vazia.")

    df_raw = build_df(tickers)
    show_table_colored(df_raw)


# ----------------- DIAGNÓSTICO -----------------
def yahoo_test():
    try:
        t = yf.Ticker("MSFT")
        d = t.history(period="5d")
        return d is not None and not d.empty
    except Exception:
        return False

def marketbeat_test():
    try:
        r = _http_get("https://www.marketbeat.com/stocks/NASDAQ/MSFT/price-target/", timeout=12)
        # 403 é comum -> consideramos "offline" no Render
        return r.status_code == 200
    except Exception:
        return False

def stockanalysis_test():
    try:
        return stockanalysis_targets("MSFT") is not None
    except Exception:
        return False


# ----------------- TABS -----------------
tab1, tab2, tab3 = st.tabs(["Investidos", "Em análise", "Diagnóstico"])

with tab1:
    ticker_manager("Investidos (editável)", "tickers_investidos", INVESTIDOS_DEFAULT)

with tab2:
    st.header("Top 5 mais recomendadas (automático) — Nasdaq-100")

    top5 = stockanalysis_top5_strong_buy_nasdaq100()
    if not top5:
        st.warning("Não consegui carregar o Top 5 automático agora (fonte indisponível ou bloqueada). Tente novamente em alguns minutos.")
    else:
        st.write("Tickers:", ", ".join(top5))
        df_top = build_df(top5)
        show_table_colored(df_top, height=360)

    st.divider()

    # Em análise “granada”: Top 5 fixo em cima + lista editável (igual investidos)
    ticker_manager("Em análise (editável)", "tickers_em_analise", EM_ANALISE_DEFAULT)

with tab3:
    st.header("Diagnóstico")

    st.write("Yahoo:", "✅" if yahoo_test() else "❌")
    st.write("MarketBeat:", "✅" if marketbeat_test() else "❌ (muito comum dar 403 no Render)")
    st.write("StockAnalysis:", "✅" if stockanalysis_test() else "❌")

    st.divider()
    st.subheader("Teste rápido (um ticker)")
    t = st.text_input("Ticker para teste", "NVDA", key="diag_ticker_test")
    if st.button("Rodar teste completo", key="diag_btn_test"):
        t = normalize_ticker(t)

        st.write("Yahoo (info):")
        try:
            info = yf.Ticker(t).info or {}
            st.write({
                "recommendationKey": info.get("recommendationKey"),
                "targetLowPrice": info.get("targetLowPrice"),
                "targetMeanPrice": info.get("targetMeanPrice"),
                "targetHighPrice": info.get("targetHighPrice"),
            })
        except Exception as e:
            st.write("erro:", e)

        st.write("MarketBeat (targets/rating):")
        st.write(marketbeat_targets_and_rating(t))

        st.write("StockAnalysis (targets):")
        st.write(stockanalysis_targets(t))
