import os
import json
import re
import requests
import pandas as pd
import streamlit as st
import yfinance as yf
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

CHART_PERIODS = {
    "1 mês": "1mo",
    "3 meses": "3mo",
    "6 meses": "6mo",
    "1 ano": "1y",
    "2 anos": "2y",
    "5 anos": "5y",
    "Máximo": "max",
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

# Ranking da recomendação (para Top 5 automático)
ANALYST_SCORE = {
    "strong_buy": 5,
    "buy": 4,
    "hold": 3,
    "sell": 2,
    "strong_sell": 1,
    None: 0,
    "none": 0,
}

TTL_SECONDS = 300  # tabela e gráfico: 5 min
TTL_LONG_SECONDS = 6 * 3600  # Top5/Nasdaq100: 6h


def _resolve_store_file() -> str:
    """
    Define onde salvar o JSON de listas.
    Prioridade:
    1) variável TICKERS_STORE_FILE
    2) /var/data/tickers_store.json (Render Disk, se existir)
    3) arquivo local no diretório do app
    """
    env_path = os.getenv("TICKERS_STORE_FILE", "").strip()
    if env_path:
        return env_path

    render_disk_dir = "/var/data"
    if os.path.isdir(render_disk_dir):
        return os.path.join(render_disk_dir, "tickers_store.json")

    return "tickers_store.json"


STORE_FILE = _resolve_store_file()  # persistência sem banco

st.set_page_config(page_title="Nasdaq Analyzer (ao vivo)", layout="wide")
st.title("Nasdaq Analyzer (ao vivo)")


# ----------------- PERSISTÊNCIA (SEM DB) -----------------
def _safe_read_json(path: str):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _safe_write_json(path: str, data: dict):
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
        return True
    except Exception:
        return False


def load_lists_from_store():
    data = _safe_read_json(STORE_FILE) or {}
    inv = data.get("investidos", INVESTIDOS_DEFAULT.copy())
    ana = data.get("em_analise", EM_ANALISE_DEFAULT.copy())
    inv = [normalize_ticker(x) for x in inv if normalize_ticker(x)]
    ana = [normalize_ticker(x) for x in ana if normalize_ticker(x)]
    return inv, ana


def save_lists_to_store(investidos: list[str], em_analise: list[str]):
    payload = {
        "investidos": investidos,
        "em_analise": em_analise,
    }
    _safe_write_json(STORE_FILE, payload)


def _migrate_legacy_store_if_needed():
    """
    Migra arquivo legado do diretório local para o caminho atual (ex.: /var/data)
    sem sobrescrever se já existir destino.
    """
    legacy = "tickers_store.json"
    if os.path.abspath(legacy) == os.path.abspath(STORE_FILE):
        return
    try:
        if os.path.exists(legacy) and not os.path.exists(STORE_FILE):
            data = _safe_read_json(legacy)
            if isinstance(data, dict):
                _safe_write_json(STORE_FILE, data)
    except Exception:
        pass


# ----------------- STATE -----------------
def normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    t = re.sub(r"[^A-Z0-9\.\-]", "", t)  # permite BRK.B / RDS-A etc
    return t


if (
    "tickers_investidos" not in st.session_state
    or "tickers_em_analise" not in st.session_state
):
    _migrate_legacy_store_if_needed()
    inv, ana = load_lists_from_store()
    st.session_state.tickers_investidos = inv
    st.session_state.tickers_em_analise = ana


def persist_now():
    save_lists_to_store(
        st.session_state.tickers_investidos, st.session_state.tickers_em_analise
    )


# ----------------- HELPERS -----------------
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


def _http_get(url: str, timeout=15):
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
    lowf, meanf, highf = _as_float(low), _as_float(mean), _as_float(high)
    if lowf is None or meanf is None or highf is None:
        return lowf, meanf, highf, None, "targets incompletos"

    ok = lowf <= meanf <= highf
    if ok:
        return lowf, meanf, highf, True, ""

    lo2, mi2, hi2 = sorted([lowf, meanf, highf])
    return lo2, mi2, hi2, False, "targets incoerentes (corrigidos)"


# ----------------- NASDAQ-100 (WIKIPEDIA) -----------------
@st.cache_data(ttl=TTL_LONG_SECONDS, show_spinner=False)
def get_nasdaq100_wikipedia() -> list[str]:
    """
    Pega tickers do Nasdaq-100 via Wikipedia.
    Em geral, funciona melhor no Render do que StockAnalysis/MarketBeat.
    """
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        r = _http_get(url, timeout=15)
        if r.status_code != 200:
            return []

        soup = BeautifulSoup(r.text, "html.parser")
        tables = soup.find_all("table", {"class": "wikitable"})
        best = None
        for tb in tables:
            headers = [th.get_text(strip=True).lower() for th in tb.find_all("th")]
            if any("ticker" in h for h in headers) and any(
                "company" in h or "security" in h for h in headers
            ):
                best = tb
                break

        if not best:
            return []

        tickers = []
        for tr in best.find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue
            tk_raw = tds[0].get_text(strip=True)
            tk = normalize_ticker(tk_raw)
            if not tk:
                continue
            tk = tk.replace(".", "-")
            if tk not in tickers:
                tickers.append(tk)
        return tickers[:120]
    except Exception:
        return []


# ----------------- YAHOO FETCH (TTL 5min) -----------------
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
        "Analistas_key": None,
        "Analistas": "—",
        "Preço": last_close,
        "Target Min": None,
        "Target Médio": None,
        "Target Máx": None,
        "Fonte": "—",
        "Alerta": "",
        "Erro": "",
    }

    try:
        info = tk.info or {}
        rec = info.get("recommendationKey")
        row["Analistas_key"] = rec
        row["Analistas"] = pretty_analyst_label(rec)
        row["Target Min"] = info.get("targetLowPrice")
        row["Target Médio"] = info.get("targetMeanPrice")
        row["Target Máx"] = info.get("targetHighPrice")
        if row["Target Médio"] is not None:
            row["Fonte"] = "Yahoo"
    except Exception:
        pass

    low, mean, high, ok, msg = validate_and_fix_targets(
        row["Target Min"], row["Target Médio"], row["Target Máx"]
    )
    row["Target Min"], row["Target Médio"], row["Target Máx"] = low, mean, high
    if ok is False:
        row["Alerta"] = "⚠️ " + msg

    if row["Target Médio"] is None:
        row["Alerta"] = (row["Alerta"] + " | " if row["Alerta"] else "") + "Sem targets (Yahoo)"
        row["Fonte"] = "Yahoo"

    return row


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def fetch_history_for_chart(ticker: str, period: str = "6mo") -> pd.DataFrame:
    ticker = normalize_ticker(ticker)
    hist = yf.Ticker(ticker).history(period=period, interval="1d")
    if hist is None or hist.empty:
        return pd.DataFrame()

    df = hist[["Close"]].dropna().copy()
    if df.empty:
        return pd.DataFrame()

    df["Retorno Diário %"] = df["Close"].pct_change() * 100.0
    base = float(df["Close"].iloc[0])
    if base > 0:
        df["Desempenho Acumulado %"] = (df["Close"] / base - 1.0) * 100.0
        # Índice normalizado ajuda a diferenciar visualmente do gráfico de preço.
        df["Índice Base 100"] = (df["Close"] / base) * 100.0
    else:
        df["Desempenho Acumulado %"] = 0.0
        df["Índice Base 100"] = 100.0
    return df


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


# ----------------- TOP 5 AUTOMÁTICO (YAHOO + NASDAQ-100) -----------------
@st.cache_data(ttl=TTL_LONG_SECONDS, show_spinner=False)
def top5_auto_nasdaq100_yahoo() -> list[str]:
    tickers = get_nasdaq100_wikipedia()
    if not tickers:
        return []

    rows = []
    for t in tickers[:110]:
        try:
            r = fetch_one(t)
            if not r or r.get("Erro"):
                continue

            rec_key = r.get("Analistas_key")
            base = ANALYST_SCORE.get(rec_key, 0)

            price = _as_float(r.get("Preço"))
            tmean = _as_float(r.get("Target Médio"))

            upside = 0.0
            if price and tmean and price > 0:
                upside = (tmean - price) / price

            score = base * 1000 + upside * 100
            rows.append((t, score))
        except Exception:
            continue

    if not rows:
        return []

    rows.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in rows[:5]]


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

    st.dataframe(styler.format(fmt), use_container_width=True, height=height)
    st.caption(
        "Dados via Yahoo (yfinance). Cache TTL 5 min + botão manual. "
        "Top 5: Nasdaq-100 (Wikipedia) + ranking Yahoo."
    )


def show_ticker_chart(tickers: list[str], key_prefix: str):
    st.subheader("Gráfico por ticker")

    tickers = [normalize_ticker(t) for t in (tickers or []) if normalize_ticker(t)]
    if not tickers:
        st.info("Sem tickers para plotar.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        selected = st.selectbox(
            "Selecione o ticker",
            options=tickers,
            key=f"{key_prefix}_chart_ticker",
        )
    with c2:
        period_label = st.selectbox(
            "Período",
            options=list(CHART_PERIODS.keys()),
            index=2,
            key=f"{key_prefix}_chart_period",
        )

    period = CHART_PERIODS[period_label]
    df = fetch_history_for_chart(selected, period=period)
    if df.empty:
        st.warning(f"Sem histórico para {selected} no período selecionado.")
        return

    st.caption(f"Ticker: {selected} | Período selecionado: {period_label}")

    st.markdown("**1) Preço de fechamento (USD)**")
    st.caption("Função: mostrar o valor nominal da ação dia a dia no período selecionado.")
    st.line_chart(df[["Close"]], use_container_width=True)

    st.markdown("**2) Índice Base 100 (desempenho relativo)**")
    st.caption(
        "Função: comparar evolução percentual sem confundir com preço. "
        "Início do período = 100; acima de 100 indica alta acumulada."
    )
    st.line_chart(df[["Índice Base 100"]], use_container_width=True)

    st.markdown("**3) Retorno diário (%)**")
    st.caption("Função: exibir a variação percentual de cada pregão (volatilidade diária).")
    st.bar_chart(df[["Retorno Diário %"]].dropna(), use_container_width=True)


# ----------------- MANAGER (IGUAL NAS DUAS ABAS) -----------------
def ticker_manager(title: str, key_state: str, default_list: list[str]):
    st.header(title)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        new_t = st.text_input(
            "Ticker", key=f"{key_state}_new_input", placeholder="Ex: AAPL, TSLA, NVDA"
        )
    with col2:
        if st.button("Adicionar", key=f"{key_state}_add_btn"):
            t = normalize_ticker(new_t)
            if not t:
                st.warning("Digite um ticker válido.")
            elif t in st.session_state[key_state]:
                st.info("Esse ticker já está na lista.")
            else:
                st.session_state[key_state].append(t)
                persist_now()
                st.rerun()
    with col3:
        if st.button("Resetar", key=f"{key_state}_reset_btn"):
            st.session_state[key_state] = default_list.copy()
            persist_now()
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
                    persist_now()
                    st.rerun()
    else:
        st.info("Lista vazia.")

    df_raw = build_df(tickers)
    show_table_colored(df_raw)
    st.divider()
    show_ticker_chart(tickers, key_state)


# ----------------- DIAGNÓSTICO -----------------
def yahoo_test():
    try:
        t = yf.Ticker("MSFT")
        d = t.history(period="5d")
        return d is not None and not d.empty
    except Exception:
        return False


def wiki_test():
    tickers = get_nasdaq100_wikipedia()
    return bool(tickers)


# ----------------- TABS -----------------
tab1, tab2, tab3 = st.tabs(["Investidos", "Em análise", "Diagnóstico"])

with tab1:
    ticker_manager("Investidos (editável)", "tickers_investidos", INVESTIDOS_DEFAULT)

with tab2:
    st.header("Top 5 mais recomendadas (automático) — Nasdaq-100")
    top5 = top5_auto_nasdaq100_yahoo()
    if not top5:
        st.warning(
            "Não consegui montar o Top 5 automático agora (Wikipedia ou Yahoo indisponível/bloqueado)."
        )
    else:
        st.write("Tickers:", ", ".join(top5))
        df_top = build_df(top5)
        show_table_colored(df_top, height=360)

    st.divider()
    ticker_manager("Em análise (editável)", "tickers_em_analise", EM_ANALISE_DEFAULT)

with tab3:
    st.header("Diagnóstico")
    st.write("Yahoo:", "✅" if yahoo_test() else "❌")
    st.write("Nasdaq-100 (Wikipedia):", "✅" if wiki_test() else "❌")

    st.divider()
    st.subheader("Sobre a coluna 'Alerta'")
    st.write(
        "- Mostra **motivos de ausência/qualidade** dos dados.\n"
        "- Ex.: **Sem targets (Yahoo)** quando o Yahoo não traz target para aquele ticker.\n"
        "- Ex.: **targets incoerentes (corrigidos)** quando Min/Médio/Máx vêm fora de ordem e o app corrige."
    )

    st.divider()
    st.subheader("Teste rápido (um ticker)")
    t = st.text_input("Ticker para teste", "NVDA", key="diag_ticker_test")
    if st.button("Rodar teste Yahoo", key="diag_btn_test"):
        st.write(fetch_one(normalize_ticker(t)))

