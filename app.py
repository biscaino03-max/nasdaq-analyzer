import os
import re
import pandas as pd
import streamlit as st
import yfinance as yf
import requests

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

TTL_SECONDS = 300  # 5 minutos

st.set_page_config(page_title="Nasdaq Analyzer (ao vivo)", layout="wide")
st.title("Nasdaq Analyzer (ao vivo)")

# ----------------- STATE -----------------
if "tickers_investidos" not in st.session_state:
    st.session_state.tickers_investidos = INVESTIDOS_DEFAULT.copy()

if "tickers_em_analise" not in st.session_state:
    st.session_state.tickers_em_analise = EM_ANALISE_DEFAULT.copy()


def normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    t = re.sub(r"[^A-Z0-9\.\-]", "", t)  # permite BRK.B / RDS-A etc
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


def _to_float(x):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return float(x)
    except Exception:
        return None


def _first_not_none(*vals):
    for v in vals:
        v2 = _to_float(v)
        if v2 is not None:
            return v2
    return None


# ----------------- FINNHUB -----------------
def _finnhub_key():
    return os.getenv("FINNHUB_API_KEY", "").strip()


def finnhub_get(url: str, params: dict):
    api_key = _finnhub_key()
    if not api_key:
        return None, {"ok": False, "reason": "FINNHUB_API_KEY vazia/não configurada"}

    try:
        r = requests.get(url, params={**params, "token": api_key}, timeout=12)
        diag = {"ok": r.status_code == 200, "status": r.status_code, "text": r.text[:500]}
        if r.status_code != 200:
            return None, diag
        data = r.json()
        diag["json_preview"] = str(data)[:500]
        return data, diag
    except Exception as e:
        return None, {"ok": False, "reason": f"Exception: {e}"}


def finnhub_price_targets(symbol: str):
    return finnhub_get("https://finnhub.io/api/v1/stock/price-target", {"symbol": symbol})


def finnhub_recommendation_key(symbol: str):
    arr, diag = finnhub_get("https://finnhub.io/api/v1/stock/recommendation", {"symbol": symbol})
    if not isinstance(arr, list) or not arr:
        return None, diag

    latest = arr[0]
    if not isinstance(latest, dict):
        return None, diag

    buy = (latest.get("buy") or 0)
    strong_buy = (latest.get("strongBuy") or 0)
    hold = (latest.get("hold") or 0)
    sell = (latest.get("sell") or 0)
    strong_sell = (latest.get("strongSell") or 0)

    if strong_buy >= max(buy, hold, sell, strong_sell) and strong_buy > 0:
        return "strong_buy", diag
    if buy >= max(hold, sell, strong_sell) and buy > 0:
        return "buy", diag
    if hold >= max(sell, strong_sell) and hold > 0:
        return "hold", diag
    if strong_sell >= max(sell, 0) and strong_sell > 0:
        return "strong_sell", diag
    if sell > 0:
        return "sell", diag

    return None, diag


# ----------------- SIDEBAR DIAGNOSTIC -----------------
with st.sidebar:
    st.header("Diagnóstico")
    has_key = bool(_finnhub_key())
    st.write("Finnhub key detectada:", "✅" if has_key else "❌")
    if has_key:
        st.caption("A chave não é mostrada por segurança.")

    test_symbol = st.text_input("Testar Finnhub com ticker", value="NVDA", key="diag_symbol_input")
    if st.button("Rodar teste Finnhub", key="diag_run_btn"):
        sym = normalize_ticker(test_symbol)

        pt, diag_pt = finnhub_price_targets(sym)
        st.subheader("Price Target")
        st.json(diag_pt)
        st.subheader("Payload (preview)")
        st.write(pt)

        rk, diag_rk = finnhub_recommendation_key(sym)
        st.subheader("Recommendation")
        st.json(diag_rk)
        st.write("recommendationKey:", rk)


# ----------------- CACHE (TTL 5min) -----------------
@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
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
        "Erro": "",
    }

    # Yahoo
    info = {}
    try:
        info = tk.info or {}
    except Exception:
        info = {}

    rec = info.get("recommendationKey")
    row["Analistas_key"] = rec
    row["Analistas"] = pretty_analyst_label(rec)

    row["Target Min"] = _to_float(info.get("targetLowPrice"))
    row["Target Médio"] = _to_float(info.get("targetMeanPrice"))
    row["Target Máx"] = _to_float(info.get("targetHighPrice"))

    # Finnhub fallback
    if row["Target Min"] is None and row["Target Médio"] is None and row["Target Máx"] is None:
        pt, _diag = finnhub_price_targets(ticker)
        if isinstance(pt, dict):
            row["Target Min"] = _first_not_none(pt.get("low"), pt.get("targetLowPrice"))
            row["Target Médio"] = _first_not_none(pt.get("mean"), pt.get("targetMeanPrice"))
            row["Target Máx"] = _first_not_none(pt.get("high"), pt.get("targetHighPrice"))

        if not row["Analistas_key"]:
            rk, _diag2 = finnhub_recommendation_key(ticker)
            row["Analistas_key"] = rk
            row["Analistas"] = pretty_analyst_label(rk)

    if row["Target Min"] is None and row["Target Médio"] is None and row["Target Máx"] is None:
        row["Erro"] = "Sem targets (Yahoo/Finnhub)"

    return row


def build_df(tickers: list[str]) -> pd.DataFrame:
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
    if v is None or pd.isna(v) or v == "":
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


def _bg_for_target(v, current_price):
    if v is None or v == "" or pd.isna(v) or current_price is None or current_price == "" or pd.isna(current_price):
        return ""
    try:
        v = float(v)
        current_price = float(current_price)
    except Exception:
        return ""
    if v > current_price:
        return "background-color: rgba(0, 200, 0, 0.12);"
    if v < current_price:
        return "background-color: rgba(255, 0, 0, 0.12);"
    return "background-color: rgba(120, 120, 120, 0.08);"


def show_table_colored(df_raw: pd.DataFrame, only_strong_buy: bool):
    st.subheader("Tabela ao vivo")

    if df_raw is None or df_raw.empty:
        st.warning("Lista vazia. Adicione um ticker.")
        return

    df = df_raw.copy()

    if only_strong_buy and "Analistas" in df.columns:
        df = df[df["Analistas"].apply(is_strong_buy_label)].copy()

    if df.empty:
        st.info("Sem linhas para mostrar (filtro ativo ou sem dados).")
        return

    final_cols = [
        "Ticker",
        "1D", "1W", "2W", "3M", "6M", "1Y",
        "Analistas",
        "Preço",
        "Target Min", "Target Médio", "Target Máx",
        "Erro",
    ]
    df = df[[c for c in final_cols if c in df.columns]]

    # troca None por ""
    df = df.where(pd.notnull(df), "")

    styler = df.style

    for c in ["1D", "1W", "2W", "3M", "6M", "1Y"]:
        if c in df.columns:
            styler = styler.applymap(_bg_for_return, subset=[c])

    if "Preço" in df.columns:
        for c in ["Target Min", "Target Médio", "Target Máx"]:
            if c in df.columns:
                styler = styler.apply(
                    lambda row, col=c: [
                        _bg_for_target(row[col], row["Preço"]) if idx == df.columns.get_loc(col) else ""
                        for idx in range(len(df.columns))
                    ],
                    axis=1
                )

    def fmt_pct(x):
        if x == "" or x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        return f"{float(x)*100:.2f}%"

    def fmt_num(x):
        if x == "" or x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        return f"{float(x):.2f}"

    fmt = {c: fmt_pct for c in ["1D", "1W", "2W", "3M", "6M", "1Y"] if c in df.columns}
    if "Preço" in df.columns:
        fmt["Preço"] = fmt_num
    for c in ["Target Min", "Target Médio", "Target Máx"]:
        if c in df.columns:
            fmt[c] = fmt_num

    styler = styler.format(fmt)

    st.dataframe(styler, use_container_width=True)
    st.caption("Dados via Yahoo (yfinance) + fallback Finnhub. Atualização automática a cada 5 min + botão manual.")


def show_rankings(df_raw: pd.DataFrame):
    st.subheader("Rankings (Top 5)")

    if df_raw is None or df_raw.empty:
        st.info("Sem dados para ranking.")
        return

    cols = st.columns(3)
    ranking_specs = [("1Y", "Top 5 — 1 Ano"), ("6M", "Top 5 — 6 Meses"), ("3M", "Top 5 — 3 Meses")]

    for col, (metric, title) in zip(cols, ranking_specs):
        with col:
            st.markdown(f"**{title}**")
            if metric not in df_raw.columns:
                st.write("—")
                continue
            temp = df_raw[["Ticker", metric]].dropna().sort_values(metric, ascending=False).head(5)
            if temp.empty:
                st.write("—")
                continue
            for idx, r in enumerate(temp.itertuples(index=False), start=1):
                st.write(f"{idx}. {r.Ticker} — {r._1*100:.2f}%")


def show_strong_buy_top5(df_raw: pd.DataFrame):
    st.subheader("🟢 Compra forte (Strong Buy)")

    if df_raw is None or df_raw.empty or "Analistas" not in df_raw.columns:
        st.info("Sem dados.")
        return

    sb = df_raw[df_raw["Analistas"].apply(is_strong_buy_label)].copy()
    if sb.empty:
        st.write("Nenhuma ação marcada como 🟢 Compra forte no momento.")
        return

    def score_row(r):
        for m in ["1Y", "6M", "3M"]:
            v = r.get(m)
            if v is not None and not pd.isna(v):
                return float(v)
        return float("-inf")

    sb["score"] = sb.apply(score_row, axis=1)
    sb = sb.sort_values("score", ascending=False).head(5)

    for idx, r in enumerate(sb.itertuples(index=False), start=1):
        row = r._asdict()
        ticker = row.get("Ticker")
        score = row.get("score")
        price = row.get("Preço")
        st.write(f"{idx}. {ticker} — desempenho (prioridade 1Y/6M/3M): {score*100:.2f}% — Preço: {price:.2f}")


def ticker_manager(title: str, key_state: str, default_list: list[str]):
    st.markdown(f"### {title}")

    tickers = st.session_state[key_state]
    left, right = st.columns([1, 3], vertical_alignment="top")

    with left:
        st.markdown("**Adicionar ação**")
        # keys mais "namespaced" para evitar qualquer conflito
        new_t = st.text_input("Ticker", key=f"tm__{key_state}__new", placeholder="Ex: AAPL, TSLA, NVDA")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Adicionar", key=f"tm__{key_state}__add"):
                t = normalize_ticker(new_t)
                if not t:
                    st.warning("Digite um ticker válido.")
                elif t in st.session_state[key_state]:
                    st.info("Esse ticker já está na lista.")
                else:
                    st.session_state[key_state].append(t)
                    st.success(f"Adicionado: {t}")
                    st.rerun()

        with c2:
            if st.button("Resetar", key=f"tm__{key_state}__reset"):
                st.session_state[key_state] = default_list.copy()
                st.success("Lista resetada.")
                st.rerun()

        st.divider()

        if st.button("Atualizar agora", key=f"tm__{key_state}__manual_refresh"):
            st.cache_data.clear()
            st.success("Atualizado manualmente.")
            st.rerun()

        st.caption(f"Atualização automática: a cada {TTL_SECONDS//60} min.")

    with right:
        st.markdown("**Tickers atuais**")
        if not tickers:
            st.info("Lista vazia.")
        else:
            cols = st.columns(len(tickers))
            for i, t in enumerate(tickers):
                with cols[i]:
                    st.write(t)
                    if st.button("❌", key=f"tm__{key_state}__del_{i}"):
                        try:
                            st.session_state[key_state].remove(t)
                        except ValueError:
                            pass
                        st.rerun()

        df_raw = build_df(st.session_state[key_state])

        show_strong_buy_top5(df_raw)
        show_rankings(df_raw)

        st.divider()
        only_sb = st.checkbox("Mostrar só 🟢 Compra forte (Strong Buy)", value=False, key=f"tm__{key_state}__only_sb")
        show_table_colored(df_raw, only_strong_buy=only_sb)


# ----------------- TABS -----------------
tab1, tab2 = st.tabs(["Investidos", "Em análise"])

with tab1:
    ticker_manager("Investidos (editável)", "tickers_investidos", INVESTIDOS_DEFAULT)

with tab2:
    ticker_manager("Em análise (editável)", "tickers_em_analise", EM_ANALISE_DEFAULT)
