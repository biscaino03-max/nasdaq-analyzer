import re
import pandas as pd
import streamlit as st
import yfinance as yf

# ----------------- CONFIG -----------------
INVESTIDOS_DEFAULT = ["AMCI", "VMAR", "VITL", "UAL", "MSFT", "DIS", "GPCR", "NVDA"]
EM_ANALISE_DEFAULT = []

WINDOWS = {"1D": 1, "1W": 5, "2W": 10, "3M": 63, "6M": 126, "1Y": 252}

TTL_SECONDS = 300  # 5 minutos (auto)
NASDAQ100_LIST_TTL = 24 * 60 * 60  # 24h (lista do Nasdaq-100 muda raramente)

ANALYST_LABELS = {
    "strong_buy": "🟢 Compra forte",
    "buy": "🔵 Compra",
    "hold": "🟡 Manter",
    "sell": "🔴 Venda",
    "strong_sell": "🟥 Venda forte",
    None: "—",
    "none": "—",
}

REC_SCORE = {
    "strong_buy": 5,
    "buy": 4,
    "hold": 3,
    "sell": 2,
    "strong_sell": 1,
    None: 0,
    "none": 0,
}

st.set_page_config(page_title="Nasdaq Analyzer (ao vivo)", layout="wide")
st.title("Nasdaq Analyzer (ao vivo)")

# ----------------- STATE -----------------
if "tickers_investidos" not in st.session_state:
    st.session_state.tickers_investidos = INVESTIDOS_DEFAULT.copy()
if "tickers_em_analise" not in st.session_state:
    st.session_state.tickers_em_analise = EM_ANALISE_DEFAULT.copy()

# token manual para forçar refresh geral (limpa cache)
if "manual_refresh_token" not in st.session_state:
    st.session_state.manual_refresh_token = 0


def normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    return re.sub(r"[^A-Z0-9\.\-]", "", t)


def pct_change(closes: pd.Series, n: int):
    if closes is None or closes.empty or len(closes) <= n:
        return None
    last = closes.iloc[-1]
    prev = closes.iloc[-1 - n]
    if pd.isna(last) or pd.isna(prev) or prev == 0:
        return None
    return float((last / prev) - 1.0)


def pretty_label(key):
    if not key:
        return ANALYST_LABELS[None]
    return ANALYST_LABELS.get(key, key)


def is_strong_buy_label(label: str) -> bool:
    return isinstance(label, str) and label.strip().startswith("🟢")


# ----------------- Nasdaq-100 tickers (web) -----------------
@st.cache_data(ttl=NASDAQ100_LIST_TTL, show_spinner=False)
def get_nasdaq100_tickers():
    """
    Tenta pegar lista de componentes do Nasdaq-100 via páginas públicas.
    1) Slickcharts (simples e costuma funcionar bem)
    2) Wikipedia como fallback
    """
    # 1) Slickcharts
    try:
        df_list = pd.read_html("https://www.slickcharts.com/nasdaq100")[0]
        # Colunas típicas: Company / Symbol / Weight ...
        if "Symbol" in df_list.columns:
            tickers = [normalize_ticker(x) for x in df_list["Symbol"].tolist()]
            tickers = [t for t in tickers if t]
            if len(tickers) >= 90:
                return sorted(list(set(tickers)))
    except Exception:
        pass

    # 2) Wikipedia fallback
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        # procurar uma tabela que tenha "Ticker" (pode variar)
        for tb in tables:
            cols = [str(c).lower() for c in tb.columns]
            if any("ticker" in c for c in cols):
                # acha coluna ticker
                tcol = None
                for c in tb.columns:
                    if "ticker" in str(c).lower():
                        tcol = c
                        break
                if tcol is not None:
                    tickers = [normalize_ticker(x) for x in tb[tcol].astype(str).tolist()]
                    tickers = [t for t in tickers if t and t != "NAN"]
                    if len(tickers) >= 90:
                        return sorted(list(set(tickers)))
    except Exception:
        pass

    return []


# ----------------- Yahoo fetch (TTL 5min) -----------------
@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def fetch_one(ticker: str, refresh_token: int):
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
        "Analyst N": None,
        "Erro": "",
    }

    try:
        info = tk.info or {}
        rec = info.get("recommendationKey")
        row["Analistas_key"] = rec
        row["Analistas"] = pretty_label(rec)
        row["Target Min"] = info.get("targetLowPrice")
        row["Target Médio"] = info.get("targetMeanPrice")
        row["Target Máx"] = info.get("targetHighPrice")
        row["Analyst N"] = info.get("numberOfAnalystOpinions")
    except Exception:
        pass

    return row


def build_df(tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    rows = []
    progress = st.progress(0)
    for i, t in enumerate(tickers):
        rows.append(fetch_one(t, st.session_state.manual_refresh_token))
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


def _bg_for_target(v, current_price):
    if v is None or pd.isna(v) or current_price is None or pd.isna(current_price):
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


def fmt_pct(x):
    if x is None or pd.isna(x):
        return ""
    return f"{float(x) * 100:.2f}%"


def fmt_price(x):
    if x is None or pd.isna(x):
        return ""
    try:
        return f"{float(x):.2f}"
    except Exception:
        return ""


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

    # Ordem pedida: Preço entre Analistas e Target Min
    final_cols = [
        "Ticker",
        "1D", "1W", "2W", "3M", "6M", "1Y",
        "Analistas",
        "Preço",
        "Target Min", "Target Médio", "Target Máx",
        "Erro",
    ]
    df = df[[c for c in final_cols if c in df.columns]]

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

    fmt = {}
    for c in ["1D", "1W", "2W", "3M", "6M", "1Y"]:
        if c in df.columns:
            fmt[c] = fmt_pct
    if "Preço" in df.columns:
        fmt["Preço"] = fmt_price
    for c in ["Target Min", "Target Médio", "Target Máx"]:
        if c in df.columns:
            fmt[c] = fmt_price

    st.dataframe(styler.format(fmt), use_container_width=True)
    st.caption("Dados via Yahoo (yfinance). Auto a cada 5 min + botão manual.")


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
                st.write(f"{idx}. {r.Ticker} — {fmt_pct(r._1)}")


def show_strong_buy_top5(df_raw: pd.DataFrame):
    st.subheader("🟢 Compra forte (Strong Buy)")
    if df_raw is None or df_raw.empty or "Analistas" not in df_raw.columns:
        st.info("Sem dados.")
        return
    sb = df_raw[df_raw["Analistas"].apply(is_strong_buy_label)].copy()
    if sb.empty:
        st.write("Nenhuma ação marcada como 🟢 Compra forte no momento.")
        return

    def perf_score_row(r):
        for m in ["1Y", "6M", "3M"]:
            v = r.get(m)
            if v is not None and not pd.isna(v):
                return float(v)
        return float("-inf")

    sb["perf_score"] = sb.apply(perf_score_row, axis=1)
    sb = sb.sort_values("perf_score", ascending=False).head(5)
    for idx, r in enumerate(sb.itertuples(index=False), start=1):
        d = r._asdict()
        st.write(f"{idx}. {d.get('Ticker')} — {d.get('Analistas')} — perf: {fmt_pct(d.get('perf_score'))} — Preço: {fmt_price(d.get('Preço'))}")


# ----------------- Nasdaq-100 Top 5 recommended (Yahoo) -----------------
def show_nasdaq100_top5_yahoo():
    st.subheader("⭐ Top 5 mais recomendadas do Nasdaq-100 (Yahoo)")

    tickers = get_nasdaq100_tickers()
    if not tickers:
        st.info("Não consegui carregar a lista do Nasdaq-100 agora. Tente novamente.")
        return

    st.caption(f"Universo: {len(tickers)} tickers do Nasdaq-100. (Pode demorar na 1ª execução.)")

    # Para não travar sempre, fazemos o cálculo sob demanda (manual) OU usa cache TTL 5min.
    df = build_df(tickers)

    if df.empty:
        st.info("Sem dados.")
        return

    # score por recomendação + desempate por número de analistas + performance
    df["rec_score"] = df["Analistas_key"].map(lambda x: REC_SCORE.get(x, 0))
    df["analyst_n"] = df.get("Analyst N", pd.Series([None]*len(df))).apply(
        lambda x: int(x) if (x is not None and not pd.isna(x)) else 0
    )

    def perf_score_row(r):
        for m in ["1Y", "6M", "3M"]:
            v = r.get(m)
            if v is not None and not pd.isna(v):
                return float(v)
        return float("-inf")

    df["perf_score"] = df.apply(perf_score_row, axis=1)

    # Ordena: melhor recomendação, mais analistas, melhor performance
    df2 = df.sort_values(["rec_score", "analyst_n", "perf_score"], ascending=[False, False, False]).head(5)

    # Mostrar
    for i, r in enumerate(df2.itertuples(index=False), start=1):
        d = r._asdict()
        st.write(
            f"{i}. {d.get('Ticker')} — {d.get('Analistas')} "
            f"(analistas: {d.get('analyst_n', 0)}) — Preço: {fmt_price(d.get('Preço'))}"
        )


def ticker_manager(title: str, key_state: str, default_list: list[str], show_nasdaq100_box: bool = False):
    st.markdown(f"### {title}")

    tickers = st.session_state[key_state]
    left, right = st.columns([1, 3], vertical_alignment="top")

    with left:
        st.markdown("**Adicionar ação**")
        new_t = st.text_input("Ticker", key=f"{key_state}_new", placeholder="Ex: AAPL, TSLA, NVDA")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Adicionar", key=f"{key_state}_add"):
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
            if st.button("Resetar", key=f"{key_state}_reset"):
                st.session_state[key_state] = default_list.copy()
                st.success("Lista resetada.")
                st.rerun()

        st.divider()

        # Manual refresh: limpa cache e muda token (para reavaliar tudo)
        if st.button("Atualizar agora", key=f"{key_state}_manual_refresh"):
            st.cache_data.clear()
            st.session_state.manual_refresh_token += 1
            st.success("Atualizado manualmente.")
            st.rerun()

        st.caption(f"Atualização automática: a cada {TTL_SECONDS//60} min (TTL).")

    with right:
        if show_nasdaq100_box:
            # Botão separado para forçar recálculo do Top 5 Nasdaq-100
            topcol1, topcol2 = st.columns([1, 2])
            with topcol1:
                if st.button("Atualizar Top 5 Nasdaq-100 agora"):
                    st.cache_data.clear()
                    st.session_state.manual_refresh_token += 1
                    st.rerun()
            with topcol2:
                st.caption("Top 5 calculado no universo Nasdaq-100 (não depende da sua lista).")

            show_nasdaq100_top5_yahoo()
            st.divider()

        st.markdown("**Tickers atuais**")
        if not tickers:
            st.info("Lista vazia.")
        else:
            cols = st.columns(len(tickers))
            for i, t in enumerate(tickers):
                with cols[i]:
                    st.write(t)
                    if st.button("❌", key=f"{key_state}_del_{i}"):
                        try:
                            st.session_state[key_state].remove(t)
                        except ValueError:
                            pass
                        st.rerun()

        df_raw = build_df(st.session_state[key_state])
        show_strong_buy_top5(df_raw)
        show_rankings(df_raw)

        st.divider()
        only_sb = st.checkbox("Mostrar só 🟢 Compra forte (Strong Buy)", value=False, key=f"{key_state}_only_sb")
        show_table_colored(df_raw, only_strong_buy=only_sb)


# ----------------- TABS -----------------
tab1, tab2 = st.tabs(["Investidos", "Em análise"])

with tab1:
    ticker_manager("Investidos (editável)", "tickers_investidos", INVESTIDOS_DEFAULT, show_nasdaq100_box=False)

with tab2:
    # Aqui aparece o Top 5 do Nasdaq-100 (Yahoo), independente da lista "Em análise"
    ticker_manager("Em análise (editável)", "tickers_em_analise", EM_ANALISE_DEFAULT, show_nasdaq100_box=True)
