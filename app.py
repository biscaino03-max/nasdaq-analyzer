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
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()


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


def _resolve_backup_store_file(path: str) -> str:
    base, ext = os.path.splitext(path)
    if not ext:
        return f"{path}.backup.json"
    return f"{base}.backup{ext}"


BACKUP_STORE_FILE = _resolve_backup_store_file(STORE_FILE)

st.set_page_config(page_title="Nasdaq Analyzer (ao vivo)", layout="wide")
st.title("Nasdaq Analyzer (ao vivo)")
top_right_placeholder = st.empty()


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
    _safe_write_json(BACKUP_STORE_FILE, payload)


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


def _append_alert(row: dict, message: str):
    if not message:
        return
    row["Alerta"] = (row["Alerta"] + " | " if row["Alerta"] else "") + message


def _merge_sources(*sources) -> str:
    uniq = []
    for s in sources:
        if not s:
            continue
        if isinstance(s, (list, tuple)):
            for item in s:
                if item and item not in uniq:
                    uniq.append(item)
        elif s not in uniq:
            uniq.append(s)
    return " + ".join(uniq) if uniq else "—"


def _build_base_row(ticker: str, closes: pd.Series, hist_source: str) -> dict:
    last_close = float(closes.iloc[-1]) if closes is not None and not closes.empty else None
    return {
        "Ticker": ticker,
        "1D": pct_change(closes, WINDOWS["1D"]) if closes is not None else None,
        "1W": pct_change(closes, WINDOWS["1W"]) if closes is not None else None,
        "2W": pct_change(closes, WINDOWS["2W"]) if closes is not None else None,
        "3M": pct_change(closes, WINDOWS["3M"]) if closes is not None else None,
        "6M": pct_change(closes, WINDOWS["6M"]) if closes is not None else None,
        "1Y": pct_change(closes, WINDOWS["1Y"]) if closes is not None else None,
        "Analistas_key": None,
        "Analistas": "—",
        "Preço": last_close,
        "Target Min": None,
        "Target Médio": None,
        "Target Máx": None,
        "Fonte": hist_source or "—",
        "Alerta": "",
        "Erro": "",
    }


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def fetch_closes_yahoo(ticker: str, period: str = "13mo") -> pd.Series:
    ticker = normalize_ticker(ticker)
    hist = yf.Ticker(ticker).history(period=period, interval="1d")
    if hist is None or hist.empty or "Close" not in hist.columns:
        return pd.Series(dtype="float64")
    closes = hist["Close"].dropna()
    return closes if not closes.empty else pd.Series(dtype="float64")


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def fetch_closes_twelvedata(ticker: str, outputsize: int = 350) -> pd.Series:
    if not TWELVE_DATA_API_KEY:
        return pd.Series(dtype="float64")
    try:
        r = _http_get(
            "https://api.twelvedata.com/time_series"
            f"?symbol={normalize_ticker(ticker)}&interval=1day&outputsize={outputsize}&apikey={TWELVE_DATA_API_KEY}",
            timeout=20,
        )
        if r.status_code != 200:
            return pd.Series(dtype="float64")
        payload = r.json()
        values = payload.get("values", [])
        if not isinstance(values, list) or not values:
            return pd.Series(dtype="float64")
        rows = []
        for item in values:
            dt_raw = item.get("datetime")
            close_raw = _as_float(item.get("close"))
            if not dt_raw or close_raw is None:
                continue
            rows.append((pd.to_datetime(dt_raw), close_raw))
        if not rows:
            return pd.Series(dtype="float64")
        df = pd.DataFrame(rows, columns=["Date", "Close"]).sort_values("Date")
        return df.set_index("Date")["Close"]
    except Exception:
        return pd.Series(dtype="float64")


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def fetch_finnhub_quote(ticker: str) -> dict:
    if not FINNHUB_API_KEY:
        return {}
    try:
        r = _http_get(
            f"https://finnhub.io/api/v1/quote?symbol={normalize_ticker(ticker)}&token={FINNHUB_API_KEY}",
            timeout=15,
        )
        if r.status_code != 200:
            return {}
        payload = r.json()
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _finnhub_rec_to_key(rec_obj: dict):
    if not isinstance(rec_obj, dict):
        return None
    counts = {
        "strong_buy": int(rec_obj.get("strongBuy", 0) or 0),
        "buy": int(rec_obj.get("buy", 0) or 0),
        "hold": int(rec_obj.get("hold", 0) or 0),
        "sell": int(rec_obj.get("sell", 0) or 0),
        "strong_sell": int(rec_obj.get("strongSell", 0) or 0),
    }
    best_key, best_value = None, -1
    for k in ["strong_buy", "buy", "hold", "sell", "strong_sell"]:
        if counts[k] > best_value:
            best_key, best_value = k, counts[k]
    return best_key if best_value > 0 else None


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def fetch_finnhub_enrichment(ticker: str) -> dict:
    if not FINNHUB_API_KEY:
        return {}
    ticker = normalize_ticker(ticker)
    out = {"analyst_key": None, "target_low": None, "target_mean": None, "target_high": None}
    try:
        rec_r = _http_get(
            f"https://finnhub.io/api/v1/stock/recommendation?symbol={ticker}&token={FINNHUB_API_KEY}",
            timeout=15,
        )
        if rec_r.status_code == 200:
            rec_payload = rec_r.json()
            if isinstance(rec_payload, list) and rec_payload:
                out["analyst_key"] = _finnhub_rec_to_key(rec_payload[0])
    except Exception:
        pass

    try:
        target_r = _http_get(
            f"https://finnhub.io/api/v1/stock/price-target?symbol={ticker}&token={FINNHUB_API_KEY}",
            timeout=15,
        )
        if target_r.status_code == 200:
            t_payload = target_r.json()
            if isinstance(t_payload, dict):
                out["target_low"] = t_payload.get("targetLow")
                out["target_mean"] = t_payload.get("targetMean")
                out["target_high"] = t_payload.get("targetHigh")
    except Exception:
        pass
    return out


def _pick_best_closes(ticker: str, period: str = "13mo"):
    closes = fetch_closes_yahoo(ticker, period=period)
    if closes is not None and not closes.empty:
        return closes, "Yahoo"

    closes = fetch_closes_twelvedata(ticker, outputsize=450)
    if closes is not None and not closes.empty:
        return closes, "TwelveData"

    return pd.Series(dtype="float64"), None


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
    closes, hist_source = _pick_best_closes(ticker, period="13mo")
    if closes is None or closes.empty:
        quote = fetch_finnhub_quote(ticker)
        price = _as_float(quote.get("c")) if quote else None
        if price is None:
            return {"Ticker": ticker, "Erro": "Sem dados (Yahoo/TwelveData/Finnhub)"}
        row = _build_base_row(ticker, pd.Series([price]), "Finnhub")
        _append_alert(row, "Sem histórico diário completo; exibindo preço atual de fallback.")
    else:
        row = _build_base_row(ticker, closes, hist_source)
        if hist_source != "Yahoo":
            _append_alert(row, f"Histórico via {hist_source} (fallback).")

    sources = [hist_source] if hist_source else []

    # Yahoo (analistas/targets) primeiro
    try:
        info = yf.Ticker(ticker).info or {}
        rec = info.get("recommendationKey")
        if rec:
            row["Analistas_key"] = rec
            row["Analistas"] = pretty_analyst_label(rec)
            sources.append("Yahoo")
        if row["Target Médio"] is None:
            row["Target Min"] = info.get("targetLowPrice")
            row["Target Médio"] = info.get("targetMeanPrice")
            row["Target Máx"] = info.get("targetHighPrice")
            if row["Target Médio"] is not None:
                sources.append("Yahoo")
    except Exception:
        pass

    # Finnhub como fallback de recomendação/targets/preço
    if row["Analistas_key"] is None or row["Target Médio"] is None or row["Preço"] is None:
        fin = fetch_finnhub_enrichment(ticker)
        if row["Analistas_key"] is None and fin.get("analyst_key"):
            row["Analistas_key"] = fin["analyst_key"]
            row["Analistas"] = pretty_analyst_label(fin["analyst_key"])
            sources.append("Finnhub")
        if row["Target Médio"] is None:
            row["Target Min"] = fin.get("target_low")
            row["Target Médio"] = fin.get("target_mean")
            row["Target Máx"] = fin.get("target_high")
            if row["Target Médio"] is not None:
                sources.append("Finnhub")
        if row["Preço"] is None:
            q = fetch_finnhub_quote(ticker)
            price = _as_float(q.get("c")) if q else None
            if price is not None:
                row["Preço"] = price
                sources.append("Finnhub")

    low, mean, high, ok, msg = validate_and_fix_targets(
        row["Target Min"], row["Target Médio"], row["Target Máx"]
    )
    row["Target Min"], row["Target Médio"], row["Target Máx"] = low, mean, high
    if ok is False:
        _append_alert(row, "⚠️ " + msg)

    if row["Target Médio"] is None:
        _append_alert(row, "Sem targets (Yahoo/Finnhub).")

    row["Fonte"] = _merge_sources(sources)
    return row


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def fetch_history_for_chart(ticker: str, period: str = "6mo") -> pd.DataFrame:
    ticker = normalize_ticker(ticker)
    closes, _source = _pick_best_closes(ticker, period=period)
    if closes is None or closes.empty:
        closes, _source = _pick_best_closes(ticker, period="13mo")
    if closes is None or closes.empty:
        return pd.DataFrame()

    # Se vier fallback com janela maior, reduz para o período escolhido.
    period_points = {
        "1mo": 23,
        "3mo": 70,
        "6mo": 140,
        "1y": 260,
        "2y": 520,
        "5y": 1300,
        "max": None,
    }
    max_points = period_points.get(period)
    if max_points:
        closes = closes.tail(max_points)

    df = closes.to_frame(name="Close").copy()

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


def _num_or_zero(v) -> float:
    n = _as_float(v)
    return n if n is not None else 0.0


@st.cache_data(ttl=TTL_LONG_SECONDS, show_spinner=False)
def top10_short_term_nasdaq100() -> pd.DataFrame:
    """
    Curto prazo (2 a 8 semanas):
    peso maior em momentum recente (1W/2W), com ajuste por analistas e upside.
    """
    tickers = get_nasdaq100_wikipedia()
    if not tickers:
        return pd.DataFrame()

    rows = []
    for t in tickers[:110]:
        try:
            r = fetch_one(t)
            if not r or r.get("Erro"):
                continue

            m1d = _num_or_zero(r.get("1D"))
            m1w = _num_or_zero(r.get("1W"))
            m2w = _num_or_zero(r.get("2W"))
            rec = ANALYST_SCORE.get(r.get("Analistas_key"), 0)

            price = _as_float(r.get("Preço"))
            tmean = _as_float(r.get("Target Médio"))
            upside = (tmean - price) / price if (price and tmean and price > 0) else 0.0

            score = (
                (m1w * 100.0) * 0.45
                + (m2w * 100.0) * 0.35
                + (m1d * 100.0) * 0.10
                + rec * 1.5
                + upside * 100.0 * 0.10
            )

            rows.append(
                {
                    "Ticker": t,
                    "Score Curto": round(score, 2),
                    "1D": round(m1d * 100.0, 2),
                    "1W": round(m1w * 100.0, 2),
                    "2W": round(m2w * 100.0, 2),
                    "Analistas": r.get("Analistas", "—"),
                    "Preço (USD)": round(float(price), 2) if price is not None else None,
                    "Fonte": r.get("Fonte", "—"),
                    "Alerta": r.get("Alerta", ""),
                    "Erro": r.get("Erro", ""),
                }
            )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("Score Curto", ascending=False).head(10).reset_index(drop=True)
    ordered_cols = [
        "Ticker",
        "Score Curto",
        "1D",
        "1W",
        "2W",
        "Analistas",
        "Preço (USD)",
        "Fonte",
        "Alerta",
        "Erro",
    ]
    return df[[c for c in ordered_cols if c in df.columns]]


@st.cache_data(ttl=TTL_LONG_SECONDS, show_spinner=False)
def top10_medium_term_nasdaq100() -> pd.DataFrame:
    """
    Médio prazo (3 a 12 meses):
    peso maior em 3M/6M/1Y, com ajuste por analistas e upside.
    """
    tickers = get_nasdaq100_wikipedia()
    if not tickers:
        return pd.DataFrame()

    rows = []
    for t in tickers[:110]:
        try:
            r = fetch_one(t)
            if not r or r.get("Erro"):
                continue

            m3m = _num_or_zero(r.get("3M"))
            m6m = _num_or_zero(r.get("6M"))
            m1y = _num_or_zero(r.get("1Y"))
            rec = ANALYST_SCORE.get(r.get("Analistas_key"), 0)

            price = _as_float(r.get("Preço"))
            tmean = _as_float(r.get("Target Médio"))
            upside = (tmean - price) / price if (price and tmean and price > 0) else 0.0

            score = (
                (m3m * 100.0) * 0.30
                + (m6m * 100.0) * 0.35
                + (m1y * 100.0) * 0.25
                + rec * 1.5
                + upside * 100.0 * 0.10
            )

            rows.append(
                {
                    "Ticker": t,
                    "Score Médio": round(score, 2),
                    "3M": round(m3m * 100.0, 2),
                    "6M": round(m6m * 100.0, 2),
                    "1Y": round(m1y * 100.0, 2),
                    "Analistas": r.get("Analistas", "—"),
                    "Preço (USD)": round(float(price), 2) if price is not None else None,
                    "Fonte": r.get("Fonte", "—"),
                    "Alerta": r.get("Alerta", ""),
                    "Erro": r.get("Erro", ""),
                }
            )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("Score Médio", ascending=False).head(10).reset_index(drop=True)
    ordered_cols = [
        "Ticker",
        "Score Médio",
        "3M",
        "6M",
        "1Y",
        "Analistas",
        "Preço (USD)",
        "Fonte",
        "Alerta",
        "Erro",
    ]
    return df[[c for c in ordered_cols if c in df.columns]]


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
    df = df.rename(
        columns={
            "Preço": "Preço (USD)",
            "Target Min": "Target Min (USD)",
            "Target Médio": "Target Médio (USD)",
            "Target Máx": "Target Máx (USD)",
        }
    )
    final_cols = [
        "Ticker",
        "1D",
        "1W",
        "2W",
        "3M",
        "6M",
        "1Y",
        "Analistas",
        "Preço (USD)",
        "Target Min (USD)",
        "Target Médio (USD)",
        "Target Máx (USD)",
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
    if "Preço (USD)" in df.columns:
        fmt["Preço (USD)"] = lambda x: "" if (x is None or pd.isna(x)) else f"{float(x):.2f}"
    for c in ["Target Min (USD)", "Target Médio (USD)", "Target Máx (USD)"]:
        if c in df.columns:
            fmt[c] = lambda x: "" if (x is None or pd.isna(x)) else f"{float(x):.2f}"

    st.dataframe(styler.format(fmt), use_container_width=True, height=height)
    st.caption(
        "Dados com fallback: Yahoo -> TwelveData (preço/histórico) "
        "e Yahoo/Finnhub (analistas/targets). Cache TTL 5 min + botão manual. "
        "Top 5: Nasdaq-100 (Wikipedia) + ranking por recomendação."
    )


def show_rank_table_colored(df_raw: pd.DataFrame, score_col: str, pct_cols: list[str], height=360):
    if df_raw is None or df_raw.empty:
        return

    def _bg_for_score(v):
        n = _as_float(v)
        if n is None:
            return ""
        if n > 0:
            return "background-color: rgba(0, 180, 0, 0.20);"
        if n < 0:
            return "background-color: rgba(220, 0, 0, 0.20);"
        return "background-color: rgba(120, 120, 120, 0.10);"

    styler = df_raw.style
    if score_col in df_raw.columns:
        styler = styler.applymap(_bg_for_score, subset=[score_col])

    valid_pct_cols = [c for c in pct_cols if c in df_raw.columns]
    for c in valid_pct_cols:
        styler = styler.applymap(_bg_for_return, subset=[c])

    fmt = {}
    if score_col in df_raw.columns:
        fmt[score_col] = lambda x: "" if (x is None or pd.isna(x)) else f"{float(x):.2f}"
    for c in valid_pct_cols:
        fmt[c] = lambda x: "" if (x is None or pd.isna(x)) else f"{float(x):.2f}%"
    if "Preço (USD)" in df_raw.columns:
        fmt["Preço (USD)"] = lambda x: "" if (x is None or pd.isna(x)) else f"{float(x):.2f}"

    st.dataframe(styler.format(fmt), use_container_width=True, height=height)


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


def finnhub_test():
    if not FINNHUB_API_KEY:
        return False
    q = fetch_finnhub_quote("MSFT")
    return bool(q and _as_float(q.get("c")) is not None)


def twelvedata_test():
    if not TWELVE_DATA_API_KEY:
        return False
    s = fetch_closes_twelvedata("MSFT", outputsize=50)
    return s is not None and not s.empty


def wiki_test():
    tickers = get_nasdaq100_wikipedia()
    return bool(tickers)


def quick_source_test(ticker: str) -> dict:
    t = normalize_ticker(ticker)
    if not t:
        return {"erro": "Ticker inválido."}

    result = {"ticker": t}

    # Yahoo
    yahoo_closes = fetch_closes_yahoo(t, period="1mo")
    result["yahoo_ok"] = yahoo_closes is not None and not yahoo_closes.empty
    result["yahoo_last_close"] = (
        round(float(yahoo_closes.iloc[-1]), 4) if result["yahoo_ok"] else None
    )

    # Finnhub
    fin_q = fetch_finnhub_quote(t)
    fin_price = _as_float(fin_q.get("c")) if isinstance(fin_q, dict) else None
    result["finnhub_ok"] = fin_price is not None
    result["finnhub_price"] = round(fin_price, 4) if fin_price is not None else None

    # Twelve Data
    td_closes = fetch_closes_twelvedata(t, outputsize=50)
    result["twelve_ok"] = td_closes is not None and not td_closes.empty
    result["twelve_last_close"] = (
        round(float(td_closes.iloc[-1]), 4) if result["twelve_ok"] else None
    )

    # Nasdaq-100 (Wikipedia): testa disponibilidade da lista + presença do ticker
    ndx_list = get_nasdaq100_wikipedia()
    result["nasdaq_ok"] = bool(ndx_list)
    result["ticker_in_nasdaq100"] = t.replace(".", "-") in ndx_list if ndx_list else False

    return result


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def fetch_usd_brl_widget_data() -> dict:
    price = None
    delta_pct = None
    analyst_key = None

    # Cotacao USD/BRL
    try:
        hist = yf.Ticker("USDBRL=X").history(period="5d", interval="1d")
        if hist is not None and not hist.empty and "Close" in hist.columns:
            closes = hist["Close"].dropna()
            if len(closes) >= 1:
                price = float(closes.iloc[-1])
            if len(closes) >= 2 and closes.iloc[-2] != 0:
                delta_pct = float((closes.iloc[-1] / closes.iloc[-2] - 1.0) * 100.0)
    except Exception:
        pass

    # Recomendacao de analistas (proxy do dolar via ETF UUP)
    try:
        info = yf.Ticker("UUP").info or {}
        analyst_key = info.get("recommendationKey")
    except Exception:
        analyst_key = None

    label = pretty_analyst_label(analyst_key)
    if analyst_key in ("strong_buy", "buy"):
        signal = "COMPRA"
    elif analyst_key in ("strong_sell", "sell"):
        signal = "VENDA"
    elif analyst_key == "hold":
        signal = "NEUTRO"
    else:
        signal = "SEM CONSENSO"

    return {
        "price": price,
        "delta_pct": delta_pct,
        "analyst_key": analyst_key,
        "analyst_label": label,
        "signal": signal,
    }


def render_top_right_usd_widget():
    data = fetch_usd_brl_widget_data()
    with top_right_placeholder.container():
        c1, c2 = st.columns([3, 2])
        with c1:
            st.write("")
        with c2:
            st.markdown("### USD/BRL")
            if data["price"] is None:
                st.warning("Cotacao indisponivel no momento.")
                return
            delta_txt = None
            if data["delta_pct"] is not None:
                delta_txt = f"{data['delta_pct']:+.2f}%"
            st.metric("Dolar americano", f"R$ {data['price']:.4f}", delta_txt)
            st.caption(
                "Sinal por analistas (proxy UUP): "
                f"{data['signal']} | {data['analyst_label']}"
            )


# ----------------- TABS -----------------
render_top_right_usd_widget()
tab1, tab2, tab3 = st.tabs(["Investidos", "Em análise", "Diagnóstico"])

with tab1:
    ticker_manager("Investidos (editável)", "tickers_investidos", INVESTIDOS_DEFAULT)

with tab2:
    st.header("Top 5 mais recomendadas (automático) — Nasdaq-100")
    st.caption("Critério do Top 5 atual: recomendação de analistas + upside de target médio.")
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
    st.subheader("Top 10 Curto Prazo (2 a 8 semanas) — Nasdaq-100")
    st.caption(
        "Critério: momentum recente (1W/2W/1D) com ajuste por recomendação de analistas e upside."
    )
    df_short = top10_short_term_nasdaq100()
    if df_short.empty:
        st.warning("Não consegui montar o Top 10 de curto prazo no momento.")
    else:
        show_rank_table_colored(
            df_short,
            score_col="Score Curto",
            pct_cols=["1D", "1W", "2W"],
            height=360,
        )
        st.write("Tickers (curto):", ", ".join(df_short["Ticker"].tolist()))

    st.divider()
    st.subheader("Top 10 Médio Prazo (3 a 12 meses) — Nasdaq-100")
    st.caption(
        "Critério: tendência/momentum (3M/6M/1Y) com ajuste por recomendação de analistas e upside."
    )
    df_medium = top10_medium_term_nasdaq100()
    if df_medium.empty:
        st.warning("Não consegui montar o Top 10 de médio prazo no momento.")
    else:
        show_rank_table_colored(
            df_medium,
            score_col="Score Médio",
            pct_cols=["3M", "6M", "1Y"],
            height=360,
        )
        st.write("Tickers (médio):", ", ".join(df_medium["Ticker"].tolist()))

    st.divider()
    ticker_manager("Em análise (editável)", "tickers_em_analise", EM_ANALISE_DEFAULT)

with tab3:
    st.header("Diagnóstico")
    st.write("Yahoo:", "✅" if yahoo_test() else "❌")
    st.write("Finnhub:", "✅" if finnhub_test() else "❌")
    st.write("Twelve Data:", "✅" if twelvedata_test() else "❌")
    st.write("Nasdaq-100 (Wikipedia):", "✅" if wiki_test() else "❌")
    st.divider()
    st.subheader("Persistência de listas")
    st.code(
        "\n".join(
            [
                f"STORE_FILE: {STORE_FILE}",
                f"STORE_FILE existe: {'sim' if os.path.exists(STORE_FILE) else 'não'}",
                f"BACKUP_STORE_FILE: {BACKUP_STORE_FILE}",
                f"BACKUP existe: {'sim' if os.path.exists(BACKUP_STORE_FILE) else 'não'}",
            ]
        ),
        language="text",
    )

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
    if st.button("Rodar teste completo", key="diag_btn_test_all"):
        test = quick_source_test(t)
        if "erro" in test:
            st.error(test["erro"])
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.write("Yahoo:", "✅" if test["yahoo_ok"] else "❌")
            with c2:
                st.write("Finnhub:", "✅" if test["finnhub_ok"] else "❌")
            with c3:
                st.write("Twelve Data:", "✅" if test["twelve_ok"] else "❌")
            with c4:
                st.write("Nasdaq:", "✅" if test["nasdaq_ok"] else "❌")

            st.write(
                {
                    "ticker": test["ticker"],
                    "yahoo_last_close": test["yahoo_last_close"],
                    "finnhub_price": test["finnhub_price"],
                    "twelve_last_close": test["twelve_last_close"],
                    "ticker_in_nasdaq100": test["ticker_in_nasdaq100"],
                }
            )

            st.markdown("**Resposta agregada atual (tabela principal):**")
            st.write(fetch_one(test["ticker"]))

            st.markdown("**Resposta agregada atual (tabela principal):**")
            st.write(fetch_one(test["ticker"]))
