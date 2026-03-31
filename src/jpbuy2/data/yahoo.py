from __future__ import annotations

from pathlib import Path
from typing import Optional

import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf


# -----------------------------
# Timezone normalisation
# -----------------------------
def _to_naive_utc_ts(x: pd.Timestamp) -> pd.Timestamp:
    """
    Return tz-naive UTC timestamp for safe comparisons.
    """
    x = pd.Timestamp(x)
    if x.tzinfo is not None:
        x = x.tz_convert("UTC").tz_localize(None)
    return x


def _index_to_naive_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df.index is tz-naive UTC for consistent filtering.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    idx = pd.to_datetime(df.index, errors="coerce")

    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
    except Exception:
        idx = pd.to_datetime(df.index, errors="coerce")

    df.index = idx
    return df


# -----------------------------
# Helpers
# -----------------------------
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise to lower snake_case cols.
    Expect at least open/high/low/close.
    Ensure volume exists.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = _flatten_columns(df).copy()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' from data. Columns: {list(df.columns)}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def _local_path(data_dir: str | Path, interval: str, ticker: str) -> Path:
    data_dir = Path(data_dir)

    if interval == "1d":
        sub = "daily"
    elif interval == "1wk":
        sub = "weekly"
    else:
        # intraday
        sub = f"intraday_{interval}"

    return data_dir / "raw" / sub / f"{ticker}.csv"


def _snapshot_path(data_dir: str | Path, interval: str, ticker: str, kind: str) -> Path:
    """
    Save snapshots of what we downloaded / what we actually used.
    Example:
      data/raw_downloads/1d/MSFT.raw.csv
      data/raw_downloads/1d/MSFT.filtered.csv
      data/raw_downloads/60m/MSFT.raw.csv
      data/raw_downloads/60m/MSFT.filtered.csv
    """
    data_dir = Path(data_dir)
    return data_dir / "raw_downloads" / interval / f"{ticker}.{kind}.csv"


def _coerce_end(end: Optional[str]) -> pd.Timestamp:
    # End is treated as exclusive in our filters; if None => tomorrow (exclusive)
    if end is None or str(end).strip() == "":
        return _to_naive_utc_ts(pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1))
    return _to_naive_utc_ts(pd.to_datetime(end))


def _latest_expected_date(interval: str, end: Optional[str]) -> pd.Timestamp:
    end_ts = _coerce_end(end)
    if interval == "1d":
        return (end_ts - pd.offsets.BDay(1)).normalize()
    if interval == "1wk":
        return (end_ts - pd.Timedelta(days=7)).normalize()
    return (end_ts - pd.Timedelta(days=1)).normalize()


def _is_local_cache_fresh(df: pd.DataFrame, interval: str, end: Optional[str]) -> bool:
    if df is None or df.empty:
        return False

    latest_local = _to_naive_utc_ts(df.index.max()).normalize()
    expected_latest = _latest_expected_date(interval=interval, end=end)

    if interval == "1d":
        tolerance_days = 1
    elif interval == "1wk":
        tolerance_days = 7
    else:
        tolerance_days = 1

    return latest_local >= (expected_latest - pd.Timedelta(days=tolerance_days))


def load_local_ohlcv(
    ticker: str,
    start: str,
    end: Optional[str],
    interval: str,
    data_dir: str | Path,
) -> pd.DataFrame:
    """
    Load cached OHLCV from disk if it exists.
    Works for:
      - 1d
      - 1wk
      - intraday intervals like 60m
    """
    p = _local_path(data_dir, interval, ticker)
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = _index_to_naive_utc(df)

    start_ts = _to_naive_utc_ts(pd.to_datetime(start))
    end_ts = _coerce_end(end)

    df = df.sort_index()
    df = df.loc[(df.index >= start_ts) & (df.index < end_ts)]
    return df


# -----------------------------
# STOOQ fallback (kept inline — no extra module)
# -----------------------------
def _stooq_symbol(ticker: str) -> str:
    """
    Convert common ticker formats to Stooq symbols.

    Stooq examples:
      - AAPL.US
      - MSFT.US
      - DTE.DE
    """
    t = ticker.strip()

    if "." in t:
        return t

    return f"{t}.US"


def _fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    """
    Fetch DAILY OHLCV from Stooq as CSV.
    Stooq provides newest-first; we sort ascending.
    """
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
    df = pd.read_csv(url)

    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError(f"Stooq CSV missing 'Date' column for {symbol}. Columns={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    rename = {}
    if "open" in df.columns:
        rename["open"] = "open"
    if "high" in df.columns:
        rename["high"] = "high"
    if "low" in df.columns:
        rename["low"] = "low"
    if "close" in df.columns:
        rename["close"] = "close"
    if "volume" in df.columns:
        rename["volume"] = "volume"

    df = df.rename(columns=rename)
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = _normalise(df)
    df = _index_to_naive_utc(df)
    return df


def _daily_to_weekly(df_d: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate DAILY to WEEKLY OHLCV:
      open = first
      high = max
      low  = min
      close= last
      volume = sum
    """
    if df_d is None or df_d.empty:
        return pd.DataFrame()

    df_d = df_d.sort_index()

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df_w = df_d.resample("W-FRI").agg(agg)
    df_w = df_w.dropna(subset=["open", "high", "low", "close"])

    # Drop the current (incomplete) week bar if today is not Friday.
    # A weekly bar labelled W-FRI closes on Friday — any bar whose label
    # is in the future (i.e. the week has not yet ended) contains only
    # partial data and must NOT be used for Golden signals or backtests.
    # Using a Monday-only bar as a "weekly close" causes false Golden
    # ON/OFF flips mid-week, as seen with INTU on 2026-03-30.
    if not df_w.empty:
        import datetime
        today = datetime.date.today()
        last_bar_date = df_w.index[-1]
        if hasattr(last_bar_date, "date"):
            last_bar_date = last_bar_date.date()
        # The bar is incomplete if its label (the coming Friday) is still in the future
        if last_bar_date > today:
            df_w = df_w.iloc[:-1]

    df_w = _normalise(df_w)
    df_w = _index_to_naive_utc(df_w)
    return df_w


def _fetch_yahoo_chart_http(
    ticker: str,
    start: str,
    end: Optional[str],
    interval: str,
) -> pd.DataFrame:
    """
    Direct Yahoo chart endpoint fallback.
    This is more robust for stale-cache repair than scraping HTML tables.
    """
    if interval not in ("1d", "1wk"):
        raise ValueError(f"HTTP chart fallback supports only 1d/1wk, got {interval}")

    start_ts = _to_naive_utc_ts(pd.to_datetime(start))
    end_ts = _coerce_end(end)

    period1 = int(start_ts.timestamp())
    period2 = int(end_ts.timestamp())

    params = urlencode(
        {
            "period1": period1,
            "period2": period2,
            "interval": interval,
            "includePrePost": "false",
            "events": "div,splits",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?{params}"
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
        },
    )

    try:
        with urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as e:
        raise ValueError(f"Yahoo HTTP chart request failed for {ticker}: {e}") from e

    chart = (payload.get("chart") or {})
    result_list = chart.get("result") or []
    if not result_list:
        err = chart.get("error")
        raise ValueError(f"Yahoo HTTP chart returned no result for {ticker}. error={err}")

    result = result_list[0]
    timestamps = result.get("timestamp") or []
    indicators = (result.get("indicators") or {}).get("quote") or []
    if not timestamps or not indicators:
        raise ValueError(f"Yahoo HTTP chart returned incomplete payload for {ticker}")

    quote = indicators[0]
    df = pd.DataFrame(
        {
            "open": quote.get("open", []),
            "high": quote.get("high", []),
            "low": quote.get("low", []),
            "close": quote.get("close", []),
            "volume": quote.get("volume", []),
        },
        index=pd.to_datetime(timestamps, unit="s", utc=True).tz_localize(None),
    )

    df = df.sort_index()
    df = _normalise(df)
    df = df.loc[(df.index >= start_ts) & (df.index < end_ts)]

    if df.empty:
        raise ValueError(f"Yahoo HTTP chart data empty after filter for {ticker} interval={interval}")

    return df


# -----------------------------
# Main daily/weekly fetch
# -----------------------------
def fetch_ohlcv(
    ticker: str,
    start: str,
    end: Optional[str],
    interval: str,
    data_dir: str | Path | None = None,
    prefer_local: bool = True,
    save_download_snapshots: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV via yfinance (primary) with direct Yahoo HTTP and Stooq fallbacks.

    Supports only:
      - 1d
      - 1wk

    Existing repo behaviour is intentionally preserved, except that stale local
    cache is now ignored and refreshed instead of being trusted forever.

    - If data_dir is set, writes the *used* data to:
        data/raw/daily/<TICKER>.csv
        data/raw/weekly/<TICKER>.csv

    - If save_download_snapshots=True, also saves:
        data/raw_downloads/<interval>/<TICKER>.raw.csv
        data/raw_downloads/<interval>/<TICKER>.filtered.csv
    """
    if interval not in ("1d", "1wk"):
        raise ValueError(f"Invalid interval '{interval}'. Use '1d' or '1wk'.")

    # 0) Prefer local cache only if it is fresh enough for the requested end-date.
    if prefer_local and data_dir is not None:
        local = load_local_ohlcv(
            ticker=ticker,
            start=start,
            end=end,
            interval=interval,
            data_dir=data_dir,
        )
        if local is not None and not local.empty and _is_local_cache_fresh(local, interval, end):
            return local

    start_ts = _to_naive_utc_ts(pd.to_datetime(start))
    end_ts = _coerce_end(end)

    # 1) Yahoo / yfinance with retries
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            df = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                raise ValueError(f"No data returned for {ticker} interval={interval} (attempt {attempt})")

            df = _normalise(df)
            df = _index_to_naive_utc(df)
            df = df.sort_index()

            # Save RAW snapshot (pre-filter)
            if data_dir is not None and save_download_snapshots:
                p_raw = _snapshot_path(data_dir, interval, ticker, "raw")
                p_raw.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(p_raw)

            # Apply internal filter (exclusive end)
            df = df.loc[(df.index >= start_ts) & (df.index < end_ts)]
            if df.empty:
                raise ValueError(f"Yahoo data empty after date filter for {ticker} interval={interval}")

            # Save FILTERED snapshot (actually used)
            if data_dir is not None and save_download_snapshots:
                p_f = _snapshot_path(data_dir, interval, ticker, "filtered")
                p_f.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(p_f)

            # Cache used data
            if data_dir is not None:
                p = _local_path(data_dir, interval, ticker)
                p.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(p)

            return df

        except Exception as e:
            last_err = e
            time.sleep(0.8 * attempt)

    # 2) Direct Yahoo HTTP chart fallback
    try:
        df = _fetch_yahoo_chart_http(ticker=ticker, start=start, end=end, interval=interval)

        if data_dir is not None and save_download_snapshots:
            p_f = _snapshot_path(data_dir, interval, ticker, "filtered_http_chart")
            p_f.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p_f)

        if data_dir is not None:
            p = _local_path(data_dir, interval, ticker)
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p)

        return df

    except Exception as http_err:
        http_last_err = http_err

    # 3) Fallback: Stooq daily -> weekly if needed
    try:
        symbol = _stooq_symbol(ticker)
        df_d = _fetch_stooq_daily(symbol)

        if data_dir is not None and save_download_snapshots:
            p_raw = _snapshot_path(data_dir, "1d", ticker, "raw_stooq_daily")
            p_raw.parent.mkdir(parents=True, exist_ok=True)
            df_d.to_csv(p_raw)

        df = df_d if interval == "1d" else _daily_to_weekly(df_d)
        df = df.sort_index()
        df = df.loc[(df.index >= start_ts) & (df.index < end_ts)]

        if df is None or df.empty:
            raise ValueError(f"Stooq data empty after date filter for {ticker} ({symbol}) interval={interval}")

        if data_dir is not None and save_download_snapshots:
            p_f = _snapshot_path(data_dir, interval, ticker, "filtered_stooq")
            p_f.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p_f)

        if data_dir is not None:
            p = _local_path(data_dir, interval, ticker)
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p)

        return df

    except Exception as e:
        raise ValueError(
            f"Failed to fetch OHLCV for {ticker} interval={interval}. "
            f"Yahoo error was: {repr(last_err)}. "
            f"Yahoo HTTP fallback error was: {repr(http_last_err)}. "
            f"Stooq fallback error was: {repr(e)}."
        )


# -----------------------------
# Intraday fetch (NEW, separate path)
# -----------------------------
def fetch_intraday_ohlcv(
    ticker: str,
    start: str,
    end: Optional[str],
    interval: str = "60m",
    data_dir: str | Path | None = None,
    prefer_local: bool = True,
    save_download_snapshots: bool = True,
) -> pd.DataFrame:
    """
    Fetch intraday OHLCV via yfinance for emergency-exit analysis.

    Recommended first use:
      interval = "60m"

    This is intentionally separate from fetch_ohlcv() so the existing
    daily/weekly pipeline remains untouched.

    Saved to:
      data/raw/intraday_<interval>/<TICKER>.csv

    Notes:
      - No Stooq fallback for intraday
      - Yahoo intraday availability is limited and unofficial
    """
    valid_intraday = {"60m", "1h", "30m", "15m", "5m", "2m", "1m"}
    if interval not in valid_intraday:
        raise ValueError(
            f"Invalid intraday interval '{interval}'. "
            f"Use one of: {sorted(valid_intraday)}"
        )

    # 0) Prefer local cache
    if prefer_local and data_dir is not None:
        local = load_local_ohlcv(
            ticker=ticker,
            start=start,
            end=end,
            interval=interval,
            data_dir=data_dir,
        )
        if local is not None and not local.empty:
            return local

    start_ts = _to_naive_utc_ts(pd.to_datetime(start))
    end_ts = _coerce_end(end)

    # Yahoo / yfinance with retries
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            df = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                raise ValueError(f"No intraday data returned for {ticker} interval={interval} (attempt {attempt})")

            df = _normalise(df)
            df = _index_to_naive_utc(df)
            df = df.sort_index()

            # Save RAW snapshot (pre-filter)
            if data_dir is not None and save_download_snapshots:
                p_raw = _snapshot_path(data_dir, interval, ticker, "raw")
                p_raw.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(p_raw)

            # Apply internal filter (exclusive end)
            df = df.loc[(df.index >= start_ts) & (df.index < end_ts)]
            if df.empty:
                raise ValueError(f"Intraday data empty after date filter for {ticker} interval={interval}")

            # Save FILTERED snapshot
            if data_dir is not None and save_download_snapshots:
                p_f = _snapshot_path(data_dir, interval, ticker, "filtered")
                p_f.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(p_f)

            # Cache used data
            if data_dir is not None:
                p = _local_path(data_dir, interval, ticker)
                p.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(p)

            return df

        except Exception as e:
            last_err = e
            time.sleep(0.8 * attempt)

    raise ValueError(
        f"Failed to fetch intraday OHLCV for {ticker} interval={interval}. "
        f"Yahoo error was: {repr(last_err)}"
    )
