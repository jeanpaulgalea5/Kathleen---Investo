from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd

from .config import settings_for, settings_for_ticker
from .strategy_select import settings_adaptive
from .data.yahoo import fetch_ohlcv, fetch_intraday_ohlcv
from .backtest import run_backtest
from .types import AssetType
from .signals.golden import compute_golden_weekly_flags


def load_watchlist_csv(path: str) -> pd.DataFrame:
    wl = pd.read_csv(path)
    if "ticker" not in wl.columns:
        raise ValueError("watchlist.csv must contain a 'ticker' column.")
    if "type" not in wl.columns:
        wl["type"] = "stock"

    wl["ticker"] = wl["ticker"].astype(str).str.strip()
    wl["type"] = wl["type"].astype(str).str.strip().str.lower()
    wl = wl[(wl["ticker"] != "") & (wl["type"].isin(["stock", "etf"]))]
    return wl.reset_index(drop=True)


def _safe_dt(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, errors="coerce")


def _build_golden_windows(flags: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Build contiguous Golden ON windows from weekly flags.

    Returns list of dicts:
      {cycle:int, on:Timestamp, off:Timestamp|NaT, weeks_on:int}
    """
    if flags is None or flags.empty or "golden_on" not in flags.columns:
        return []

    on_series = flags["golden_on"].astype(bool).fillna(False)

    windows = []
    in_regime = False
    start = None
    cycle = 0

    for idx, is_on in on_series.items():
        if (not in_regime) and is_on:
            in_regime = True
            start = idx
            cycle += 1

        if in_regime and (not is_on):
            off = idx  # first OFF week date
            weeks_on = max(0, int(flags.loc[start:off].shape[0] - 1)) if start is not None else 0
            windows.append({"cycle": cycle, "on": start, "off": off, "weeks_on": weeks_on})
            in_regime = False
            start = None

    if in_regime and start is not None:
        weeks_on = int(flags.loc[start:].shape[0])
        windows.append({"cycle": cycle, "on": start, "off": pd.NaT, "weeks_on": weeks_on})

    return windows


def _trades_to_df(ticker: str, trades: list) -> pd.DataFrame:
    """
    Convert engine Trade objects to a normalised DataFrame.
    """
    rows = []
    for t in trades:
        d = asdict(t)
        rows.append(
            {
                "Ticker": ticker,
                "Entry Date": d.get("entry_date") or "",
                "Entry Price": d.get("entry_px"),
                "Exit Date": d.get("exit_date") or "",
                "Exit Price": d.get("exit_px"),
                "Exit Reason": d.get("reason_exit") or d.get("exit_reason") or "",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Entry DT"] = df["Entry Date"].apply(_safe_dt)
    df["Exit DT"] = df["Exit Date"].apply(_safe_dt)

    # Return %
    def _ret(row):
        try:
            ep = float(row["Entry Price"])
            xp = float(row["Exit Price"])
            if ep > 0:
                return (xp / ep - 1.0) * 100.0
        except Exception:
            pass
        return ""

    df["Return %"] = df.apply(_ret, axis=1)
    return df


def build_windows_report_for_ticker(
    ticker: str,
    golden_flags: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per Golden window. If a Silver trade entry falls inside the window,
    populate Silver Entry / Exit columns. Otherwise blanks.
    """
    windows = _build_golden_windows(golden_flags)

    # If no golden windows, still return empty frame with correct columns
    out_rows = []

    for w in windows:
        on_dt = w["on"]
        off_dt = w["off"]  # NaT means still ON at end of available weekly data

        # Match: first trade whose entry date is within [on_dt, off_dt)
        matched = None
        if trades_df is not None and not trades_df.empty:
            # Only consider valid entry datetimes
            cand = trades_df.dropna(subset=["Entry DT"]).copy()

            # Window bounds
            start = pd.Timestamp(on_dt).normalize()
            if pd.isna(off_dt):
                # Open-ended window; match any entry >= start
                cand = cand[cand["Entry DT"] >= start]
            else:
                end = pd.Timestamp(off_dt).normalize()
                cand = cand[(cand["Entry DT"] >= start) & (cand["Entry DT"] < end)]

            if not cand.empty:
                # Take earliest entry in the window
                cand = cand.sort_values("Entry DT")
                matched = cand.iloc[0]

        # Output row
        row = {
            "Ticker": ticker,
            "Golden Cycle": w["cycle"],
            "Golden ON": str(pd.Timestamp(on_dt).date()) if pd.notna(on_dt) else "",
            "Golden OFF": str(pd.Timestamp(off_dt).date()) if pd.notna(off_dt) else "",
            "Weeks ON": w["weeks_on"],
            "Silver Entry Date": "",
            "Silver Entry Price": "",
            "Exit Date": "",
            "Exit Price": "",
            "Return %": "",
            "Exit Reason": "",
        }

        if matched is not None:
            row["Silver Entry Date"] = matched.get("Entry Date", "")
            row["Silver Entry Price"] = matched.get("Entry Price", "")
            row["Exit Date"] = matched.get("Exit Date", "")
            row["Exit Price"] = matched.get("Exit Price", "")
            row["Return %"] = matched.get("Return %", "")
            row["Exit Reason"] = matched.get("Exit Reason", "")

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def write_combined_trade_report(
    watchlist_csv: str,
    out_dir: str,
    start: str,
    end: str,
    data_dir: Optional[str],
) -> dict:
    """
    Produces ONE consolidated JP-BUY-style report that is:
      - one row per Golden window per ticker
      - includes Silver entry/price and exit details if a trade occurred, else blanks

    Outputs:
      - backtest_windows.csv  (main report)
      - backtest_summary.csv  (optional)
    """
    wl = load_watchlist_csv(watchlist_csv)

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    all_windows = []
    summary_rows = []

    for _, r in wl.iterrows():
        ticker = r["ticker"]
        asset_type: AssetType = r["type"]  # type: ignore

        s = settings_for_ticker(ticker, asset_type)

        # Weekly + Golden flags
        df_w = fetch_ohlcv(ticker, start=start, end=end, interval="1wk", data_dir=data_dir)

        # Daily + optional intraday + backtest
        df_d = fetch_ohlcv(ticker, start=start, end=end, interval="1d", data_dir=data_dir)

        # Adaptive strategy: auto-selects best params from ticker's own history.
        # Uses cache (data/strategy_cache/<ticker>.json) — recomputes only when
        # new data has arrived. Falls back to settings_for_ticker if data too short.
        if asset_type == "stock" and len(df_d) >= 260:
            try:
                s = settings_adaptive(
                    ticker=ticker, df_d=df_d, df_w=df_w,
                    data_dir=data_dir, verbose=True,
                )
            except Exception:
                pass  # keep settings_for_ticker as fallback

        golden_flags = compute_golden_weekly_flags(df_w, s)

        df_i = None
        if bool(getattr(s, "use_intraday_emergency_exit", False)):
            try:
                df_i = fetch_intraday_ohlcv(
                    ticker=ticker,
                    start=start,
                    end=end,
                    interval=getattr(s, "intraday_emergency_interval", "60m"),
                    data_dir=data_dir,
                    prefer_local=True,
                )
            except Exception as e:
                print(f"WARNING: intraday data unavailable for {ticker}: {e}")
                df_i = None

        backtest_error = ""

        try:
            out = run_backtest(df_d, df_w, asset_type, s, ticker=ticker, df_i=df_i)
            trades = out.get("trades", [])
        except ValueError as exc:
            if "Not enough history to backtest." in str(exc):
                out = {"trades": []}
                trades = []
                backtest_error = "Insufficient history"
            else:
                raise
    
        trades_df = _trades_to_df(ticker, trades)
        windows_df = build_windows_report_for_ticker(ticker, golden_flags, trades_df)

        if not windows_df.empty:
            all_windows.append(windows_df)

        # Summary: based on realised trades only
        trades_n = 0
        win_rate = ""
        avg_ret = ""
        compounded = ""

        if trades_df is not None and not trades_df.empty:
            rets = pd.to_numeric(trades_df["Return %"], errors="coerce").dropna()
            trades_n = int(len(rets))
            if trades_n > 0:
                win_rate = float((rets > 0).mean() * 100.0)
                avg_ret = float(rets.mean())
                compounded = float((1.0 + rets / 100.0).prod() - 1.0)

        summary_rows.append(
            {
                "Ticker": ticker,
                "Type": asset_type,
                "Trades": trades_n,
                "Win Rate %": win_rate,
                "Avg Return %": avg_ret,
                "Compounded Return": compounded,
                "Backtest Status": backtest_error or "OK",
            }
        )

    df_windows = (
        pd.concat(all_windows, ignore_index=True)
        if all_windows
        else pd.DataFrame(
            columns=[
                "Ticker",
                "Golden Cycle",
                "Golden ON",
                "Golden OFF",
                "Weeks ON",
                "Silver Entry Date",
                "Silver Entry Price",
                "Exit Date",
                "Exit Price",
                "Return %",
                "Exit Reason",
            ]
        )
    )

    df_summary = pd.DataFrame(summary_rows)

    windows_path = outp / "backtest_windows.csv"
    summary_path = outp / "backtest_summary.csv"

    df_windows.to_csv(windows_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    return {
        "windows_path": str(windows_path),
        "summary_path": str(summary_path),
        "tickers": int(len(wl)),
        "total_windows": int(len(df_windows)),
    }
