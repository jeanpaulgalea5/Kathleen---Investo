from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import ETFEntrySettings
from .signals import compute_etf_entry_features, latest_etf_signal


def _load_price_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    date_col = None
    for candidate in ("date", "datetime", "timestamp"):
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        unnamed_cols = [c for c in df.columns if c.startswith("unnamed:")]
        if unnamed_cols:
            date_col = unnamed_cols[0]

    if date_col is None and len(df.columns) > 0:
        first_col = df.columns[0]
        parsed = pd.to_datetime(df[first_col], errors="coerce")
        if parsed.notna().mean() > 0.8:
            date_col = first_col

    if date_col is None:
        raise ValueError(f"Could not identify date column in CSV: {path}. Columns={list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()].copy().set_index(date_col).sort_index()

    for col in ("adj_close", "close", "high", "low", "open", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def select_window_entries(features: pd.DataFrame, cooldown_days: int, max_windows_per_year: int) -> pd.DataFrame:
    candidates = features[features["buy_gate"]].copy()
    if candidates.empty:
        return candidates

    dates = list(candidates.index)
    windows: list[list[pd.Timestamp]] = []
    current_window = [dates[0]]
    for dt in dates[1:]:
        if (dt - current_window[-1]).days <= cooldown_days:
            current_window.append(dt)
        else:
            windows.append(current_window)
            current_window = [dt]
    windows.append(current_window)

    selected_rows = []
    for window in windows:
        block = candidates.loc[window].copy()
        block["window_score"] = (
            block["strong_buy"].astype(int) * 3.0
            + block["bullish_day"].astype(int) * 1.5
            + block["macd_slope_cross_up"].astype(int) * 1.5
            + block["macd_improving"].astype(int) * 1.0
            + (block["drawdown_20d"].fillna(0.0) * 100.0) * 0.8
            + (-block["rsi14"].fillna(999.0)) * 0.1
        )
        best_idx = block["window_score"].idxmax()
        best_row = block.loc[[best_idx]].copy()
        best_row["window_open"] = window[0]
        best_row["window_close"] = window[-1]
        best_row["window_days"] = len(window)
        selected_rows.append(best_row)

    out = pd.concat(selected_rows).sort_index()
    kept = []
    for year, grp in out.groupby(out.index.year):
        grp = grp.sort_values(["window_score", "rsi14"], ascending=[False, True])
        kept.append(grp.head(max_windows_per_year))
    out = pd.concat(kept).sort_index() if kept else out.iloc[:0].copy()
    out["selected"] = True
    return out


def summarise_entries(entries: pd.DataFrame, ticker: str) -> dict:
    if entries.empty:
        return {"ticker": ticker, "signals": 0, "avg_1y_return": None, "median_1y_return": None, "hit_rate": None, "avg_days_between": None}
    valid = entries["fwd_1y_return"].dropna()
    avg_ret = float(valid.mean()) if not valid.empty else None
    med_ret = float(valid.median()) if not valid.empty else None
    hit_rate = float((valid > 0).mean()) if not valid.empty else None
    avg_days_between = None
    if len(entries.index) > 1:
        date_index = pd.to_datetime(entries.index, errors="coerce")
        date_index = pd.Series(date_index).dropna().sort_values()
        if len(date_index) > 1:
            gaps = date_index.diff().dt.days.dropna()
            avg_days_between = float(gaps.mean()) if not gaps.empty else None
    return {"ticker": ticker, "signals": int(len(entries)), "avg_1y_return": avg_ret, "median_1y_return": med_ret, "hit_rate": hit_rate, "avg_days_between": avg_days_between}


def optimise_ticker(ticker: str, daily_csv: str | Path, settings: ETFEntrySettings | None = None) -> dict:
    s = settings or ETFEntrySettings()
    profile = dict(s.default_profile)
    profile.update(s.per_ticker_profiles.get(ticker, {}))

    df_d = _load_price_csv(daily_csv)
    feats = compute_etf_entry_features(df_d=df_d, ticker=ticker, s=s, profile=profile)

    if feats.empty or len(feats) < s.min_history_days:
        return {
            "ticker": ticker,
            "profile": profile,
            "settings": asdict(s),
            "summary": summarise_entries(pd.DataFrame(), ticker),
            "latest": {"ticker": ticker, "signal": "WAIT", "reason": "Not enough data."},
            "windows": pd.DataFrame(),
            "entries": pd.DataFrame(),
        }

    cooldown_days = int(profile.get("cooldown_days", s.cooldown_days))
    max_windows_per_year = int(profile.get("max_windows_per_year", s.max_windows_per_year))

    windows_df = feats[feats["buy_gate"]].copy()
    entries_df = select_window_entries(feats, cooldown_days, max_windows_per_year).copy()

    return {
        "ticker": ticker,
        "profile": profile,
        "settings": asdict(s),
        "summary": summarise_entries(entries_df if not entries_df.empty else pd.DataFrame(), ticker),
        "latest": latest_etf_signal(df_d=df_d, ticker=ticker, s=s, profile=profile),
        "windows": windows_df.reset_index(),
        "entries": entries_df.reset_index(),
    }


def optimise_many(tickers: Iterable[str], data_dir: str | Path = "data/raw", settings: ETFEntrySettings | None = None) -> dict[str, pd.DataFrame]:
    s = settings or ETFEntrySettings()
    data_dir = Path(data_dir)

    summaries = []
    latest_rows = []
    profile_rows = []
    window_frames = []
    entry_frames = []

    for ticker in tickers:
        daily_csv = data_dir / "daily" / f"{ticker}.csv"
        if not daily_csv.exists():
            summaries.append(summarise_entries(pd.DataFrame(), ticker))
            latest_rows.append({"ticker": ticker, "signal": "WAIT", "reason": "Missing daily CSV."})
            continue

        result = optimise_ticker(ticker=ticker, daily_csv=daily_csv, settings=s)
        summaries.append(result["summary"])
        latest_rows.append(result["latest"])
        profile_rows.append({"ticker": ticker, **result["profile"]})

        windows = result["windows"]
        if not windows.empty:
            windows.insert(0, "ticker", ticker)
            window_frames.append(windows)

        entries = result["entries"]
        if not entries.empty:
            entries.insert(0, "ticker", ticker)
            entry_frames.append(entries)

    return {
        "summary": pd.DataFrame(summaries),
        "latest": pd.DataFrame(latest_rows),
        "profiles": pd.DataFrame(profile_rows),
        "windows": pd.concat(window_frames, ignore_index=True) if window_frames else pd.DataFrame(),
        "entries": pd.concat(entry_frames, ignore_index=True) if entry_frames else pd.DataFrame(),
    }
