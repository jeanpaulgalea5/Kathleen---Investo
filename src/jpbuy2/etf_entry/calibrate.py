from __future__ import annotations

from pathlib import Path
from itertools import combinations
import json
import math

import pandas as pd

from .config import ETFEntrySettings
from .optimize import _load_price_csv, select_window_entries
from .signals import compute_etf_entry_features


def _candidate_month_sets() -> list[tuple[int, ...]]:
    combos: list[tuple[int, ...]] = []
    for k in (3, 4):
        combos.extend(list(combinations(range(1, 13), k)))
    return combos


def _score_result(entries: pd.DataFrame, years: float) -> float:
    valid = entries["fwd_1y_return"].dropna()
    if valid.empty:
        return -999.0

    windows_per_year = len(entries) / years if years > 0 else 0.0
    if windows_per_year < 0.8 or windows_per_year > 2.2:
        return -999.0

    penalty = 0.0
    penalty += abs(windows_per_year - 1.2) * 0.01

    return float(valid.mean()) + 0.03 * float((valid > 0).mean()) - penalty


def calibrate_ticker(
    ticker: str,
    daily_csv: str | Path,
    weekly_csv: str | Path,
    settings: ETFEntrySettings | None = None,
) -> dict:
    s = settings or ETFEntrySettings()
    df_d = _load_price_csv(daily_csv)
    df_w = _load_price_csv(weekly_csv)

    years = (df_d.index.max() - df_d.index.min()).days / 365.25
    best: dict | None = None

    for months in _candidate_month_sets():
        for rsi_max in (40.0, 42.0, 45.0):
            for drawdown_min in (0.03, 0.04, 0.05):
                for band in (0.08, 0.10, 0.12):
                    for weekly_mode in ("pos", "pos_or_improve"):
                        profile = {
                            "allowed_months": list(months),
                            "rsi_max": rsi_max,
                            "drawdown_min": drawdown_min,
                            "price_to_sma200_band": band,
                            "weekly_mode": weekly_mode,
                        }
                        feats = compute_etf_entry_features(
                            df_d=df_d,
                            df_w=df_w,
                            ticker=ticker,
                            s=s,
                            profile=profile,
                        )
                        entries = select_window_entries(feats, s)
                        if entries.empty:
                            continue

                        score = _score_result(entries, years)
                        if best is None or score > best["score"]:
                            valid = entries["fwd_1y_return"].dropna()
                            best = {
                                "ticker": ticker,
                                "score": score,
                                "allowed_months": list(months),
                                "rsi_max": rsi_max,
                                "drawdown_min": drawdown_min,
                                "price_to_sma200_band": band,
                                "weekly_mode": weekly_mode,
                                "signals": int(len(entries)),
                                "windows_per_year": float(len(entries) / years) if years > 0 else None,
                                "avg_1y_return": float(valid.mean()) if not valid.empty else None,
                                "hit_rate": float((valid > 0).mean()) if not valid.empty else None,
                            }

    if best is None:
        best = {
            "ticker": ticker,
            **s.default_profile,
            "signals": 0,
            "windows_per_year": None,
            "avg_1y_return": None,
            "hit_rate": None,
            "score": None,
        }

    return best


def calibrate_many(
    tickers: list[str],
    data_dir: str | Path = "data/raw",
    out_json: str | Path | None = None,
    settings: ETFEntrySettings | None = None,
) -> dict[str, dict]:
    s = settings or ETFEntrySettings()
    data_dir = Path(data_dir)

    out: dict[str, dict] = {}
    for ticker in tickers:
        daily_csv = data_dir / "daily" / f"{ticker}.csv"
        weekly_csv = data_dir / "weekly" / f"{ticker}.csv"
        if daily_csv.exists() and weekly_csv.exists():
            out[ticker] = calibrate_ticker(
                ticker=ticker,
                daily_csv=daily_csv,
                weekly_csv=weekly_csv,
                settings=s,
            )

    if out_json is not None:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    return out
