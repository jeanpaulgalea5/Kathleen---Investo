from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from jpbuy2.commodity_entry import CommodityEntrySettings, optimise_many
from jpbuy2.data.watchlist_loader import get_commodity_rows


SUMMARY_COLS = [
    "ticker",
    "commodity_type",
    "signals",
    "entry_cost_pct",
    "avg_2y_return",
    "median_2y_return",
    "avg_3y_return",
    "median_3y_return",
    "hit_rate_2y_net",
    "hit_rate_3y_net",
    "avg_days_between",
]

LATEST_COLS = [
    "ticker",
    "commodity_type",
    "date",
    "signal",
    "close",
    "rsi14",
    "drawdown",
    "drawdown_entry_th",
    "drawdown_overshoot_x",
    "drawdown_overshoot_label",
    "macd_hist",
    "structural_bull",
    "primed",
    "reason",
]

PROFILE_COLS = [
    "ticker",
    "commodity_type",
    "entry_cost_pct",
    "adaptive_lookback_days",
    "drawdown_lookback_days",
    "drawdown_entry_quantile",
    "rsi_support_quantile",
    "require_bullish_day",
    "cooldown_days",
    "max_windows_per_year",
    "monthly_ma_period",
    "weekly_ma_period",
    "weekly_ma_slope_weeks",
    "deep_drawdown_min",
    "daily_rsi_cap",
    "strong_buy_extra_drawdown",
]

WINDOW_COLS = [
    "ticker",
    "commodity_type",
    "index",
    "close",
    "rsi14",
    "macd_hist",
    "macd_improving",
    "macd_slope",
    "macd_slope_cross_up",
    "bullish_day",
    "drawdown_lookback",
    "drawdown_entry_th",
    "rsi_support_th",
    "rsi_support_ok",
    "structural_bull",
    "deep_dip",
    "turn_confirm",
    "buy_gate",
    "primed",
    "strong_buy",
    "fwd_1y_return",
    "fwd_2y_return",
    "fwd_3y_return",
]

ENTRY_COLS = [
    "ticker",
    "commodity_type",
    "index",
    "close",
    "rsi14",
    "macd_hist",
    "macd_improving",
    "macd_slope",
    "macd_slope_cross_up",
    "bullish_day",
    "drawdown_lookback",
    "drawdown_entry_th",
    "rsi_support_th",
    "rsi_support_ok",
    "structural_bull",
    "deep_dip",
    "turn_confirm",
    "buy_gate",
    "primed",
    "strong_buy",
    "fwd_1y_return",
    "fwd_2y_return",
    "fwd_3y_return",
    "window_score",
    "window_open",
    "window_close",
    "window_days",
    "selected",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run commodity entry optimiser.")
    p.add_argument(
        "--watchlist",
        default="watchlist.csv",
        help="Watchlist file containing commodity rows",
    )
    p.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing daily/ subfolder",
    )
    p.add_argument(
        "--out-dir",
        default="reports/latest/commodity_entry",
        help="Where to write commodity optimiser outputs",
    )
    return p


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[cols]


def main() -> int:
    args = build_parser().parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = CommodityEntrySettings()
    rows = get_commodity_rows(args.watchlist)

    print(f"Loaded {len(rows)} commodity rows from watchlist.")
    if not rows:
        print("No commodity rows found in watchlist.")

        pd.DataFrame(columns=SUMMARY_COLS).to_csv(out_dir / "commodity_entry_summary.csv", index=False)
        pd.DataFrame(columns=LATEST_COLS).to_csv(out_dir / "commodity_entry_latest.csv", index=False)
        pd.DataFrame(columns=PROFILE_COLS).to_csv(out_dir / "commodity_entry_profiles.csv", index=False)
        pd.DataFrame(columns=WINDOW_COLS).to_csv(out_dir / "commodity_entry_windows.csv", index=False)
        pd.DataFrame(columns=ENTRY_COLS).to_csv(out_dir / "commodity_entry_history.csv", index=False)
        return 0

    result = optimise_many(
        rows=rows,
        data_dir=args.data_dir,
        settings=settings,
    )

    summary_df = _ensure_columns(result.get("summary", pd.DataFrame()), SUMMARY_COLS)
    latest_df = _ensure_columns(result.get("latest", pd.DataFrame()), LATEST_COLS)
    profiles_df = _ensure_columns(result.get("profiles", pd.DataFrame()), PROFILE_COLS)
    windows_df = _ensure_columns(result.get("windows", pd.DataFrame()), WINDOW_COLS)
    entries_df = _ensure_columns(result.get("entries", pd.DataFrame()), ENTRY_COLS)

    summary_df.to_csv(out_dir / "commodity_entry_summary.csv", index=False)
    latest_df.to_csv(out_dir / "commodity_entry_latest.csv", index=False)
    profiles_df.to_csv(out_dir / "commodity_entry_profiles.csv", index=False)
    windows_df.to_csv(out_dir / "commodity_entry_windows.csv", index=False)
    entries_df.to_csv(out_dir / "commodity_entry_history.csv", index=False)

    print(f"Wrote outputs to: {out_dir}")
    print(f"summary rows: {len(summary_df)}")
    print(f"latest rows: {len(latest_df)}")
    print(f"profiles rows: {len(profiles_df)}")
    print(f"windows rows: {len(windows_df)}")
    print(f"entries rows: {len(entries_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
