from __future__ import annotations

import os
import sys
import pandas as pd

from jpbuy2.config import Settings
from jpbuy2.signals.golden import compute_golden_weekly_flags
from jpbuy2.data.yahoo import fetch_ohlcv


def analyse(ticker: str, start: str = "2014-01-01", end: str = None) -> None:
    s = Settings()

    df_w = fetch_ohlcv(
        ticker,
        start=start,
        end=end,
        interval="1wk",
        data_dir="data",
        prefer_local=True,
    )

    flags = compute_golden_weekly_flags(df_w, s).copy()
    if flags.empty:
        print("No weekly data/flags.")
        return

    close = df_w.loc[flags.index, "close"].astype(float)
    fwd_4w = (close.shift(-4) / close - 1.0) * 100.0
    fwd_8w = (close.shift(-8) / close - 1.0) * 100.0

    flags["debug_fwd_4w_pct"] = fwd_4w
    flags["debug_fwd_8w_pct"] = fwd_8w

    # Stats
    n_entry_base = int(flags["debug_golden_entry_base"].sum())
    n_entry_actual = int(flags["debug_golden_entry"].sum())
    n_entry_blocked = int(flags["debug_entry_blocked_by_gate"].sum())

    n_exit_base = int(flags["debug_golden_exit_base"].sum())
    n_exit_blocked = int(flags["debug_exit_blocked_by_adx_di"].sum())
    n_override = int(flags["debug_override_confirmed"].sum())

    print(f"\n=== {ticker} ===")
    print(f"Entry base signals:   {n_entry_base}")
    print(f"Entry executed:       {n_entry_actual}")
    print(f"Entry blocked (gate): {n_entry_blocked}")

    print(f"\nExit base signals:        {n_exit_base}")
    print(f"Exit blocked (ADX+DI up): {n_exit_blocked}")
    print(f"Exit override used:       {n_override}")

    blocked = flags[flags["debug_exit_blocked_by_adx_di"]].copy()
    if not blocked.empty:
        print("\nBlocked exit forward returns (means):")
        print(f"  mean fwd 4w  %: {blocked['debug_fwd_4w_pct'].mean():.2f}")
        print(f"  mean fwd 8w  %: {blocked['debug_fwd_8w_pct'].mean():.2f}")
    else:
        print("\nNo blocked exits for this ticker under current rules.")

    # Persist outputs for audit
    os.makedirs("data/runs", exist_ok=True)
    flags_path = f"data/runs/{ticker}_golden_flags.csv"
    flags.to_csv(flags_path)

    print(f"\nSaved flags to: {flags_path}")


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else "BKNG"
    analyse(t)
