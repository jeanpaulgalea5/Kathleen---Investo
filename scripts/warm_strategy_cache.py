#!/usr/bin/env python3
"""
warm_strategy_cache.py
----------------------
Pre-warms the adaptive strategy cache for all stock tickers in the watchlist.

Run this:
  - Once after first installing the adaptive strategy module
  - After a major market regime change (use --force)
  - After adding several new tickers

Usage:
  python scripts/warm_strategy_cache.py
  python scripts/warm_strategy_cache.py --watchlist watchlist.csv --force
  python scripts/warm_strategy_cache.py --ticker MSFT --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# --- ensure src/ is on the path when run directly ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from jpbuy2.strategy_select import select_strategy, _cache_is_valid


def load_daily(ticker: str, data_dir: str) -> pd.DataFrame | None:
    safe = ticker.replace("/", "_").replace("\\", "_")
    path = Path(data_dir) / "raw" / "daily" / f"{safe}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df.sort_index().dropna(subset=["close", "high", "low"])


def to_weekly(df_d: pd.DataFrame) -> pd.DataFrame:
    df_d.index = pd.to_datetime(df_d.index)
    return df_d.resample("W-FRI").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()


def warm_ticker(ticker: str, data_dir: str, force: bool, min_trades: int) -> dict:
    if not force and _cache_is_valid(ticker, data_dir):
        return {"ticker": ticker, "status": "cache_hit", "best": None}

    df_d = load_daily(ticker, data_dir)
    if df_d is None:
        return {"ticker": ticker, "status": "no_data", "best": None}
    if len(df_d) < 260:
        return {"ticker": ticker, "status": "too_short", "best": None}

    df_w = to_weekly(df_d)
    if len(df_w) < 60:
        return {"ticker": ticker, "status": "too_short", "best": None}

    t0 = time.time()
    best, _s, cums = select_strategy(
        ticker=ticker, df_d=df_d, df_w=df_w,
        data_dir=data_dir, min_trades=min_trades, force=force,
    )
    elapsed = time.time() - t0

    # Read override flags from the freshly written cache
    import json as _json
    cache_path = Path(data_dir) / "strategy_cache" / f"{ticker}.json"
    cache_data  = _json.loads(cache_path.read_text()) if cache_path.exists() else {}
    override_3y = cache_data.get("recent_override")
    override_1y = cache_data.get("regime_change_1y")
    cum_3y      = cache_data.get("recent_cum_3y")   # % on 3Y window
    cum_1y_raw  = cache_data.get("recent_cum_1y")   # % on 1Y window (filter-by-exit)

    # Full-history net P/L on the selected strategy
    from jpbuy2.backtest.engine import run_backtest
    from jpbuy2.backtest.metrics import trades_to_df
    from jpbuy2.strategy_select import STRATEGY_PROFILES

    s_best = STRATEGY_PROFILES.get(best)
    net_full = net_1y = None
    cutoff_1y = pd.Timestamp(df_d.index[-1]) - pd.Timedelta(days=365)
    try:
        r = run_backtest(df_d, df_w, "stock", s_best)
        dt = trades_to_df(r["trades"])
        if not dt.empty:
            net_full = dt["ret"].sum() * 10_000
            dt["exit_date"] = pd.to_datetime(dt["exit_date"])
            dt_1y = dt[dt["exit_date"] >= cutoff_1y]
            net_1y = dt_1y["ret"].sum() * 10_000 if not dt_1y.empty else 0.0
    except Exception:
        pass

    # Override label
    if override_1y:
        ovr = f"←1Y:{override_1y}"
    elif override_3y:
        ovr = f"←3Y:{override_3y}"
    else:
        ovr = ""

    return {
        "ticker":   ticker,
        "status":   "computed",
        "best":     best,
        "full_pct": round(cums.get(best) or 0.0, 1),
        "cum_3y":   round(cum_3y, 1) if cum_3y is not None else None,
        "net_full": round(net_full) if net_full is not None else None,
        "net_1y":   round(net_1y)   if net_1y   is not None else None,
        "override": ovr,
        "elapsed":  round(elapsed, 1),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pre-warm adaptive strategy cache")
    p.add_argument("--watchlist", default="watchlist.csv")
    p.add_argument("--data-dir",  default="data")
    p.add_argument("--force",     action="store_true",
                   help="Ignore existing cache and recompute all tickers")
    p.add_argument("--ticker",    default=None,
                   help="Warm a single ticker only")
    p.add_argument("--min-trades", type=int, default=6)
    args = p.parse_args(argv)

    if args.ticker:
        tickers = [args.ticker.strip()]
    else:
        wl = pd.read_csv(args.watchlist)
        tickers = (
            wl[wl["type"].str.lower() == "stock"]["ticker"]
            .str.strip().unique().tolist()
        )

    total = len(tickers)
    print(f"\nWarming strategy cache — {total} tickers  "
          f"({'force recompute' if args.force else 'skipping cached'})\n")
    print(f"  {'Ticker':<14} {'Status':<10} {'Strategy':<20} {'Full%':>7} {'3Y%':>7} {'Net Full€':>11} {'Net 1Y€':>9}  {'Override'}")
    print("  " + "-" * 110)

    computed = cached = skipped = 0
    for i, ticker in enumerate(tickers, 1):
        r = warm_ticker(ticker, args.data_dir, args.force, args.min_trades)
        st = r["status"]

        if st == "computed":
            computed += 1
            full_pct = f"{r['full_pct']:>+6.1f}%" if r.get('full_pct') is not None else "      —"
            cum_3y   = f"{r['cum_3y']:>+6.1f}%"   if r.get('cum_3y')   is not None else "      —"
            net_full = f"{r['net_full']:>+10,.0f}€" if r.get('net_full') is not None else "          —"
            net_1y   = f"{r['net_1y']:>+8,.0f}€"   if r.get('net_1y')   is not None else "        —"
            ovr      = r.get('override') or ""
            line = (f"  {ticker:<14} {'computed':<10} {r['best']:<20} "
                    f"{full_pct} {cum_3y} {net_full} {net_1y}  {ovr}  {r['elapsed']:.1f}s")
        elif st == "cache_hit":
            cached += 1
            line = f"  {ticker:<14} {'cached':<10} (skipped — up to date)"
        else:
            skipped += 1
            line = f"  {ticker:<14} {st:<10} (no data or too short)"

        print(line, flush=True)

    print("\n" + "  " + "-" * 110)
    print(f"  Done — {computed} computed  {cached} cached  {skipped} skipped")
    print(f"  Cache: {args.data_dir}/strategy_cache/")
    print(f"\n  Columns:")
    print(f"    Full%     = selected strategy full-history cumulative return")
    print(f"    3Y%       = same strategy on last 3 years (hybrid Layer 1 trigger)")
    print(f"    Net Full€ = total net P/L full history on €10,000/trade")
    print(f"    Net 1Y€   = net P/L last 12 months on €10,000/trade  ← key signal")
    print(f"    Override  = ←3Y or ←1Y when hybrid layer switched strategy\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
