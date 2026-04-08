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

def load_weekly(ticker: str, data_dir: str):
    safe = ticker.replace("/", "_").replace("\\", "_")
    path = Path(data_dir) / "raw" / "weekly" / f"{safe}.csv"
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

    df_w = load_weekly(ticker, data_dir)
    if df_w is None:
        return {"ticker": ticker, "status": "no_weekly", "best": None}
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
    decision_window = cache_data.get("decision_window")
    best_full = cache_data.get("best_full")
    best_4y = cache_data.get("best_4y")
    best_3y = cache_data.get("best_3y")

    cums_full = cache_data.get("cums_full", {})
    cums_4y = cache_data.get("cums_4y", {})
    cums_3y = cache_data.get("cums_3y", {})
    cums_2y = cache_data.get("cums_2y", {})

    full_pct = cums_full.get(best_full) if best_full else None
    pct_4y = cums_4y.get(best_4y) if best_4y else None
    pct_3y = cums_3y.get(best_3y) if best_3y else None
    pct_2y = cums_2y.get(best) if best else None
  
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
    ovr = decision_window or ""

    return {
        "ticker":   ticker,
        "status":   "computed",
        "best":     best,
        "full_pct": round(full_pct, 1) if full_pct is not None else None,
        "pct_4y":   round(pct_4y, 1) if pct_4y is not None else None,
        "pct_3y":   round(pct_3y, 1) if pct_3y is not None else None,
        "pct_2y":   round(pct_2y, 1) if pct_2y is not None else None,
        "net_full": round(net_full) if net_full is not None else None,
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
    print(f"  {'Ticker':<14} {'Status':<10} {'Strategy':<20} {'Full%':>7} {'4Y%':>7} {'3Y%':>7} {'2Y%':>7} {'Net Full€':>11}  {'Window'}")
    print("  " + "-" * 110)

    computed = cached = skipped = 0
    for i, ticker in enumerate(tickers, 1):
        r = warm_ticker(ticker, args.data_dir, args.force, args.min_trades)
        st = r["status"]

        if st == "computed":
            computed += 1
            full_pct = f"{r['full_pct']:>+6.1f}%" if r.get('full_pct') is not None else "      —"
            pct_4y   = f"{r['pct_4y']:>+6.1f}%"   if r.get('pct_4y')   is not None else "      —"
            pct_3y   = f"{r['pct_3y']:>+6.1f}%"   if r.get('pct_3y')   is not None else "      —"
            pct_2y   = f"{r['pct_2y']:>+6.1f}%"   if r.get('pct_2y')   is not None else "      —"
            net_full = f"{r['net_full']:>+10,.0f}€" if r.get('net_full') is not None else "          —"
            ovr      = r.get('override') or ""
            line = (f"  {ticker:<14} {'computed':<10} {r['best']:<20} "
                    f"{full_pct} {pct_4y} {pct_3y} {pct_2y} {net_full}  {ovr}  {r['elapsed']:.1f}s")
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
    print(f"    Full%     = full-history reference only")
    print(f"    4Y%       = best 4Y-window strategy cumulative return")
    print(f"    3Y%       = best 3Y-window strategy cumulative return")
    print(f"    2Y%       = selected strategy cumulative return over the last 2 years")
    print(f"    Net Full€ = total net P/L full history on €10,000/trade")
    print(f"    Window    = selector basis now standardised to 2Y")
    print(f"                 Technicals use full history; selection uses best 2Y return\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
