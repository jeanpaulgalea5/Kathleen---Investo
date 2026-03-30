from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .types import AssetType
from .config import settings_for, settings_for_ticker
from .strategy_select import settings_adaptive
from .data.yahoo import fetch_ohlcv, fetch_intraday_ohlcv
from .backtest import run_backtest
from .signals.golden import compute_golden_weekly
from .signals.silver import silver_signal
from .report import write_combined_trade_report


def _parse_asset_type(s: str) -> AssetType:
    x = (s or "").strip().lower()
    if x in ("stock", "shares", "share", "equity"):
        return "stock"  # type: ignore
    if x in ("etf",):
        return "etf"  # type: ignore
    raise argparse.ArgumentTypeError("type must be 'stock' or 'etf'")


def _ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def cmd_scan(args: argparse.Namespace) -> int:
    """
    Single-ticker scan:
      - computes weekly Golden (as-of)
      - computes daily Silver (entry-only) gated by Golden
    """
    ticker = args.ticker.strip()
    asset_type: AssetType = args.type
    s = settings_for_ticker(ticker, asset_type)

    df_w = fetch_ohlcv(
        ticker, start=args.start, end=args.end, interval="1wk", data_dir=args.data_dir
    )
    df_d = fetch_ohlcv(
        ticker, start=args.start, end=args.end, interval="1d", data_dir=args.data_dir
    )

    df_i = None
    if bool(getattr(s, "use_intraday_emergency_exit", False)):
        try:
            df_i = fetch_intraday_ohlcv(
                ticker=ticker,
                start=args.start,
                end=args.end,
                interval=getattr(s, "intraday_emergency_interval", "60m"),
                data_dir=args.data_dir,
                prefer_local=True,
            )
        except Exception as e:
            print(f"WARNING: intraday data unavailable for {ticker}: {e}")
            df_i = None

    out = run_backtest(df_d, df_w, asset_type, s, ticker=ticker, df_i=df_i)
    
    golden_on = bool(g.get("golden_on", False))

    sil = silver_signal(df_d, asset_type, s, golden_on=golden_on)
    buy = bool(sil.get("silver_buy", False))

    asof = args.asof or ""
    print(f"{ticker} | asof={asof or 'latest'} | golden={golden_on} | buy={buy}")
    print(f"Silver: {sil.get('reason', '')}")

    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """
    Backtest one ticker and print a quick summary to stdout.
    Trades are governed by:
      - entry = Silver (only when Golden ON)
      - exit = Golden OFF only
    """
    ticker = args.ticker.strip()
    asset_type: AssetType = args.type
    s = settings_for_ticker(ticker, asset_type)

    df_w = fetch_ohlcv(
        ticker, start=args.start, end=args.end, interval="1wk", data_dir=args.data_dir
    )
    df_d = fetch_ohlcv(
        ticker, start=args.start, end=args.end, interval="1d", data_dir=args.data_dir
    )

    # Adaptive: auto-select best strategy from ticker's own history
    if asset_type == "stock" and len(df_d) >= 260:
        try:
            s = settings_adaptive(
                ticker=ticker, df_d=df_d, df_w=df_w,
                data_dir=args.data_dir, verbose=True,
            )
        except Exception:
            pass  # keep fallback

    out = run_backtest(df_d, df_w, asset_type, s)
    trades = out.get("trades", []) or []

    n = len(trades)
    wins = 0
    rets = []

    for t in trades:
        ep = getattr(t, "entry_px", None)
        xp = getattr(t, "exit_px", None)
        if ep and xp and ep > 0:
            r = (xp / ep - 1.0) * 100.0
            rets.append(r)
            if r > 0:
                wins += 1

    win_rate = (wins / len(rets) * 100.0) if rets else 0.0
    avg_ret = (sum(rets) / len(rets)) if rets else 0.0

    print(f"{ticker} | trades={n} | win_rate={win_rate:.2f}% | avg_ret={avg_ret:.2f}%")

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8") as f:
            f.write("Ticker,Entry Date,Entry Price,Exit Date,Exit Price,Exit Reason\n")
            for t in trades:
                f.write(
                    f"{ticker},{t.entry_date},{t.entry_px},"
                    f"{t.exit_date or ''},{t.exit_px or ''},{t.reason_exit or ''}\n"
                )
        print(f"Wrote: {out_csv}")

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """
    Generates the consolidated Golden-window report (single CSV),
    with Silver entry/exit filled where a trade occurred.
    """
    info = write_combined_trade_report(
        watchlist_csv=args.watchlist,
        out_dir=args.out_dir,
        start=args.start,
        end=args.end,
        data_dir=args.data_dir,
    )

    print("Wrote combined backtest reports:")
    if "windows_path" in info:
        print(f" - {info['windows_path']}")
    elif "trades_path" in info:
        print(f" - {info['trades_path']}")

    if "summary_path" in info:
        print(f" - {info['summary_path']}")

    if "tickers" in info:
        print(f"Tickers processed: {info['tickers']}")
    if "total_windows" in info:
        print(f"Total golden windows: {info['total_windows']}")
    if "total_trades" in info:
        print(f"Total trades: {info['total_trades']}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jpbuy2",
        description="JPBUY2 – Golden weekly regime + Silver entry timing",
    )
    p.set_defaults(func=None)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--start", default="2010-01-01", help="Start date (YYYY-MM-DD)")
        sub.add_argument("--end", default=None, help="End date (YYYY-MM-DD) or blank for latest")
        # ✅ IMPORTANT: default to repo data folder so downloads are persisted
        sub.add_argument("--data-dir", default="data", help="Local data cache dir (default: data)")

    sp = p.add_subparsers(dest="cmd", required=True)

    scan = sp.add_parser("scan", help="Scan one ticker (golden + silver)")
    scan.add_argument("ticker", help="Ticker, e.g. MSFT")
    scan.add_argument("--type", default="stock", type=_parse_asset_type, help="stock|etf")
    scan.add_argument("--asof", default=None, help="Label only (optional)")
    add_common(scan)
    scan.set_defaults(func=cmd_scan)

    bt = sp.add_parser("backtest", help="Backtest one ticker")
    bt.add_argument("ticker", help="Ticker, e.g. MSFT")
    bt.add_argument("--type", default="stock", type=_parse_asset_type, help="stock|etf")
    bt.add_argument("--out-csv", default=None, help="Optional path to write raw trades CSV")
    add_common(bt)
    bt.set_defaults(func=cmd_backtest)

    rp = sp.add_parser("report", help="Generate consolidated report from watchlist")
    rp.add_argument("--watchlist", default="watchlist.csv", help="CSV with columns: ticker,type")
    rp.add_argument("--out-dir", default="artifacts/reports", type=_ensure_dir, help="Output directory")
    add_common(rp)
    rp.set_defaults(func=cmd_report)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.func is None:
        parser.print_help()
        return 2

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as e:
        print(f"ERROR: {e}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
