from __future__ import annotations

import argparse
from pathlib import Path

from jpbuy2.etf_entry import ETFEntrySettings, optimise_many
from jpbuy2.data.watchlist_loader import get_etf_tickers


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run pure JP-BUY style ETF pullback optimiser."
    )
    p.add_argument(
        "--watchlist",
        default="watchlist.csv",
        help="Watchlist file containing tickers and ETF/type classification",
    )
    p.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing daily/ subfolder",
    )
    p.add_argument(
        "--out-dir",
        default="reports/latest/etf_entry",
        help="Where to write ETF optimiser outputs",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = ETFEntrySettings()
    tickers = get_etf_tickers(args.watchlist)
    print(f"Loaded {len(tickers)} ETF tickers from watchlist: {tickers}")

    result = optimise_many(
        tickers=tickers,
        data_dir=args.data_dir,
        settings=settings,
    )

    result["summary"].to_csv(out_dir / "etf_entry_summary.csv", index=False)
    result["latest"].to_csv(out_dir / "etf_entry_latest.csv", index=False)
    result["profiles"].to_csv(out_dir / "etf_entry_profiles.csv", index=False)
    result["windows"].to_csv(out_dir / "etf_entry_windows.csv", index=False)
    result["entries"].to_csv(out_dir / "etf_entry_history.csv", index=False)

    print(f"Wrote outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
