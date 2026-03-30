import argparse
from pathlib import Path
import pandas as pd

from jpbuy2.data.yahoo import fetch_intraday_ohlcv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--watchlist", required=True, help="CSV file with a 'ticker' column")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--interval", default="60m")
    p.add_argument("--start", required=True, help="Start date, e.g. 2024-01-01")
    p.add_argument("--end", default=None, help="Optional end date")
    args = p.parse_args()

    wl = pd.read_csv(args.watchlist)
    if "ticker" not in wl.columns:
        raise ValueError("watchlist must contain a 'ticker' column")

    tickers = [str(t).strip() for t in wl["ticker"].tolist() if str(t).strip()]

    out_dir = Path(args.data_dir) / "raw" / f"intraday_{args.interval}"
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []

    for t in tickers:
        out_path = out_dir / f"{t}.csv"

        try:
            print(f"Fetching intraday {args.interval} for {t}...")
            df = fetch_intraday_ohlcv(
                ticker=t,
                start=args.start,
                end=args.end,
                interval=args.interval,
                data_dir=args.data_dir,
                prefer_local=False,
            )
            df.to_csv(out_path)
            print(f"Saved {t}: {len(df)} rows")

        except Exception as e:
            failed.append((t, str(e)))
            print(f"FAILED {t}: {e}")

    if failed:
        print("\nThe following tickers failed:")
        for t, msg in failed:
            print(f"- {t}: {msg}")


if __name__ == "__main__":
    main()
