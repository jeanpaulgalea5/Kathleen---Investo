#!/usr/bin/env python3
from __future__ import annotations

import argparse

from jpbuy2.reporting.daily_dashboard import generate_daily_dashboard


def main() -> int:
    p = argparse.ArgumentParser(description="Generate Investo Daily dashboard")
    p.add_argument("--watchlist", default="watchlist.csv")
    p.add_argument("--holdings-xlsx", default="Investments.xlsx")
    p.add_argument("--holdings-sheet", default="Transactions")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="reports/latest/daily_dashboard")
    p.add_argument("--archive-dir", default="reports/archive/daily_dashboard")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--send-email", action="store_true")
    args = p.parse_args()

    info = generate_daily_dashboard(
        watchlist_csv=args.watchlist,
        holdings_xlsx=args.holdings_xlsx,
        holdings_sheet=args.holdings_sheet,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        archive_dir=args.archive_dir,
        start=args.start,
        end=args.end,
        send_email=args.send_email,
    )

    print("Daily dashboard generated:")
    for k, v in info.items():
        print(f" - {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
