import argparse
import json
import time
from pathlib import Path

import pandas as pd

from jpbuy2.data.yahoo import fetch_ohlcv


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    return df


def merge_append(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        df = new.copy()
    else:
        df = pd.concat([old, new])
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

    return df


def sanitise_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove any Saturday/Sunday rows from DAILY files.

    This is the hard safety-net for cases where a source or patch logic
    accidentally introduces a weekend date. We do NOT invent replacement
    rows; we simply keep the latest real trading day already present.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    out = out.sort_index()

    # Monday=0 ... Sunday=6 ; keep Mon-Fri only
    out = out[out.index.dayofweek < 5]

    # Re-deduplicate after filtering just to stay safe
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    return out


def _safe_fetch(
    ticker: str,
    start: str,
    end: str,
    interval: str,
) -> pd.DataFrame:
    return fetch_ohlcv(
        ticker,
        start=start,
        end=end,
        interval=interval,
        data_dir=None,
        prefer_local=False,
    )


def _latest_expected_date(interval: str, end: str) -> pd.Timestamp:
    end_ts = pd.Timestamp(end)
    if interval == "1d":
        return (end_ts - pd.offsets.BDay(1)).normalize()
    if interval == "1wk":
        return (end_ts - pd.Timedelta(days=7)).normalize()
    raise ValueError(f"Unsupported interval: {interval}")


def _is_stale(df: pd.DataFrame, interval: str, end: str) -> tuple[bool, pd.Timestamp | None, pd.Timestamp]:
    expected = _latest_expected_date(interval=interval, end=end)
    if df is None or df.empty:
        return True, None, expected

    latest = pd.Timestamp(df.index.max()).normalize()

    if interval == "1d":
        tolerance = pd.Timedelta(days=1)
    else:
        tolerance = pd.Timedelta(days=7)

    stale = latest < (expected - tolerance)
    return stale, latest, expected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--watchlist", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--force", action="store_true")
    p.add_argument("--pause-seconds", type=float, default=0.35)
    p.add_argument("--max-failures", type=int, default=12)
    args = p.parse_args()

    wl = pd.read_csv(args.watchlist)

    if "ticker" not in wl.columns:
        raise ValueError("watchlist must contain column 'ticker'")

    tickers = [str(t).strip() for t in wl["ticker"].tolist() if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))  # de-duplicate, keep order

    data_dir = Path(args.data_dir)
    daily_dir = data_dir / "raw" / "daily"
    weekly_dir = data_dir / "raw" / "weekly"
    reports_dir = data_dir / "_prefetch_reports"

    daily_dir.mkdir(parents=True, exist_ok=True)
    weekly_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    failures: list[dict] = []
    repairs: list[dict] = []
    successes = 0

    for i, t in enumerate(tickers, start=1):
        print(f"\n===== {i}/{len(tickers)} {t} =====")

        daily_path = daily_dir / f"{t}.csv"
        weekly_path = weekly_dir / f"{t}.csv"

        try:
            # -------------------------
            # DAILY
            # -------------------------
            old_daily = load_existing(daily_path)
            old_daily = sanitise_daily_series(old_daily)

            if args.force or old_daily.empty:
                print("Downloading FULL daily history")
                new_daily = _safe_fetch(
                    ticker=t,
                    start=args.start,
                    end=args.end,
                    interval="1d",
                )
                final_daily = new_daily
            else:
                last_date = old_daily.index.max()
                start_fetch = last_date - pd.Timedelta(days=14)

                print(f"Updating daily from {start_fetch.date()}")
                new_daily = _safe_fetch(
                    ticker=t,
                    start=start_fetch.strftime("%Y-%m-%d"),
                    end=args.end,
                    interval="1d",
                )
                final_daily = merge_append(old_daily, new_daily)

            # HARD SAFETY-NET: never save weekend rows in daily files
            pre_filter_len = len(final_daily)
            final_daily = sanitise_daily_series(final_daily)
            removed_weekend_rows = pre_filter_len - len(final_daily)

            stale_daily, latest_daily, expected_daily = _is_stale(final_daily, "1d", args.end)

            repairs.append(
                {
                    "ticker": t,
                    "interval": "1d",
                    "latest_local": "" if latest_daily is None else str(latest_daily.date()),
                    "expected_latest": str(expected_daily.date()),
                    "removed_weekend_rows": removed_weekend_rows,
                    "status": "stale_after_refresh" if stale_daily else "ok",
                }
            )

            final_daily.to_csv(daily_path)
            print(
                f"Daily rows: {len(final_daily)} | "
                f"latest={final_daily.index.max()} | "
                f"removed_weekend_rows={removed_weekend_rows}"
            )

            time.sleep(args.pause_seconds)

            # -------------------------
            # WEEKLY
            # -------------------------
            old_weekly = load_existing(weekly_path)

            if args.force or old_weekly.empty:
                print("Downloading FULL weekly history")
                new_weekly = _safe_fetch(
                    ticker=t,
                    start=args.start,
                    end=args.end,
                    interval="1wk",
                )
                final_weekly = new_weekly
            else:
                last_date = old_weekly.index.max()
                start_fetch = last_date - pd.Timedelta(days=60)

                print(f"Updating weekly from {start_fetch.date()}")
                new_weekly = _safe_fetch(
                    ticker=t,
                    start=start_fetch.strftime("%Y-%m-%d"),
                    end=args.end,
                    interval="1wk",
                )
                final_weekly = merge_append(old_weekly, new_weekly)

            stale_weekly, latest_weekly, expected_weekly = _is_stale(final_weekly, "1wk", args.end)

            repairs.append(
                {
                    "ticker": t,
                    "interval": "1wk",
                    "latest_local": "" if latest_weekly is None else str(latest_weekly.date()),
                    "expected_latest": str(expected_weekly.date()),
                    "removed_weekend_rows": 0,
                    "status": "stale_after_refresh" if stale_weekly else "ok",
                }
            )

            final_weekly.to_csv(weekly_path)
            print(f"Weekly rows: {len(final_weekly)} | latest={final_weekly.index.max()}")

            successes += 1
            time.sleep(args.pause_seconds)

        except Exception as e:
            print(f"ERROR for {t}: {e}")
            failures.append(
                {
                    "ticker": t,
                    "error": str(e),
                }
            )

            if len(failures) >= args.max_failures:
                print(f"Too many failures ({len(failures)}). Stopping prefetch.")
                break

            continue

    summary = {
        "watchlist_count": len(tickers),
        "success_count": successes,
        "failure_count": len(failures),
        "stale_count": sum(1 for r in repairs if r["status"] != "ok"),
        "failed_tickers": failures,
        "start": args.start,
        "end": args.end,
    }

    summary_json = reports_dir / "prefetch_summary.json"
    failures_csv = reports_dir / "prefetch_failures.csv"
    repairs_csv = reports_dir / "prefetch_repairs.csv"

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if failures:
        pd.DataFrame(failures).to_csv(failures_csv, index=False)
    else:
        pd.DataFrame(columns=["ticker", "error"]).to_csv(failures_csv, index=False)

    if repairs:
        pd.DataFrame(repairs).to_csv(repairs_csv, index=False)
    else:
        pd.DataFrame(
            columns=["ticker", "interval", "latest_local", "expected_latest", "removed_weekend_rows", "status"]
        ).to_csv(repairs_csv, index=False)

    print("\n===== PREFETCH SUMMARY =====")
    print(json.dumps(summary, indent=2))

    if failures and len(failures) >= args.max_failures:
        raise RuntimeError(
            f"Prefetch aborted after {len(failures)} failures. "
            f"See {summary_json} and {failures_csv}."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
