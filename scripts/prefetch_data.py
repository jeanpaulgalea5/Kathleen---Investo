import argparse
import json
import time
from pathlib import Path

import pandas as pd

from jpbuy2.data.yahoo import fetch_ohlcv

# ---------------------------------------------------------------------------
# FX-converted tickers
#
# Some watchlist tickers are EUR-denominated Revolut instruments that have no
# direct yfinance feed.  For these, we fetch the underlying USD-quoted source
# ticker, then divide by EURUSD=X to produce EUR prices, and save the result
# under the watchlist ticker name.
#
# Format:  { watchlist_ticker: (yfinance_source_ticker, fx_rate_ticker) }
# Conversion applied:  price_eur = price_usd / eurusd_rate
# ---------------------------------------------------------------------------
TICKER_FX_MAP: dict[str, tuple[str, str]] = {
    "XAG": ("XAG=X", "EURUSD=X"),   # Revolut silver (EUR/oz) ← silver spot (USD/oz) / EURUSD
    "XAU": ("GC=F",  "EURUSD=X"),   # Revolut gold   (EUR/oz) ← gold futures (USD/oz) / EURUSD
}


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


def _fetch_fx_converted(
    watchlist_ticker: str,
    source_ticker: str,
    fx_ticker: str,
    start: str,
    end: str,
    data_dir: Path,
) -> pd.DataFrame:
    """
    Fetch a USD-quoted source (e.g. XAG=X) and convert to EUR by dividing
    through the FX rate (e.g. EURUSD=X), then return OHLCV denominated in EUR.

    The FX rate file is expected to already be present on disk from an earlier
    pass through the prefetch loop (EURUSD=X appears before XAG/XAU in the
    watchlist).  If the local file is absent we fall back to a live fetch.
    """
    # --- fetch or load source USD price ---
    src_df = _safe_fetch(source_ticker, start=start, end=end, interval="1d")

    if src_df is None or src_df.empty:
        raise ValueError(f"No data returned for source ticker '{source_ticker}'")

    src_df.columns = [str(c).strip().lower() for c in src_df.columns]
    src_df.index = pd.to_datetime(src_df.index, errors="coerce")
    src_df = src_df.sort_index()

    # --- load FX rate (prefer local cache, fall back to live fetch) ---
    fx_path = data_dir / "raw" / "daily" / f"{fx_ticker}.csv"
    if fx_path.exists():
        fx_df = pd.read_csv(fx_path, index_col=0, parse_dates=True)
        fx_df.index = pd.to_datetime(fx_df.index, errors="coerce")
        fx_df = fx_df.sort_index()
    else:
        fx_df = _safe_fetch(fx_ticker, start=start, end=end, interval="1d")
        if fx_df is None or fx_df.empty:
            raise ValueError(f"Could not load FX rate ticker '{fx_ticker}'")

    fx_df.columns = [str(c).strip().lower() for c in fx_df.columns]
    fx_rate = fx_df["close"].rename("fx_rate")

    # --- align on dates (inner join to avoid NaN gaps) ---
    aligned = src_df.join(fx_rate, how="left")
    # Forward-fill FX rate over weekends / holidays (max 3 days)
    aligned["fx_rate"] = aligned["fx_rate"].fillna(method="ffill", limit=3)
    aligned = aligned.dropna(subset=["fx_rate", "close"])

    # --- convert all price columns to EUR ---
    price_cols = [c for c in ("open", "high", "low", "close", "adj_close") if c in aligned.columns]
    out = aligned.copy()
    for col in price_cols:
        out[col] = aligned[col] / aligned["fx_rate"]

    out = out.drop(columns=["fx_rate"])

    print(
        f"  FX-converted {source_ticker} → {watchlist_ticker} (EUR): "
        f"{len(out)} rows, latest close = {out['close'].iloc[-1]:.4f} EUR"
    )
    return out


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
            # FX-CONVERTED TICKERS
            # e.g. XAG / XAU: fetch USD source, divide by EURUSD=X → EUR
            # -------------------------
            if t in TICKER_FX_MAP:
                source_ticker, fx_ticker = TICKER_FX_MAP[t]
                print(f"FX-conversion mode: {t} ← {source_ticker} / {fx_ticker}")

                fx_daily = _fetch_fx_converted(
                    watchlist_ticker=t,
                    source_ticker=source_ticker,
                    fx_ticker=fx_ticker,
                    start=args.start,
                    end=args.end,
                    data_dir=data_dir,
                )

                # Save daily converted file under watchlist ticker name
                fx_daily = sanitise_daily_series(fx_daily)
                fx_daily_path = daily_dir / f"{t}.csv"

                # Merge with any existing local data (incremental update)
                old_fx = load_existing(fx_daily_path)
                old_fx = sanitise_daily_series(old_fx)
                final_daily = merge_append(old_fx, fx_daily)
                final_daily = sanitise_daily_series(final_daily)

                stale_daily, latest_daily, expected_daily = _is_stale(final_daily, "1d", args.end)

                repairs.append({
                    "ticker": t,
                    "interval": "1d",
                    "latest_local": "" if latest_daily is None else str(latest_daily.date()),
                    "expected_latest": str(expected_daily.date()),
                    "removed_weekend_rows": 0,
                    "status": "stale_after_refresh" if stale_daily else "ok",
                })

                final_daily.to_csv(fx_daily_path)
                print(f"Saved {len(final_daily)} EUR-converted rows to {fx_daily_path}")

                # Also produce weekly by resampling the converted daily
                from jpbuy2.data.yahoo import _daily_to_weekly  # noqa: PLC0415
                final_weekly = _daily_to_weekly(final_daily)
                weekly_path = weekly_dir / f"{t}.csv"
                old_weekly = load_existing(weekly_path)
                final_weekly = merge_append(old_weekly, final_weekly)
                stale_weekly, latest_weekly, expected_weekly = _is_stale(final_weekly, "1wk", args.end)

                repairs.append({
                    "ticker": t,
                    "interval": "1wk",
                    "latest_local": "" if latest_weekly is None else str(latest_weekly.date()),
                    "expected_latest": str(expected_weekly.date()),
                    "removed_weekend_rows": 0,
                    "status": "stale_after_refresh" if stale_weekly else "ok",
                })

                final_weekly.to_csv(weekly_path)
                print(f"Saved {len(final_weekly)} EUR-converted weekly rows to {weekly_path}")

                successes += 1
                time.sleep(args.pause_seconds)
                continue  # skip normal fetch path for this ticker

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
