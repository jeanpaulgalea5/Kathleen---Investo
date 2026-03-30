from __future__ import annotations

import pandas as pd
from dataclasses import dataclass

from ..config import Settings
from ..types import AssetType
from ..signals.blockers import blockers_daily
from ..signals.silver import silver_signal
from ..signals.golden import compute_golden_weekly_flags


@dataclass
class Trade:
    entry_date: str
    entry_px: float
    exit_date: str | None = None
    exit_px: float | None = None
    reason_exit: str | None = None


def _weekly_index_map(df_d: pd.DataFrame, df_w: pd.DataFrame) -> pd.Series:
    """
    Map each daily date to the latest available weekly date (as-of mapping).
    """
    w_dates = pd.Index(df_w.index)
    out = []
    for d in df_d.index:
        pos = w_dates.searchsorted(d, side="right") - 1
        out.append(w_dates[pos] if pos >= 0 else pd.NaT)
    return pd.Series(out, index=df_d.index)


def _find_intraday_stop_hit(
    df_i: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts_exclusive: pd.Timestamp,
    stop_px: float,
    use_low: bool = True,
) -> pd.Timestamp | None:
    """
    Find the first intraday bar between [start_ts, end_ts_exclusive) that breaches stop_px.

    IMPORTANT:
    We use the NEXT DAY after entry as the intraday emergency-start point, to avoid
    false stop hits from intraday bars earlier on the same calendar day before the
    daily entry actually became valid.
    """
    if df_i is None or df_i.empty:
        return None

    col = "low" if use_low and "low" in df_i.columns else "close"
    if col not in df_i.columns:
        return None

    sl = df_i.loc[(df_i.index >= start_ts) & (df_i.index < end_ts_exclusive)]
    if sl.empty:
        return None

    hits = sl[sl[col].astype(float) <= float(stop_px)]
    if hits.empty:
        return None

    return pd.Timestamp(hits.index[0])


def run_backtest(
    df_d: pd.DataFrame,
    df_w: pd.DataFrame,
    asset_type: AssetType,
    s: Settings,
    ticker: str | None = None,
    df_i: pd.DataFrame | None = None,
) -> dict:
    """
    Required behaviour:
      - Golden is WEEKLY and stateful.
      - Silver is ENTRY ONLY (daily).
      - Entry only when Golden is ON.
      - Main exit only when Golden is OFF.
      - Optional intraday emergency exit overlay can close the trade earlier
        if a severe intraday crash occurs.

    The intraday overlay is OFF by default via config.
    """
    if len(df_d) < 260 or len(df_w) < 60:
        raise ValueError("Not enough history to backtest.")

    df_d = df_d.copy()
    df_w = df_w.copy()

    intraday_enabled = bool(getattr(s, "use_intraday_emergency_exit", False))
    intraday_stop_pct = float(getattr(s, "intraday_emergency_stop_pct", 0.12))
    intraday_use_low = bool(getattr(s, "intraday_emergency_use_low", True))

    if df_i is not None and not df_i.empty:
        df_i = df_i.copy()
        df_i.index = pd.to_datetime(df_i.index, errors="coerce")
        df_i = df_i.sort_index()

    # Precompute Golden flags ONCE (so engine & report agree)
    flags = compute_golden_weekly_flags(df_w, s)
    if flags is None or flags.empty or "golden_on" not in flags.columns:
        raise ValueError("Golden weekly flags could not be computed.")

    # Align flags to df_w index
    golden_on_by_week = (
        flags["golden_on"]
        .reindex(df_w.index)
        .fillna(False)
        .astype(bool)
    )

    # Pull weekly exit reason
    golden_exit_reason_by_week = (
        flags.get("golden_exit_reason", pd.Series("", index=flags.index))
        .reindex(df_w.index)
        .fillna("")
        .astype(str)
    )

    # Map daily -> weekly (as-of)
    df_d["wk_asof"] = _weekly_index_map(df_d, df_w)

    trades: list[Trade] = []
    in_pos = False
    t: Trade | None = None

    # If using break-of-stabilisation-high execution, we can "arm" an entry level
    pending_break_high: float | None = None

    for i in range(5, len(df_d) - 1):
        row = df_d.iloc[i]
        day_ts = pd.Timestamp(df_d.index[i]).normalize()
        next_day_ts = day_ts + pd.Timedelta(days=1)
        date = str(day_ts.date())

        wk_date = row["wk_asof"]
        if pd.isna(wk_date):
            continue

        try:
            golden_on = bool(golden_on_by_week.loc[wk_date])
        except Exception:
            golden_on = False

        # -------------------------
        # OPTIONAL INTRADAY EMERGENCY EXIT
        # -------------------------
        if in_pos and t is not None and intraday_enabled and df_i is not None and not df_i.empty:
            # Start checking from the NEXT calendar day after entry, so we do not
            # accidentally stop out on intraday bars earlier on the same entry date.
            intraday_start_ts = pd.Timestamp(t.entry_date).normalize() + pd.Timedelta(days=1)

            # Only check once we are beyond entry day
            if next_day_ts > intraday_start_ts:
                stop_px = float(t.entry_px) * (1.0 - intraday_stop_pct)

                hit_ts = _find_intraday_stop_hit(
                    df_i=df_i,
                    start_ts=intraday_start_ts,
                    end_ts_exclusive=next_day_ts,
                    stop_px=stop_px,
                    use_low=intraday_use_low,
                )

                if hit_ts is not None:
                    t.exit_date = str(pd.Timestamp(hit_ts).date())
                    t.exit_px = float(stop_px)
                    t.reason_exit = "INTRADAY_EMERGENCY_STOP"

                    trades.append(t)
                    in_pos = False
                    t = None
                    pending_break_high = None
                    continue

        # -------------------------
        # EXIT: MAIN EXIT ONLY when Golden is OFF
        # -------------------------
        if in_pos and t is not None:
            if not golden_on:
                t.exit_date = date
                t.exit_px = float(row["close"])

                try:
                    reason = str(golden_exit_reason_by_week.loc[wk_date]).strip()
                except Exception:
                    reason = ""

                t.reason_exit = reason if reason else "GOLDEN_OFF"

                trades.append(t)
                in_pos = False
                t = None
                pending_break_high = None
                continue

        # -------------------------
        # ENTRY: ONLY when Golden is ON
        # -------------------------
        if not in_pos:
            # If we have a pending break level, keep it alive until:
            # - it breaks (entry happens), OR
            # - Golden turns OFF (cancel it)
            if pending_break_high is not None:
                if not golden_on:
                    pending_break_high = None
                    continue

                if float(row["high"]) > float(pending_break_high):
                    t = Trade(entry_date=date, entry_px=float(pending_break_high))
                    in_pos = True
                    pending_break_high = None
                continue

            # No pending: only evaluate Silver signals if Golden is ON
            if not golden_on:
                continue

            d_slice = df_d.iloc[: i + 1]

            sil = silver_signal(d_slice, asset_type, s, golden_on=True)
            if not bool(sil.get("silver_buy", False)):
                continue

            blocked, _ = blockers_daily(d_slice, s)
            if blocked:
                continue

            if bool(getattr(s, "use_break_of_stabilisation_high", False)):
                pending_break_high = float(sil["stabilisation_high"])
                continue
            else:
                t = Trade(entry_date=date, entry_px=float(row["close"]))
                in_pos = True

    # close any open trade at end so CSV has exits
    if in_pos and t is not None:
        t.exit_date = str(pd.Timestamp(df_d.index[-1]).date())
        t.exit_px = float(df_d["close"].iloc[-1])
        t.reason_exit = "BACKTEST_END"
        trades.append(t)

    return {"trades": trades}
