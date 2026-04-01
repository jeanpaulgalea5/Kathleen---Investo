from __future__ import annotations

import json
import math
import os
import smtplib
from dataclasses import asdict
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..backtest import run_backtest
from ..config import settings_for
from ..data.yahoo import fetch_ohlcv, fetch_intraday_ohlcv
from ..indicators.macd import compute_macd
from ..indicators.rsi import rsi_wilder
from ..report import load_watchlist_csv
from ..signals.blockers import blockers_daily
from ..signals.golden import compute_golden_weekly_flags
from ..signals.silver import silver_signal
from .exit_monitor import load_current_holdings_from_workbook, build_exit_monitor
from ..etf_entry import ETFEntrySettings, latest_etf_signal, optimise_ticker

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


TZ_DISPLAY = "Europe/Malta"


def _tz() -> timezone:
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(TZ_DISPLAY)  # type: ignore[arg-type]
    except Exception:
        return timezone.utc


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _fmt_num(x: Any, digits: int = 2, suffix: str = "") -> str:
    v = _safe_float(x)
    if v is None:
        return ""
    return f"{v:.{digits}f}{suffix}"


def _fmt_pct(x: Any, digits: int = 2) -> str:
    return _fmt_num(x, digits=digits, suffix="%")


def _fmt_qty(x: Any) -> str:
    v = _safe_float(x)
    if v is None:
        return ""
    return f"{v:.4f}".rstrip("0").rstrip(".")


def _fmt_cost(x: Any, ccy: str = "") -> str:
    v = _safe_float(x)
    if v is None:
        return ""

    c = str(ccy or "").strip().upper()
    symbol_map = {
        "EUR": "€",
        "USD": "$",
        "GBP": "£",
    }

    prefix = symbol_map.get(c, f"{c} " if c else "")
    return f"{prefix}{v:,.2f}"

def _parse_pct_value(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip().replace("%", "")
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _parse_price_value(x: Any) -> Optional[float]:
    return _safe_float(x)


def _holding_pl_pct(net_cost: Any, price: Any, qty: Any) -> Optional[float]:
    cost = _safe_float(net_cost)
    px = _parse_price_value(price)
    q = _safe_float(qty)
    if cost is None or px is None or q is None or cost == 0:
        return None
    current_value = px * q
    return (current_value / cost - 1.0) * 100.0


def _exit_reason_label(signal: Any, golden: Any) -> str:
    sig = str(signal or "").strip().upper()
    g = str(golden or "").strip().upper()

    mapping = {
        "TRAILING_STOP": "Trailing stop broken",
        "RSI_BAND_DROP": "RSI support lost",
        "BACKTEST_END": "No active strategy signal",
    }

    base = mapping.get(sig, sig.replace("_", " ").title() if sig else "")

    if g == "OFF":
        if base:
            return f"{base}; weekly trend off"
        return "Weekly trend off"

    return base


def _exit_urgency(golden: Any, signal: Any, day_pct: Any) -> str:
    g = str(golden or "").strip().upper()
    sig = str(signal or "").strip().upper()
    d1 = _parse_pct_value(day_pct)

    if g == "OFF":
        return "High"
    if sig in {"TRAILING_STOP", "RSI_BAND_DROP"} and d1 is not None and d1 < 0:
        return "High"
    if sig:
        return "Medium"
    return "Review"

def _exit_conflict(golden: Any, signal: Any) -> str:
    g = str(golden or "").strip().upper()
    s = str(signal or "").strip().upper()

    if g == "ON" and s:
        return "⚠ Trend vs Exit"
    return ""

def _prudent_target(target_price: Any, haircut: float = 0.04) -> Optional[float]:
    tp = _safe_float(target_price)
    if tp is None:
        return None
    return tp * (1.0 - haircut)


def _upside_pct(current_price: Any, target_price: Any) -> Optional[float]:
    px = _safe_float(current_price)
    tp = _safe_float(target_price)
    if px is None or tp is None or px <= 0:
        return None
    return (tp / px - 1.0) * 100.0

def _normalise_price_for_display_and_target(
    ticker: str,
    currency: str,
    price: Any,
    target_price: Any,
) -> Optional[float]:
    px = _safe_float(price)
    tp = _safe_float(target_price)
    t = str(ticker or "").strip().upper()
    ccy = str(currency or "").strip().upper()

    if px is None:
        return None

    # Yahoo often returns LSE GBP equities in pence, while our target prices
    # are stored in pounds. Only rescale where the mismatch is obvious.
    if ccy == "GBP" and t.endswith(".L") and px >= 100:
        if tp is None or tp < 100:
            return px / 100.0

    return px

def _target_status(prudent_upside_pct: Any) -> str:
    up = _safe_float(prudent_upside_pct)
    if up is None:
        return "No Target"
    if up >= 20:
        return "Attractive"
    if up >= 10:
        return "Moderate"
    if up >= 0:
        return "Near Target"
    return "Above Target"
    
def _priority_score(
    target_status: str,
    entry_zone: str,
    win_rate_3y: Any,
    trades_n: Any,
    zone_effectiveness_score: int = 0,
    cycle_edge: str = "",
) -> int:
    score = 0

    target_points = {
        "Attractive": 4,
        "Moderate": 2,
        "Near Target": 0,
        "Above Target": -3,
        "No Target": 0,
    }
    score += target_points.get(str(target_status or "").strip(), 0)

    entry_points = {
        "Lower Quartile": 4,
        "Normal Range": 2,
        "Upper Quartile": 0,
        "Extended": -2,
        "Limited history": 0,
    }
    score += entry_points.get(str(entry_zone or "").strip(), 0)

    if entry_zone == "Extended":
        score += min(0, int(zone_effectiveness_score))
    else:
        score += int(zone_effectiveness_score)

    wr3 = _safe_float(win_rate_3y)
    if wr3 is not None:
        if wr3 >= 80:
            score += 3
        elif wr3 >= 70:
            score += 2
        elif wr3 >= 60:
            score += 1

    tr = _safe_float(trades_n)
    if tr is not None:
        if tr >= 50:
            score += 2
        elif tr >= 40:
            score += 1

    cycle_edge = str(cycle_edge or "").strip()
    if cycle_edge == "Late Risk":
        score -= 2
    elif cycle_edge == "Exhausted":
        score -= 4
    elif cycle_edge == "Early Edge":
        score += 2

    return int(score)


def _priority_stars(score: Any) -> str:
    s = _safe_float(score)
    if s is None:
        return ""
    if s >= 8:
        return "⭐⭐⭐"
    if s >= 5:
        return "⭐⭐"
    return "⭐"

def _current_opportunity_score(
    cycle_edge: str,
    cycle_age_pct: Any,
    cycle_win_rate_3y: Any,
    cycle_median_return_3y: Any,
    prudent_upside_pct: Any,
) -> int:
    score = 0

    edge_points = {
        "Early Edge": 30,
        "Mid OK": 18,
        "Late Risk": 4,
        "Exhausted": 0,
        "Limited history": 10,
    }
    score += edge_points.get(str(cycle_edge or "").strip(), 0)

    age = _parse_pct_value(cycle_age_pct)
    if age is not None:
        if age <= 20:
            score += 20
        elif age <= 35:
            score += 16
        elif age <= 50:
            score += 12
        elif age <= 70:
            score += 7
        elif age <= 85:
            score += 3
    # --- cycle decay factor ---
    decay = 1.0
    age = _parse_pct_value(cycle_age_pct)

    if age is not None:
        if age <= 30:
            decay = 1.0
        elif age <= 50:
            decay = 0.85
        elif age <= 70:
            decay = 0.55
        elif age <= 85:
            decay = 0.4
        else:
            decay = 0.2
    # --- cycle confidence ---
    wr_val = _parse_pct_value(cycle_win_rate_3y)
    cycle_confidence = None
    if wr_val is not None:
        cycle_confidence = wr_val * decay      
    
    med = _parse_pct_value(cycle_median_return_3y)
    if med is not None:
        if med >= 12:
            score += int(20 * decay)
        elif med >= 8:
            score += 15
        elif med >= 4:
            score += 10
        elif med > 0:
            score += 5

    up = _parse_pct_value(prudent_upside_pct)
    if up is not None:
        if up >= 20:
            score += int(15 * decay)
        elif up >= 12:
            score += 11
        elif up >= 8:
            score += 8
        elif up >= 4:
            score += 4

    wr = _parse_pct_value(cycle_win_rate_3y)
    if wr is not None:
        if wr >= 80:
            score += int(15 * decay)
        elif wr >= 70:
            score += 12
        elif wr >= 60:
            score += 8
        elif wr >= 50:
            score += 4

    return int(score)


def _current_opportunity_stars(score: Any) -> str:
    s = _safe_float(score)
    if s is None:
        return ""
    if s >= 78:
        return "⭐⭐⭐"
    if s >= 62:
        return "⭐⭐"
    if s >= 45:
        return "⭐"
    return ""

def _entry_zone_thresholds(entry_rsi_values: list[float]) -> tuple[float, float, float]:
    rsi_series = pd.Series(entry_rsi_values, dtype=float)
    q1 = float(rsi_series.quantile(0.25))
    q3 = float(rsi_series.quantile(0.75))
    iqr = q3 - q1
    stretch_cutoff = q3 + max(2.0, 0.25 * iqr)
    return q1, q3, stretch_cutoff


def _classify_entry_zone_from_thresholds(
    current_rsi: Any,
    q1: float,
    q3: float,
    stretch_cutoff: float,
) -> str:
    rsi = _safe_float(current_rsi)
    if rsi is None:
        return "Limited history"
    if rsi <= q1:
        return "Lower Quartile"
    if rsi < q3:
        return "Normal Range"
    if rsi <= stretch_cutoff:
        return "Upper Quartile"
    return "Extended"


def _entry_zone_effectiveness(df_d: pd.DataFrame, trades: list[Any]) -> dict[str, dict[str, float]]:
    if df_d is None or df_d.empty or not trades:
        return {}

    entry_samples: list[dict[str, Any]] = []

    for t in trades:
        try:
            d = asdict(t)
        except Exception:
            continue

        entry_date_raw = d.get("entry_date")
        if not entry_date_raw:
            continue

        entry_ts = pd.to_datetime(entry_date_raw, errors="coerce")
        if pd.isna(entry_ts):
            continue

        try:
            entry_ts = pd.Timestamp(entry_ts).tz_localize(None)
        except Exception:
            entry_ts = pd.Timestamp(entry_ts)

        try:
            history_slice = df_d.loc[df_d.index <= entry_ts]
        except Exception:
            continue

        if history_slice.empty:
            continue

        closes = history_slice["close"].astype(float)
        entry_rsi_series = rsi_wilder(closes, period=14)
        entry_rsi = _safe_float(entry_rsi_series.iloc[-1] if len(entry_rsi_series) else None)

        ep = _safe_float(d.get("entry_px"))
        xp = _safe_float(d.get("exit_px"))
        if entry_rsi is None or ep is None or xp is None or ep <= 0:
            continue

        ret = (xp / ep - 1.0) * 100.0
        entry_samples.append({"entry_rsi": entry_rsi, "ret": ret})

    if len(entry_samples) < 4:
        return {}

    entry_rsi_values = [float(x["entry_rsi"]) for x in entry_samples]
    q1, q3, stretch_cutoff = _entry_zone_thresholds(entry_rsi_values)

    buckets: dict[str, list[float]] = {
        "Lower Quartile": [],
        "Normal Range": [],
        "Upper Quartile": [],
        "Extended": [],
    }

    for sample in entry_samples:
        zone = _classify_entry_zone_from_thresholds(
            sample["entry_rsi"],
            q1=q1,
            q3=q3,
            stretch_cutoff=stretch_cutoff,
        )
        if zone in buckets:
            buckets[zone].append(float(sample["ret"]))

    stats: dict[str, dict[str, float]] = {}
    for zone, rets in buckets.items():
        if not rets:
            continue
        stats[zone] = {
            "count": float(len(rets)),
            "win_rate": float(sum(1 for r in rets if r > 0) / len(rets) * 100.0),
            "avg_return": float(pd.Series(rets).mean()),
        }

    return stats


def _entry_zone_effectiveness_score(
    entry_zone: str,
    zone_stats: dict[str, dict[str, float]],
) -> int:
    stats = zone_stats.get(str(entry_zone or "").strip(), {})
    count = _safe_float(stats.get("count"))
    win_rate = _safe_float(stats.get("win_rate"))
    avg_return = _safe_float(stats.get("avg_return"))

    if count is None or count < 3 or win_rate is None or avg_return is None:
        return 0

    score = 0

    if win_rate >= 70:
        score += 1
    elif win_rate < 50:
        score -= 1

    if avg_return >= 8:
        score += 1
    elif avg_return <= 0:
        score -= 1

    return max(-2, min(2, score))
def _golden_cycle_segments(flags: pd.DataFrame) -> list[dict[str, Any]]:
    if flags is None or flags.empty or "golden_on" not in flags.columns:
        return []

    f = flags.copy()
    f = f.sort_index()

    segments: list[dict[str, Any]] = []
    in_cycle = False
    start_idx = None

    idx_list = list(f.index)

    for i, idx in enumerate(idx_list):
        on = bool(f.loc[idx, "golden_on"])

        if on and not in_cycle:
            in_cycle = True
            start_idx = idx
        elif not on and in_cycle:
            end_idx = idx_list[i - 1] if i > 0 else idx
            segments.append(
                {
                    "start": pd.Timestamp(start_idx).tz_localize(None) if getattr(pd.Timestamp(start_idx), "tzinfo", None) is not None else pd.Timestamp(start_idx),
                    "end": pd.Timestamp(end_idx).tz_localize(None) if getattr(pd.Timestamp(end_idx), "tzinfo", None) is not None else pd.Timestamp(end_idx),
                }
            )
            in_cycle = False
            start_idx = None

    if in_cycle and start_idx is not None:
        end_idx = idx_list[-1]
        segments.append(
            {
                "start": pd.Timestamp(start_idx).tz_localize(None) if getattr(pd.Timestamp(start_idx), "tzinfo", None) is not None else pd.Timestamp(start_idx),
                "end": pd.Timestamp(end_idx).tz_localize(None) if getattr(pd.Timestamp(end_idx), "tzinfo", None) is not None else pd.Timestamp(end_idx),
            }
        )

    return segments


def _cycle_progress_bucket(progress_pct: Optional[float]) -> str:
    p = _safe_float(progress_pct)
    if p is None:
        return "Limited history"
    if p <= 25:
        return "Fresh"
    if p <= 50:
        return "Mid"
    if p <= 75:
        return "Mature"
    return "Late"


def _cycle_edge_label(
    cycle_age_pct: Optional[float],
    cycle_win_rate_3y: Optional[float],
    cycle_median_return_3y: Optional[float],
) -> str:
    age = _safe_float(cycle_age_pct)
    wr = _safe_float(cycle_win_rate_3y)
    med = _safe_float(cycle_median_return_3y)

    if age is None or wr is None or med is None:
        return "Limited history"

    if age > 100 or (age > 75 and (wr < 50 or med <= 0)):
        return "Exhausted"
    if age > 75 or wr < 50 or med <= 0:
        return "Late Risk"
    if age <= 35 and wr >= 70 and med > 0:
        return "Early Edge"
    return "Mid OK"


def _cycle_age_stats(
    flags: pd.DataFrame,
    trades: list[Any],
    now_ts: pd.Timestamp,
    last_n_years: int = 3,
) -> dict[str, Any]:
    if flags is None or flags.empty or not trades:
        return {
            "cycle_age_pct": None,
            "cycle_stage": "Limited history",
            "cycle_win_rate_3y": None,
            "cycle_median_return_3y": None,
            "cycle_edge": "Limited history",
            "median_cycle_weeks": None,
            "bucket_trade_count_3y": 0,
        }

    segments = _golden_cycle_segments(flags)
    if not segments:
        return {
            "cycle_age_pct": None,
            "cycle_stage": "Limited history",
            "cycle_win_rate_3y": None,
            "cycle_median_return_3y": None,
            "cycle_edge": "Limited history",
            "median_cycle_weeks": None,
            "bucket_trade_count_3y": 0,
        }

    # Historical completed cycle lengths in weeks
    hist_cycle_weeks: list[float] = []
    for seg in segments[:-1]:
        days = (seg["end"] - seg["start"]).days
        if days > 0:
            hist_cycle_weeks.append(days / 7.0)

    median_cycle_weeks = float(pd.Series(hist_cycle_weeks).median()) if hist_cycle_weeks else None

    # Current live cycle age proxy from last open/ongoing segment
    current_seg = segments[-1]
    current_elapsed_weeks = max(0.0, (now_ts - current_seg["start"]).days / 7.0)
    cycle_age_pct = None
    if median_cycle_weeks and median_cycle_weeks > 0:
        cycle_age_pct = (current_elapsed_weeks / median_cycle_weeks) * 100.0

    cycle_stage = _cycle_progress_bucket(cycle_age_pct)

    # Build historical entry-bucket stats over last N years
    cutoff = now_ts - pd.DateOffset(years=last_n_years)
    bucket_rets: dict[str, list[float]] = {
        "Fresh": [],
        "Mid": [],
        "Mature": [],
        "Late": [],
    }

    for t in trades:
        d = asdict(t)
        entry_dt = pd.to_datetime(d.get("entry_date"), errors="coerce")
        exit_dt = pd.to_datetime(d.get("exit_date"), errors="coerce")
        ep = _safe_float(d.get("entry_px"))
        xp = _safe_float(d.get("exit_px"))

        if pd.isna(entry_dt) or ep is None or xp is None or ep <= 0:
            continue

        try:
            if getattr(entry_dt, "tzinfo", None) is not None:
                entry_dt = entry_dt.tz_localize(None)
        except Exception:
            pass

        # Use exit date when available, else entry date, for 3Y inclusion
        ref_dt = exit_dt if pd.notna(exit_dt) else entry_dt
        try:
            if pd.notna(ref_dt) and getattr(ref_dt, "tzinfo", None) is not None:
                ref_dt = ref_dt.tz_localize(None)
        except Exception:
            pass

        if pd.isna(ref_dt) or ref_dt < cutoff:
            continue

        matching_seg = None
        for seg in segments:
            if seg["start"] <= entry_dt <= seg["end"]:
                matching_seg = seg
                break

        if matching_seg is None:
            continue

        seg_days = (matching_seg["end"] - matching_seg["start"]).days
        if seg_days <= 0:
            continue

        progress_pct = ((entry_dt - matching_seg["start"]).days / seg_days) * 100.0
        bucket = _cycle_progress_bucket(progress_pct)
        if bucket == "Limited history":
            continue

        ret = (xp / ep - 1.0) * 100.0
        bucket_rets[bucket].append(ret)

    full_rets_3y: list[float] = []
    for t in trades:
        d = asdict(t)
        entry_dt = pd.to_datetime(d.get("entry_date"), errors="coerce")
        exit_dt = pd.to_datetime(d.get("exit_date"), errors="coerce")
        ep = _safe_float(d.get("entry_px"))
        xp = _safe_float(d.get("exit_px"))

        if pd.isna(entry_dt) or ep is None or xp is None or ep <= 0:
            continue

        ref_dt = exit_dt if pd.notna(exit_dt) else entry_dt
        try:
            if pd.notna(ref_dt) and getattr(ref_dt, "tzinfo", None) is not None:
                ref_dt = ref_dt.tz_localize(None)
        except Exception:
            pass

        if pd.isna(ref_dt) or ref_dt < cutoff:
            continue

        full_rets_3y.append((xp / ep - 1.0) * 100.0)

    # Shrink bucket stats toward full-cycle 3Y stats
    current_bucket_rets = bucket_rets.get(cycle_stage, [])
    bucket_n = len(current_bucket_rets)

    full_wr_3y = None
    full_med_3y = None
    if full_rets_3y:
        full_wr_3y = sum(1 for r in full_rets_3y if r > 0) / len(full_rets_3y) * 100.0
        full_med_3y = float(pd.Series(full_rets_3y).median())

    bucket_wr = None
    bucket_med = None
    if current_bucket_rets:
        bucket_wr = sum(1 for r in current_bucket_rets if r > 0) / len(current_bucket_rets) * 100.0
        bucket_med = float(pd.Series(current_bucket_rets).median())

    if bucket_n > 0 and full_wr_3y is not None and full_med_3y is not None:
        w = min(1.0, bucket_n / 5.0)
        cycle_win_rate_3y = (w * bucket_wr) + ((1.0 - w) * full_wr_3y) if bucket_wr is not None else full_wr_3y
        cycle_median_return_3y = (w * bucket_med) + ((1.0 - w) * full_med_3y) if bucket_med is not None else full_med_3y
    else:
        cycle_win_rate_3y = full_wr_3y
        cycle_median_return_3y = full_med_3y

    cycle_edge = _cycle_edge_label(
        cycle_age_pct=cycle_age_pct,
        cycle_win_rate_3y=cycle_win_rate_3y,
        cycle_median_return_3y=cycle_median_return_3y,
    )

    return {
        "cycle_age_pct": cycle_age_pct,
        "cycle_stage": cycle_stage,
        "cycle_win_rate_3y": cycle_win_rate_3y,
        "cycle_median_return_3y": cycle_median_return_3y,
        "cycle_edge": cycle_edge,
        "median_cycle_weeks": median_cycle_weeks,
        "bucket_trade_count_3y": bucket_n,
    }

def _entry_zone_label(df_d: pd.DataFrame, trades: list[Any], daily_rsi_latest: Any) -> str:
    current_rsi = _safe_float(daily_rsi_latest)
    if current_rsi is None:
        return ""

    entry_rsi_values: list[float] = []
    if df_d is None or df_d.empty or not trades:
        return "Limited history"

    for t in trades:
        try:
            d = asdict(t)
        except Exception:
            continue

        entry_date_raw = d.get("entry_date")
        if not entry_date_raw:
            continue

        entry_ts = pd.to_datetime(entry_date_raw, errors="coerce")
        if pd.isna(entry_ts):
            continue

        try:
            entry_ts = pd.Timestamp(entry_ts).tz_localize(None)
        except Exception:
            entry_ts = pd.Timestamp(entry_ts)

        try:
            history_slice = df_d.loc[df_d.index <= entry_ts]
        except Exception:
            continue

        if history_slice.empty:
            continue

        closes = history_slice["close"].astype(float)
        entry_rsi_series = rsi_wilder(closes, period=14)
        entry_rsi = _safe_float(entry_rsi_series.iloc[-1] if len(entry_rsi_series) else None)
        if entry_rsi is not None:
            entry_rsi_values.append(entry_rsi)

    if len(entry_rsi_values) < 4:
        return "Limited history"

    q1, q3, stretch_cutoff = _entry_zone_thresholds(entry_rsi_values)
    return _classify_entry_zone_from_thresholds(
        current_rsi,
        q1=q1,
        q3=q3,
        stretch_cutoff=stretch_cutoff,
    )

def _dashboard_status_label(
    *,
    golden_on: bool,
    silver_buy: bool,
    silver_window: bool,
    blocked: bool,
    use_break_of_stabilisation_high: bool,
) -> str:
    """
    Dashboard status should reflect execution reality.

    Silver Window is informational only and must not gate BUY/BLOCKED/ARMED.
    Execution is driven by Golden + Silver BUY (+ blocker / breakout mode).
    """
    if not golden_on:
        return "WAIT"

    if blocked and silver_buy:
        return "BLOCKED"

    if silver_buy:
        if use_break_of_stabilisation_high:
            return "ARMED"
        return "BUY"

    return "WATCH"
    
def _reconstruct_engine_entry_state(
    *,
    df_d: pd.DataFrame,
    df_w: pd.DataFrame,
    asset_type: str,
    s: Any,
) -> dict[str, Any]:
    """
    Reconstruct the current engine entry state using the same logic as run_backtest(),
    but only to determine today's effective state for the dashboard.

    This is necessary because the engine can keep a pending breakout level alive
    across days, while the raw Silver signal only reflects today's setup.

    Returns:
        {
            "state": "WAIT" | "WATCH" | "BLOCKED" | "ARMED" | "BUY" | "IN_POS",
            "pending_break_high": float | None,
            "silver_today": bool,
            "blocked_today": bool,
            "golden_today": bool,
        }
    """
    from ..backtest.engine import _weekly_index_map

    if df_d is None or df_d.empty or df_w is None or df_w.empty:
        return {
            "state": "WAIT",
            "pending_break_high": None,
            "silver_today": False,
            "blocked_today": False,
            "golden_today": False,
        }

    df_d = df_d.copy()
    df_w = df_w.copy()

    flags = compute_golden_weekly_flags(df_w, s)
    if flags is None or flags.empty or "golden_on" not in flags.columns:
        return {
            "state": "WAIT",
            "pending_break_high": None,
            "silver_today": False,
            "blocked_today": False,
            "golden_today": False,
        }

    golden_on_by_week = (
        flags["golden_on"]
        .reindex(df_w.index)
        .fillna(False)
        .astype(bool)
    )

    df_d["wk_asof"] = _weekly_index_map(df_d, df_w)

    use_break_of_stabilisation_high = bool(
        getattr(s, "use_break_of_stabilisation_high", False)
    )

    in_pos = False
    pending_break_high: float | None = None

    silver_today = False
    blocked_today = False
    golden_today = False

    # Mirror the engine loop as closely as possible
    for i in range(5, len(df_d)):
        row = df_d.iloc[i]
        wk_date = row["wk_asof"]

        if pd.isna(wk_date):
            continue

        try:
            golden_on = bool(golden_on_by_week.loc[wk_date])
        except Exception:
            golden_on = False

        if i == len(df_d) - 1:
            golden_today = golden_on

        # If engine is already in a position, dashboard should not call this BUY/ARMED.
        if in_pos:
            if not golden_on:
                in_pos = False
                pending_break_high = None
            continue

        # Engine keeps pending breakout alive until breakout or Golden OFF
        if pending_break_high is not None:
            if not golden_on:
                pending_break_high = None
            else:
                if float(row["high"]) > float(pending_break_high):
                    in_pos = True
                    pending_break_high = None
            continue

        # No pending: only evaluate fresh Silver if Golden is ON
        if not golden_on:
            continue

        d_slice = df_d.iloc[: i + 1]

        sil = silver_signal(d_slice, asset_type, s, golden_on=True)
        silver_buy = bool(sil.get("silver_buy", False))

        blocked, _ = blockers_daily(d_slice, s)

        if i == len(df_d) - 1:
            silver_today = silver_buy
            blocked_today = blocked

        if not silver_buy:
            continue

        if blocked:
            continue

        if use_break_of_stabilisation_high:
            pending_break_high = float(sil["stabilisation_high"])
            continue
        else:
            in_pos = True

    # Final state exactly as engine-style interpretation
    if in_pos:
        state = "IN_POS"
    elif pending_break_high is not None and golden_today:
        state = "ARMED"
    elif not golden_today:
        state = "WAIT"
    elif silver_today and blocked_today:
        state = "BLOCKED"
    elif silver_today and not use_break_of_stabilisation_high:
        state = "BUY"
    else:
        state = "WATCH"

    return {
        "state": state,
        "pending_break_high": pending_break_high,
        "silver_today": silver_today,
        "blocked_today": blocked_today,
        "golden_today": golden_today,
    }

def _is_non_gold_etf(asset_type: str, ticker: str, name: str) -> bool:
    at = str(asset_type or "").strip().lower()
    t = str(ticker or "").strip().upper()
    n = str(name or "").strip().lower()

    if at != "etf":
        return False

    # Exclude gold / hedge ETFs for now
    gold_like = {
        "XAD5.MI",
        "XAU",
        "XAUUSD",
    }
    if t in gold_like:
        return False

    if "gold" in n:
        return False

    return True


def _load_etf_daily_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    date_col = None
    for candidate in ("date", "datetime", "timestamp"):
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        unnamed_cols = [c for c in df.columns if c.startswith("unnamed:")]
        if unnamed_cols:
            date_col = unnamed_cols[0]

    if date_col is None and len(df.columns) > 0:
        first_col = df.columns[0]
        parsed = pd.to_datetime(df[first_col], errors="coerce")
        if parsed.notna().mean() > 0.8:
            date_col = first_col

    if date_col is None:
        raise ValueError(f"Could not identify date column in CSV: {path}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()].copy().set_index(date_col).sort_index()

    for col in ("adj_close", "close", "high", "low", "open", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _build_etf_monitor_row(
    *,
    data_dir: str,
    ticker: str,
    asset_type: str,
    platform: str,
    name: str,
    currency: str,
    target_price: Any,
    held_tickers: set[str] | None = None,
) -> dict[str, Any]:
    held_tickers = held_tickers or set()
    ticker_norm = str(ticker).strip().upper()
    held_now = ticker_norm in held_tickers

    daily_csv = Path(data_dir) / "raw" / "daily" / f"{ticker}.csv"
    if not daily_csv.exists():
        return {
            "Ticker": ticker,
            "Name": str(name or "").strip(),
            "Type": asset_type,
            "Platform": platform,
            "Currency": currency,
            "Status": "WAIT",
            "Backtest State": "WAIT",
            "Price": "",
            "Target Price": _fmt_num(target_price, 2),
            "Prudent Target": _fmt_num(_prudent_target(target_price), 2),
            "Prudent Upside %": _fmt_pct(_upside_pct(None, _prudent_target(target_price)), 1),
            "Target Status": "No Target",
            "Priority": "",
            "Priority Score": 0,
            "1D %": "",
            "Golden": "",
            "Silver": "",
            "Silver Window": "",
            "Breakout Level": "",
            "Entry Zone": "ETF Pullback",
            "Zone Edge": "",
            "Zone Win Rate %": "",
            "Weekly RSI": "",
            "Daily RSI": "",
            "Weekly MACD Slope": "",
            "Daily MACD Slope": "",
            "Blocker": "Missing daily CSV",
            "Trades": 0,
            "Win Rate %": "",
            "Win Rate last 3 Years": "",
            "Median Return %": "",
            "Compounded Return %": "",
            "Latest Exit Reason": "",
        }

    etf_settings = ETFEntrySettings()

    # Latest live signal
    df_d = _load_etf_daily_csv(daily_csv)
    latest = latest_etf_signal(df_d=df_d, ticker=ticker, s=etf_settings)

    # Historical summary for dashboard ranking
    opt = optimise_ticker(
        ticker=ticker,
        daily_csv=daily_csv,
        settings=etf_settings,
    )
    summary = opt.get("summary", {}) or {}

    signal = str(latest.get("signal", "WAIT")).strip().upper()
    if held_now:
        status = "IN_POS"
    elif signal in {"BUY", "STRONG_BUY"}:
        status = "BUY"
    elif signal == "PRIMED":
        status = "PRIMED"
    else:
        status = "WATCH"

    price = _safe_float(latest.get("close"))
    raw_target_price = _safe_float(target_price)
    prudent_target = _prudent_target(raw_target_price, haircut=0.04)
    prudent_upside_pct = _upside_pct(price, prudent_target)
    target_status = _target_status(prudent_upside_pct)

    win_rate = _safe_float(summary.get("hit_rate"))
    if win_rate is not None:
        win_rate = win_rate * 100.0

    avg_ret = _safe_float(summary.get("avg_1y_return"))
    if avg_ret is not None:
        avg_ret = avg_ret * 100.0

    median_ret = _safe_float(summary.get("median_1y_return"))
    if median_ret is not None:
        median_ret = median_ret * 100.0

    trades_n = int(summary.get("signals", 0) or 0)

    # ETF priority: live trigger first, then historical quality, then upside
    priority_score = 0
    if signal == "STRONG_BUY":
        priority_score += 8
    elif signal == "BUY":
        priority_score += 5
    elif signal == "PRIMED":
        priority_score += 3

    if win_rate is not None:
        if win_rate >= 80:
            priority_score += 3
        elif win_rate >= 70:
            priority_score += 2
        elif win_rate >= 60:
            priority_score += 1

    if avg_ret is not None:
        if avg_ret >= 15:
            priority_score += 3
        elif avg_ret >= 10:
            priority_score += 2
        elif avg_ret >= 5:
            priority_score += 1

    if prudent_upside_pct is not None:
        if prudent_upside_pct >= 20:
            priority_score += 2
        elif prudent_upside_pct >= 10:
            priority_score += 1

    priority = _priority_stars(priority_score)

    latest_reason = str(latest.get("reason", "") or "")

    return {
        "Ticker": ticker,
        "Name": str(name or "").strip(),
        "Type": asset_type,
        "Platform": platform,
        "Currency": currency,
        "Status": status,
        "Backtest State": signal,
        "Price": _fmt_num(price, 2),
        "Target Price": _fmt_num(raw_target_price, 2),
        "Prudent Target": _fmt_num(prudent_target, 2),
        "Prudent Upside %": _fmt_pct(prudent_upside_pct, 1),
        "Target Status": target_status,
        "Priority": priority,
        "Priority Score": priority_score,
        "Opportunity Score": priority_score,
        "1D %": "",
        "Golden": "",
        "Silver": signal if signal in {"BUY", "STRONG_BUY", "PRIMED"} else "NO",
        "Silver Window": "",
        "Breakout Level": "",
        "Entry Zone": "ETF Pullback",
        "Zone Edge": "",
        "Zone Win Rate %": _fmt_pct(win_rate, 1),
        "Weekly RSI": "",
        "Daily RSI": _fmt_num(latest.get("rsi14"), 1),
        "Weekly MACD Slope": "",
        "Daily MACD Slope": _fmt_num(latest.get("macd_hist"), 3),
        "Drawdown Overshoot": (
            f"{latest.get('drawdown_overshoot_x')}x [{latest.get('drawdown_overshoot_label')}]"
            if latest.get("drawdown_overshoot_x") is not None else ""
        ),
        "Blocker": "" if status in {"BUY", "PRIMED"} else latest_reason,
        "Trades": trades_n,
        "Win Rate %": _fmt_pct(win_rate, 1),
        "Win Rate last 3 Years": _fmt_pct(win_rate, 1),
        "Median Return %": _fmt_pct(median_ret, 2),
        "Compounded Return %": _fmt_pct(avg_ret, 2),
        "Latest Exit Reason": latest_reason,
    }

def _build_backtest_reconciliation(
    monitor_df: pd.DataFrame,
    exit_df: pd.DataFrame,
    holdings_view_df: pd.DataFrame,
    backtest_windows_csv: str | Path,
) -> pd.DataFrame:
    """
    Reconcile dashboard state vs latest open/closed state implied by backtest_windows.csv.
    """
    try:
        bt = pd.read_csv(backtest_windows_csv)
    except Exception:
        return pd.DataFrame()

    if bt.empty or "Ticker" not in bt.columns:
        return pd.DataFrame()

    bt = bt.copy()
    bt["Ticker"] = bt["Ticker"].astype(str).str.strip().str.upper()

    # Keep latest row per ticker
    bt["_golden_on_dt"] = pd.to_datetime(bt.get("Golden ON"), errors="coerce")
    bt = bt.sort_values(["Ticker", "_golden_on_dt"]).groupby("Ticker", as_index=False).tail(1)

    held_tickers = set()

    if not exit_df.empty and "Ticker" in exit_df.columns:
        held_tickers.update(
            {
                str(x).strip().upper()
                for x in exit_df["Ticker"].dropna().tolist()
                if str(x).strip()
            }
        )

    if not holdings_view_df.empty and "Ticker" in holdings_view_df.columns:
        held_tickers.update(
            {
                str(x).strip().upper()
                for x in holdings_view_df["Ticker"].dropna().tolist()
                if str(x).strip()
            }
        )

    rows: list[dict[str, Any]] = []

    for _, r in monitor_df.iterrows():
        ticker = str(r.get("Ticker", "")).strip().upper()
        if not ticker:
            continue

        bt_row = bt[bt["Ticker"] == ticker]
        if bt_row.empty:
            rows.append(
                {
                    "Ticker": ticker,
                    "Dashboard Golden": r.get("Golden", ""),
                    "Dashboard Status": r.get("Status", ""),
                    "Backtest State": r.get("Backtest State", ""),
                    "Backtest Golden": "N/A",
                    "Backtest Silver": "N/A",
                    "Held": "YES" if ticker in held_tickers else "NO",
                    "Golden Match": "N/A",
                    "Silver / Position Match": "N/A",
                    "Reconciliation": "NO BACKTEST ROW",
                }
            )
            continue

        bt_last = bt_row.iloc[0]

        golden_on_dt = pd.to_datetime(bt_last.get("Golden ON"), errors="coerce")
        golden_off_dt = pd.to_datetime(bt_last.get("Golden OFF"), errors="coerce")
        silver_entry_dt = pd.to_datetime(bt_last.get("Silver Entry Date"), errors="coerce")
        exit_dt = pd.to_datetime(bt_last.get("Exit Date"), errors="coerce")
        exit_reason_bt = str(bt_last.get("Exit Reason", "")).strip().upper()

        golden_on_bt = pd.notna(golden_on_dt) and pd.isna(golden_off_dt)

        # Treat BACKTEST_END as still open for reconciliation purposes.
        silver_on_bt = pd.notna(silver_entry_dt) and (
            pd.isna(exit_dt) or exit_reason_bt == "BACKTEST_END"
        )

        dashboard_golden_on = str(r.get("Golden", "")).strip().upper() == "ON"
        dashboard_status = str(r.get("Status", "")).strip().upper()
        backtest_state = str(r.get("Backtest State", "")).strip().upper()
        held_now = ticker in held_tickers

        golden_match = dashboard_golden_on == golden_on_bt

        # Reconcile strategy state against the engine-style backtest state,
        # not against the dashboard action label. Dashboard Status is
        # portfolio-aware and can legitimately be WATCH even when the
        # engine-style state is IN_POS for names not currently held.
        if silver_on_bt:
            silver_match = backtest_state == "IN_POS"
        else:
            silver_match = backtest_state != "IN_POS"

        recon = "OK" if golden_match and silver_match else "MISMATCH"

        rows.append(
            {
                "Ticker": ticker,
                "Dashboard Golden": "ON" if dashboard_golden_on else "OFF",
                "Dashboard Status": dashboard_status,
                "Backtest State": backtest_state,
                "Backtest Golden": "ON" if golden_on_bt else "OFF",
                "Backtest Silver": "ON" if silver_on_bt else "OFF",
                "Held": "YES" if held_now else "NO",
                "Golden Match": "YES" if golden_match else "NO",
                "Silver / Position Match": "YES" if silver_match else "NO",
                "Reconciliation": recon,
            }
        )

    return pd.DataFrame(rows)


def _reconciliation_summary_html(recon_df: pd.DataFrame) -> str:
    if recon_df is None or recon_df.empty:
        return """
        <div class="card section">
          <h2>Backtest Reconciliation</h2>
          <div class="muted">No reconciliation rows were produced.</div>
        </div>
        """

    total = len(recon_df)
    ok_count = int((recon_df["Reconciliation"] == "OK").sum())
    mismatch_count = total - ok_count

    cols = [
        "Ticker",
        "Dashboard Golden",
        "Dashboard Status",
        "Backtest State",
        "Backtest Golden",
        "Backtest Silver",
        "Held",
        "Golden Match",
        "Silver / Position Match",
        "Reconciliation",
    ]

    return f"""
    <div class="card section">
      <h2>Backtest Reconciliation</h2>
      <div class="muted">
        TThis section checks the latest dashboard state against the latest row per ticker in reports/latest/backtest_windows.csv.
        Golden is treated as ON when the latest backtest row has Golden ON populated and Golden OFF blank.
        Backtest Silver is treated as ON when the latest backtest row has Silver Entry Date populated and either Exit Date is blank or Exit Reason = BACKTEST_END.
      </div>
      <div class="chips" style="margin-top:10px;">
        <div class="chip"><span>Rows checked</span><b>{total}</b></div>
        <div class="chip"><span>OK</span><b>{ok_count}</b></div>
        <div class="chip"><span>MISMATCH</span><b>{mismatch_count}</b></div>
      </div>
      {_table_html(recon_df, cols, 'No reconciliation rows were produced.')}
    </div>
    """

def _entry_zone_legend_html() -> str:
    return """
    <div class="legend">
      <h3>BUY section legend</h3>
      <div class="muted">
        Entry Zone is relative to each ticker's own historical RSI at past strategy entry points, not a fixed universal RSI rule.
      </div>
      <ul>
        <li><b>Lower Quartile</b> – current daily RSI is in the lowest 25% of this ticker's historical entry RSI readings.</li>
        <li><b>Normal Range</b> – current daily RSI is between the ticker's 25th and 75th percentile historical entry RSI readings.</li>
        <li><b>Upper Quartile</b> – current daily RSI is above the ticker's normal entry range but still within its usual historical behaviour.</li>
        <li><b>Extended</b> – current daily RSI is materially above the ticker's normal entry range and may be stretched even for that name.</li>
        <li><b>Limited history</b> – too few historical trades were available to build reliable ticker-specific quartiles.</li>
        <li><b>Prudent Target</b> – Investing.com target price reduced by 4% as a prudence haircut before calculating upside.</li>
        <li><b>Priority / Opportunity Score</b> – current-opportunity ranking for stock BUY candidates, combining cycle timing now, cycle-conditioned win rate, cycle-conditioned median return, and prudent upside still left.</li>
        <li><b>Priority ⭐⭐⭐</b> – top current setup right now: usually fresh or favourable cycle position, good cycle-conditioned return profile, and enough upside left.</li>
        <li><b>Priority ⭐⭐</b> – solid current setup, but with one or two weaker elements such as less upside, weaker cycle-conditioned return, or more mature cycle position.</li>
        <li><b>Priority ⭐</b> – valid system BUY, but currently less attractive than the other names on the board.</li>
        <li><b>BUY</b> – immediate execution state only when the system is configured to enter directly without breakout confirmation.</li>
        <li><b>ARMED</b> – valid setup detected and breakout level armed; the system is waiting for price to break the stabilisation high before actual entry.</li>
        <li><b>BLOCKED</b> – setup is valid, but a blocker is preventing entry.</li>
        <li><b>IN_POS</b> – the backtest engine would already be in an active strategy position.</li>
        <li><b>WATCH</b> – Golden regime is ON, but no active executable setup is currently present.</li>
        <li><b>WAIT</b> – Golden regime is OFF.</li>
        <li><b>Zone Win Rate %</b> – percentage of past trades for this ticker that were profitable when entries occurred within the current Entry Zone. This reflects how effective that zone has historically been for that specific name.</li>
        <li><b>Cycle Age %</b> – live proxy of how advanced the current Golden cycle is, measured as elapsed weeks since the current Golden regime started divided by the ticker’s median historical Golden cycle length.</li>
        <li><b>Cycle Stage</b> – Fresh, Mid, Mature or Late, based on the live cycle-age proxy.</li>
        <li><b>Cycle Win Rate 3Y</b> – last 3 years win rate for historical entries made at a similar point in the Golden cycle to today, with small-sample shrinkage toward the full-cycle 3Y win rate.</li>
        <li><b>Cycle Median Return 3Y</b> – last 3 years median return for historical entries made at a similar point in the Golden cycle to today, again shrunk toward the full-cycle 3Y median when the bucket is sparse.</li>
        <li><b>Cycle Edge</b> – summary marker of whether buying at this stage of the Golden cycle has historically been favourable for this ticker: Early Edge, Mid OK, Late Risk or Exhausted.</li>      
      </ul>
    </div>
    """


def _timing_label(
    golden_on: bool,
    silver_buy: bool,
    silver_window: bool,
    daily_rsi_val: Any,
    daily_macd_slope_val: Any,
) -> str:
    if not (golden_on and silver_buy):
        return ""

    rsi = _safe_float(daily_rsi_val)
    macd = _safe_float(daily_macd_slope_val)

    if rsi is None:
        return ""

    if rsi < 50 and (macd is not None and macd > 0):
        return "BUY ZONE"
    if 50 <= rsi <= 65:
        return "WAIT ZONE"
    return "OVERBOUGHT"


def _read_git_sha(repo_root: Path) -> str:
    head = repo_root / ".git" / "HEAD"
    if not head.exists():
        return ""
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref:"):
            ref_path = repo_root / ".git" / ref.split(" ", 1)[1]
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()[:12]
        return ref[:12]
    except Exception:
        return ""

def _format_cell_html(col: str, val: Any) -> str:
    sval = "" if pd.isna(val) else str(val)

    if col == "Ticker" and sval:
        url = f"https://finance.yahoo.com/quote/{sval}"
        return f'<a href="{url}" target="_blank">{escape(sval)}</a>'

    if col in {"P/L %", "1D %", "Prudent Upside %", "Win Rate %", "Win Rate last 3 Years", "Zone Win Rate %"}:
        num = _parse_pct_value(sval)
        if num is None:
            return escape(sval)

        css = "neutral"
        if col in {"P/L %", "1D %", "Prudent Upside %"}:
            if num > 0:
                css = "pos"
            elif num < 0:
                css = "neg"
        else:
            if num >= 70:
                css = "good"
            elif num < 50:
                css = "weak"

        return f'<span class="{css}">{escape(sval)}</span>'

    if col == "Urgency":
        css_map = {
            "High": "high",
            "Medium": "medium",
            "Review": "review",
        }
        return f'<span class="{css_map.get(sval, "neutral")}">{escape(sval)}</span>'

    if col == "Zone Edge":
        num = _safe_float(val)
        if num is None:
            return escape(sval)
        if num > 0:
            return f'<span class="pos">+{int(num)}</span>'
        if num < 0:
            return f'<span class="neg">{int(num)}</span>'
        return f'<span class="neutral">0</span>'

    if col == "Target Status":
        css_map = {
            "Attractive": "good",
            "Moderate": "medium",
            "Near Target": "warn",
            "Above Target": "neg",
            "No Target": "neutral",
        }
        return f'<span class="{css_map.get(sval, "neutral")}">{escape(sval)}</span>'

    if col == "Entry Zone":
        css_map = {
            "Lower Quartile": "good",
            "Normal Range": "neutral",
            "Upper Quartile": "warn",
            "Extended": "neg",
            "Limited history": "review",
        }
        return f'<span class="{css_map.get(sval, "neutral")}">{escape(sval)}</span>'

    if col == "Priority" and sval:
        return f'<span class="warn">{escape(sval)}</span>'

    if col == "Conflict" and sval:
        return f'<span class="warn">{escape(sval)}</span>'

    if col == "Cycle Edge":
        css_map = {
            "Early Edge": "good",
            "Mid OK": "neutral",
            "Late Risk": "warn",
            "Exhausted": "neg",
            "Limited history": "review",
        }
        return f'<span class="{css_map.get(sval, "neutral")}">{escape(sval)}</span>'

    if col == "Cycle Stage":
        css_map = {
            "Fresh": "good",
            "Mid": "neutral",
            "Mature": "warn",
            "Late": "neg",
            "Limited history": "review",
        }
        return f'<span class="{css_map.get(sval, "neutral")}">{escape(sval)}</span>'

    if col == "Cycle Age %":
        val_num = _parse_pct_value(sval)
        if val_num is None:
            return escape(sval)
        if val_num > 85:
            return f'<span class="neg">{escape(sval)}</span>'
        if val_num > 70:
            return f'<span class="warn">{escape(sval)}</span>'
        if val_num < 30:
            return f'<span class="good">{escape(sval)}</span>'
        return f'<span class="neutral">{escape(sval)}</span>'

    if col == "Opportunity Score":
        num = _safe_float(val)
        if num is None:
            return escape(sval)
        if num >= 80:
            return f'<span class="good">{escape(sval)}</span>'
        if num >= 60:
            return f'<span class="warn">{escape(sval)}</span>'
        if num >= 40:
            return f'<span class="neutral">{escape(sval)}</span>'
        return f'<span class="weak">{escape(sval)}</span>'

    if col == "Cycle Confidence %":
        num = _parse_pct_value(sval)
        if num is None:
            return escape(sval)
        if num >= 70:
            return f'<span class="good">{escape(sval)}</span>'
        if num >= 50:
            return f'<span class="warn">{escape(sval)}</span>'
        return f'<span class="weak">{escape(sval)}</span>'
        
    return escape(sval)

def _table_html(df: pd.DataFrame, columns: list[str], empty_message: str) -> str:
    if df is None or df.empty:
        return f'<div class="muted">{escape(empty_message)}</div>'

    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            val = row.get(col, "")
            cell_html = _format_cell_html(col, val)
            cells.append(f"<td>{cell_html}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    head = "".join(f"<th>{escape(c)}</th>" for c in columns)
    return (
        "<table><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )

def _build_html(
    monitor_df: pd.DataFrame,
    exit_df: pd.DataFrame,
    holdings_view_df: pd.DataFrame,
    audit: dict[str, Any],
    recon_df: pd.DataFrame,
    commodity_df: pd.DataFrame | None = None,
) -> str:
    held_tickers: set[str] = set()

    if exit_df is not None and not exit_df.empty and "Ticker" in exit_df.columns:
        held_tickers.update(
            {
                str(x).strip().upper()
                for x in exit_df["Ticker"].dropna().tolist()
                if str(x).strip()
            }
        )

    if holdings_view_df is not None and not holdings_view_df.empty and "Ticker" in holdings_view_df.columns:
        held_tickers.update(
            {
                str(x).strip().upper()
                for x in holdings_view_df["Ticker"].dropna().tolist()
                if str(x).strip()
            }
        )

    buy_df = monitor_df[monitor_df["Status"] == "BUY"].copy()
    buy_df["Ticker"] = buy_df["Ticker"].astype(str).str.strip().str.upper()
    buy_df["Type"] = buy_df["Type"].astype(str).str.strip().str.upper()
    buy_df["_win_rate_3y_num"] = pd.to_numeric(
        buy_df["Win Rate last 3 Years"].astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    )
    buy_df = buy_df[buy_df["_win_rate_3y_num"].fillna(0) >= 60.0].copy()
    buy_df = buy_df.drop(columns=["_win_rate_3y_num"], errors="ignore")

    # Fresh stock / ETF buys that are NOT currently held
    unheld_buy_df = buy_df.copy()
    if held_tickers:
        unheld_buy_df = unheld_buy_df[
            ~unheld_buy_df["Ticker"].isin(held_tickers)
        ].copy()

    stock_buy_df = unheld_buy_df[unheld_buy_df["Type"] != "ETF"].copy()
    etf_buy_df = unheld_buy_df[unheld_buy_df["Type"] == "ETF"].copy()

    # ETFs already held, but with active add/buy signal
    etf_add_df = monitor_df.copy()
    etf_add_df["Ticker"] = etf_add_df["Ticker"].astype(str).str.strip().str.upper()
    etf_add_df["Type"] = etf_add_df["Type"].astype(str).str.strip().str.upper()
    etf_add_df["Backtest State"] = etf_add_df["Backtest State"].astype(str).str.strip().str.upper()

    etf_add_df = etf_add_df[
        (etf_add_df["Type"] == "ETF")
        & (etf_add_df["Ticker"].isin(held_tickers))
        & (etf_add_df["Backtest State"].isin(["BUY", "STRONG_BUY", "PRIMED"]))
    ].copy()

    # PRIMED ETFs — unheld ETFs only with PRIMED signal.
    # Held ETFs with PRIMED already surface in "ETF add opportunities" above.
    primed_etf_df = monitor_df.copy()
    primed_etf_df["Ticker"] = primed_etf_df["Ticker"].astype(str).str.strip().str.upper()
    primed_etf_df["Type"] = primed_etf_df["Type"].astype(str).str.strip().str.upper()
    primed_etf_df["Backtest State"] = primed_etf_df["Backtest State"].astype(str).str.strip().str.upper()
    primed_etf_df = primed_etf_df[
        (primed_etf_df["Type"] == "ETF")
        & (primed_etf_df["Backtest State"] == "PRIMED")
        & ~primed_etf_df["Ticker"].isin(held_tickers)
    ].copy()
    if not primed_etf_df.empty and "Priority Score" in primed_etf_df.columns:
        primed_etf_df.sort_values(
            by=["Priority Score", "Win Rate last 3 Years", "Trades", "Ticker"],
            ascending=[False, False, False, True],
            inplace=True,
        )

    if not stock_buy_df.empty and "Priority Score" in stock_buy_df.columns:
        stock_buy_df.sort_values(
            by=["Priority Score", "Cycle Win Rate 3Y", "Prudent Upside %", "Ticker"],
            ascending=[False, False, False, True],
            inplace=True,
        )

    for _df in (etf_buy_df, etf_add_df):
        if not _df.empty and "Priority Score" in _df.columns:
            _df.sort_values(
                by=["Priority Score", "Win Rate last 3 Years", "Trades", "Ticker"],
                ascending=[False, False, False, True],
                inplace=True,
            )

    blocked_df = monitor_df[monitor_df["Status"] == "BLOCKED"].copy()
    if held_tickers:
        blocked_df = blocked_df[
            ~blocked_df["Ticker"].astype(str).str.strip().str.upper().isin(held_tickers)
        ].copy()
    
    exit_now_df = (
        exit_df[exit_df["Status"] == "EXIT NOW"].copy()
        if not exit_df.empty
        else pd.DataFrame()
    )

    watch_count = int((monitor_df["Status"] == "WATCH").sum()) if not monitor_df.empty else 0
    wait_count = int((monitor_df["Status"] == "WAIT").sum()) if not monitor_df.empty else 0
    armed_count = int((monitor_df["Status"] == "ARMED").sum()) if not monitor_df.empty else 0
    buy_count = int(len(buy_df)) if 'buy_df' in locals() else 0
    in_pos_count = int((monitor_df["Status"] == "IN_POS").sum()) if not monitor_df.empty else 0
        # ── Commodity signals ─────────────────────────────────────────────
    commodity_active_df = pd.DataFrame()
    commodity_primed_df = pd.DataFrame()
    if commodity_df is not None and not commodity_df.empty and "signal" in commodity_df.columns:
        commodity_active_df = commodity_df[
            commodity_df["signal"].isin(["STRONG_BUY", "BUY"])
        ].copy()
        commodity_primed_df = commodity_df[
            commodity_df["signal"] == "PRIMED"
        ].copy()

    stock_buy_count = int(len(stock_buy_df)) if 'stock_buy_df' in locals() else 0
    
    etf_buy_count = int(len(etf_buy_df)) if 'etf_buy_df' in locals() else 0
    etf_add_count = int(len(etf_add_df)) if 'etf_add_df' in locals() else 0
    primed_etf_count = int(len(primed_etf_df)) if 'primed_etf_df' in locals() else 0

    commodity_active_count = int(len(commodity_active_df))
    commodity_primed_count = int(len(commodity_primed_df))

    chips = [
        ("Watchlist", str(len(monitor_df))),
        ("BUY", str(buy_count)),
        ("Commodity BUY", str(commodity_active_count)),
        ("BUY Stocks", str(stock_buy_count)),
        ("BUY ETFs", str(etf_buy_count)),
        ("ETF Adds", str(etf_add_count)),
        ("PRIMED ETFs", str(primed_etf_count)),
        ("ARMED", str(armed_count)),
        ("BLOCKED", str(len(blocked_df))),
        ("IN_POS", str(in_pos_count)),
        ("WATCH", str(watch_count)),
        ("WAIT", str(wait_count)),
        ("Held", str(len(exit_df))),
        ("EXIT NOW", str(len(exit_now_df))),
    ]
    chips_html = "".join(
        f'<div class="chip"><span>{escape(k)}</span><b>{escape(v)}</b></div>'
        for k, v in chips
    )

    stock_buy_cols = [
        "Priority",
        "Opportunity Score",
        "Name",
        "Ticker",
        "Platform",
        "Currency",
        "Cycle Edge",
        "Cycle Stage",
        "Cycle Age %",
        "Cycle Win Rate 3Y",
        "Cycle Confidence %",
        "Cycle Median Return 3Y",
        "Win Rate %",
        "Win Rate last 3 Years",
        "Price",
        "Target Price",
        "Prudent Target",
        "Prudent Upside %",
        "Target Status",
        "Entry Zone",
        "Zone Win Rate %",
    ]
    
    etf_buy_cols = [
        "Priority",
        "Name",
        "Ticker",
        "Platform",
        "Currency",
        "Backtest State",
        "Price",
        "Daily RSI",
        "Daily MACD Slope",
        "Drawdown Overshoot",
        "Win Rate %",
        "Median Return %",
        "Compounded Return %",
        "Entry Zone",
        "Latest Exit Reason",
    ]

    commodity_cols = [
        "ticker",
        "commodity_type",
        "signal",
        "close",
        "rsi14",
        "drawdown",
        "drawdown_entry_th",
        "drawdown_overshoot_x",
        "drawdown_overshoot_label",
        "structural_bull",
    ]
    exit_action_cols = [
        "Ticker",
        "Name",
        "Platform",
        "Quantity",
        "Currency",
        "Net Cost",
        "Price",
        "P/L %",
        "Exit Reason",
        "Conflict",
        "Urgency",
        "Status",
    ]

    blocked_cols = [
        "Ticker",
        "Name",
        "Platform",
        "Status",
        "Price",
        "Golden",
        "Silver",
        "Blocker",
        "Trades",
        "Win Rate %",
        "Median Return %",
    ]

    exit_monitor_cols = [
        "Ticker",
        "Name",
        "Platform",
        "Quantity",
        "Currency",
        "Net Cost",
        "Price",
        "P/L %",
        "1D %",
        "Golden",
        "Silver",
        "Exit Reason",
        "Conflict",
        "Urgency",
        "Status",
    ]

    monitor_cols = [
        "Ticker",
        "Name",
        "Status",
        "Price",
        "1D %",
        "Golden",
        "Silver",
        "Silver Window",
        "Entry Zone",
        "Weekly RSI",
        "Daily RSI",
        "Weekly MACD Slope",
        "Daily MACD Slope",
        "Trades",
        "Win Rate %",
        "Win Rate last 3 Years",
        "Median Return %",
        "Latest Exit Reason",
    ]

    holdings_cols = [
        "Ticker",
        "Name",
        "Platform",
        "Quantity",
        "Currency",
        "Net Cost",
    ]

    audit_rows = "".join(
        f"<tr><th>{escape(str(k))}</th><td>{escape(str(v))}</td></tr>"
        for k, v in audit.items()
    )

    generated_at = escape(str(audit.get("generated_at_local", "")))

    has_exit_now = exit_now_df is not None and not exit_now_df.empty
    has_buy = any([
        stock_buy_df is not None and not stock_buy_df.empty,
        etf_buy_df is not None and not etf_buy_df.empty,
        etf_add_df is not None and not etf_add_df.empty,
        primed_etf_df is not None and not primed_etf_df.empty,
        not commodity_active_df.empty,
    ])
    has_blocked = blocked_df is not None and not blocked_df.empty

    if has_exit_now:
        action_panel_class = "card-alert"
    elif has_buy or has_blocked:
        action_panel_class = "card-review"
    else:
        action_panel_class = ""

    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>Daily dashboard</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;background:#f6f8fb;color:#1f2937;margin:0;padding:24px;}}
.wrap{{max-width:1900px;margin:0 auto;}}
.card{{background:#fff;border:1px solid #d8dee8;border-radius:14px;padding:18px 20px;margin:0 0 16px 0;}}
.card-alert{{border:2px solid #dc2626 !important;box-shadow:0 0 0 2px rgba(220,38,38,0.15);}}
.card-review{{border:2px solid #d97706 !important;box-shadow:0 0 0 2px rgba(217,119,6,0.12);}}
.panel-grid{{display:grid;grid-template-columns:1fr;gap:14px;margin-top:14px;}}
.subcard{{border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;background:#ffffff;}}
.subcard h3{{margin:0 0 8px 0;font-size:16px;}}
.subcard p{{margin:0 0 12px 0;font-size:13px;color:#4b5563;}}
.subcard-exit{{border:2px solid #dc2626;background:#fef2f2;}}
.subcard-commodity{{border:1px solid #86efac;background:#f0fdf4;}}
h1{{margin:0 0 6px 0;font-size:28px;}}
h2{{margin:0 0 12px 0;font-size:19px;}}
h3{{margin:18px 0 10px 0;font-size:15px;}}
.muted{{color:#6b7280;font-size:13px;}}
.chips{{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px;}}
.chip{{background:#f3f4f6;border:1px solid #e5e7eb;border-radius:12px;padding:10px 14px;min-width:110px;}}
.chip span{{display:block;font-size:12px;color:#6b7280;}}
.chip b{{display:block;font-size:20px;margin-top:4px;}}
table{{width:100%;border-collapse:collapse;font-size:13px;}}
th,td{{border-bottom:1px solid #e5e7eb;padding:8px 10px;text-align:left;vertical-align:top;}}
th{{background:#f9fafb;position:sticky;top:0;}}
.pos{{color:#15803d;font-weight:600;}}
.neg{{color:#b91c1c;font-weight:600;}}
.warn{{color:#b45309;font-weight:600;}}
.high{{color:#b91c1c;font-weight:700;}}
.medium{{color:#b45309;font-weight:600;}}
.review{{color:#6b7280;font-weight:600;}}
.good{{color:#15803d;font-weight:600;}}
.weak{{color:#b91c1c;font-weight:600;}}
.neutral{{color:#374151;font-weight:600;}}
.section{{overflow:auto;}}
.audit th{{width:260px;background:#fff;}}
.legend{{margin-top:12px;padding:12px 14px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;}}
.legend ul{{margin:8px 0 0 18px;padding:0;}}
.legend li{{margin:4px 0;}}
.group-stocks{{border:2px solid #86efac;border-radius:14px;padding:14px 16px;margin-bottom:4px;background:#f0fdf4;}}
.group-etfs{{border:2px solid #bfdbfe;border-radius:14px;padding:14px 16px;margin-bottom:4px;background:#eff6ff;}}
.group-commodity{{border:2px solid #fcd34d;border-radius:14px;padding:14px 16px;margin-bottom:4px;background:#fffbeb;}}
.group-label{{display:block;font-size:17px;font-weight:700;text-decoration:underline;text-transform:uppercase;letter-spacing:0.04em;margin:0 0 12px 2px;padding-bottom:4px;}}
.group-stocks .group-label{{color:#15803d;}}
.group-etfs .group-label{{color:#1d4ed8;}}
.group-commodity .group-label{{color:#b45309;}}
.group-stocks .subcard{{margin-bottom:10px;}}
.group-etfs .subcard{{margin-bottom:10px;}}
.group-commodity .subcard{{margin-bottom:10px;}}
.subcard-buy{{border:1px solid #86efac;background:#f0fdf4;}}
.subcard-blocked{{border:1px solid #fcd34d;background:#fffbeb;}}
</style>
</head>
<body>
<div class=\"wrap\">
  <div class=\"card {action_panel_class}\">
    <h1>Daily dashboard</h1>
    <div class=\"muted\">Generated: {generated_at}</div>
    <div class=\"chips\">{chips_html}</div>
  </div>

  <div class=\"card section {action_panel_class}\">
  <h2>Action Panel</h2>
  <div class=\"muted\">Grouped by asset class. Stocks → ETFs → Commodities → Exits.</div>

  <div class=\"panel-grid\">

    <div class=\"group-stocks\">
      <span class=\"group-label\">Stocks</span>

     <div class=\"subcard subcard-exit\">
       <h3>🚨 EXIT NOW holdings</h3>
       <p>You already hold these positions. Review for immediate action.</p>
       {_table_html(exit_now_df, exit_action_cols, 'No immediate exit actions today.')}
     </div>

      <div class=\"subcard subcard-buy\">
        <h3>✅ BUY candidates — Stocks</h3>
        <p>You do not currently hold these stock names. These are fresh executable buys from the main strategy engine.</p>
        {_table_html(stock_buy_df, stock_buy_cols, 'No stock BUY candidates today.')}
      </div>

      <div class=\"subcard subcard-blocked\">
        <h3>⏳ Blocked set-ups — Stocks near-miss</h3>
        <p>These are close but not actionable yet. Review the blocker before taking any entry decision.</p>
        {_table_html(blocked_df, blocked_cols, 'No blocked set-ups today.')}
      </div>

    </div>

    <div class=\"group-etfs\">
      <span class=\"group-label\">ETFs</span>

      <div class=\"subcard subcard-buy\">
        <h3>✅ BUY candidates — ETFs</h3>
        <p>You do not currently hold these ETF names. These are fresh ETF pullback opportunities not yet in the portfolio.</p>
        {_table_html(etf_buy_df, etf_buy_cols, 'No fresh ETF BUY candidates today.')}
      </div>

      <div class=\"subcard subcard-buy\">
        <h3>➕ ETF add opportunities</h3>
        <p>These ETFs are already held, but the ETF entry engine currently shows a BUY, STRONG_BUY, or PRIMED add signal.</p>
        {_table_html(etf_add_df, etf_buy_cols, 'No ETF add opportunities today.')}
      </div>

      <div class=\"subcard subcard-blocked\">
        <h3>⚡ PRIMED ETFs — signal imminent</h3>
        <p>Drawdown gate already met and MACD improving. BUY may fire on the next bar. Have capital ready.</p>
        {_table_html(primed_etf_df, etf_buy_cols, 'No ETFs currently primed.')}
      </div>

    </div>

    <div class=\"group-commodity\">
      <span class=\"group-label\">Commodities</span>

      <div class=\"subcard subcard-buy\">
        <h3>🥇 Commodity entry signals — BUY / STRONG BUY</h3>
        <p>Live signals from the commodity entry engine. STRONG_BUY = exceptional dip bypass active. Check the commodity entry report for full detail.</p>
        {_table_html(commodity_active_df, commodity_cols, 'No commodity BUY or STRONG_BUY signals today.')}
      </div>

      <div class=\"subcard subcard-blocked\">
        <h3>⏳ Commodity — PRIMED (signal imminent)</h3>
        <p>Drawdown gate met but awaiting weekly momentum confirmation. Monitor closely.</p>
        {_table_html(commodity_primed_df, commodity_cols, 'No commodity signals currently primed.')}
      </div>

    </div>

    

  </div>

  {_entry_zone_legend_html()}

</div>
  <div class=\"card section\">
    <h2>Exit Monitor</h2>
    <div class=\"muted\">All currently held watchlist positions. Immediate exits are surfaced above in the action panel.</div>
    {_table_html(exit_df, exit_monitor_cols, 'No currently held watchlist names were matched from Investments.xlsx.')}
  </div>

  <div class=\"card section\">
    <h2>Watchlist Monitor</h2>
    {_table_html(monitor_df, monitor_cols, 'No monitor rows were produced.')}
  </div>

  <div class=\"card section\">
    <h2>Holdings</h2>
    <div class=\"muted\">Aggregated directly from Investments.xlsx → Holdings by Code + Platform.</div>
    {_table_html(holdings_view_df, holdings_cols, 'No holdings rows were read from Investments.xlsx.')}
  </div>
  
  {_reconciliation_summary_html(recon_df)}
  
  <div class=\"card section audit\">
    <h2>Audit</h2>
    <table><tbody>{audit_rows}</tbody></table>
  </div>
</div>
</body>
</html>"""



def _send_email(subject: str, html_body: str) -> None:
    email_from = os.getenv("EMAIL_FROM", "").strip()
    email_to = os.getenv("EMAIL_TO", "").strip()
    host = os.getenv("SMTP_HOST", "").strip()
    port = os.getenv("SMTP_PORT", "").strip()
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()

    if not all([email_from, email_to, host, port, user, password]):
        print("Daily dashboard email not configured; skipping send.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = email_to
    msg.attach(MIMEText("Daily dashboard (HTML).", "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(host, int(port)) as smtp:
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)


def _build_monitor_row(
    data_dir: str,
    start: str,
    end: str | None,
    ticker: str,
    asset_type: str,
    platform: str,
    name: str = "",
    currency: str = "",
    target_price: Any = None,
    held_tickers: set[str] | None = None,
) -> dict[str, Any]:
    s = settings_for(asset_type)  # type: ignore[arg-type]

    if _is_non_gold_etf(asset_type=asset_type, ticker=ticker, name=name):
        return _build_etf_monitor_row(
            data_dir=data_dir,
            ticker=ticker,
            asset_type=asset_type,
            platform=platform,
            name=name,
            currency=currency,
            target_price=target_price,
            held_tickers=held_tickers,
        )

    df_w = fetch_ohlcv(ticker, start=start, end=end, interval="1wk", data_dir=data_dir)
    df_d = fetch_ohlcv(ticker, start=start, end=end, interval="1d", data_dir=data_dir)

    df_i = None
    if bool(getattr(s, "use_intraday_emergency_exit", False)):
        try:
            df_i = fetch_intraday_ohlcv(
                ticker=ticker,
                start=start,
                end=end,
                interval=getattr(s, "intraday_emergency_interval", "60m"),
                data_dir=data_dir,
                prefer_local=True,
            )
        except Exception as exc:
            print(f"WARNING: intraday data unavailable for {ticker}: {exc}")
            df_i = None

    flags = compute_golden_weekly_flags(df_w, s)
    latest_w = flags.iloc[-1]

    golden_on = bool(latest_w.get("golden_on", False))
    silver_window = bool(latest_w.get("silver_window", False))

    silver = silver_signal(df_d, asset_type, s, golden_on=golden_on)  # type: ignore[arg-type]
    silver_buy = bool(silver.get("silver_buy", False))

    blocked, block_reason = blockers_daily(df_d, s)

    engine_state = _reconstruct_engine_entry_state(
        df_d=df_d,
        df_w=df_w,
        asset_type=asset_type,
        s=s,
    )

    backtest_error = ""
    
    try:
        bt = run_backtest(df_d, df_w, asset_type, s, ticker=ticker, df_i=df_i)  # type: ignore[arg-type]
        trades = bt.get("trades", []) or []
    except ValueError as exc:
        if "Not enough history to backtest." in str(exc):
            bt = {"trades": []}
            trades = []
            backtest_error = "Insufficient history"
        else:
            raise
    price = float(df_d["close"].iloc[-1])
    prev_close = float(df_d["close"].iloc[-2]) if len(df_d) > 1 else price
    day_pct = ((price / prev_close) - 1.0) * 100.0 if prev_close else None

    daily_rsi = rsi_wilder(df_d["close"].astype(float), period=14)
    daily_macd = compute_macd(df_d["close"].astype(float), 12, 26, 9)

    daily_rsi_latest = daily_rsi.iloc[-1] if len(daily_rsi) else None
    daily_macd_slope_latest = daily_macd["slope"].iloc[-1] if not daily_macd.empty else None

    rets: list[float] = []
    rets_3y: list[float] = []
    latest_exit_reason = ""

    now_ts = pd.Timestamp.utcnow().tz_localize(None)
    cutoff_3y = now_ts - pd.DateOffset(years=3)

    for t in trades:
        d = asdict(t)
        ep = _safe_float(d.get("entry_px"))
        xp = _safe_float(d.get("exit_px"))
        if ep and xp and ep > 0:
            ret = (xp / ep - 1.0) * 100.0
            rets.append(ret)

            exit_dt_raw = d.get("exit_date") or d.get("entry_date")
            exit_dt = pd.to_datetime(exit_dt_raw, errors="coerce")
            if pd.notna(exit_dt):
                try:
                    if getattr(exit_dt, "tzinfo", None) is not None:
                        exit_dt = exit_dt.tz_localize(None)
                except Exception:
                    pass
                if exit_dt >= cutoff_3y:
                    rets_3y.append(ret)

    if trades:
        latest_exit_reason = str(asdict(trades[-1]).get("reason_exit") or "")

    trades_n = len(rets)
    win_rate = (sum(1 for r in rets if r > 0) / trades_n * 100.0) if trades_n else None
    win_rate_3y = (sum(1 for r in rets_3y if r > 0) / len(rets_3y) * 100.0) if rets_3y else None
    median_ret = float(pd.Series(rets).median()) if rets else None
    compounded = ((pd.Series(rets) / 100.0 + 1.0).prod() - 1.0) * 100.0 if rets else None

    use_break_of_stabilisation_high = bool(
        getattr(s, "use_break_of_stabilisation_high", False)
    )

    ticker_norm = str(ticker).strip().upper()
    held_now = ticker_norm in (held_tickers or set())

    raw_engine_state = str(engine_state.get("state", "WAIT")).upper()

    # Dashboard BUY = executable now only:
    # Golden ON + Silver VALID + not blocked + not already held
    if held_now and not golden_on:
        status = "EXIT NOW"        # backtest says exit — Golden went OFF while holding
    elif not golden_on:
        status = "WAIT"            # not held, Golden OFF = nothing to do
    elif blocked and silver_buy:
        status = "BLOCKED"
    elif (not held_now) and golden_on and silver_buy:
        status = "BUY"
    elif held_now and golden_on:
        status = "IN_POS"
    else:
        status = "WATCH"
        
    entry_zone = _entry_zone_label(
        df_d=df_d,
        trades=trades,
        daily_rsi_latest=daily_rsi_latest,
    )

    zone_stats = _entry_zone_effectiveness(df_d=df_d, trades=trades)
    zone_effectiveness_score = _entry_zone_effectiveness_score(
        entry_zone=entry_zone,
        zone_stats=zone_stats,
    )

    raw_target_price = _safe_float(target_price)
    display_price = _normalise_price_for_display_and_target(
        ticker=ticker,
        currency=currency,
        price=price,
        target_price=raw_target_price,
    )
    prudent_target = _prudent_target(raw_target_price, haircut=0.04)
    prudent_upside_pct = _upside_pct(display_price, prudent_target)
    target_status = _target_status(prudent_upside_pct)

    cycle_stats = _cycle_age_stats(
        flags=flags,
        trades=trades,
        now_ts=now_ts,
        last_n_years=3,
    )

    # --- cycle confidence calculation ---
    age_val = _parse_pct_value(_fmt_pct(cycle_stats.get("cycle_age_pct"), 0))
    wr_val = _parse_pct_value(_fmt_pct(cycle_stats.get("cycle_win_rate_3y"), 1))

    decay = 1.0
    if age_val is not None:
        if age_val <= 30:
            decay = 1.0
        elif age_val <= 50:
            decay = 0.85
        elif age_val <= 70:
            decay = 0.65
        elif age_val <= 85:
            decay = 0.4
        else:
            decay = 0.2

    cycle_confidence = None
    if wr_val is not None:
        cycle_confidence = wr_val * decay
        
    current_opportunity_score = _current_opportunity_score(
        cycle_edge=str(cycle_stats.get("cycle_edge", "") or ""),
        cycle_age_pct=_fmt_pct(cycle_stats.get("cycle_age_pct"), 0),
        cycle_win_rate_3y=_fmt_pct(cycle_stats.get("cycle_win_rate_3y"), 1),
        cycle_median_return_3y=_fmt_pct(cycle_stats.get("cycle_median_return_3y"), 2),
        prudent_upside_pct=_fmt_pct(prudent_upside_pct, 1),
    )
    priority_score = current_opportunity_score
    priority = _current_opportunity_stars(priority_score)
    
        
    return {
        "Ticker": ticker,
        "Name": str(name or "").strip(),
        "Type": asset_type,
        "Platform": platform,
        "Currency": currency,
        "Status": status,
        "Backtest State": raw_engine_state,
        "Price": _fmt_num(display_price, 2),
        "Target Price": _fmt_num(raw_target_price, 2),
        "Prudent Target": _fmt_num(prudent_target, 2),
        "Prudent Upside %": _fmt_pct(prudent_upside_pct, 1),
        "Target Status": target_status,
        "Priority": priority,
        "Priority Score": priority_score,
        "Opportunity Score": priority_score,
        "1D %": _fmt_pct(day_pct, 2),
        "Golden": "ON" if golden_on else "OFF",
        "Silver": "VALID" if silver_buy else "NO",
        "Silver Window": "OPEN" if silver_window else "OFF",
        "Breakout Level": _fmt_num(engine_state.get("pending_break_high"), 2),
        "Entry Zone": entry_zone,
        "Zone Edge": _fmt_num(zone_effectiveness_score, 0),
        "Zone Win Rate %": _fmt_pct(
            zone_stats.get(entry_zone, {}).get("win_rate"),
            1,
        ),
        "Weekly RSI": _fmt_num(latest_w.get("rsi14"), 1),
        "Daily RSI": _fmt_num(daily_rsi_latest, 1),
        "Weekly MACD Slope": _fmt_num(latest_w.get("macd_slope"), 3),
        "Daily MACD Slope": _fmt_num(daily_macd_slope_latest, 3),
        "Blocker": block_reason if blocked else "",
        "Trades": trades_n,
        "Win Rate %": _fmt_pct(win_rate, 1),
        "Win Rate last 3 Years": _fmt_pct(win_rate_3y, 1),
        "Cycle Age %": _fmt_pct(cycle_stats.get("cycle_age_pct"), 0),
        "Cycle Stage": str(cycle_stats.get("cycle_stage", "") or ""),
        "Cycle Win Rate 3Y": _fmt_pct(cycle_stats.get("cycle_win_rate_3y"), 1),
        "Cycle Confidence %": _fmt_pct(cycle_confidence, 1),
        "Cycle Median Return 3Y": _fmt_pct(cycle_stats.get("cycle_median_return_3y"), 2),
        "Cycle Edge": str(cycle_stats.get("cycle_edge", "") or ""),
        "Median Return %": _fmt_pct(median_ret, 2),
        "Compounded Return %": _fmt_pct(compounded, 2),
        "Latest Exit Reason": backtest_error or latest_exit_reason,
    }


def generate_daily_dashboard(
    *,
    watchlist_csv: str = "watchlist.csv",
    holdings_xlsx: str = "Investments.xlsx",
    holdings_sheet: str = "Holdings",
    data_dir: str = "data",
    out_dir: str = "reports/latest/daily_dashboard",
    archive_dir: str = "reports/archive/daily_dashboard",
    start: str = "2010-01-01",
    end: str | None = None,
    send_email: bool = False,
) -> dict[str, Any]:
    repo_root = Path.cwd()
    wl = load_watchlist_csv(watchlist_csv)

    rows: list[dict[str, Any]] = []

    current_holdings_df = load_current_holdings_from_workbook(
        path=holdings_xlsx,
        transactions_sheet=holdings_sheet,
    )

    held_tickers = set()
    if not current_holdings_df.empty and "Ticker" in current_holdings_df.columns:
        held_tickers = {
            str(x).strip().upper()
            for x in current_holdings_df["Ticker"].dropna().tolist()
            if str(x).strip()
        }
    for _, r in wl.iterrows():
        ticker = str(r.get("ticker", "")).strip()
        asset_type = str(r.get("type", "stock")).strip().lower()
        platform = str(r.get("Platform", "")).strip() if "Platform" in wl.columns else ""
        name = str(r.get("Name", "") or "").strip()
        currency = str(r.get("Currency", "") or "").strip()
        target_price = r.get("Target Price", None)

        rows.append(
            _build_monitor_row(
                data_dir=data_dir,
                start=start,
                end=end,
                ticker=ticker,
                asset_type=asset_type,
                platform=platform,
                name=name,
                currency=currency,
                target_price=target_price,
                held_tickers=held_tickers,
            )
        )

    monitor_df = pd.DataFrame(rows)

    status_order = {"BUY": 0, "ARMED": 1, "PRIMED": 2, "BLOCKED": 3, "IN_POS": 4, "WATCH": 5, "WAIT": 6}
    monitor_df["_sort"] = monitor_df["Status"].map(status_order).fillna(9)
    monitor_df = (
        monitor_df.sort_values(["_sort", "Ticker"])
        .drop(columns=["_sort"])
        .reset_index(drop=True)
    )

    
    holdings_view_df = current_holdings_df.copy()
    if not holdings_view_df.empty:
        holdings_view_df["Quantity"] = holdings_view_df["Quantity"].map(_fmt_qty)
        holdings_view_df["Net Cost"] = holdings_view_df.apply(
            lambda r: _fmt_cost(r.get("Net Cost"), r.get("Currency", "")),
            axis=1,
        )
        holdings_view_df = holdings_view_df[
            ["Ticker", "Name", "Platform", "Quantity", "Currency", "Net Cost"]
        ].copy()

    exit_holdings_df = current_holdings_df.copy()

    if not exit_holdings_df.empty and "Type" in exit_holdings_df.columns:
        exit_holdings_df = exit_holdings_df[
            ~exit_holdings_df["Type"].astype(str).str.strip().str.upper().isin({"ETF", "GOLDETF", "COMMODITY"})
        ].copy()

    exit_df = build_exit_monitor(exit_holdings_df, monitor_df)
    if not exit_df.empty:
        exit_df = exit_df.drop(columns=["Name"], errors="ignore").merge(
            monitor_df[["Ticker", "Name"]],
            on="Ticker",
            how="left",
        )
        
        exit_df["P/L %"] = exit_df.apply(
            lambda r: _fmt_pct(
                _holding_pl_pct(
                    net_cost=r.get("Net Cost"),
                    price=r.get("Price"),
                    qty=r.get("Quantity"),
                ),
                1,
            ),
            axis=1,
        )

        exit_df["Exit Reason"] = exit_df.apply(
            lambda r: _exit_reason_label(
                signal=r.get("Exit Signal"),
                golden=r.get("Golden"),
            ),
            axis=1,
        )

        exit_df["Urgency"] = exit_df.apply(
            lambda r: _exit_urgency(
                golden=r.get("Golden"),
                signal=r.get("Exit Signal"),
                day_pct=r.get("1D %"),
            ),
            axis=1,
        )

        exit_df["Conflict"] = exit_df.apply(
            lambda r: _exit_conflict(
                golden=r.get("Golden"),
                signal=r.get("Exit Signal"),
            ),
            axis=1,
        )
        
        exit_df["Quantity"] = exit_df["Quantity"].map(_fmt_qty)
        exit_df["Net Cost"] = exit_df.apply(
            lambda r: _fmt_cost(r.get("Net Cost"), r.get("Currency", "")),
            axis=1,
        )

        exit_df["__status_rank"] = exit_df["Status"].map(
            {"EXIT NOW": 0, "WATCH EXIT": 1}
        ).fillna(9)
        
        exit_df["__urgency_rank"] = exit_df["Urgency"].map(
            {"High": 0, "Medium": 1, "Review": 2}
        ).fillna(9)

        exit_df["__pl_num"] = (
            exit_df["P/L %"].astype(str).str.replace("%", "", regex=False)
        )
        exit_df["__pl_num"] = pd.to_numeric(exit_df["__pl_num"], errors="coerce").fillna(-9999)

        exit_df = exit_df.sort_values(
            by=["__status_rank", "__urgency_rank", "__pl_num"],
            ascending=[True, True, False],
        ).drop(columns=["__status_rank", "__urgency_rank", "__pl_num"])

    out_path = Path(out_dir)
    archive_path = Path(archive_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    archive_path.mkdir(parents=True, exist_ok=True)

    now_local = datetime.now(_tz())
    stamp = now_local.strftime("%Y-%m-%d")

    held_tickers = set()
    if not exit_df.empty and "Ticker" in exit_df.columns:
        held_tickers = {
            str(x).strip().upper()
            for x in exit_df["Ticker"].dropna().tolist()
            if str(x).strip()
        }

    buy_candidates_count = int(len(unheld_buy_df)) if 'unheld_buy_df' in locals() else 0

    audit = {
        "generated_at_local": now_local.strftime("%d/%m/%Y %H:%M:%S"),
        "watchlist_csv": watchlist_csv,
        "holdings_xlsx": holdings_xlsx,
        "holdings_sheet": holdings_sheet,
        "data_dir": data_dir,
        "start": start,
        "end": end or "latest",
        "tickers_processed": len(monitor_df),
        "current_holdings_rows": len(current_holdings_df),
        "held_watchlist_positions": len(exit_df),
        "buy_count": buy_candidates_count,
        "blocked_count": int((monitor_df["Status"] == "BLOCKED").sum()),
        "watch_count": int((monitor_df["Status"] == "WATCH").sum()),
        "wait_count": int((monitor_df["Status"] == "WAIT").sum()),
        "exit_now_count": int((exit_df["Status"] == "EXIT NOW").sum()) if not exit_df.empty else 0,
        "git_sha": _read_git_sha(repo_root),
    }

    recon_df = _build_backtest_reconciliation(
        monitor_df=monitor_df,
        exit_df=exit_df,
        holdings_view_df=holdings_view_df,
        backtest_windows_csv="reports/latest/backtest_windows.csv",
    )

    audit["reconciliation_rows"] = len(recon_df)
    audit["reconciliation_ok"] = (
        int((recon_df["Reconciliation"] == "OK").sum())
        if not recon_df.empty and "Reconciliation" in recon_df.columns
        else 0
    )
    audit["reconciliation_mismatch"] = (
        int((recon_df["Reconciliation"] == "MISMATCH").sum())
        if not recon_df.empty and "Reconciliation" in recon_df.columns
        else 0
    )

    # Load latest commodity signals
    _commodity_df = pd.DataFrame()
    _commodity_path = Path("reports/latest/commodity_entry/commodity_entry_latest.csv")
    if _commodity_path.exists():
        try:
            _commodity_df = pd.read_csv(_commodity_path)
        except Exception:
            _commodity_df = pd.DataFrame()

    html = _build_html(monitor_df, exit_df, holdings_view_df, audit, recon_df, commodity_df=_commodity_df)
    
    html_latest = out_path / "daily_dashboard.html"
    csv_latest = out_path / "daily_dashboard_monitor.csv"
    holdings_latest = out_path / "daily_dashboard_exit_monitor.csv"
    summary_latest = out_path / "daily_dashboard_summary.json"

    html_archive = archive_path / f"daily_dashboard_{stamp}.html"
    csv_archive = archive_path / f"daily_dashboard_monitor_{stamp}.csv"
    exit_csv_archive = archive_path / f"daily_dashboard_exit_monitor_{stamp}.csv"
    summary_archive = archive_path / f"daily_dashboard_summary_{stamp}.json"

    html_latest.write_text(html, encoding="utf-8")
    html_archive.write_text(html, encoding="utf-8")

    monitor_df.to_csv(csv_latest, index=False)
    monitor_df.to_csv(csv_archive, index=False)

    exit_df.to_csv(holdings_latest, index=False)
    exit_df.to_csv(exit_csv_archive, index=False)

    summary_json = json.dumps(audit, indent=2)
    summary_latest.write_text(summary_json, encoding="utf-8")
    summary_archive.write_text(summary_json, encoding="utf-8")

    if send_email:
        subject = f"Kathleen - Daily dashboard – {now_local.strftime('%d/%m/%Y %H:%M')}"
        _send_email(subject, html)

    return {
        "html_latest": str(html_latest),
        "html_archive": str(html_archive),
        "csv_latest": str(csv_latest),
        "csv_archive": str(csv_archive),
        "exit_csv_latest": str(holdings_latest),
        "exit_csv_archive": str(exit_csv_archive),
        "json_latest": str(summary_latest),
        "json_archive": str(summary_archive),
        "audit": audit,
    }
