from __future__ import annotations

"""
Golden (weekly) regime detector.

Golden is WEEKLY and stateful (months-long regime).
Silver is entry-timing only (handled elsewhere).
Golden exits are the only exits (Silver never exits).

Key design:
- Entry conditions evaluated only when Golden is OFF.
- Exit conditions evaluated only when Golden is ON.
- golden_entry/golden_exit are state transitions.

Whipsaw vs drawdown handling (IMPORTANT):
- "Soft exits" can be gated by min duration (to reduce chattering).
- "Hard exits" must ALWAYS be allowed immediately (to prevent heavy losses).

This version includes:
  1) Live-safe quantiles (NO LOOKAHEAD): quantiles computed on rsi.shift(1)
  2) Regime persistence: golden_min_weeks_on gates SOFT exits only
  3) Trend-break exit (hard, ROBUST): configurable buffered MA40 break confirmed
     + MA40 slope down + configurable negative MACD-slope confirmation
  4) Trailing stop (hard): hybrid fixed-% and ATR stop
  5) Hard stop from entry (hard): close <= entry_close*(1 - hard_stop_pct)
  6) Fast re-entry after hard exit: limited-time weekly re-entry branch
  7) Settings audit columns to verify effective parameters
"""

from typing import List

import pandas as pd

from ..config import Settings
from ..indicators import macd, rsi_wilder
from ..indicators.atr import atr_wilder


def _rolling_or_expanding_quantile(s: pd.Series, q: float, window: int, min_periods: int) -> pd.Series:
    """Rolling quantile with an expanding fallback for early history."""
    s = s.astype(float)
    roll = s.rolling(window=window, min_periods=min_periods).quantile(q)
    exp = s.expanding(min_periods=min_periods).quantile(q)
    return roll.where(~roll.isna(), exp)


def _confirm_n_consecutive(series_bool: pd.Series, n: int) -> pd.Series:
    """
    True if current bar and previous n-1 bars are all True.
    For n=1, returns the original boolean series.
    """
    confirmed = series_bool.fillna(False).astype(bool).copy()
    for k in range(1, max(1, n)):
        confirmed = confirmed & series_bool.shift(k).fillna(False).astype(bool)
    return confirmed


def compute_golden_weekly_flags(df_w: pd.DataFrame, s: Settings) -> pd.DataFrame:
    if df_w is None or df_w.empty:
        return pd.DataFrame()

    for col in ("close", "high", "low"):
        if col not in df_w.columns:
            raise ValueError(f"Weekly dataframe missing required column '{col}'")

    close = df_w["close"].astype(float)

    # Core indicators
    rsi = rsi_wilder(close, 14)
    m = macd(close, 12, 26, 9)
    macd_fast = m["macd"].astype(float)
    macd_slope = macd_fast.diff()
    atr = atr_wilder(df_w["high"].astype(float), df_w["low"].astype(float), close, period=14)

    out = pd.DataFrame(
        {
            "close": close,
            "rsi14": rsi,
            "macd_fast": macd_fast,
            "macd_slope": macd_slope,
            "atr14": atr,
        },
        index=df_w.index,
    ).dropna()

    if out.empty or len(out) < 20:
        out["golden_entry"] = False
        out["golden_exit"] = False
        out["golden_exit_reason"] = ""
        out["golden_on"] = False
        out["silver_window"] = False
        out["rsi_q1"] = float("nan")
        out["rsi_med"] = float("nan")
        out["rsi_q3"] = float("nan")
        out["rsi_high_p"] = float("nan")
        out["high_rsi_winner"] = False
        out["audit_min_on_weeks"] = float("nan")
        out["audit_trailing_stop_pct"] = float("nan")
        out["audit_trailing_atr_mult"] = float("nan")
        out["audit_hard_stop_pct"] = float("nan")
        out["audit_ma_break_buffer"] = float("nan")
        out["audit_ma_break_confirm_weeks"] = float("nan")
        out["audit_macd_neg_confirm_weeks"] = float("nan")
        out["audit_reentry_enabled"] = float("nan")
        out["audit_reentry_window_weeks"] = float("nan")
        return out

    close_s = out["close"].astype(float)
    atr_s = out["atr14"].astype(float)
    rsi_s = out["rsi14"].astype(float)
    slope_s = out["macd_slope"].astype(float)
    slope_prev = slope_s.shift(1)
    rsi_prev = rsi_s.shift(1)

    # -------------------------
    # Rolling RSI thresholds — LIVE SAFE (NO LOOKAHEAD)
    # -------------------------
    lookback_weeks = int(getattr(s, "golden_rsi_quantile_window_weeks", 156))
    min_q_periods = int(getattr(s, "golden_rsi_quantile_min_weeks", max(20, lookback_weeks // 2)))

    rsi_hist = rsi_s.shift(1)

    rsi_q1_s = _rolling_or_expanding_quantile(rsi_hist, 0.25, lookback_weeks, min_q_periods)
    rsi_med_s = _rolling_or_expanding_quantile(rsi_hist, 0.50, lookback_weeks, min_q_periods)
    rsi_q3_s = _rolling_or_expanding_quantile(rsi_hist, 0.75, lookback_weeks, min_q_periods)

    rsi_high_q = float(getattr(s, "golden_rsi_high_quantile", 0.85))
    rsi_high_s = _rolling_or_expanding_quantile(rsi_hist, rsi_high_q, lookback_weeks, min_q_periods)

    # -------------------------
    # High-RSI winner classification
    # -------------------------
    recent = rsi_s.tail(min(lookback_weeks, len(rsi_s)))
    if not recent.empty:
        recent_q1 = float(recent.quantile(0.25))
        recent_med = float(recent.quantile(0.50))
        high_rsi_winner = (recent_q1 > 45.0) and (recent_med > 55.0)
    else:
        high_rsi_winner = False

    # -------------------------
    # Golden ENTRY
    # -------------------------
    rsi_entry_buffer = float(getattr(s, "golden_rsi_entry_buffer", 20.0))
    rsi_entry_cap = (rsi_q1_s + rsi_entry_buffer).astype(float)

    entry_strict = (slope_s > 0.0) & (slope_prev > 0.0) & (rsi_s <= rsi_entry_cap)

    relaxed_max_weeks = int(getattr(s, "golden_relaxed_recovery_weeks", 12))
    cond_q1_cross_up = (rsi_prev <= rsi_q1_s) & (rsi_s > rsi_q1_s) & (slope_s > 0.0)

    entry_relaxed = pd.Series(False, index=out.index)
    idxs = [i for i, v in enumerate(cond_q1_cross_up.values) if bool(v)]
    n = len(out)
    for i in idxs:
        j = min(i + relaxed_max_weeks, n - 1)
        med_target = rsi_med_s.iloc[i]
        if pd.notna(med_target) and (rsi_s.iloc[i : j + 1] >= med_target).any():
            entry_relaxed.iloc[i] = True

    if high_rsi_winner:
        entry_high = (slope_s > 0.0) & rsi_s.between(rsi_med_s - 5.0, rsi_q3_s, inclusive="both")
    else:
        entry_high = pd.Series(False, index=out.index)

    golden_entry_raw = (entry_strict | entry_relaxed | entry_high).fillna(False).astype(bool)

    # -------------------------
    # Golden EXIT components
    # -------------------------
    # Soft exits
    in_high_band = (rsi_s >= rsi_high_s).fillna(False)
    in_high_band_prev = (rsi_prev >= rsi_high_s).fillna(False)
    two_neg_slope_in_high = (slope_s < 0.0) & (slope_prev < 0.0) & in_high_band
    exiting_high_band = in_high_band_prev & (rsi_s < rsi_high_s)
    soft_exit_raw = (two_neg_slope_in_high | exiting_high_band).fillna(False).astype(bool)

    # Hard exits
    ma40 = close_s.rolling(40, min_periods=40).mean()
    ma40_slope = ma40.diff()

    ma_break_buffer = float(getattr(s, "golden_ma_break_buffer", 0.0))
    ma_break_confirm_weeks = int(getattr(s, "golden_ma_break_confirm_weeks", 1))
    macd_neg_confirm_weeks = int(getattr(s, "golden_trend_break_macd_neg_weeks", 1))

    below_ma40 = close_s < (ma40 * (1.0 - ma_break_buffer))
    below_ma40_confirmed = _confirm_n_consecutive(below_ma40, ma_break_confirm_weeks)

    macd_neg = slope_s < 0.0
    macd_neg_confirmed = _confirm_n_consecutive(macd_neg, macd_neg_confirm_weeks)

    trend_break = below_ma40_confirmed & (ma40_slope < 0.0) & macd_neg_confirmed

    trailing_stop_pct = float(getattr(s, "golden_trailing_stop_pct", 0.18))
    trailing_atr_mult = float(getattr(s, "golden_trailing_atr_mult", 1.4))
    hard_stop_pct = float(getattr(s, "golden_hard_stop_from_entry_pct", 0.11))

    # -------------------------
    # Re-entry controls
    # -------------------------
    enable_reentry = bool(getattr(s, "golden_enable_reentry_after_hard_exit", True))
    reentry_window_weeks = int(getattr(s, "golden_reentry_window_weeks", 8))
    reentry_require_close_above_ma40 = bool(getattr(s, "golden_reentry_require_close_above_ma40", True))
    reentry_require_rsi_above_q1 = bool(getattr(s, "golden_reentry_require_rsi_above_q1", True))
    reentry_require_positive_macd_slope = bool(getattr(s, "golden_reentry_require_positive_macd_slope", True))

    # -------------------------
    # Stateful regime
    # -------------------------
    min_on_weeks = int(getattr(s, "golden_min_weeks_on", 8))

    on_flags: List[bool] = []
    entry_flags: List[bool] = []
    exit_flags: List[bool] = []
    reasons: List[str] = []

    in_regime = False
    weeks_in_regime = 0
    entry_close = float("nan")
    peak_close = float("nan")

    last_hard_exit_reason = ""
    weeks_since_hard_exit = 999999

    for idx in out.index:
        base_entry_now = bool(golden_entry_raw.loc[idx])
        soft_exit_now = bool(soft_exit_raw.loc[idx])
        trend_break_now = bool(trend_break.loc[idx])

        close_now = float(close_s.loc[idx])
        ma40_now = float(ma40.loc[idx]) if pd.notna(ma40.loc[idx]) else float("nan")
        rsi_now = float(rsi_s.loc[idx]) if pd.notna(rsi_s.loc[idx]) else float("nan")
        rsi_q1_now = float(rsi_q1_s.loc[idx]) if pd.notna(rsi_q1_s.loc[idx]) else float("nan")
        slope_now = float(slope_s.loc[idx]) if pd.notna(slope_s.loc[idx]) else float("nan")

        did_entry = False
        did_exit = False
        reason = ""

        if not in_regime:
            weeks_in_regime = 0
            entry_close = float("nan")
            peak_close = float("nan")

            reentry_now = False
            if enable_reentry and weeks_since_hard_exit <= reentry_window_weeks:
                reentry_now = True

                if reentry_require_positive_macd_slope:
                    reentry_now = reentry_now and pd.notna(slope_now) and (slope_now > 0.0)

                if reentry_require_close_above_ma40:
                    reentry_now = reentry_now and pd.notna(ma40_now) and (close_now > ma40_now)

                if reentry_require_rsi_above_q1:
                    reentry_now = reentry_now and pd.notna(rsi_q1_now) and pd.notna(rsi_now) and (rsi_now > rsi_q1_now)

                reentry_now = reentry_now and (last_hard_exit_reason in {
                    "TRAILING_STOP",
                    "TREND_BREAK_MA40_TWO_NEG_SLOPE",
                    "HARD_STOP_FROM_ENTRY",
                })

            entry_now = base_entry_now or reentry_now

            if entry_now:
                in_regime = True
                weeks_in_regime = 1
                did_entry = True
                entry_close = close_now
                peak_close = close_now
                weeks_since_hard_exit = 999999
                last_hard_exit_reason = ""
            else:
                weeks_since_hard_exit += 1

        else:
            weeks_in_regime += 1
            peak_close = max(peak_close, close_now) if pd.notna(peak_close) else close_now

            atr_now = float(atr_s.loc[idx]) if pd.notna(atr_s.loc[idx]) else float("nan")
            pct_stop_px = peak_close * (1.0 - trailing_stop_pct) if pd.notna(peak_close) else float("nan")
            atr_stop_px = (
                peak_close - trailing_atr_mult * atr_now
                if (pd.notna(peak_close) and pd.notna(atr_now))
                else float("nan")
            )
            stop_px = (
                max(pct_stop_px, atr_stop_px)
                if (pd.notna(pct_stop_px) and pd.notna(atr_stop_px))
                else pct_stop_px
            )
            trail_stop_now = pd.notna(stop_px) and (close_now <= stop_px)

            hard_stop_now = pd.notna(entry_close) and (close_now <= entry_close * (1.0 - hard_stop_pct))

            hard_exit_now = bool(trend_break_now or trail_stop_now or hard_stop_now)

            if hard_exit_now:
                in_regime = False
                did_exit = True
                weeks_in_regime = 0
                weeks_since_hard_exit = 0

                if hard_stop_now:
                    reason = "HARD_STOP_FROM_ENTRY"
                elif trail_stop_now:
                    reason = "TRAILING_STOP"
                elif trend_break_now:
                    reason = "TREND_BREAK_MA40_TWO_NEG_SLOPE"
                else:
                    reason = "HARD_EXIT"

                last_hard_exit_reason = reason

            elif soft_exit_now and (weeks_in_regime >= max(1, min_on_weeks)):
                in_regime = False
                did_exit = True
                weeks_in_regime = 0
                weeks_since_hard_exit = 999999
                last_hard_exit_reason = ""

                if bool(two_neg_slope_in_high.loc[idx]):
                    reason = "TWO_NEG_MACD_SLOPE_IN_HIGH_BAND"
                elif bool(exiting_high_band.loc[idx]):
                    reason = "RSI_BAND_DROP"
                else:
                    reason = "SOFT_EXIT"

        on_flags.append(in_regime)
        entry_flags.append(did_entry)
        exit_flags.append(did_exit)
        reasons.append(reason)

    out["golden_entry"] = pd.Series(entry_flags, index=out.index).astype(bool)
    out["golden_exit"] = pd.Series(exit_flags, index=out.index).astype(bool)
    out["golden_exit_reason"] = pd.Series(reasons, index=out.index, dtype="object")
    out["golden_on"] = pd.Series(on_flags, index=out.index).astype(bool)

    # -------------------------
    # Silver Window (weekly gate)
    # -------------------------
    if high_rsi_winner:
        silver_window = slope_s > 0.0
    else:
        silver_window = (slope_s > 0.02) & rsi_s.between(40.0, 55.0, inclusive="both")

    out["silver_window"] = (silver_window.fillna(False) & out["golden_on"]).astype(bool)

    # Rolling thresholds for audit
    out["rsi_q1"] = rsi_q1_s.astype(float)
    out["rsi_med"] = rsi_med_s.astype(float)
    out["rsi_q3"] = rsi_q3_s.astype(float)
    out["rsi_high_p"] = rsi_high_s.astype(float)
    out["high_rsi_winner"] = bool(high_rsi_winner)

    # Settings audit columns
    out["audit_min_on_weeks"] = float(min_on_weeks)
    out["audit_trailing_stop_pct"] = float(trailing_stop_pct)
    out["audit_trailing_atr_mult"] = float(trailing_atr_mult)
    out["audit_hard_stop_pct"] = float(hard_stop_pct)
    out["audit_ma_break_buffer"] = float(ma_break_buffer)
    out["audit_ma_break_confirm_weeks"] = float(ma_break_confirm_weeks)
    out["audit_macd_neg_confirm_weeks"] = float(macd_neg_confirm_weeks)
    out["audit_reentry_enabled"] = float(enable_reentry)
    out["audit_reentry_window_weeks"] = float(reentry_window_weeks)

    return out


def compute_golden_weekly(df_w: pd.DataFrame, s: Settings) -> dict:
    """
    Backwards-compatible wrapper expected by parts of the repo.
    """
    flags = compute_golden_weekly_flags(df_w, s)
    if flags is None or flags.empty:
        return {
            "golden_on": False,
            "golden_exit": False,
            "silver_window": False,
            "golden_entry": False,
            "weekly_rsi": float("nan"),
            "weekly_macd_slope": float("nan"),
            "rsi_q1": float("nan"),
            "rsi_med": float("nan"),
            "rsi_q3": float("nan"),
            "rsi_high_p": float("nan"),
            "high_rsi_winner": False,
            "golden_exit_reason": "",
        }

    last = flags.iloc[-1]
    return {
        "golden_on": bool(last["golden_on"]),
        "golden_exit": bool(last["golden_exit"]),
        "silver_window": bool(last["silver_window"]),
        "golden_entry": bool(last["golden_entry"]),
        "weekly_rsi": float(last["rsi14"]),
        "weekly_macd_slope": float(last["macd_slope"]),
        "rsi_q1": float(last["rsi_q1"]),
        "rsi_med": float(last["rsi_med"]),
        "rsi_q3": float(last["rsi_q3"]),
        "rsi_high_p": float(last["rsi_high_p"]),
        "high_rsi_winner": bool(last["high_rsi_winner"]),
        "golden_exit_reason": str(last.get("golden_exit_reason", "")),
    }
