"""
strategy_select.py
------------------
Adaptive per-ticker strategy selector for Investo.

HOW IT WORKS
============
1. On first call for a ticker, runs all configured strategy profiles against
   the full available history and picks the one with the highest cumulative return.
2. The result is cached to data/strategy_cache/<TICKER>.json
3. Cache is auto-invalidated when the strategy fingerprint changes
   (profiles, params, or hybrid selection constants).
4. Any ticker — new or existing — is handled automatically.
   No hardcoded map. No manual maintenance.

STRATEGY PROFILES
=================
S0_BASE         min_on=8,  atr=1.4                       (baseline)
S1_FAST         min_on=4,  atr=1.0                       (choppy / mean-reverting)
S2_HOLD12       min_on=12, atr=1.4                       (medium hold)
S3_HOLD16       min_on=16, atr=1.4                       (secular compounder)
S4_WIDEATR      min_on=8,  atr=2.2                       (volatile trending)
S5_COMBO        min_on=12, atr=2.0, ma_confirm=2         (momentum + wide trail)
S6_TIGHT        min_on=6,  atr=1.0, hard_stop=7%         (tight risk)
S2_HOLD12_BUF   S2_HOLD12 + rsi_buf=30, rsi_high_q=0.92 (medium hold + BUF)
S3_HOLD16_BUF   S3_HOLD16 + rsi_buf=30, rsi_high_q=0.92 (secular + BUF)
S5_COMBO_BUF    S5_COMBO  + rsi_buf=30, rsi_high_q=0.92 (momentum + BUF)
S8_DI_GUARD     min_on=4,  atr=1.0, require +DI>-DI at entry (downtrend guard)
S8_DI_GUARD_BUF S8_DI_GUARD + rsi_buf=30, rsi_high_q=0.92
S11_VOLATILE_FAST min_on=3, atr=0.9, hard_stop=6%, max_dist_ma=5% (noisy volatile)
S12_RECOVERY    min_on=6,  atr=1.6, rsi_buf=25, adx_min=18 (low-momentum recovery)

USAGE
=====
    from jpbuy2.strategy_select import settings_adaptive

    s = settings_adaptive(
        ticker   = "MSFT",
        df_d     = df_daily,      # full daily DataFrame (pre-loaded)
        df_w     = df_weekly,     # full weekly DataFrame (pre-loaded)
        data_dir = "data",        # repo data root (for cache)
        min_trades = 6,           # skip tickers with too little history
        force    = False,         # True = ignore cache, recompute
    )
    # s is a Settings object — drop-in replacement for settings_for()
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import pandas as pd
import hashlib

from .config import Settings
from .backtest.engine import run_backtest
from .backtest.metrics import trades_to_df


# ---------------------------------------------------------------------------
# Strategy profiles (all defined relative to base — no magic numbers elsewhere)
# ---------------------------------------------------------------------------

_BASE = Settings()

# BUF parameters — widen RSI entry gate and raise overbought exit threshold.
# Applied on top of hold strategies (S2/S3/S5) to let winners run longer.
# Validated across 32 tickers: +€1,204 avg net P/L uplift per ticker on €10k/trade.
# NOT applied to S1_FAST, S4_WIDEATR, S6_TIGHT — those need fast exits.
_BUF_KWARGS = dict(
    golden_rsi_entry_buffer=30.0,   # was 20.0 — wider entry gate on trending stocks
    golden_rsi_high_quantile=0.92,  # was 0.85 — exit only at true RSI extreme
)

STRATEGY_PROFILES: dict[str, Settings] = {
    # --- Original profiles (unchanged) ---
    "S0_BASE":        Settings(),
    "S1_FAST":        replace(_BASE, golden_min_weeks_on=4,  golden_trailing_atr_mult=1.0,
                               golden_ma_break_buffer=0.015,  # ← PHASE 1 ADDED
                               golden_hard_stop_from_entry_pct=0.06),  # ← PHASE 1 ADDED
    "S2_HOLD12":      replace(_BASE, golden_min_weeks_on=12, golden_trailing_atr_mult=1.4),
    "S3_HOLD16":      replace(_BASE, golden_min_weeks_on=16, golden_trailing_atr_mult=1.4),
    "S4_WIDEATR":     replace(_BASE, golden_min_weeks_on=8,  golden_trailing_atr_mult=2.2),
    "S5_COMBO":       replace(_BASE, golden_min_weeks_on=12, golden_trailing_atr_mult=2.0,
                               golden_ma_break_confirm_weeks=2),
    "S6_TIGHT":       replace(_BASE, golden_min_weeks_on=6,  golden_trailing_atr_mult=1.0,
                               golden_hard_stop_from_entry_pct=0.08,  # ← PHASE 1 CHANGED (was 0.07)
                               golden_ma_break_buffer=0.02),  # ← PHASE 1 ADDED
    # --- BUF variants — hold strategies with wider RSI gate + higher exit threshold ---
    # Adaptive selector picks these automatically when they outperform the plain version.
    # INTU and MAP.MC tested and stay on their plain variants (S3_HOLD16 / S5_COMBO).
    "S2_HOLD12_BUF":  replace(_BASE, golden_min_weeks_on=12, golden_trailing_atr_mult=1.4,
                               golden_ma_break_buffer=0.02,  # ← PHASE 1 ADDED
                               **_BUF_KWARGS),
    "S3_HOLD16_BUF":  replace(_BASE, golden_min_weeks_on=16, golden_trailing_atr_mult=1.4,
                               **_BUF_KWARGS),
    "S5_COMBO_BUF":   replace(_BASE, golden_min_weeks_on=12, golden_trailing_atr_mult=2.0,
                               golden_ma_break_confirm_weeks=2,
                               golden_ma_break_buffer=0.03,  # ← PHASE 1 ADDED
                               golden_hard_stop_from_entry_pct=0.065,  # ← PHASE 1 ADDED
                               **_BUF_KWARGS),
    # --- DI Guard: blocks entry when -DI >= +DI (bearish directional trend) ---
    # Targets tickers in confirmed downtrends. Validated: blocks 92% of losing
    # entries on AMP.MI, TEP.PA, HYQ.DE, NEXI.MI, NOV.DE, DSY.PA.
    "S8_DI_GUARD":     replace(_BASE, golden_min_weeks_on=4, golden_trailing_atr_mult=1.0,
                                silver_require_bullish_di=True,
                                golden_ma_break_buffer=0.02,  # ← PHASE 1 ADDED
                                golden_hard_stop_from_entry_pct=0.06),  # ← PHASE 1 ADDED
    "S8_DI_GUARD_BUF": replace(_BASE, golden_min_weeks_on=4, golden_trailing_atr_mult=1.0,
                                silver_require_bullish_di=True,
                                golden_ma_break_buffer=0.025,  # ← PHASE 1 ADDED
                                golden_hard_stop_from_entry_pct=0.06,  # ← PHASE 1 ADDED
                                **_BUF_KWARGS),
    # --- Specialised profiles ---
    # S11: noisy volatile stocks — exit faster than S1_FAST, tight dist filter
    "S11_VOLATILE_FAST": replace(_BASE, golden_min_weeks_on=3, golden_trailing_atr_mult=0.9,
                                  golden_hard_stop_from_entry_pct=0.06,
                                  silver_max_dist_from_ma=0.05),
    # S12_COMPOUNDER_HOLD: patient hold profile for strong trend / quality compounders
    "S12_COMPOUNDER_HOLD": replace(
        _BASE,
        golden_min_weeks_on=10,
        golden_trailing_stop_pct=0.17,
        golden_hard_stop_from_entry_pct=0.095,
        golden_ma_break_confirm_weeks=2,
        golden_trend_break_macd_neg_weeks=1,
    ),
    # S12: low-momentum recovery — wait for ADX confirmation, wider trail
    "S12_RECOVERY":    replace(_BASE, golden_min_weeks_on=6, golden_trailing_atr_mult=1.6,
                                golden_rsi_entry_buffer=25.0, silver_adx_entry_min=18.0,
                                golden_ma_break_buffer=0.02,  # ← PHASE 1 ADDED
                                golden_hard_stop_from_entry_pct=0.07),  # ← PHASE 1 ADDED
   # S13_QUALITY_HOLD: patient profile for quality dividend/steady compounders
   # No MA break buffer (let trends develop), wide trail, RSI protection
   "S13_QUALITY_HOLD": replace(_BASE,
       golden_min_weeks_on=14,              # Patient hold
       golden_trailing_atr_mult=1.7,        # Wide trail to let it run
       golden_hard_stop_from_entry_pct=0.065,  # Moderate downside protection
       golden_rsi_entry_buffer=30.0,        # RSI filter
       golden_rsi_high_quantile=0.92,       # High RSI exit threshold
       silver_max_dist_from_ma=0.08,        # Wider entry tolerance
    ),
}

STRATEGY_PARAMS: dict[str, dict] = {
    "S0_BASE":        {},
    "S1_FAST":        {"golden_min_weeks_on": 4,  "golden_trailing_atr_mult": 1.0,
                        "golden_ma_break_buffer": 0.015,  # ← PHASE 1 ADDED
                        "golden_hard_stop_from_entry_pct": 0.06},  # ← PHASE 1 ADDED
    "S2_HOLD12":      {"golden_min_weeks_on": 12},
    "S3_HOLD16":      {"golden_min_weeks_on": 16},
    "S4_WIDEATR":     {"golden_trailing_atr_mult": 2.2},
    "S5_COMBO":       {"golden_min_weeks_on": 12, "golden_trailing_atr_mult": 2.0,
                        "golden_ma_break_confirm_weeks": 2},
    "S6_TIGHT":       {"golden_min_weeks_on": 6,  "golden_trailing_atr_mult": 1.0,
                        "golden_hard_stop_from_entry_pct": 0.08,  # ← PHASE 1 CHANGED (was 0.07)
                        "golden_ma_break_buffer": 0.02},  # ← PHASE 1 ADDED
    "S2_HOLD12_BUF":  {"golden_min_weeks_on": 12, "golden_trailing_atr_mult": 1.4,
                        "golden_ma_break_buffer": 0.02,  # ← PHASE 1 ADDED
                        "golden_rsi_entry_buffer": 30.0, "golden_rsi_high_quantile": 0.92},
    "S3_HOLD16_BUF":  {"golden_min_weeks_on": 16, "golden_trailing_atr_mult": 1.4,
                        "golden_rsi_entry_buffer": 30.0, "golden_rsi_high_quantile": 0.92},
    "S5_COMBO_BUF":   {"golden_min_weeks_on": 12, "golden_trailing_atr_mult": 2.0,
                        "golden_ma_break_confirm_weeks": 2,
                        "golden_ma_break_buffer": 0.03,  # ← PHASE 1 ADDED
                        "golden_hard_stop_from_entry_pct": 0.065,  # ← PHASE 1 ADDED
                        "golden_rsi_entry_buffer": 30.0, "golden_rsi_high_quantile": 0.92},
    "S8_DI_GUARD":     {"golden_min_weeks_on": 4,  "golden_trailing_atr_mult": 1.0,
                         "silver_require_bullish_di": True,
                         "golden_ma_break_buffer": 0.02,  # ← PHASE 1 ADDED
                         "golden_hard_stop_from_entry_pct": 0.06},  # ← PHASE 1 ADDED
    "S8_DI_GUARD_BUF": {"golden_min_weeks_on": 4,  "golden_trailing_atr_mult": 1.0,
                         "silver_require_bullish_di": True,
                         "golden_ma_break_buffer": 0.025,  # ← PHASE 1 ADDED
                         "golden_hard_stop_from_entry_pct": 0.06,  # ← PHASE 1 ADDED
                         "golden_rsi_entry_buffer": 30.0, "golden_rsi_high_quantile": 0.92},
    "S11_VOLATILE_FAST": {"golden_min_weeks_on": 3, "golden_trailing_atr_mult": 0.9,
                           "golden_hard_stop_from_entry_pct": 0.06,
                           "silver_max_dist_from_ma": 0.05},
     "S12_COMPOUNDER_HOLD": {
         "golden_min_weeks_on": 10,
         "golden_trailing_stop_pct": 0.17,
         "golden_hard_stop_from_entry_pct": 0.095,
         "golden_ma_break_confirm_weeks": 2,
         "golden_trend_break_macd_neg_weeks": 1,},
   
    "S12_RECOVERY":    {"golden_min_weeks_on": 6, "golden_trailing_atr_mult": 1.6,
                         "golden_rsi_entry_buffer": 25.0, "silver_adx_entry_min": 18.0,
                         "golden_ma_break_buffer": 0.02,  # ← PHASE 1 ADDED
                         "golden_hard_stop_from_entry_pct": 0.07},  # ← PHASE 1 ADDED
    "S13_QUALITY_HOLD": {
         "golden_min_weeks_on": 14,
         "golden_trailing_atr_mult": 1.7,
         "golden_hard_stop_from_entry_pct": 0.065,
         "golden_rsi_entry_buffer": 30.0,
         "golden_rsi_high_quantile": 0.92,
         "silver_max_dist_from_ma": 0.08,
     },
}



# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(ticker: str, data_dir: str) -> Path:
    cache_dir = Path(data_dir) / "strategy_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = ticker.replace("/", "_").replace("\\", "_")
    return cache_dir / f"{safe}.json"


def _daily_csv_path(ticker: str, data_dir: str) -> Path:
    safe = ticker.replace("/", "_").replace("\\", "_")
    return Path(data_dir) / "raw" / "daily" / f"{safe}.csv"


def _cache_is_valid(ticker: str, data_dir: str) -> bool:
    """
    Cache is valid only if:
      - file exists
      - it was computed under the current strategy fingerprint
    """
    path = _cache_path(ticker, data_dir)
    if not path.exists():
        return False

    try:
        with path.open() as f:
            cached = json.load(f)
        return cached.get("fingerprint") == _strategy_fingerprint()
    except Exception:
        return False


def _read_cache(ticker: str, data_dir: str) -> dict | None:
    try:
        with _cache_path(ticker, data_dir).open() as f:
            return json.load(f)
    except Exception:
        return None


def _write_cache(ticker: str, data_dir: str, payload: dict) -> None:
    try:
        with _cache_path(ticker, data_dir).open("w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass  # cache write failure is non-fatal


# ---------------------------------------------------------------------------
# Core selector
# ---------------------------------------------------------------------------

def _run_all_strategies(
    df_d: pd.DataFrame,
    df_w: pd.DataFrame,
    min_trades: int = 6,
) -> dict[str, float | None]:
    """
    Run every strategy profile and return cumulative return per strategy.
    Returns None for a strategy if it produced too few trades.
    """
    cums: dict[str, float | None] = {}
    for sname, s in STRATEGY_PROFILES.items():
        try:
            result = run_backtest(df_d, df_w, "stock", s)
            dt = trades_to_df(result["trades"])
            if dt.empty or len(dt) < min_trades:
                cums[sname] = None
                continue
            cums[sname] = float((dt["ret"] + 1).prod() - 1)
        except Exception:
            cums[sname] = None
    return cums


def _pick_best(cums: dict[str, float | None]) -> str:
    """Pick the strategy name with the highest cumulative return."""
    valid = {k: v for k, v in cums.items() if v is not None}
    if not valid:
        return "S0_BASE"
    return max(valid, key=lambda k: valid[k])


# ---------------------------------------------------------------------------
# Window selection constants
# ---------------------------------------------------------------------------
_WINDOW_4Y_DAYS = 4 * 365
_WINDOW_3Y_DAYS = 3 * 365
_WINDOW_2Y_DAYS = 2 * 365
_WINDOW_1Y_DAYS = 365

# Minimum relative uplift required for a recent-window winner to override FULL.
# Example: 0.05 = recent-window winner must beat FULL by at least 5 percentage points.
_WINDOW_OVERRIDE_MIN_DELTA = 0.05

# ---------------------------------------------------------------------------
# Cache versioning
# ---------------------------------------------------------------------------

_STRATEGY_CACHE_VERSION = "2026-04-07-v4"

def _strategy_fingerprint() -> str:
    """
    Fingerprint the current strategy universe so stale caches are invalidated
    whenever profiles or parameters change.
    """
    payload = {
       "version": _STRATEGY_CACHE_VERSION,
       "profiles": sorted(STRATEGY_PROFILES.keys()),
       "params": STRATEGY_PARAMS,
       "window_4y_days": _WINDOW_4Y_DAYS,
       "window_3y_days": _WINDOW_3Y_DAYS,
       "window_2y_days": _WINDOW_2Y_DAYS,
       "window_1y_days": _WINDOW_1Y_DAYS,
       "window_override_min_delta": _WINDOW_OVERRIDE_MIN_DELTA,
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def _run_all_strategies_full_trades(
    df_d: pd.DataFrame,
    df_w: pd.DataFrame,
    min_trades: int = 3,
) -> dict[str, pd.DataFrame | None]:
    """
    Run every strategy profile on full history and return the full trades DataFrame
    per strategy. Returns None if insufficient trades.
    """
    out: dict[str, pd.DataFrame | None] = {}

    for sname, s in STRATEGY_PROFILES.items():
        try:
            result = run_backtest(df_d, df_w, "stock", s)
            dt = trades_to_df(result["trades"])

            if dt.empty or len(dt) < min_trades:
                out[sname] = None
                continue

            dt = dt.copy()

            if "entry_date" in dt.columns:
                dt["entry_date"] = pd.to_datetime(dt["entry_date"], errors="coerce")
            if "exit_date" in dt.columns:
                dt["exit_date"] = pd.to_datetime(dt["exit_date"], errors="coerce")
            elif "exit_ts" in dt.columns:
                dt["exit_date"] = pd.to_datetime(dt["exit_ts"], errors="coerce")
            else:
                out[sname] = None
                continue

            dt = dt.dropna(subset=["exit_date"]).copy()

            if dt.empty or len(dt) < min_trades:
                out[sname] = None
                continue

            out[sname] = dt

        except Exception:
            out[sname] = None

    return out


def _cum_from_trades(dt: pd.DataFrame | None, min_trades: int = 3) -> float | None:
    """
    Compound full trade list return.
    """
    if dt is None or dt.empty or len(dt) < min_trades:
        return None
    return float((dt["ret"] + 1).prod() - 1)


def _score_trades_window(
    dt: pd.DataFrame | None,
    asof: pd.Timestamp,
    days: int,
    min_trades: int = 3,
) -> float | None:
    """
    Score only trades whose exit_date falls within the recent window.
    Technicals are still built from full-history backtest.
    """
    if dt is None or dt.empty:
        return None

    cutoff = asof - pd.Timedelta(days=days)
    sl = dt[dt["exit_date"] >= cutoff].copy()

    if sl.empty or len(sl) < min_trades:
        return None

    return float((sl["ret"] + 1).prod() - 1)


def _window_scores_from_full_trades(
    all_trades: dict[str, pd.DataFrame | None],
    asof: pd.Timestamp,
    days: int,
    min_trades: int = 3,
) -> dict[str, float | None]:
    """
    Compute recent-window compounded return from full-history trade lists.
    """
    scores: dict[str, float | None] = {}
    for sname, dt in all_trades.items():
        scores[sname] = _score_trades_window(
            dt=dt,
            asof=asof,
            days=days,
            min_trades=min_trades,
        )
    return scores

def select_strategy(
    ticker: str,
    df_d: pd.DataFrame,
    df_w: pd.DataFrame,
    data_dir: str = "data",
    min_trades: int = 6,
    force: bool = False,
) -> tuple[str, Settings, dict]:
    """
    Select the best strategy for a ticker using full-history technicals
    and recent-window scoring from the resulting trade lists.

    Candidate decision bases:
    - FULL history
    - last 4 years
    - last 3 years

    FULL remains primary.
    4Y / 3Y can override only if they are genuinely better and positive.
    """
    if not force and _cache_is_valid(ticker, data_dir):
        cached = _read_cache(ticker, data_dir)
        if cached and "best" in cached:
            best_name = cached["best"]
            s = STRATEGY_PROFILES.get(best_name, Settings())
            return best_name, s, cached.get("cums", {})

    # Ensure datetime index
    df_d = df_d.copy()
    df_d.index = pd.to_datetime(df_d.index)

    # Run all strategies once on full-history technical context
    all_trades = _run_all_strategies_full_trades(df_d, df_w, min_trades=3)
    asof = pd.Timestamp(df_d.index[-1])

    # FULL-history compounded return
    cums_full: dict[str, float | None] = {
        sname: _cum_from_trades(dt, min_trades=3)
        for sname, dt in all_trades.items()
    }

    # Recent windows scored from full-history trades
    cums_4y = _window_scores_from_full_trades(
        all_trades, asof, _WINDOW_4Y_DAYS, min_trades=3
    )
    cums_3y = _window_scores_from_full_trades(
        all_trades, asof, _WINDOW_3Y_DAYS, min_trades=3
    )
    cums_2y = _window_scores_from_full_trades(
        all_trades, asof, _WINDOW_2Y_DAYS, min_trades=2
    )
    cums_1y = _window_scores_from_full_trades(
        all_trades, asof, _WINDOW_1Y_DAYS, min_trades=1
    )

    best_full = _pick_best(cums_full)
    best_4y = _pick_best(cums_4y)
    best_3y = _pick_best(cums_3y)
    best_2y = _pick_best(cums_2y)
    best_1y = _pick_best(cums_1y)

    best_2y_score = cums_2y.get(best_2y)

    # Strict selector basis = best realised cumulative return over the last 2 years
    if best_2y_score is not None:
        selected_window = "2Y"
        best_name = best_2y
        selected_cums = cums_2y
    else:
        # No valid 2Y trade history -> neutral fallback, not FULL and not 1Y
        selected_window = "2Y"
        best_name = "S0_BASE"
        selected_cums = cums_2y

    payload = {
        "ticker": ticker,
        "best": best_name,
        "decision_window": selected_window,
        "params": STRATEGY_PARAMS.get(best_name, {}),
        "cums": {
            k: round(v * 100, 2) if v is not None else None
            for k, v in selected_cums.items()
        },
        "cums_full": {
            k: round(v * 100, 2) if v is not None else None
            for k, v in cums_full.items()
        },
        "cums_4y": {
            k: round(v * 100, 2) if v is not None else None
            for k, v in cums_4y.items()
        },
        "cums_3y": {
            k: round(v * 100, 2) if v is not None else None
            for k, v in cums_3y.items()
        },
        "cums_2y": {
            k: round(v * 100, 2) if v is not None else None
            for k, v in cums_2y.items()
        },
        "cums_1y": {
            k: round(v * 100, 2) if v is not None else None
            for k, v in cums_1y.items()
        },
        "best_full": best_full,
        "best_4y": best_4y,
        "best_3y": best_3y,
        "computed": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cache_version": _STRATEGY_CACHE_VERSION,
        "fingerprint": _strategy_fingerprint(),
    }

    _write_cache(ticker, data_dir, payload)

    return best_name, STRATEGY_PROFILES[best_name], payload["cums"]


def settings_adaptive(
    ticker: str,
    df_d: pd.DataFrame,
    df_w: pd.DataFrame,
    data_dir: str = "data",
    min_trades: int = 6,
    force: bool = False,
    verbose: bool = False,
) -> Settings:
    """
    Drop-in replacement for settings_for().
    Returns the optimal Settings for this ticker, auto-computed and cached.

    Parameters
    ----------
    ticker     : ticker string (used for cache key)
    df_d       : daily OHLCV DataFrame (already loaded)
    df_w       : weekly OHLCV DataFrame (already loaded)
    data_dir   : repo data root directory (default: "data")
    min_trades : minimum backtest trades required to trust a strategy (default: 6)
    force      : if True, ignore cache and recompute (default: False)
    verbose    : if True, print selected strategy to stdout (default: False)
    """
    best_name, s, cums = select_strategy(
        ticker=ticker, df_d=df_d, df_w=df_w,
        data_dir=data_dir, min_trades=min_trades, force=force,
    )
    if verbose:
        best_cum = cums.get(best_name)
        base_cum = cums.get("S0_BASE")
        print(
            f"[strategy_select] {ticker}: {best_name}"
            + (f" cum={best_cum:+.1f}%" if best_cum is not None else "")
            + (f" (base={base_cum:+.1f}%)" if base_cum is not None else "")
        )
    return s
