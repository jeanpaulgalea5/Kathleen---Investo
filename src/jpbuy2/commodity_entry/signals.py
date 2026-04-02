from __future__ import annotations

from typing import Any

import pandas as pd

from .config import CommodityEntrySettings


def _normalise_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index, errors="coerce")
    return df.sort_index()


def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    avg_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_up / avg_down.replace(0.0, pd.NA)
    return 100.0 - (100.0 / (1.0 + rs))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _macd_hist(close: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line - signal_line


def _rolling_quantile(series: pd.Series, lookback: int, q: float, min_obs: int | None = None) -> pd.Series:
    if min_obs is None:
        min_obs = max(120, lookback // 3)
    return series.shift(1).rolling(lookback, min_periods=min_obs).quantile(q)


def _rolling_min_max_cap(series: pd.Series, lookback: int, frac: float, min_obs: int | None = None) -> pd.Series:
    if min_obs is None:
        min_obs = max(52, lookback // 3)
    lo = series.shift(1).rolling(lookback, min_periods=min_obs).min()
    hi = series.shift(1).rolling(lookback, min_periods=min_obs).max()
    return lo + frac * (hi - lo)


def _resolve_profile(
    commodity_type: str,
    s: CommodityEntrySettings,
    profile: dict | None = None,
) -> dict:
    if profile is not None:
        return profile

    ctype = str(commodity_type or "generic").strip().lower()
    base = dict(s.default_profile)
    base.update(s.per_type_profiles.get(ctype, {}))
    return base


def _to_monthly(df_d: pd.DataFrame) -> pd.DataFrame:
    if df_d.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=df_d.index)
    out["close"] = df_d["close"].astype(float)
    return out.resample("ME").last().dropna(subset=["close"])


def _to_weekly(df_d: pd.DataFrame) -> pd.DataFrame:
    if df_d.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=df_d.index)
    out["close"] = df_d["close"].astype(float)
    return out.resample("W-FRI").last().dropna(subset=["close"])


def compute_commodity_entry_features(
    df_d: pd.DataFrame,
    ticker: str,
    commodity_type: str,
    s: CommodityEntrySettings,
    profile: dict | None = None,
) -> pd.DataFrame:
    if df_d is None or df_d.empty:
        return pd.DataFrame()

    profile = _resolve_profile(commodity_type=commodity_type, s=s, profile=profile)
    df_d = _normalise_ohlc(df_d)

    if "close" not in df_d.columns:
        raise ValueError("Daily dataframe missing required column 'close'")

    lookback = int(profile.get("adaptive_lookback_days", 756))
    dd_lookback = int(profile.get("drawdown_lookback_days", 20))
    dd_entry_q = float(profile.get("drawdown_entry_quantile", 0.88))
    rsi_support_q = float(profile.get("rsi_support_quantile", 0.35))
    require_bullish_day = bool(profile.get("require_bullish_day", True))
    deep_drawdown_min = float(profile.get("deep_drawdown_min", 0.10))
    daily_rsi_cap = float(profile.get("daily_rsi_cap", 45.0))
    monthly_ma_period = int(profile.get("monthly_ma_period", 10))
    weekly_ma_period = int(profile.get("weekly_ma_period", 40))
    weekly_ma_slope_weeks = int(profile.get("weekly_ma_slope_weeks", 4))
    strong_buy_extra_drawdown = float(profile.get("strong_buy_extra_drawdown", 0.04))

    out = pd.DataFrame(index=df_d.index)
    out["close"] = df_d["close"].astype(float)

    out["rsi14"] = _rsi_wilder(out["close"], s.rsi_period)
    out["macd_hist"] = _macd_hist(out["close"], s.macd_fast, s.macd_slow, s.macd_signal)
    out["macd_improving"] = out["macd_hist"] > out["macd_hist"].shift(1)
    out["macd_slope"] = out["macd_hist"].diff()
    out["macd_slope_cross_up"] = (out["macd_slope"] > 0) & (out["macd_slope"].shift(1) <= 0)
    out["bullish_day"] = out["close"] > out["close"].shift(1)

    rolling_high = out["close"].rolling(dd_lookback, min_periods=dd_lookback).max()
    out["drawdown_lookback"] = (rolling_high - out["close"]) / rolling_high

    positive_dd = out["drawdown_lookback"].where(out["drawdown_lookback"] > 0)
    out["drawdown_entry_th"] = _rolling_quantile(positive_dd, lookback=lookback, q=dd_entry_q)
    out["rsi_support_th"] = _rolling_quantile(out["rsi14"], lookback=lookback, q=rsi_support_q)
    out["rsi_support_ok"] = out["rsi14"] <= out["rsi_support_th"]

    monthly = _to_monthly(df_d)
    weekly = _to_weekly(df_d)

    monthly_close = monthly["close"] if not monthly.empty else pd.Series(dtype=float)
    monthly_macd = _macd_hist(monthly_close, s.macd_fast, s.macd_slow, s.macd_signal)
    monthly_ma = monthly_close.rolling(monthly_ma_period, min_periods=monthly_ma_period).mean()

    weekly_close = weekly["close"] if not weekly.empty else pd.Series(dtype=float)
    weekly_ma = weekly_close.rolling(weekly_ma_period, min_periods=weekly_ma_period).mean()

    monthly_macd_pos = (monthly_macd > 0).reindex(out.index, method="ffill").fillna(False)
    monthly_above_ma = (monthly_close > monthly_ma).reindex(out.index, method="ffill").fillna(False)
    weekly_ma_slope_up = (weekly_ma > weekly_ma.shift(weekly_ma_slope_weeks)).reindex(out.index, method="ffill").fillna(False)

    regime_score = monthly_macd_pos.astype(int) + monthly_above_ma.astype(int) + weekly_ma_slope_up.astype(int)
    out["structural_bull"] = regime_score >= 2

    weekly_rsi = _rsi_wilder(weekly["close"], s.rsi_period) if not weekly.empty else pd.Series(dtype=float)
    weekly_macd_hist = _macd_hist(weekly["close"], s.macd_fast, s.macd_slow, s.macd_signal) if not weekly.empty else pd.Series(dtype=float)

    weekly_rsi_cap = _rolling_min_max_cap(weekly_rsi, lookback=156, frac=0.70)
    weekly_macd_improving_2w = (
        (weekly_macd_hist > weekly_macd_hist.shift(1))
        & (weekly_macd_hist.shift(1) > weekly_macd_hist.shift(2))
    )

    weekly_rolling_high_12w = weekly["close"].rolling(12, min_periods=12).max() if not weekly.empty else pd.Series(dtype=float)
    weekly_drawdown_12w = ((weekly_rolling_high_12w - weekly["close"]) / weekly_rolling_high_12w) if not weekly.empty else pd.Series(dtype=float)

    weekly_gate_gold = (
        ((weekly_macd_hist > 0) | weekly_macd_improving_2w.fillna(False))
        & (weekly_rsi <= weekly_rsi_cap)
        & (weekly_drawdown_12w >= 0.03)
    )

    out["weekly_gate_gold"] = weekly_gate_gold.reindex(out.index, method="ffill").fillna(False)

    # ── Silver-specific weekly gate ──────────────────────────────────────────
    # Silver's weekly MACD can stay negative for extended periods even during
    # structural bull phases (high industrial-demand volatility).  We therefore
    # accept any N-week MACD improvement OR positive MACD, with a more lenient
    # RSI cap and a deeper minimum drawdown requirement.
    _silver_weekly_rsi_cap_frac = float(profile.get("silver_weekly_rsi_cap_frac", 0.70))
    _silver_weekly_macd_lookback = int(profile.get("silver_weekly_macd_lookback", 2))
    _silver_weekly_deep_dip_min = float(profile.get("silver_weekly_deep_dip_min", 0.12))

    weekly_rsi_cap_silver = _rolling_min_max_cap(weekly_rsi, lookback=156, frac=_silver_weekly_rsi_cap_frac)
    weekly_macd_improving_silver = weekly_macd_hist > weekly_macd_hist.shift(_silver_weekly_macd_lookback)
    weekly_deep_dip_silver = weekly_drawdown_12w >= _silver_weekly_deep_dip_min

    weekly_gate_silver = (
        ((weekly_macd_hist > 0) | weekly_macd_improving_silver.fillna(False))
        & (weekly_rsi <= weekly_rsi_cap_silver)
        & weekly_deep_dip_silver
    )

    out["weekly_gate_silver"] = weekly_gate_silver.reindex(out.index, method="ffill").fillna(False)

    # ── Silver-specific turn confirmation ────────────────────────────────────
    # Silver turns faster and with more noise than gold.  We use an adaptive
    # RSI cap (wider lookback fraction) and require only that MACD hist is
    # improving vs yesterday — not the stricter slope-cross-up used by default.
    _silver_adaptive_rsi_frac = float(profile.get("silver_adaptive_rsi_cap_frac", 0.78))
    daily_rsi_cap_silver = _rolling_min_max_cap(out["rsi14"], lookback=252, frac=_silver_adaptive_rsi_frac)
    out["turn_confirm_silver"] = (
        out["macd_improving"].fillna(False)
        & (out["rsi14"] <= daily_rsi_cap_silver)
    ).fillna(False)

    bullish_ok = out["bullish_day"] if require_bullish_day else True

    out["deep_dip"] = (
        (out["drawdown_lookback"] >= out["drawdown_entry_th"])
        & (out["drawdown_lookback"] >= deep_drawdown_min)
    ).fillna(False)

    out["turn_confirm_default"] = (
        out["macd_improving"].fillna(False)
        & out["macd_improving"].shift(1).fillna(False)
        & out["macd_slope_cross_up"].fillna(False)
        & bullish_ok
        & (out["rsi14"] <= daily_rsi_cap)
    ).fillna(False)

    daily_rsi_cap_gold = _rolling_min_max_cap(out["rsi14"], lookback=252, frac=0.75)
    out["turn_confirm_gold"] = (
        out["macd_improving"].fillna(False)
        & (out["rsi14"] <= daily_rsi_cap_gold)
    ).fillna(False)

    ctype = str(profile.get("commodity_type", commodity_type or "generic")).strip().lower()
    is_gold = ctype == "gold"
    is_silver = ctype == "silver"

    # ─── GOLD BRANCH ───
    if is_gold:
        out["turn_confirm"] = out["turn_confirm_gold"]

        # Exceptional macro dip bypass:
        # When drawdown is 2x+ the historical threshold AND structural
        # bull is intact AND daily momentum is turning, bypass the
        # weekly gate. This captures fast liquidation selloffs that
        # take 2-4 weeks to show weekly MACD recovery.
        exceptional_dip = (
            out["structural_bull"].fillna(False)
            & out["deep_dip"]
            & out["turn_confirm"]
            & (out["drawdown_lookback"] >= out["drawdown_entry_th"] * 2.0)
        ).fillna(False)

        out["buy_gate"] = (
            (out["weekly_gate_gold"].fillna(False) | exceptional_dip)
            & out["deep_dip"]
            & out["turn_confirm"]
        ).fillna(False)

        out["primed"] = (
            (out["weekly_gate_gold"].fillna(False) | exceptional_dip)
            & out["deep_dip"]
            & ~out["buy_gate"]
        ).fillna(False)

    # ─── SILVER BRANCH ───
    elif is_silver:
        out["turn_confirm"] = out["turn_confirm_silver"]

        _silver_exceptional_multiplier = float(
            profile.get("silver_exceptional_dip_multiplier", 1.8)
        )

        # Silver exceptional dip bypass:
        # Lower multiplier than gold (1.8x vs 2.0x) because silver swings are
        # structurally larger.  Does NOT require structural_bull — deeply
        # oversold silver at extreme drawdown is a valid standalone entry even
        # when the weekly trend has not yet turned.
        exceptional_dip_silver = (
            out["deep_dip"]
            & out["turn_confirm"]
            & (out["drawdown_lookback"] >= out["drawdown_entry_th"] * _silver_exceptional_multiplier)
        ).fillna(False)

        out["buy_gate"] = (
            (out["weekly_gate_silver"].fillna(False) | exceptional_dip_silver)
            & out["deep_dip"]
            & out["turn_confirm"]
        ).fillna(False)

        out["primed"] = (
            (out["weekly_gate_silver"].fillna(False) | exceptional_dip_silver)
            & out["deep_dip"]
            & out["macd_improving"].fillna(False)
            & ~out["buy_gate"]
        ).fillna(False)

    # ─── GENERIC BRANCH ───
    else:
        out["turn_confirm"] = out["turn_confirm_default"]
        out["buy_gate"] = (
            out["structural_bull"]
            & out["deep_dip"]
            & out["turn_confirm"]
        ).fillna(False)

        out["primed"] = (
            out["structural_bull"]
            & out["deep_dip"]
            & out["macd_improving"].fillna(False)
            & ~out["buy_gate"]
        ).fillna(False)

    out["strong_buy"] = (
        out["buy_gate"]
        & (
            (out["drawdown_lookback"] >= (out["drawdown_entry_th"] + strong_buy_extra_drawdown))
            | out["rsi_support_ok"].fillna(False)
        )
    ).fillna(False)

    out["fwd_1y_return"] = out["close"].shift(-s.forward_days_1y) / out["close"] - 1.0
    out["fwd_2y_return"] = out["close"].shift(-s.forward_days_2y) / out["close"] - 1.0
    out["fwd_3y_return"] = out["close"].shift(-s.forward_days_3y) / out["close"] - 1.0

    return out


def latest_commodity_signal(
    df_d: pd.DataFrame,
    ticker: str,
    commodity_type: str,
    s: CommodityEntrySettings,
    profile: dict | None = None,
) -> dict[str, Any]:
    feats = compute_commodity_entry_features(
        df_d=df_d,
        ticker=ticker,
        commodity_type=commodity_type,
        s=s,
        profile=profile,
    )
    if feats.empty:
        return {
            "ticker": ticker,
            "commodity_type": commodity_type,
            "signal": "WAIT",
            "reason": "No usable commodity data.",
        }

    row = feats.iloc[-1]

    if bool(row.get("strong_buy", False)):
        signal = "STRONG_BUY"
    elif bool(row.get("buy_gate", False)):
        signal = "BUY"
    elif bool(row.get("primed", False)):
        signal = "PRIMED"
    else:
        signal = "WAIT"

    reasons: list[str] = [f"commodity_type={str(commodity_type or 'generic').lower()}"]
    if bool(row.get("structural_bull", False)):
        reasons.append("structural_bull")
    if bool(row.get("weekly_gate_gold", False)):
        reasons.append("weekly_gate_gold")
    if bool(row.get("weekly_gate_silver", False)):
        reasons.append("weekly_gate_silver")
    if bool(row.get("exceptional_dip", False)):
        reasons.append("exceptional_dip_bypass")
    if bool(row.get("deep_dip", False)):
        reasons.append("deep_dip")
    if bool(row.get("turn_confirm", False)):
        reasons.append("turn_confirm")
    if signal == "PRIMED":
        reasons.append("awaiting_full_turn")
    if pd.notna(row.get("rsi14")):
        reasons.append(f"rsi14={float(row['rsi14']):.2f}")
    if pd.notna(row.get("drawdown_lookback")):
        reasons.append(f"dd={float(row['drawdown_lookback']):.3f}")
    if pd.notna(row.get("drawdown_entry_th")):
        reasons.append(f"dd_th={float(row['drawdown_entry_th']):.3f}")

    dd_now = float(row["drawdown_lookback"]) if pd.notna(row.get("drawdown_lookback")) else None
    dd_th = float(row["drawdown_entry_th"]) if pd.notna(row.get("drawdown_entry_th")) else None

    overshoot_x = None
    overshoot_label = None
    if dd_now is not None and dd_th is not None and dd_th > 0:
        overshoot_x = round(dd_now / dd_th, 2)
        if overshoot_x >= 2.0:
            overshoot_label = "exceptional"
        elif overshoot_x >= 1.5:
            overshoot_label = "strong"
        elif overshoot_x >= 1.1:
            overshoot_label = "moderate"
        else:
            overshoot_label = "mild"

    return {
        "ticker": ticker,
        "commodity_type": str(commodity_type or "generic").lower(),
        "date": str(pd.Timestamp(feats.index[-1]).date()),
        "signal": signal,
        "close": float(row["close"]),
        "rsi14": float(row["rsi14"]) if pd.notna(row["rsi14"]) else None,
        "drawdown": dd_now,
        "drawdown_entry_th": dd_th,
        "drawdown_overshoot_x": overshoot_x,
        "drawdown_overshoot_label": overshoot_label,
        "macd_hist": float(row["macd_hist"]) if pd.notna(row["macd_hist"]) else None,
        "structural_bull": bool(row.get("structural_bull", False)),
        "weekly_gate_gold": bool(row.get("weekly_gate_gold", False)),
        "weekly_gate_silver": bool(row.get("weekly_gate_silver", False)),
        "primed": bool(row.get("primed", False)),
        "reason": ", ".join(reasons),
    }
