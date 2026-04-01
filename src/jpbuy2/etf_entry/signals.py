from __future__ import annotations

from typing import Any

import pandas as pd

from .config import ETFEntrySettings


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


def _macd_hist(close: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line - signal_line


def _resolve_profile(ticker: str, s: ETFEntrySettings, profile: dict | None = None) -> dict:
    if profile is not None:
        return profile
    base = dict(s.default_profile)
    base.update(s.per_ticker_profiles.get(ticker, {}))
    return base


def _rolling_quantile(series: pd.Series, lookback: int, q: float, min_obs: int | None = None) -> pd.Series:
    if min_obs is None:
        min_obs = max(80, lookback // 3)
    return series.shift(1).rolling(lookback, min_periods=min_obs).quantile(q)


def compute_etf_entry_features(
    df_d: pd.DataFrame,
    ticker: str,
    s: ETFEntrySettings,
    profile: dict | None = None,
) -> pd.DataFrame:
    if df_d is None or df_d.empty:
        return pd.DataFrame()

    profile = _resolve_profile(ticker=ticker, s=s, profile=profile)
    df_d = _normalise_ohlc(df_d)

    if "close" not in df_d.columns:
        raise ValueError("Daily dataframe missing required column 'close'")

    lookback = int(profile.get("adaptive_lookback_days", s.adaptive_lookback_days))
    dd_entry_q = float(profile.get("drawdown_entry_quantile", s.drawdown_entry_quantile))
    rsi_support_q = float(profile.get("rsi_support_quantile", s.rsi_support_quantile))
    require_bullish_day = bool(profile.get("require_bullish_day", True))
    grace_days = int(profile.get("drawdown_grace_days", s.drawdown_grace_days))
    grace_buy_min_ratio = float(profile.get("grace_buy_min_ratio", s.grace_buy_min_ratio))
    primed_min_ratio = float(profile.get("primed_min_ratio", s.primed_min_ratio))
    buy_window_days = int(profile.get("buy_window_days", s.buy_window_days))

    out = pd.DataFrame(index=df_d.index)
    out["close"] = df_d["close"].astype(float)

    out["rsi14"] = _rsi_wilder(out["close"], s.rsi_period)
    out["macd_hist"] = _macd_hist(
        out["close"],
        fast=s.macd_fast,
        slow=s.macd_slow,
        signal=s.macd_signal,
    )
    out["macd_improving"] = out["macd_hist"] > out["macd_hist"].shift(1)
    out["macd_slope"] = out["macd_hist"].diff()
    out["macd_slope_cross_up"] = (out["macd_slope"] > 0) & (out["macd_slope"].shift(1) <= 0)
    out["bullish_day"] = out["close"] > out["close"].shift(1)

    rolling_high = out["close"].rolling(s.drawdown_lookback_days, min_periods=s.drawdown_lookback_days).max()
    out["drawdown_20d"] = (rolling_high - out["close"]) / rolling_high
    positive_dd = out["drawdown_20d"].where(out["drawdown_20d"] > 0)
    out["drawdown_entry_th"] = _rolling_quantile(positive_dd, lookback=lookback, q=dd_entry_q)

    # Grace window: peak drawdown over the last N bars. Allows the MACD cross-up
    # to fire a BUY even if price has partially recovered since the dip low.
    out["drawdown_peak_grace"] = (
        out["drawdown_20d"].rolling(grace_days, min_periods=1).max()
    )
    out["drawdown_gate_live"] = out["drawdown_20d"] >= out["drawdown_entry_th"]
    # Grace gate: peak was above threshold recently AND current drawdown still
    # meaningful (>= grace_buy_min_ratio * threshold). Prevents buying into a healed dip.
    out["drawdown_gate_grace"] = (
        out["drawdown_peak_grace"] >= out["drawdown_entry_th"]
    ) & (
        out["drawdown_20d"] >= out["drawdown_entry_th"] * grace_buy_min_ratio
    )
    out["drawdown_gate"] = out["drawdown_gate_live"] | out["drawdown_gate_grace"]

    # RSI is supportive only; not a hard veto.
    out["rsi_support_th"] = _rolling_quantile(out["rsi14"], lookback=lookback, q=rsi_support_q)
    out["rsi_support_ok"] = out["rsi14"] <= out["rsi_support_th"]

    bullish_ok = out["bullish_day"] if require_bullish_day else True

    out["buy_gate"] = (
        out["drawdown_gate"].fillna(False)
        & out["macd_slope_cross_up"].fillna(False)
        & bullish_ok
    ).fillna(False)

    # Track whether this entry used the grace window (price partially recovered)
    out["grace_entry"] = (
        out["buy_gate"]
        & out["drawdown_gate_grace"].fillna(False)
        & ~out["drawdown_gate_live"].fillna(False)
    ).fillna(False)

    out["signal_lane"] = "DRAWDOWN_MACD_TURN"

    out["strong_buy"] = (
        out["buy_gate"]
        & (
            (out["drawdown_20d"] >= _rolling_quantile(positive_dd, lookback=lookback, q=0.85))
            | out["rsi_support_ok"].fillna(False)
        )
    ).fillna(False)

    # Buy window: once a BUY fires, keep the signal open while the dip is still
    # active (drawdown_gate_live = True, i.e. overshoot >= 1x). The signal closes
    # automatically when the drawdown heals back below the threshold.
    # buy_window_days is a safety cap — the primary expiry is drawdown_gate_live.
    out["buy_signal_recent"] = (
        out["buy_gate"].rolling(buy_window_days, min_periods=1).max().astype(bool)
    )
    out["buy_window"] = (
        out["buy_signal_recent"]
        & out["drawdown_gate_live"].fillna(False)
        & ~out["buy_gate"]  # buy_gate day itself is already the trigger
    ).fillna(False)

    # PRIMED: drawdown gate satisfied + MACD improving + no cross-up yet.
    # Expires when drawdown heals below primed_min_ratio * threshold.
    out["primed"] = (
        out["drawdown_gate"].fillna(False)
        & out["macd_improving"].fillna(False)
        & ~out["buy_gate"]
        & ~out["buy_window"]
        & (out["drawdown_20d"] >= out["drawdown_entry_th"] * primed_min_ratio)
    ).fillna(False)

    out["fwd_1y_return"] = out["close"].shift(-s.forward_days) / out["close"] - 1.0
    return out


def latest_etf_signal(
    df_d: pd.DataFrame,
    ticker: str,
    s: ETFEntrySettings,
    profile: dict | None = None,
) -> dict[str, Any]:
    feats = compute_etf_entry_features(df_d=df_d, ticker=ticker, s=s, profile=profile)
    if feats.empty:
        return {"ticker": ticker, "signal": "WAIT", "reason": "No usable ETF data."}

    row = feats.iloc[-1]

    if bool(row.get("strong_buy", False)):
        signal = "STRONG_BUY"
    elif bool(row.get("buy_gate", False)):
        signal = "BUY"
    elif bool(row.get("buy_window", False)):
        signal = "BUY"
    elif bool(row.get("primed", False)):
        signal = "PRIMED"
    else:
        signal = "WAIT"

    reasons: list[str] = []
    reasons.append("drawdown_macd_turn")
    if pd.notna(row.get("rsi14")):
        reasons.append(f"rsi14={float(row['rsi14']):.2f}")
    if pd.notna(row.get("rsi_support_th")):
        reasons.append(f"rsi_sup={float(row['rsi_support_th']):.2f}")
    if pd.notna(row.get("drawdown_20d")):
        reasons.append(f"dd20={float(row['drawdown_20d']):.3f}")
    if pd.notna(row.get("drawdown_entry_th")):
        reasons.append(f"dd_th={float(row['drawdown_entry_th']):.3f}")
    if bool(row.get("bullish_day", False)):
        reasons.append("bullish_day")
    if bool(row.get("macd_slope_cross_up", False)):
        reasons.append("macd_slope_cross_up")
    elif bool(row.get("macd_improving", False)):
        reasons.append("macd_improving")
    if signal == "PRIMED":
        reasons.append("awaiting_macd_cross")
    if bool(row.get("grace_entry", False)):
        reasons.append("grace_window_entry")
    if bool(row.get("buy_window", False)):
        reasons.append("buy_window_open")

    dd20 = float(row["drawdown_20d"]) if pd.notna(row.get("drawdown_20d")) else None
    dd_th = float(row["drawdown_entry_th"]) if pd.notna(row.get("drawdown_entry_th")) else None

    if dd20 is not None and dd_th is not None and dd_th > 0:
        overshoot_x = round(dd20 / dd_th, 2)
        if overshoot_x >= 4.0:
            overshoot_label = "exceptional"
        elif overshoot_x >= 2.5:
            overshoot_label = "strong"
        elif overshoot_x >= 1.5:
            overshoot_label = "moderate"
        else:
            overshoot_label = "mild"
    else:
        overshoot_x = None
        overshoot_label = None

    return {
        "ticker": ticker,
        "date": str(pd.Timestamp(feats.index[-1]).date()),
        "signal": signal,
        "close": float(row["close"]),
        "rsi14": float(row["rsi14"]) if pd.notna(row["rsi14"]) else None,
        "rsi_support_th": float(row["rsi_support_th"]) if pd.notna(row["rsi_support_th"]) else None,
        "drawdown_20d": dd20,
        "drawdown_entry_th": dd_th,
        "drawdown_overshoot_x": overshoot_x,
        "drawdown_overshoot_label": overshoot_label,
        "macd_hist": float(row["macd_hist"]) if pd.notna(row["macd_hist"]) else None,
        "primed": bool(row.get("primed", False)),
        "buy_window_active": bool(row.get("buy_window", False)),
        "grace_entry": bool(row.get("grace_entry", False)),
        "reason": ", ".join(reasons) if reasons else "No ETF setup.",
    }
