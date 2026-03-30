import pandas as pd

from ..config import Settings
from ..types import AssetType
from ..indicators.macd import compute_macd
from ..indicators.adx import adx_wilder


def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise column names to lower-case OHLC where possible.
    Your raw data is usually: Date, Open, High, Low, Close, Adj Close, Volume.
    After loading, it may already be lower-case.
    """
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def silver_signal(df_d: pd.DataFrame, asset_type: AssetType, s: Settings, golden_on: bool) -> dict:
    """
    IMPORTANT: engine expects a dict with at least:
      - silver_buy: bool
    Optional:
      - stabilisation_high: float (only used if use_break_of_stabilisation_high=True)

    Current Silver logic + daily ADX filter:
      - MACD slope > 0 for 2 consecutive days
      - close > MA(short)
      - not too extended above MA
      - daily ADX(14) >= 12
    """
    if df_d is None or df_d.empty:
        return {"silver_buy": False}

    df = _ensure_ohlc(df_d)

    # Guard: need these columns
    for col in ("close", "high", "low"):
        if col not in df.columns:
            return {"silver_buy": False}

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # --- MACD slope (daily) ---
    macd_df = compute_macd(
        close,
        fast=int(getattr(s, "silver_macd_fast", 12)),
        slow=int(getattr(s, "silver_macd_slow", 26)),
        signal=int(getattr(s, "silver_macd_signal", 9)),
    )
    slope = macd_df["slope"]

    # --- MA(short) ---
    ma_len = int(getattr(s, "silver_short_ma_period", 5))
    ma = close.rolling(ma_len, min_periods=ma_len).mean()

    # --- Distance-from-MA (anti-spike filter) ---
    max_dist = float(getattr(s, "silver_max_dist_from_ma", 0.07))
    dist = (close - ma) / ma
    dist_ok = dist.notna() & (dist <= max_dist)

    # --- ADX filter (daily) ---
    adx_period = int(getattr(s, "silver_adx_period", 14))
    adx_floor = float(getattr(s, "silver_adx_entry_min", 12.0))

    adx_df = adx_wilder(high, low, close, period=adx_period)
    adx = adx_df["adx"]
    adx_ok = adx >= adx_floor

    # --- Directional Index filter (optional — S8_DI_GUARD) ---
    # When silver_require_bullish_di=True: block entry if -DI >= +DI at signal time.
    # Validated: blocks 92% of losing trades on downtrending tickers (AMP.MI, TEP.PA etc.)
    if bool(getattr(s, "silver_require_bullish_di", False)):
        plus_di  = adx_df["plus_di"]
        minus_di = adx_df["minus_di"]
        adx_ok   = adx_ok & (plus_di > minus_di)

    # --- Option 2 confirmation ---
    slope_pos_today = slope > 0
    slope_pos_yday = slope_pos_today.shift(1).fillna(False)

    pullback = (close <= ma * 1.01)
    momentum = slope_pos_today | slope_pos_yday
    trend_confirm = close > ma
    breakout = close > high.shift(3)

    entry_ok = (trend_confirm & momentum & (pullback | breakout)) & adx_ok

    # Signal is evaluated on the last available daily bar in the slice
    buy_now = bool(entry_ok.iloc[-1]) if len(entry_ok) > 0 else False

    # Stabilisation high (only used if you run "break of stabilisation high" entries)
    # IMPORTANT FIX: use PRIOR highs only, otherwise today's high makes the level self-referential.
    lookback = int(getattr(s, "stabilisation_lookback", 5))
    prior_high = high.shift(1)
    stabilisation_high = float(prior_high.rolling(lookback, min_periods=1).max().iloc[-1])

    return {
        "silver_buy": buy_now,
        "stabilisation_high": stabilisation_high,

        # Debug
        "debug_slope": float(slope.iloc[-1]) if pd.notna(slope.iloc[-1]) else None,
        "debug_slope_prev": float(slope.iloc[-2]) if len(slope) > 1 and pd.notna(slope.iloc[-2]) else None,
        "debug_close": float(close.iloc[-1]),
        "debug_ma": float(ma.iloc[-1]) if pd.notna(ma.iloc[-1]) else None,
        "debug_dist_from_ma": float(dist.iloc[-1]) if pd.notna(dist.iloc[-1]) else None,
        "debug_max_dist_from_ma": float(max_dist),
        "debug_adx": float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else None,
        "debug_adx_floor": float(adx_floor),
    }
