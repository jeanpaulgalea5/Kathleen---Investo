import pandas as pd
from ..config import Settings
from ..indicators import adx_wilder, atr_wilder

def blockers_daily(df_d: pd.DataFrame, s: Settings) -> tuple[bool, str | None]:
    """
    Returns (blocked, reason)
    """
    if len(df_d) < 5:
        return True, "Not enough daily data for blockers"

    low = df_d["low"]
    close = df_d["close"]
    high = df_d["high"]

    # 1) Lower-lows blocker: still making lower lows
    l0 = float(low.iloc[-1])
    l1 = float(low.iloc[-2])
    l2 = float(low.iloc[-3])
    if l0 < min(l1, l2):
        return True, "Lower-low still forming (price still breaking down)"

    # 2) Strong downtrend blocker: ADX high, -DI dominant, ADX rising
    adx_df = adx_wilder(high, low, close, 14)
    adx = adx_df["adx"]
    plus_di = adx_df["plus_di"]
    minus_di = adx_df["minus_di"]

    adx_t = float(adx.iloc[-1])
    di_plus_t = float(plus_di.iloc[-1])
    di_minus_t = float(minus_di.iloc[-1])

    # ADX rising over last N days?
    n = s.blocker_adx_rising_days
    if len(adx) >= n + 1:
        rising = all(float(adx.iloc[-k]) > float(adx.iloc[-k-1]) for k in range(1, n+1))
    else:
        rising = False

    if (adx_t > s.blocker_adx_threshold) and (di_minus_t > di_plus_t) and rising:
        return True, "Strong downtrend (ADX high & rising; -DI dominant)"

    # 3) Volatility still expanding on down closes
    atr = atr_wilder(high, low, close, 14)
    m = s.blocker_atr_rising_days
    if len(atr) >= m + 1:
        atr_rising = all(float(atr.iloc[-k]) > float(atr.iloc[-k-1]) for k in range(1, m+1))
    else:
        atr_rising = False

    if atr_rising and float(close.iloc[-1]) < float(close.iloc[-2]):
        return True, "Volatility still expanding on down closes (ATR rising)"

    return False, None
