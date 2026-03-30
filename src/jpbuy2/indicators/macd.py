import pandas as pd


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Returns a DataFrame with:
      - macd: MACD line (EMA(fast) - EMA(slow))
      - signal: signal line (EMA(macd, signal))
      - hist: macd - signal
      - slope: 1-bar slope of MACD line (macd.diff())
    """
    close = close.astype(float)

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    slope = macd_line.diff()

    out = pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "hist": hist,
            "slope": slope,
        },
        index=close.index,
    )
    return out


# Backwards/alt naming compatibility (in case other modules import `macd`)
def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    return compute_macd(close=close, fast=fast, slow=slow, signal=signal)
