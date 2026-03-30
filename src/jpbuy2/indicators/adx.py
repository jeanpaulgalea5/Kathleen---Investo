import pandas as pd
import numpy as np

def adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_dm_sm = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    minus_dm_sm = minus_dm.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (plus_dm_sm / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm_sm / atr.replace(0, np.nan))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    out = pd.DataFrame({"adx": adx, "plus_di": plus_di, "minus_di": minus_di})
    return out.bfill()
