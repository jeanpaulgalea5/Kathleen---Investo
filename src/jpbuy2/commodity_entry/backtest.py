from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CommodityEntrySettings
from .signals import latest_commodity_signal


def run_commodity_backtest(
    ticker: str,
    data_path: str | Path,
    commodity_type: str = "gold",
    entry_cost_pct: float = 0.10,
):
    """
    Simple accumulation-style backtest:
    - buy when latest_commodity_signal returns BUY or STRONG_BUY
    - hold forever
    - track invested capital vs final market value
    """

    df = pd.read_csv(data_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.sort_values("date").reset_index(drop=True)

    position = 0.0
    invested = 0.0
    trades = []

    settings = CommodityEntrySettings()

    for i in range(30, len(df)):
        window = df.iloc[: i + 1].copy()

        latest = latest_commodity_signal(
            df_d=window,
            ticker=ticker,
            commodity_type=commodity_type,
            s=settings,
        )

        signal = str(latest.get("signal", "WAIT")).upper()

        price = float(window.iloc[-1]["close"])
        date = window.iloc[-1]["date"]

        if signal in {"BUY", "STRONG_BUY"}:
            qty = 1.0
            cost = price * (1 + entry_cost_pct)

            position += qty
            invested += cost

            trades.append(
                {
                    "date": date,
                    "signal": signal,
                    "price": price,
                    "cost_with_fee": cost,
                    "position": position,
                }
            )

    if position == 0:
        return {
            "ticker": ticker,
            "trades": 0,
            "invested": 0.0,
            "value": 0.0,
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "trades_log": [],
        }

    final_price = float(df.iloc[-1]["close"])
    value = position * final_price

    pnl = value - invested
    pnl_pct = pnl / invested if invested > 0 else 0.0

    return {
        "ticker": ticker,
        "trades": len(trades),
        "invested": invested,
        "value": value,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "trades_log": trades,
    }
