from __future__ import annotations
from dataclasses import asdict
import pandas as pd
from .engine import Trade

def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    rows = []
    for t in trades:
        rows.append({
            "entry_date": t.entry_date,
            "entry_px": t.entry_px,
            "exit_date": t.exit_date,
            "exit_px": t.exit_px,
            "exit_reason": t.reason_exit
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ret"] = (df["exit_px"] / df["entry_px"]) - 1.0
    return df

def summary(df_trades: pd.DataFrame) -> dict:
    if df_trades is None or df_trades.empty:
        return {"trades": 0, "win_rate": None, "avg_ret": None, "cum_ret": 0.0}
    wins = (df_trades["ret"] > 0).sum()
    n = len(df_trades)
    return {
        "trades": n,
        "win_rate": float(wins / n),
        "avg_ret": float(df_trades["ret"].mean()),
        "cum_ret": float((df_trades["ret"] + 1).prod() - 1),
    }
