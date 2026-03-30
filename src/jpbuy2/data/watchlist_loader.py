from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_watchlist(path: str | Path) -> pd.DataFrame:
    """
    Load watchlist file used by Investo.

    Accepts either:
    - ticker + category
    - ticker + type
    - symbol + category/type
    """
    path = Path(path)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df.columns = [c.lower().strip() for c in df.columns]

    if "ticker" not in df.columns and "symbol" not in df.columns:
        raise ValueError("Watchlist must contain 'ticker' or 'symbol' column")

    if "ticker" not in df.columns:
        df["ticker"] = df["symbol"]

    if "category" not in df.columns:
        if "type" in df.columns:
            df["category"] = df["type"]
        else:
            raise ValueError("Watchlist must contain 'category' or 'type' column")

    df["category"] = df["category"].astype(str).str.upper().str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip()

    return df


def get_etf_tickers(path: str | Path) -> list[str]:
    """
    Extract ETF tickers from the watchlist.
    """
    df = load_watchlist(path)
    etf_df = df[df["category"] == "ETF"]
    tickers = sorted(etf_df["ticker"].dropna().unique())
    return list(tickers)
