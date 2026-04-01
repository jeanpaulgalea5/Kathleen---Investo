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

    if "type" not in df.columns:
        df["type"] = df["category"]

    df["category"] = df["category"].astype(str).str.upper().str.strip()
    df["type"] = df["type"].astype(str).str.lower().str.strip()
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


def get_commodity_rows(path: str | Path) -> list[dict]:
    """
    Extract commodity rows from the watchlist, including commodity_type where available.

    Expected flexible inputs:
    - category = COMMODITY
    - or type = commodity

    Optional columns:
    - commodity_type
    - name
    """
    df = load_watchlist(path)

    cmdf = df[
        (df["category"] == "COMMODITY")
        | (df["type"] == "commodity")
    ].copy()

    if cmdf.empty:
        return []

    def _pick_commodity_type(row: pd.Series) -> str:
        raw = ""

        if "commodity_type" in row.index:
            raw = str(row.get("commodity_type", "") or "").strip().lower()

        if not raw and "name" in row.index:
            raw = str(row.get("name", "") or "").strip().lower()

        if "gold" in raw:
            return "gold"
        if "silver" in raw:
            return "silver"
        if any(x in raw for x in ["copper", "aluminium", "aluminum", "zinc", "nickel"]):
            return "industrial"
        if any(x in raw for x in ["oil", "gas", "brent", "wti", "energy"]):
            return "energy"

        return "generic"

    rows: list[dict] = []
    for _, r in cmdf.iterrows():
        ticker = str(r.get("ticker", "")).strip()
        if not ticker:
            continue

        rows.append(
            {
                "ticker": ticker,
                "commodity_type": _pick_commodity_type(r),
            }
        )

    return rows
