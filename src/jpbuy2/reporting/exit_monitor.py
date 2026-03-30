from __future__ import annotations

from typing import Any, Optional

import pandas as pd


QTY_TOLERANCE = 1e-6


def _safe_float(x: Any) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _normalise_platform(x: Any) -> str:
    return str(x or "").strip().lower()


def _normalise_ticker(x: Any) -> str:
    return str(x or "").strip().upper()

def _normalise_currency(x: Any) -> str:
    c = str(x or "").strip().upper()
    mapping = {
        "EURO": "EUR",
        "EUR": "EUR",
        "USDOLLAR": "USD",
        "US DOLLAR": "USD",
        "USD": "USD",
        "GBPSTERLING": "GBP",
        "POUND": "GBP",
        "GBP": "GBP",
    }
    return mapping.get(c, c)
    
def _pick_native_cost_and_ccy(row: pd.Series) -> tuple[Optional[float], str]:
    eur = _safe_float(row.get("NetCost_EUR"))
    fc = _safe_float(row.get("NetCost_FC"))
    ccy = _normalise_currency(row.get("Currency", ""))

    # For non-EUR holdings, prefer foreign-currency cost so that
    # cost and live market price are compared like-for-like.
    if ccy and ccy != "EUR":
        if fc is not None and abs(fc) > 0:
            return fc, ccy
        if eur is not None and abs(eur) > 0:
            return eur, "EUR"
        return None, ccy

    # For EUR holdings, use EUR cost first.
    if eur is not None and abs(eur) > 0:
        return eur, "EUR"

    if fc is not None and abs(fc) > 0:
        return fc, ccy or "EUR"

    return None, ccy or "EUR"

def _build_open_position_snapshot(txn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build current open-position holdings by Code + Platform using average-cost basis.

    Important:
    - Sold rows must reduce quantity and reduce cost basis by average cost,
      not by sale proceeds.
    - If a position goes flat, cost basis resets to zero.
    - A later rebuy starts a fresh position.
    """
    df = txn_df.copy()

    if "Date" in df.columns:
        df["_sort_dt"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        df["_sort_dt"] = pd.NaT

    df["_row_order"] = range(len(df))
    df = df.sort_values(["Code", "Platform", "_sort_dt", "_row_order"]).copy()

    rows: list[dict[str, Any]] = []

    for (code, platform), g in df.groupby(["Code", "Platform"], sort=False):
        qty_open = 0.0
        cost_eur_open = 0.0
        cost_fc_open = 0.0

        last_name = ""
        last_type = ""
        last_ccy = ""

        for _, r in g.iterrows():
            status = str(r.get("Status", "") or "").strip().lower()
            qty = _safe_float(r.get("Quantity")) or 0.0
            total_eur = _safe_float(r.get("TotalCost_EUR")) or 0.0
            total_fc = _safe_float(r.get("TotalCost_FC")) or 0.0
            charges = _safe_float(r.get("Charges")) or 0.0
            ccy = _normalise_currency(r.get("Currency", ""))

            last_name = str(r.get("Name", "") or "").strip()
            last_type = str(r.get("Type", "") or "").strip()
            last_ccy = ccy or last_ccy

            if qty <= 0:
                continue

            if status == "bought":
                # Include buy-side charges in cost basis
                qty_open += qty
                if ccy and ccy != "EUR":
                    cost_fc_open += total_fc + charges
                    cost_eur_open += total_eur
                else:
                    cost_eur_open += total_eur + charges
                    cost_fc_open += total_fc

            elif status == "sold":
                if qty_open <= QTY_TOLERANCE:
                    # Nothing open to reduce; ignore for open-position snapshot
                    qty_open = 0.0
                    cost_eur_open = 0.0
                    cost_fc_open = 0.0
                    continue

                sell_qty = min(qty, qty_open)

                avg_cost_eur = (cost_eur_open / qty_open) if qty_open > QTY_TOLERANCE else 0.0
                avg_cost_fc = (cost_fc_open / qty_open) if qty_open > QTY_TOLERANCE else 0.0

                # Reduce remaining open cost basis by average cost, NOT by sale proceeds
                cost_eur_open -= avg_cost_eur * sell_qty
                cost_fc_open -= avg_cost_fc * sell_qty
                qty_open -= sell_qty

                # If flat (or effectively flat), reset cleanly so a future rebuy starts fresh
                if qty_open <= QTY_TOLERANCE:
                    qty_open = 0.0
                    cost_eur_open = 0.0
                    cost_fc_open = 0.0

        if qty_open > QTY_TOLERANCE:
            rows.append(
                {
                    "Ticker": _normalise_ticker(code),
                    "Platform": _normalise_platform(platform),
                    "Name": last_name,
                    "Type": last_type,
                    "Quantity": qty_open,
                    "Currency": last_ccy or "EUR",
                    "NetCost_EUR": cost_eur_open,
                    "NetCost_FC": cost_fc_open,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=["Ticker", "Platform", "Name", "Type", "Quantity", "Currency", "Net Cost"]
        )

    final_rows = []
    for _, r in out.iterrows():
        net_cost, ccy = _pick_native_cost_and_ccy(r)
        final_rows.append(
            {
                "Ticker": _normalise_ticker(r.get("Ticker")),
                "Platform": _normalise_platform(r.get("Platform")),
                "Name": str(r.get("Name", "") or "").strip(),
                "Type": str(r.get("Type", "") or "").strip(),
                "Quantity": float(r.get("Quantity", 0.0)),
                "Currency": ccy,
                "Net Cost": net_cost,
            }
        )

    final_df = pd.DataFrame(final_rows)
    if final_df.empty:
        return pd.DataFrame(
            columns=["Ticker", "Platform", "Name", "Type", "Quantity", "Currency", "Net Cost"]
        )

    return final_df.sort_values(["Ticker", "Platform"]).reset_index(drop=True)
    
def load_current_holdings_from_workbook(
    path: str = "Investments.xlsx",
    transactions_sheet: str = "Transactions",
) -> pd.DataFrame:
    """
    Build current holdings from Investo's Transactions sheet using OPEN-POSITION
    average-cost basis.

    Logic:
      - Bought => increase quantity and cost basis
      - Sold   => reduce quantity and reduce cost basis by average cost
      - if position goes flat, reset cost basis to zero
      - aggregate by Code + Platform
      - keep only net positive current positions
    """
    df = pd.read_excel(path, sheet_name=transactions_sheet)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"Code", "Platform", "Quantity", "Status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Transactions sheet missing required columns: {sorted(missing)}")

    if "Name" not in df.columns:
        df["Name"] = ""
    if "Type" not in df.columns:
        df["Type"] = ""
    if "Currency" not in df.columns:
        df["Currency"] = ""
    if "TotalCost_EUR" not in df.columns:
        df["TotalCost_EUR"] = None
    if "TotalCost_FC" not in df.columns:
        df["TotalCost_FC"] = None
    if "Charges" not in df.columns:
        df["Charges"] = 0.0
    if "Date" not in df.columns:
        df["Date"] = None

    df["Code"] = df["Code"].map(_normalise_ticker)
    df["Platform"] = df["Platform"].map(_normalise_platform)
    df["Status"] = df["Status"].astype(str).str.strip().str.lower()
    df["Currency"] = df["Currency"].map(_normalise_currency)

    df = df[df["Code"] != ""].copy()

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
    df["TotalCost_EUR"] = pd.to_numeric(df["TotalCost_EUR"], errors="coerce").fillna(0.0)
    df["TotalCost_FC"] = pd.to_numeric(df["TotalCost_FC"], errors="coerce").fillna(0.0)
    df["Charges"] = pd.to_numeric(df["Charges"], errors="coerce").fillna(0.0)

    out = _build_open_position_snapshot(df)

    if out.empty:
        return pd.DataFrame(
            columns=["Ticker", "Platform", "Name", "Type", "Quantity", "Currency", "Net Cost"]
        )

    return out


def build_exit_monitor(current_holdings_df: pd.DataFrame, monitor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match derived current holdings to watchlist monitor rows.
    Only currently held watchlist names are shown.
    """
    if current_holdings_df is None or current_holdings_df.empty:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Platform",
                "Name",
                "Quantity",
                "Currency",
                "Net Cost",
                "Price",
                "1D %",
                "Golden",
                "Silver",
                "Exit Signal",
                "Status",
            ]
        )

    if monitor_df is None or monitor_df.empty:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Platform",
                "Name",
                "Quantity",
                "Currency",
                "Net Cost",
                "Price",
                "1D %",
                "Golden",
                "Silver",
                "Exit Signal",
                "Status",
            ]
        )

    m = monitor_df.copy()
    m["_ticker_key"] = m["Ticker"].astype(str).str.strip().str.upper()
    m["_platform_key"] = m["Platform"].astype(str).str.strip().str.lower()

    h = current_holdings_df.copy()
    h["_ticker_key"] = h["Ticker"].astype(str).str.strip().str.upper()
    h["_platform_key"] = h["Platform"].astype(str).str.strip().str.lower()

    merged = h.merge(
        m,
        how="inner",
        on=["_ticker_key", "_platform_key"],
        suffixes=("_hold", "_mon"),
    )

    rows: list[dict[str, Any]] = []
    for _, r in merged.iterrows():
        golden = str(r.get("Golden", "")).strip().upper()
        latest_exit_reason = str(r.get("Latest Exit Reason", "") or "").strip()

        if golden == "OFF":
            status = "EXIT NOW"
        elif latest_exit_reason:
            status = "WATCH EXIT"
        else:
            status = "HOLD"

        rows.append(
            {
                "Ticker": str(r.get("Ticker_hold", "") or r.get("Ticker_mon", "")).strip(),
                "Platform": str(r.get("Platform_hold", "")).strip(),
                "Name": str(r.get("Name_mon", "") or r.get("Name_hold", "")).strip(),
                "Quantity": _safe_float(r.get("Quantity")) or 0.0,
                "Currency": str(r.get("Currency_hold", "")).strip(),
                "Net Cost": _safe_float(r.get("Net Cost")),
                "Price": r.get("Price", ""),
                "1D %": r.get("1D %", ""),
                "Golden": r.get("Golden", ""),
                "Silver": r.get("Silver", ""),
                "Exit Signal": latest_exit_reason,
                "Status": status,
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    status_order = {"EXIT NOW": 0, "WATCH EXIT": 1, "HOLD": 2}
    out["_sort"] = out["Status"].map(status_order).fillna(9)
    out = out.sort_values(["_sort", "Ticker", "Platform"]).drop(columns=["_sort"]).reset_index(drop=True)

    return out
