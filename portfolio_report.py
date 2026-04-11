#!/usr/bin/env python3
"""Investo – Valuation report + realised + YTD money-weighted return (MWR)

Transactions-sheet driven report.

Reads "Investments.xlsx" (Transactions sheet) and produces:
  - portfolio_positions.csv
  - portfolio_realised_trades.csv
  - portfolio_realised_trades_summary.csv
  - portfolio_realised_by_year.csv
  - portfolio_ytd_returns.csv
  - portfolio_report.html
  - portfolio_report_email.html

Emailing (optional):
  set SEND_VALUATION_EMAIL=true and SMTP/EMAIL env vars.

Notes
- Uses weighted-average cost basis to compute realised P/L per sell.
- YTD MWR uses XIRR on cashflows within the year, plus a start-of-year value
  (if units were held at 1 Jan) and an end-of-report value.
- If a Yahoo price is missing, valuation/MWR becomes N/A for that instrument,
  but the script does not crash.
"""

from __future__ import annotations

import math
import os
import smtplib
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None


PORTFOLIO_XLSX = Path(os.getenv("PORTFOLIO_XLSX", "Investments.xlsx"))
OUT_DIR = Path(os.getenv("PORTFOLIO_OUT_DIR", "./reports/valuation_report"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEND_VALUATION_EMAIL = os.getenv("SEND_VALUATION_EMAIL", "").strip().lower() in {"1", "true", "yes"}
UNITS_EPS = float(os.getenv("UNITS_EPS", "0.001"))
REPORT_YEAR = int(os.getenv("REPORT_YEAR", str(datetime.now().year)))

# Commodity proxy mapping aligned with daily dashboard
# XAU and XAG are sourced in USD and converted to EUR when needed.
XAU_PROXY_TICKER = os.getenv("XAU_PROXY_TICKER", "GC=F")
XAU_PROXY_CCY = os.getenv("XAU_PROXY_CCY", "USD")
XAG_PROXY_TICKER = os.getenv("XAG_PROXY_TICKER", "SI=F")
XAG_PROXY_CCY = os.getenv("XAG_PROXY_CCY", "USD")

def _now_str() -> str:
    return datetime.now().strftime("%d/%m/%Y %H:%M")


def _norm_currency(v) -> str:
    s = ("" if v is None else str(v)).strip().upper()
    if s in {"EURO", "EUR"}:
        return "EUR"
    if s in {"USD", "US DOLLAR", "US$", "$"}:
        return "USD"
    if s in {"GBP", "POUND", "POUNDS", "£"}:
        return "GBP"
    return s or "EUR"


def _price_feed(code: str, holding_ccy: str) -> tuple[str, str]:
    """
    Returns (yahoo_ticker_to_fetch, price_currency).

    Rules:
      - XAU -> fetch XAU proxy in USD and FX-convert.
      - Moneybase/EODHD-style LSE codes like 'ITRKL.XC' / 'BARCL.XC'
        map to Yahoo LSE ticker '<BASE>.L'
    """
    code_u = (code or "").strip().upper()

    if code_u == "XAU":
        return XAU_PROXY_TICKER, _norm_currency(XAU_PROXY_CCY)

    if code_u == "XAG":
        return XAG_PROXY_TICKER, _norm_currency(XAG_PROXY_CCY)

    if code_u.endswith("L.XC") and len(code_u) > 4:
        base = code_u[:-4]
        if base.endswith("L") and len(base) > 1:
            base = base[:-1]
        if base:
            return f"{base}.L", "GBP"

    return code, _norm_currency(holding_ccy)


def _adjust_yahoo_price_units(ticker: str, px: float | None) -> float | None:
    """
    Normalise Yahoo/yfinance price units.

    Yahoo reports London Stock Exchange tickers (.L) in pence.
    Convert pence to pounds.
    """
    if px is None or (isinstance(px, float) and math.isnan(px)):
        return None

    t = (ticker or "").strip().upper()
    if t.endswith(".L"):
        return float(px) / 100.0

    return float(px)


def _extract_close_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = [str(x) for x in df.columns.get_level_values(0)]
        if "Close" in lvl0:
            close = df.xs("Close", axis=1, level=0)
        elif "Adj Close" in lvl0:
            close = df.xs("Adj Close", axis=1, level=0)
        else:
            return None

        if isinstance(close, pd.DataFrame):
            if close.empty:
                return None
            s = close.iloc[:, 0]
        else:
            s = close

        s = pd.to_numeric(s, errors="coerce").dropna()
        return None if s.empty else s

    if "Close" in df.columns:
        s = pd.to_numeric(df["Close"], errors="coerce").dropna()
        return None if s.empty else s

    if "Adj Close" in df.columns:
        s = pd.to_numeric(df["Adj Close"], errors="coerce").dropna()
        return None if s.empty else s

    return None


def _safe_last_close(ticker: str) -> Optional[float]:
    if yf is None:
        return None

    t = (ticker or "").strip()
    if not t:
        return None

    try:
        df = yf.download(
            t,
            period="10d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
        )
        s = _extract_close_series(df)
        if s is not None:
            return _adjust_yahoo_price_units(t, float(s.iloc[-1]))

        df = yf.download(
            t,
            period="1mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
        )
        s = _extract_close_series(df)
        if s is not None:
            return _adjust_yahoo_price_units(t, float(s.iloc[-1]))

        h = yf.Ticker(t).history(period="1mo", auto_adjust=False)
        s = _extract_close_series(h)
        return None if s is None else _adjust_yahoo_price_units(t, float(s.iloc[-1]))

    except Exception:
        return None


def _safe_close_on_or_after(ticker: str, target: date) -> Optional[float]:
    if yf is None:
        return None

    t = (ticker or "").strip()
    if not t:
        return None

    try:
        start = pd.Timestamp(target - timedelta(days=7))
        end = pd.Timestamp(target + timedelta(days=10))

        df = yf.download(
            t,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
        )
        s = _extract_close_series(df)
        if s is not None:
            s.index = pd.to_datetime(s.index)
            s = s[s.index.date >= target]
            if not s.empty:
                return _adjust_yahoo_price_units(t, float(s.iloc[0]))

        h = yf.Ticker(t).history(start=start, end=end, auto_adjust=False)
        s = _extract_close_series(h)
        if s is not None:
            s.index = pd.to_datetime(s.index)
            s = s[s.index.date >= target]
            if not s.empty:
                return _adjust_yahoo_price_units(t, float(s.iloc[0]))

        return None

    except Exception:
        return None


def _fx_ccy_per_eur(ccy: str) -> Optional[float]:
    c = _norm_currency(ccy)
    if c == "EUR":
        return 1.0
    if c == "USD":
        return _safe_last_close("EURUSD=X")
    if c == "GBP":
        return _safe_last_close("EURGBP=X")
    return None


def _fx_ccy_per_eur_on_or_after(ccy: str, d: date) -> Optional[float]:
    c = _norm_currency(ccy)
    if c == "EUR":
        return 1.0
    if c == "USD":
        return _safe_close_on_or_after("EURUSD=X", d)
    if c == "GBP":
        return _safe_close_on_or_after("EURGBP=X", d)
    return None


def _ccy_to_eur(amount_ccy: float, ccy: str, fx_ccy_per_eur: Optional[float]) -> Optional[float]:
    c = _norm_currency(ccy)
    if amount_ccy is None or (isinstance(amount_ccy, float) and math.isnan(amount_ccy)):
        return None
    if c == "EUR":
        return float(amount_ccy)
    if fx_ccy_per_eur is None or fx_ccy_per_eur == 0:
        return None
    return float(amount_ccy) / float(fx_ccy_per_eur)


def load_transactions(xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name="Transactions", engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    required = {
        "Date",
        "Name",
        "Code",
        "Quantity",
        "Platform",
        "TotalCost_EUR",
        "Charges",
        "Currency",
        "Status",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Transactions sheet is missing columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Name"] = df["Name"].fillna("").astype(str).str.strip()
    df["Code"] = df["Code"].fillna("").astype(str).str.strip()
    df["Platform"] = df["Platform"].fillna("").astype(str).str.strip()
    df["Currency"] = df["Currency"].map(_norm_currency)
    df["Status"] = df["Status"].fillna("").astype(str).str.strip().str.lower()

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
    df["TotalCost_EUR"] = pd.to_numeric(df["TotalCost_EUR"], errors="coerce").fillna(0.0)
    df["Charges"] = pd.to_numeric(df["Charges"], errors="coerce").fillna(0.0)

    for col in ["Price_Per_Share_FC", "Price_Per_Share_EUR", "TotalCost_FC"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["Code"].astype(str).str.len() > 0].copy()
    return df


@dataclass
class BasisState:
    units: float = 0.0
    cost_eur: float = 0.0


def compute_positions_and_realised(tx: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    open_rows: List[dict] = []
    realised_trades: List[dict] = []

    for (code, platform), g in tx.groupby(["Code", "Platform"], dropna=False):
        g = g.sort_values(["Date"]).reset_index(drop=True)

        name = g["Name"].iloc[-1]
        ccy = g["Currency"].dropna().iloc[-1] if g["Currency"].dropna().size else "EUR"

        st = BasisState()

        for r in g.itertuples(index=False):
            status = str(getattr(r, "Status", "")).lower()
            dt = getattr(r, "Date", pd.NaT)
            qty = float(getattr(r, "Quantity", 0.0) or 0.0)
            if qty == 0:
                continue

            total_eur = float(getattr(r, "TotalCost_EUR", 0.0) or 0.0)
            charges = float(getattr(r, "Charges", 0.0) or 0.0)

            if status.startswith("bought"):
                st.units += qty
                st.cost_eur += total_eur + charges

            elif status.startswith("sold"):
                proceeds_eur = total_eur - charges
                avg_cost = (st.cost_eur / st.units) if st.units > 0 else 0.0
                cost_sold = avg_cost * qty
                realised = proceeds_eur - cost_sold

                st.cost_eur -= cost_sold
                st.units -= qty

                sell_px_native = None
                if hasattr(r, "Price_Per_Share_FC") and not pd.isna(getattr(r, "Price_Per_Share_FC")):
                    sell_px_native = float(getattr(r, "Price_Per_Share_FC"))
                elif hasattr(r, "TotalCost_FC") and not pd.isna(getattr(r, "TotalCost_FC")) and qty != 0:
                    sell_px_native = float(getattr(r, "TotalCost_FC")) / qty

                yr = int(pd.Timestamp(dt).year) if pd.notna(dt) else None

                realised_trades.append(
                    {
                        "Date": pd.Timestamp(dt) if pd.notna(dt) else pd.NaT,
                        "Year": yr,
                        "Investment": getattr(r, "Name", ""),
                        "Code": code,
                        "Platform": platform,
                        "Ccy": ccy,
                        "Units sold": qty,
                        "Sell price (ccy)": sell_px_native,
                        "Proceeds (EUR)": proceeds_eur,
                        "Cost basis sold (EUR)": cost_sold,
                        "Realised P/L (EUR)": realised,
                    }
                )

        if abs(st.units) > UNITS_EPS:
            open_rows.append(
                {
                    "Investment": name,
                    "Code": code,
                    "Platform": platform,
                    "Ccy": ccy,
                    "Net amount": st.units,
                    "Cost (EUR)": st.cost_eur,
                }
            )

    open_df = pd.DataFrame(open_rows)
    trades_df = pd.DataFrame(realised_trades)

    if trades_df.empty:
        realised_by_year = pd.DataFrame(columns=["Year", "Realised P/L (EUR)", "Sell trades", "Winners", "Losers"])
    else:
        trades_df["Year"] = pd.to_numeric(trades_df["Year"], errors="coerce").astype("Int64")
        min_year = int(trades_df["Year"].dropna().min())
        max_year = int(trades_df["Year"].dropna().max())
        years = list(range(min_year, max_year + 1))

        agg = (
            trades_df.groupby("Year", dropna=True)
            .agg(
                **{
                    "Realised P/L (EUR)": ("Realised P/L (EUR)", "sum"),
                    "Sell trades": ("Realised P/L (EUR)", "count"),
                    "Winners": ("Realised P/L (EUR)", lambda s: int((s > 0).sum())),
                    "Losers": ("Realised P/L (EUR)", lambda s: int((s < 0).sum())),
                }
            )
            .reset_index()
        )

        rows = []
        for y in years:
            row = agg[agg["Year"] == y]
            if row.empty:
                rows.append({"Year": y, "Realised P/L (EUR)": 0.0, "Sell trades": 0, "Winners": 0, "Losers": 0})
            else:
                rows.append(row.iloc[0].to_dict())

        realised_by_year = pd.DataFrame(rows)
        realised_by_year = pd.concat(
            [
                realised_by_year,
                pd.DataFrame(
                    [
                        {
                            "Year": "TOTAL",
                            "Realised P/L (EUR)": float(realised_by_year["Realised P/L (EUR)"].sum()),
                            "Sell trades": int(realised_by_year["Sell trades"].sum()),
                            "Winners": int(realised_by_year["Winners"].sum()),
                            "Losers": int(realised_by_year["Losers"].sum()),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    return open_df, trades_df.sort_values(["Date"]) if not trades_df.empty else trades_df, realised_by_year


def build_realised_trades_summary(tx: pd.DataFrame, realised_trades: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Year",
        "Investment",
        "Code",
        "Platform",
        "Ccy",
        "First buy",
        "First sell",
        "Last sell",
        "Sell trades",
        "Units sold",
        "Proceeds (EUR)",
        "Cost basis sold (EUR)",
        "Realised P/L (EUR)",
    ]

    if realised_trades is None or realised_trades.empty:
        return pd.DataFrame(columns=cols)

    t = tx.copy()
    t["Date"] = pd.to_datetime(t["Date"], errors="coerce")
    t["Status"] = t["Status"].fillna("").astype(str).str.lower()
    buys = t[t["Status"].str.startswith("bought")].copy()
    first_buy = (
        buys.groupby(["Code", "Platform"], dropna=False)["Date"]
        .min()
        .rename("First buy")
        .reset_index()
    )

    s = realised_trades.copy()
    s["Date"] = pd.to_datetime(s["Date"], errors="coerce")
    s["Year"] = pd.to_numeric(s["Year"], errors="coerce").astype("Int64")

    grp_cols = ["Year", "Investment", "Code", "Platform", "Ccy"]
    agg = (
        s.groupby(grp_cols, dropna=False)
        .agg(
            **{
                "First sell": ("Date", "min"),
                "Last sell": ("Date", "max"),
                "Sell trades": ("Date", "count"),
                "Units sold": ("Units sold", "sum"),
                "Proceeds (EUR)": ("Proceeds (EUR)", "sum"),
                "Cost basis sold (EUR)": ("Cost basis sold (EUR)", "sum"),
                "Realised P/L (EUR)": ("Realised P/L (EUR)", "sum"),
            }
        )
        .reset_index()
    )

    out = agg.merge(first_buy, how="left", on=["Code", "Platform"])
    out = out[cols].copy()
    out = out.sort_values(["Year", "Realised P/L (EUR)"], ascending=[True, False]).reset_index(drop=True)
    return out


def enrich_valuation(open_df: pd.DataFrame) -> pd.DataFrame:
    if open_df.empty:
        return open_df

    df = open_df.copy()

    price_tickers: List[str] = []
    price_ccys: List[str] = []
    for r in df.itertuples(index=False):
        pt, pccy = _price_feed(getattr(r, "Code"), getattr(r, "Ccy"))
        price_tickers.append(pt)
        price_ccys.append(pccy)

    df["Ccy"] = price_ccys

    fx_map: Dict[str, Optional[float]] = {}
    for ccy in sorted(set(df["Ccy"].astype(str).map(_norm_currency))):
        fx_map[ccy] = _fx_ccy_per_eur(ccy)

    last_prices, missing = [], []
    for pt in price_tickers:
        px = _safe_last_close(pt)
        last_prices.append(px)
        missing.append(px is None)

    df["Last price (ccy)"] = last_prices
    df["Price_Missing"] = missing

    df["Value (ccy)"] = df["Net amount"] * df["Last price (ccy)"]
    df.loc[df["Price_Missing"], "Value (ccy)"] = math.nan

    df["FX→EUR"] = df["Ccy"].map(lambda c: fx_map.get(_norm_currency(c)))
    df["Value (EUR)"] = df.apply(lambda r: _ccy_to_eur(r["Value (ccy)"], r["Ccy"], r["FX→EUR"]), axis=1)

    df["Unrealised (EUR)"] = df["Value (EUR)"] - df["Cost (EUR)"]
    df["Unrealised %"] = (df["Unrealised (EUR)"] / df["Cost (EUR)"].replace({0.0: math.nan})) * 100.0

    priced = df[(~df["Price_Missing"]) & df["Value (EUR)"].notna()].copy()
    total_cost = float(pd.to_numeric(priced["Cost (EUR)"], errors="coerce").sum())
    total_val = float(pd.to_numeric(priced["Value (EUR)"], errors="coerce").sum())
    total_upl = total_val - total_cost
    total_pct = (total_upl / total_cost * 100.0) if total_cost else math.nan

    unpriced_cost = float(pd.to_numeric(df[df["Price_Missing"]]["Cost (EUR)"], errors="coerce").sum())

    totals_row = {
        "Investment": "TOTALS (priced only)",
        "Code": "",
        "Platform": "",
        "Ccy": "",
        "Net amount": math.nan,
        "Cost (EUR)": total_cost,
        "Last price (ccy)": math.nan,
        "Value (ccy)": math.nan,
        "FX→EUR": math.nan,
        "Value (EUR)": total_val,
        "Unrealised (EUR)": total_upl,
        "Unrealised %": total_pct,
        "Price_Missing": False,
    }

    unpriced_row = {
        "Investment": "Unpriced cost at risk",
        "Code": "",
        "Platform": "",
        "Ccy": "",
        "Net amount": math.nan,
        "Cost (EUR)": unpriced_cost,
        "Last price (ccy)": math.nan,
        "Value (ccy)": math.nan,
        "FX→EUR": math.nan,
        "Value (EUR)": math.nan,
        "Unrealised (EUR)": math.nan,
        "Unrealised %": math.nan,
        "Price_Missing": True,
    }

    df = pd.concat([df, pd.DataFrame([totals_row, unpriced_row])], ignore_index=True)

    cols = [
        "Investment",
        "Code",
        "Platform",
        "Ccy",
        "Net amount",
        "Cost (EUR)",
        "Last price (ccy)",
        "Value (ccy)",
        "FX→EUR",
        "Value (EUR)",
        "Unrealised (EUR)",
        "Unrealised %",
        "Price_Missing",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = math.nan

    return df[cols]


def _xirr(cashflows: List[Tuple[date, float]]) -> Optional[float]:
    cashflows = [(d, float(a)) for d, a in cashflows if d is not None and not math.isnan(float(a))]
    if len(cashflows) < 2:
        return None

    amounts = [a for _, a in cashflows]
    if not (any(a < 0 for a in amounts) and any(a > 0 for a in amounts)):
        return None

    cashflows.sort(key=lambda x: x[0])
    d0 = cashflows[0][0]
    ts = [(d - d0).days / 365.0 for d, _ in cashflows]

    def _safe_exp(x: float) -> float:
        if x > 700:
            return float("inf")
        if x < -700:
            return 0.0
        return math.exp(x)

    def npv(r: float) -> float:
        if r <= -0.999999:
            return float("inf")
        lr = math.log1p(r)
        total = 0.0
        for t, (_, a) in zip(ts, cashflows):
            disc = _safe_exp(-t * lr)
            if disc == float("inf"):
                return math.copysign(float("inf"), a)
            total += a * disc
        return total

    lo = -0.9999
    hi = 10.0

    f_lo = npv(lo)
    f_hi = npv(hi)

    tries = 0
    while math.isfinite(f_lo) and math.isfinite(f_hi) and (f_lo == 0 or f_hi == 0 or (f_lo > 0) == (f_hi > 0)) and tries < 12:
        hi *= 2.0
        f_hi = npv(hi)
        tries += 1

    if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
        return None

    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    if (f_lo > 0) == (f_hi > 0):
        return None

    for _ in range(200):
        mid = (lo + hi) / 2.0
        f_mid = npv(mid)
        if not math.isfinite(f_mid):
            return None
        if abs(f_mid) < 1e-10:
            return mid
        if (f_lo > 0) == (f_mid > 0):
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
        if abs(hi - lo) < 1e-12:
            break

    return (lo + hi) / 2.0


def build_ytd_returns(tx: pd.DataFrame, realised_trades: pd.DataFrame, report_year: int) -> Tuple[pd.DataFrame, Optional[float]]:
    y0 = date(report_year, 1, 1)
    asof = datetime.now().date()

    rows: List[dict] = []
    portfolio_flows: Dict[date, float] = {}

    for (code, platform), g in tx.groupby(["Code", "Platform"], dropna=False):
        g = g.sort_values(["Date"]).reset_index(drop=True)
        name = g["Name"].iloc[-1]
        ccy = g["Currency"].dropna().iloc[-1] if g["Currency"].dropna().size else "EUR"

        # Reconstruct opening position at 1 Jan and current ending units
        # Reconstruct opening position at 1 Jan and current ending units
        units_start = 0.0
        units_end = 0.0
        opening_cost_basis_eur = 0.0
        opening_units_at_y0 = 0.0

        running_units = 0.0
        running_cost_eur = 0.0

        for r in g.itertuples(index=False):
            qty = float(getattr(r, "Quantity", 0.0) or 0.0)
            status = str(getattr(r, "Status", "")).lower()
            dt = getattr(r, "Date", pd.NaT)
            d = pd.Timestamp(dt).date() if pd.notna(dt) else None
            total_eur = float(getattr(r, "TotalCost_EUR", 0.0) or 0.0)
            charges = float(getattr(r, "Charges", 0.0) or 0.0)

            if status.startswith("bought"):
                running_units += qty
                running_cost_eur += (total_eur + charges)
            elif status.startswith("sold"):
                proceeds = float(total_eur - charges)
                if abs(running_units) > UNITS_EPS:
                    avg_cost = running_cost_eur / running_units
                    cost_sold = avg_cost * qty
                    running_units -= qty
                    running_cost_eur -= cost_sold
                else:
                    running_units -= qty

            if d is not None and d < y0:
                units_start = running_units
                opening_units_at_y0 = running_units
                opening_cost_basis_eur = running_cost_eur

        units_end = running_units

        price_ticker, price_ccy = _price_feed(code, ccy)

        if abs(units_start) <= UNITS_EPS:
            start_value_eur: Optional[float] = 0.0
        else:
            start_px = _safe_close_on_or_after(price_ticker, y0)
            start_fx = _fx_ccy_per_eur_on_or_after(price_ccy, y0)
            start_value_eur = None if start_px is None else _ccy_to_eur(units_start * start_px, price_ccy, start_fx)

        if abs(units_end) <= UNITS_EPS:
            end_value_eur: Optional[float] = 0.0
        else:
            end_px = _safe_last_close(price_ticker)
            end_fx = _fx_ccy_per_eur(price_ccy)
            end_value_eur = None if end_px is None else _ccy_to_eur(units_end * end_px, price_ccy, end_fx)

        buys_eur = 0.0
        sells_eur = 0.0
        flows: List[Tuple[date, float]] = []

        if start_value_eur is not None and math.isfinite(float(start_value_eur)) and float(start_value_eur) != 0.0:
            flows.append((y0, -float(start_value_eur)))
            portfolio_flows[y0] = portfolio_flows.get(y0, 0.0) - float(start_value_eur)

        # YTD layer tracking
        realised_ytd_eur = 0.0

        # Accounting layer for realised P/L
        accounting_open_units_remaining = float(units_start) if abs(units_start) > UNITS_EPS else 0.0
        accounting_open_cost_remaining = float(opening_cost_basis_eur) if math.isfinite(float(opening_cost_basis_eur)) else 0.0

        # Performance layer for YTD unrealised
        perf_open_units_remaining = float(units_start) if abs(units_start) > UNITS_EPS else 0.0
        opening_start_value_eur = float(start_value_eur or 0.0)
        opening_value_per_unit_eur = (
            opening_start_value_eur / perf_open_units_remaining
            if abs(perf_open_units_remaining) > UNITS_EPS
            else 0.0
        )

        current_year_buy_lots_accounting: list[dict[str, float]] = []
        current_year_buy_lots_perf: list[dict[str, float]] = []

        for r in g.itertuples(index=False):
            status = str(getattr(r, "Status", "")).lower()
            dt = getattr(r, "Date", pd.NaT)
            d = pd.Timestamp(dt).date() if pd.notna(dt) else None
            if d is None or d < y0 or d > asof:
                continue

            qty = float(getattr(r, "Quantity", 0.0) or 0.0)
            if qty == 0:
                continue

            total_eur = float(getattr(r, "TotalCost_EUR", 0.0) or 0.0)
            charges = float(getattr(r, "Charges", 0.0) or 0.0)

            if status.startswith("bought"):
                cash = float(total_eur + charges)
                buys_eur += cash
                flows.append((d, -cash))
                portfolio_flows[d] = portfolio_flows.get(d, 0.0) - cash

                current_year_buy_lots_accounting.append(
                    {"units": float(qty), "cost_eur": float(cash)}
                )
                current_year_buy_lots_perf.append(
                    {"units": float(qty), "cost_eur": float(cash)}
                )

            elif status.startswith("sold"):
                proceeds = float(total_eur - charges)
                sells_eur += proceeds
                flows.append((d, proceeds))
                portfolio_flows[d] = portfolio_flows.get(d, 0.0) + proceeds

                qty_to_match = float(qty)

                # -------- accounting realised P/L --------
                matched_accounting_cost_eur = 0.0
                qty_left_accounting = qty_to_match

                if qty_left_accounting > UNITS_EPS and accounting_open_units_remaining > UNITS_EPS:
                    used = min(accounting_open_units_remaining, qty_left_accounting)
                    avg_open_cost = (
                        accounting_open_cost_remaining / accounting_open_units_remaining
                        if accounting_open_units_remaining > UNITS_EPS else 0.0
                    )
                    matched_accounting_cost_eur += used * avg_open_cost
                    accounting_open_units_remaining -= used
                    accounting_open_cost_remaining -= used * avg_open_cost
                    qty_left_accounting -= used

                if qty_left_accounting > UNITS_EPS:
                    for lot in current_year_buy_lots_accounting:
                        lot_units = float(lot["units"])
                        if lot_units <= UNITS_EPS:
                            continue
                        used = min(lot_units, qty_left_accounting)
                        lot_cost_per_unit = float(lot["cost_eur"]) / lot_units if lot_units > UNITS_EPS else 0.0
                        matched_accounting_cost_eur += used * lot_cost_per_unit
                        lot["units"] = lot_units - used
                        lot["cost_eur"] = float(lot["cost_eur"]) - (used * lot_cost_per_unit)
                        qty_left_accounting -= used
                        if qty_left_accounting <= UNITS_EPS:
                            break

                realised_ytd_eur += (proceeds - matched_accounting_cost_eur)

                # -------- performance layer consumption for unrealised YTD --------
                qty_left_perf = qty_to_match

                if qty_left_perf > UNITS_EPS and perf_open_units_remaining > UNITS_EPS:
                    used = min(perf_open_units_remaining, qty_left_perf)
                    perf_open_units_remaining -= used
                    qty_left_perf -= used

                if qty_left_perf > UNITS_EPS:
                    for lot in current_year_buy_lots_perf:
                        lot_units = float(lot["units"])
                        if lot_units <= UNITS_EPS:
                            continue
                        used = min(lot_units, qty_left_perf)
                        lot_cost_per_unit = float(lot["cost_eur"]) / lot_units if lot_units > UNITS_EPS else 0.0
                        lot["units"] = lot_units - used
                        lot["cost_eur"] = float(lot["cost_eur"]) - (used * lot_cost_per_unit)
                        qty_left_perf -= used
                        if qty_left_perf <= UNITS_EPS:
                            break

        if end_value_eur is not None and math.isfinite(float(end_value_eur)) and float(end_value_eur) != 0.0:
            flows.append((asof, float(end_value_eur)))
            portfolio_flows[asof] = portfolio_flows.get(asof, 0.0) + float(end_value_eur)

        unrealised_ytd_eur = None
        if end_value_eur is not None:
            remaining_perf_units = float(perf_open_units_remaining) + sum(float(lot["units"]) for lot in current_year_buy_lots_perf)

            current_value_per_unit_eur = (
                float(end_value_eur) / remaining_perf_units
                if abs(remaining_perf_units) > UNITS_EPS
                else 0.0
            )

            opening_layer_current_value = perf_open_units_remaining * current_value_per_unit_eur
            opening_layer_start_value = perf_open_units_remaining * opening_value_per_unit_eur

            current_year_layer_current_value = sum(float(lot["units"]) for lot in current_year_buy_lots_perf) * current_value_per_unit_eur
            current_year_layer_cost = sum(float(lot["cost_eur"]) for lot in current_year_buy_lots_perf)

            unrealised_ytd_eur = (
                (opening_layer_current_value - opening_layer_start_value)
                + (current_year_layer_current_value - current_year_layer_cost)
            )

        total_return_eur = None if unrealised_ytd_eur is None else (float(realised_ytd_eur) + float(unrealised_ytd_eur))

        xirr_ann = _xirr(flows)
        days_elapsed = max((asof - y0).days, 1)
        frac_year = days_elapsed / 365.0

        if xirr_ann is None:
            mwr_ytd_pct = math.nan
            mwr_ann_pct = math.nan
        else:
            mwr_ann_pct = xirr_ann * 100.0
            mwr_ytd_pct = (((1.0 + xirr_ann) ** frac_year) - 1.0) * 100.0

        simple_base_eur = None
        simple_return_pct = None

        if start_value_eur is not None:
            simple_base_eur = float(start_value_eur) + float(buys_eur)

        if (
            simple_base_eur is not None
            and simple_base_eur > 0
            and total_return_eur is not None
        ):
            simple_return_pct = (float(total_return_eur) / float(simple_base_eur)) * 100.0

        rows.append(
            {
                "Investment": name,
                "Code": code,
                "Platform": platform,
                "Ccy": ccy,
                "Start value (EUR)": start_value_eur,
                "Buys (EUR)": buys_eur,
                "Sells (EUR)": sells_eur,
                "End value (EUR)": end_value_eur,
                "Realised YTD (EUR)": realised_ytd_eur,
                "Unrealised YTD (EUR)": unrealised_ytd_eur,
                "Total return YTD (EUR)": total_return_eur,
                "Simple Return %": simple_return_pct,
                "MWR % (YTD)": mwr_ytd_pct,
                "MWR % (ann.)": mwr_ann_pct,
            }
        )

    ytd_df = pd.DataFrame(rows)
    if ytd_df.empty:
        return ytd_df, None

    ytd_key_cols = [
        "Start value (EUR)",
        "Buys (EUR)",
        "Sells (EUR)",
        "End value (EUR)",
        "Realised YTD (EUR)",
        "Unrealised YTD (EUR)",
        "Total return YTD (EUR)",
    ]
    try:
        key = ytd_df[ytd_key_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        keep = key.abs().sum(axis=1) > 1e-9
        ytd_df = ytd_df.loc[keep].reset_index(drop=True)
    except Exception:
        pass

    sum_cols = [
        "Start value (EUR)",
        "Buys (EUR)",
        "Sells (EUR)",
        "End value (EUR)",
        "Realised YTD (EUR)",
        "Unrealised YTD (EUR)",
        "Total return YTD (EUR)",
    ]

    totals = {"Investment": "TOTAL (priced only)", "Code": "", "Platform": "", "Ccy": ""}
    for c in sum_cols:
        totals[c] = float(pd.to_numeric(ytd_df[c], errors="coerce").sum(min_count=1))

    total_base = totals.get("Start value (EUR)", 0.0) + totals.get("Buys (EUR)", 0.0)
    if total_base and total_base > 0:
        totals["Simple Return %"] = (totals.get("Total return YTD (EUR)", 0.0) / total_base) * 100.0
    else:
        totals["Simple Return %"] = math.nan

    port_flows = sorted([(d, a) for d, a in portfolio_flows.items() if a != 0.0], key=lambda x: x[0])
    port_xirr = _xirr(port_flows)
    totals["MWR % (YTD)"] = (((1.0 + port_xirr) ** ((asof - y0).days / 365.0) - 1.0) * 100.0) if port_xirr is not None else math.nan
    totals["MWR % (ann.)"] = (port_xirr * 100.0) if port_xirr is not None else math.nan

    ytd_df = pd.concat([ytd_df, pd.DataFrame([totals])], ignore_index=True)

    if len(ytd_df) > 1:
        body = ytd_df.iloc[:-1].copy()
        body["__mwr_ytd"] = pd.to_numeric(body.get("MWR % (YTD)"), errors="coerce")
        body = body.sort_values(["__mwr_ytd"], ascending=False, na_position="last").drop(columns=["__mwr_ytd"], errors="ignore")
        ytd_df = pd.concat([body, ytd_df.iloc[[-1]]], ignore_index=True)

    return ytd_df, port_xirr


def _fmt(x, decimals=2) -> str:
    try:
        if x is None:
            return "N/A"
        if isinstance(x, str):
            return x
        if isinstance(x, (int, float)):
            if math.isnan(x):
                return "N/A"
            return f"{x:,.{decimals}f}"
        return str(x)
    except Exception:
        return str(x)


def _fmt_pct(x) -> str:
    try:
        if x is None:
            return "—"
        if isinstance(x, float) and math.isnan(x):
            return "—"
        v = float(x)
        if abs(v) < 0.005:
            v = 0.0
        return f"{v:,.2f}%"
    except Exception:
        return "—"


def _df_to_html(df: pd.DataFrame, pct_cols: Iterable[str] = ()) -> str:
    if df.empty:
        return "<p>N/A</p>"

    out = df.copy()
    for c in out.columns:
        if c in pct_cols:
            out[c] = out[c].map(_fmt_pct)
        else:
            out[c] = out[c].map(lambda v: _fmt(v, 2) if isinstance(v, (int, float)) else (v if v else ""))

    return out.to_html(index=False, border=0, justify="left")


def write_html(
    valuation: pd.DataFrame,
    realised_summary: pd.DataFrame,
    realised_trades: pd.DataFrame,
    realised_by_year: pd.DataFrame,
    ytd: pd.DataFrame,
    include_realised_trades_detail: bool = True,
) -> str:
    miss = []
    if not valuation.empty and "Price_Missing" in valuation.columns:
        m = valuation[valuation["Price_Missing"] == True]  # noqa: E712
        miss = m["Code"].dropna().astype(str).tolist()
    missing_str = ", ".join(miss) if miss else "None"

    asof = _now_str()

    priced_val_eur = None
    unrealised_eur = None
    total_realised = None
    simple_return_total = None
    realised_ytd_total = None
    unrealised_ytd_total = None

    try:
        if not valuation.empty:
            totals_row = valuation[
                valuation["Investment"].astype(str).str.startswith("TOTALS")
            ]
            if not totals_row.empty:
                priced_val_eur = totals_row["Value (EUR)"].iloc[0]
                unrealised_eur = totals_row["Unrealised (EUR)"].iloc[0]
    except Exception:
        pass

    try:
        if not realised_by_year.empty:
            total_row = realised_by_year[
                realised_by_year["Year"].astype(str).str.upper() == "TOTAL"
            ]
            if not total_row.empty:
                total_realised = total_row["Realised P/L (EUR)"].iloc[0]
    except Exception:
        pass

    try:
        if not ytd.empty:
            total_row = ytd[
                ytd["Investment"].astype(str).str.startswith("TOTAL")
            ]
            if not total_row.empty:
                if "Simple Return %" in total_row.columns:
                    simple_return_total = total_row["Simple Return %"].iloc[0]
                if "Realised YTD (EUR)" in total_row.columns:
                    realised_ytd_total = total_row["Realised YTD (EUR)"].iloc[0]
                if "Unrealised YTD (EUR)" in total_row.columns:
                    unrealised_ytd_total = total_row["Unrealised YTD (EUR)"].iloc[0]
    except Exception:
        pass

    def _fmt_cell(col: str, val) -> str:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "<span class='muted'>N/A</span>"

        if col in {"First buy", "First sell", "Last sell", "Date"}:
            try:
                dt = pd.to_datetime(val, errors="coerce")
                if pd.notna(dt):
                    return dt.strftime("%Y-%m-%d")
            except Exception:
                pass

        if isinstance(val, str):
            sval = val
        elif col in {"Unrealised %", "Simple Return %", "MWR % (YTD)", "MWR % (ann.)"}:
            sval = _fmt_pct(val)
        else:
            sval = _fmt(val, 2)

        num = None
        try:
            if isinstance(val, (int, float)) and not math.isnan(float(val)):
                num = float(val)
            elif isinstance(val, str):
                cleaned = val.replace("%", "").replace(",", "").strip()
                num = float(cleaned)
        except Exception:
            num = None

        pct_cols = {"Unrealised %", "Simple Return %", "MWR % (YTD)", "MWR % (ann.)"}
        pnl_cols = {
            "Unrealised (EUR)",
            "Realised P/L (EUR)",
            "Realised YTD (EUR)",
            "Unrealised YTD (EUR)",
            "Total return YTD (EUR)",
        }

        if col in pct_cols or col in pnl_cols:
            if num is not None:
                if abs(num) < 0.005:
                    sval = "0.00%" if col in pct_cols else "0.00"
                    return f"<span class='neutral'>{sval}</span>"
                if num > 0:
                    return f"<span class='pos'>{sval}</span>"
                if num < 0:
                    return f"<span class='neg'>{sval}</span>"

        return str(sval)

    def _table_html(df: pd.DataFrame, pct_cols: Iterable[str] = ()) -> str:
        if df.empty:
            return "<div class='muted'>N/A</div>"

        cols = list(df.columns)
        rows_html = []

        for _, row in df.iterrows():
            cells = []
            for col in cols:
                val = row.get(col, None)
                cells.append(f"<td>{_fmt_cell(col, val)}</td>")
            rows_html.append("<tr>" + "".join(cells) + "</tr>")

        head = "".join(f"<th>{col}</th>" for col in cols)
        return (
            "<table class='report-table'>"
            "<thead><tr>" + head + "</tr></thead>"
            "<tbody>" + "".join(rows_html) + "</tbody>"
            "</table>"
        )

    realised_summary_block = f"""
    <div class="card section">
      <h2>Realised trades summary</h2>
      <div class="muted">Grouped realised trade outcome by position and year.</div>
      {_table_html(realised_summary)}
    </div>
    """

    realised_trades_block = (
        f"""
        <div class="card section">
          <h2>Realised trade ledger</h2>
          <div class="muted">Detailed sell-side realised trade history.</div>
          {_table_html(realised_trades)}
        </div>
        """
        if include_realised_trades_detail
        else ""
    )

    def _fmt_chip_money(x, ccy: str = "EUR") -> str:
        try:
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return "—"

            v = float(x)
            symbol = "€" if str(ccy).upper() == "EUR" else f"{ccy} "

            if abs(v) < 0.005:
                v = 0.0

            if v < 0:
                return f"<span class='neg'>({symbol}{abs(v):,.2f})</span>"

            return f"<span class='pos-neutral'>{symbol}{v:,.2f}</span>"
        except Exception:
            return "—"

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Investo Valuation Report</title>
  <style>
    body {{
      font-family: Arial, Helvetica, sans-serif;
      background: #f6f8fb;
      color: #1f2937;
      margin: 0;
      padding: 24px;
    }}
    .wrap {{
      max-width: 1900px;
      margin: 0 auto;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #d8dee8;
      border-radius: 14px;
      padding: 18px 20px;
      margin: 0 0 16px 0;
    }}
    .hero {{
      background: #ffffff;
      border: 1px solid #d8dee8;
      border-radius: 16px;
      padding: 22px 24px;
      margin-bottom: 16px;
    }}
    h1 {{
      margin: 0 0 6px 0;
      font-size: 28px;
    }}
    h2 {{
      margin: 0 0 10px 0;
      font-size: 19px;
    }}
    .muted {{
      color: #6b7280;
      font-size: 13px;
    }}
    .chips {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
    }}
    .chip {{
      background: #f3f4f6;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 12px 16px;
      min-width: 190px;
    }}
    .chip span {{
      display: block;
      font-size: 12px;
      color: #6b7280;
    }}
    .chip b,
    .chip b span {{
      display: block;
      font-size: 32px;
      line-height: 1.1;
      margin-top: 6px;
      font-weight: 700;
    }}
    .chip .neg {{
      color: #b91c1c;
      font-weight: 700;
    }}
    .chip .pos-neutral {{
      color: #1f2937;
      font-weight: 700;
    }}
    .section {{
      overflow: auto;
    }}
    table.report-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      margin-top: 10px;
    }}
    .report-table th,
    .report-table td {{
      border-bottom: 1px solid #e5e7eb;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    .report-table th {{
      background: #f9fafb;
      position: sticky;
      top: 0;
      z-index: 1;
      font-weight: 700;
    }}
    .report-table tbody tr:nth-child(even) {{
      background: #fcfcfd;
    }}
    .report-table tbody tr:hover {{
      background: #f3f4f6;
    }}
    .pos {{
      color: #15803d;
      font-weight: 600;
    }}
    .neg {{
      color: #b91c1c;
      font-weight: 600;
    }}
    .neutral {{
      color: #374151;
      font-weight: 600;
    }}
    .footer {{
      color: #6b7280;
      font-size: 12px;
      margin-top: 4px;
    }}
  </style>
</head>
<body>
  <div class="wrap">

    <div class="hero">
      <h1>Investo Valuation Report</h1>
      <div class="muted">Generated: {asof}</div>

     <div class="chips">
        <div class="chip"><span>Portfolio value</span><b>{_fmt_chip_money(priced_val_eur, "EUR")}</b></div>
        <div class="chip"><span>Unrealised P/L</span><b>{_fmt_chip_money(unrealised_eur, "EUR")}</b></div>
        <div class="chip"><span>Realised P/L</span><b>{_fmt_chip_money(total_realised, "EUR")}</b></div>
        <div class="chip"><span>Unrealised P/L current year</span><b>{_fmt_chip_money(unrealised_ytd_total, "EUR")}</b></div>
        <div class="chip"><span>Realised P/L current year</span><b>{_fmt_chip_money(realised_ytd_total, "EUR")}</b></div>
        <div class="chip"><span>YTD plain return</span><b>{_fmt_pct(simple_return_total)}</b></div>
      </div>
    </div>

    <div class="card section">
      <h2>Valuation summary</h2>
      <div class="muted">Open positions with latest market valuation and unrealised performance.</div>
      {_table_html(valuation.drop(columns=["Price_Missing"], errors="ignore"))}
    </div>

    {realised_summary_block}

    {realised_trades_block}

    <div class="card section">
      <h2>Realised P/L by year</h2>
      <div class="muted">Year-by-year sell-side realised outcome.</div>
      {_table_html(realised_by_year)}
    </div>

    <div class="card section">
      <h2>YTD performance</h2>
      <div class="muted">Includes both plain return for readability and money-weighted return for timing-sensitive analysis.</div>
      {_table_html(ytd)}
    </div>

    <div class="footer">Generated by <b>portfolio_report.py</b></div>
  </div>
</body>
</html>
"""

def send_email(subject: str, html_body: str, attachments: List[Path]) -> None:
    host = os.getenv("SMTP_HOST", "")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    pwd = os.getenv("SMTP_PASSWORD", "")
    email_from = os.getenv("EMAIL_FROM", "")
    email_to = [e.strip() for e in os.getenv("EMAIL_TO", "").split(",") if e.strip()]

    if not (host and email_from and email_to):
        print("Valuation email not configured; skipping send.")
        return

    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = ", ".join(email_to)

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText("Investo Valuation Report (HTML).", "plain"))
    alt.attach(MIMEText(html_body, "html"))
    msg.attach(alt)

    for p in attachments:
        try:
            if not p.exists():
                print(f"Attachment missing (skip): {p}")
                continue
            part = MIMEApplication(p.read_bytes(), Name=p.name)
            part["Content-Disposition"] = f'attachment; filename="{p.name}"'
            msg.attach(part)
        except Exception as e:
            print(f"Failed to attach {p}: {e}")

    with smtplib.SMTP(host, port, timeout=30) as server:
        server.starttls()
        if user and pwd:
            server.login(user, pwd)
        server.sendmail(email_from, email_to, msg.as_string())


def main() -> int:
    if not PORTFOLIO_XLSX.exists():
        raise FileNotFoundError(f"Missing portfolio file: {PORTFOLIO_XLSX}")

    tx = load_transactions(PORTFOLIO_XLSX)

    open_df, realised_trades, realised_by_year = compute_positions_and_realised(tx)
    valuation = enrich_valuation(open_df)

    realised_summary = build_realised_trades_summary(tx, realised_trades)

    if not valuation.empty and "Unrealised %" in valuation.columns:
        try:
            _v = valuation.copy()
            inv = _v.get("Investment").fillna("").astype(str)
            special_mask = inv.str.startswith("TOTALS") | (inv == "Unpriced cost at risk")

            body = _v.loc[~special_mask].copy()
            special = _v.loc[special_mask].copy()

            body["__unr_pct"] = pd.to_numeric(body["Unrealised %"], errors="coerce")
            body = body.sort_values(["__unr_pct"], ascending=False, na_position="last").drop(columns=["__unr_pct"], errors="ignore")

            valuation = pd.concat([body, special], ignore_index=True).reset_index(drop=True)
        except Exception:
            pass

    ytd, _ = build_ytd_returns(tx, realised_trades, REPORT_YEAR)

    positions_csv = OUT_DIR / "portfolio_positions.csv"
    realised_trades_csv = OUT_DIR / "portfolio_realised_trades.csv"
    realised_summary_csv = OUT_DIR / "portfolio_realised_trades_summary.csv"
    realised_by_year_csv = OUT_DIR / "portfolio_realised_by_year.csv"
    ytd_csv = OUT_DIR / "portfolio_ytd_returns.csv"
    report_html = OUT_DIR / "portfolio_report.html"
    report_email_html = OUT_DIR / "portfolio_report_email.html"

    valuation.to_csv(positions_csv, index=False)
    realised_trades.to_csv(realised_trades_csv, index=False)
    realised_summary.to_csv(realised_summary_csv, index=False)
    realised_by_year.to_csv(realised_by_year_csv, index=False)
    ytd.to_csv(ytd_csv, index=False)

    html_full = write_html(
        valuation,
        realised_summary,
        realised_trades,
        realised_by_year,
        ytd,
        include_realised_trades_detail=True,
    )
    report_html.write_text(html_full, encoding="utf-8")

    html_email = write_html(
        valuation,
        realised_summary,
        realised_trades,
        realised_by_year,
        ytd,
        include_realised_trades_detail=False,
    )
    report_email_html.write_text(html_email, encoding="utf-8")

    print(
        "Wrote:",
        positions_csv.name,
        realised_trades_csv.name,
        realised_summary_csv.name,
        realised_by_year_csv.name,
        ytd_csv.name,
        report_html.name,
        report_email_html.name,
    )

    if SEND_VALUATION_EMAIL:
        subject = f"Investo -Live - Valuation Report – {_now_str()}"
        send_email(
            subject,
            html_email,
            [report_email_html, positions_csv, realised_by_year_csv, ytd_csv],
        )
        print("Valuation email sent.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
