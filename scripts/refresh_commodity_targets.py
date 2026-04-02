from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

import pandas as pd
import requests


HEADERS = {
    "User-Agent": "Mozilla/5.0 (InvestoBot)",
}


# =========================
# FETCH
# =========================

def fetch_text(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.text


# =========================
# PARSERS
# =========================

def extract_gold_target(text: str) -> float:
    """
    Extract LBMA gold consensus (~4742)
    """
    patterns = [
        r"consensus[^$]{0,20}\$?([\d,]{4,6})/oz",
        r"\$([\d,]{4,6})/oz.*consensus",
    ]

    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return float(m.group(1).replace(",", ""))

    raise RuntimeError("Gold target not found")


def extract_silver_target(text: str) -> float:
    """
    Extract LBMA silver consensus (~75–80)
    """
    patterns = [
        r"consensus[^$]{0,20}\$?([\d]{2,3})/oz",
        r"\$([\d]{2,3})/oz.*silver",
    ]

    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if 20 < val < 200:  # sanity check
                return val

    raise RuntimeError("Silver target not found")


# =========================
# BUILD
# =========================

def build_targets() -> pd.DataFrame:

    gold_url = "https://ahasignals.com/gold-forecast-tracker/"
    silver_url = "https://ahasignals.com/research/silver-market/silver-market-consensus-fragility-analysis-2026/"

    gold_text = fetch_text(gold_url)
    silver_text = fetch_text(silver_url)

    gold = extract_gold_target(gold_text)
    silver = extract_silver_target(silver_text)

    df = pd.DataFrame([
        {
            "ticker": "XAU",
            "target_usd": gold,
            "last_update": date.today().isoformat(),
            "source": gold_url,
        },
        {
            "ticker": "XAG",
            "target_usd": silver,
            "last_update": date.today().isoformat(),
            "source": silver_url,
        },
    ])

    return df


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/commodities_targets.csv")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = build_targets()

    if df.empty:
        raise RuntimeError("No data produced")

    df.to_csv(out, index=False)

    print("Updated commodity targets:")
    print(df)


if __name__ == "__main__":
    main()
