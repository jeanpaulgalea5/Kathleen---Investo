from dataclasses import dataclass
from typing import Optional, Literal

AssetType = Literal["stock", "etf"]

@dataclass(frozen=True)
class SignalResult:
    ticker: str
    asof: str
    golden_on: bool
    silver_buy: bool
    reason: str
    blocked: bool
    block_reason: Optional[str] = None
