from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ETFEntrySettings:
    """
    ETF dip-entry settings for non-gold ETFs.

    Strategy objective:
    - catch real yearly pullbacks
    - confirm the turn with MACD histogram slope
    - avoid fixed RSI gates becoming stale
    - adapt automatically for new ETFs from the watchlist
    """

    forward_days: int = 252
    min_history_days: int = 260

    # Core indicators
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Adaptive windows
    adaptive_lookback_days: int = 504   # c. 2 years
    drawdown_lookback_days: int = 20

    # Entry threshold: current 20d pullback must be at least this percentile
    # of that ETF's own prior positive 20d pullbacks.
    drawdown_entry_quantile: float = 0.65

    # Optional RSI support: used for scoring/reporting only, not as a hard gate.
    rsi_support_quantile: float = 0.60

    # signal clustering / selection
    cooldown_days: int = 12
    max_windows_per_year: int = 4

    # Keep hook for future per-ticker overrides, but default remains adaptive.
    per_ticker_profiles: dict[str, dict] = field(default_factory=dict)

    default_profile: dict = field(
        default_factory=lambda: {
            "adaptive_lookback_days": 504,
            "drawdown_entry_quantile": 0.65,
            "rsi_support_quantile": 0.60,
            "require_bullish_day": True,
            "cooldown_days": 12,
            "max_windows_per_year": 4,
        }
    )
