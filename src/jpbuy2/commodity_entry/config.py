from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CommodityEntrySettings:
    """
    Commodity dip-entry settings.

    Design:
    - separate commodity engine from ETF engine
    - support multiple commodity types via profiles
    - default to long-term accumulation behaviour
    - remain entry-only for now
    """

    min_history_days: int = 520

    # Indicators
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Forward horizons
    forward_days_1y: int = 252
    forward_days_2y: int = 504
    forward_days_3y: int = 756

    default_profile: dict = field(
        default_factory=lambda: {
            "commodity_type": "generic",
            "entry_cost_pct": 0.00,
            "adaptive_lookback_days": 756,
            "drawdown_lookback_days": 20,
            "drawdown_entry_quantile": 0.88,
            "rsi_support_quantile": 0.35,
            "require_bullish_day": True,
            "cooldown_days": 30,
            "max_windows_per_year": 3,
            "monthly_ma_period": 10,
            "weekly_ma_period": 40,
            "weekly_ma_slope_weeks": 4,
            "deep_drawdown_min": 0.10,
            "daily_rsi_cap": 45.0,
            "strong_buy_extra_drawdown": 0.04,
        }
    )

    per_type_profiles: dict[str, dict] = field(
        default_factory=lambda: {
            "gold": {
                "commodity_type": "gold",
                "entry_cost_pct": 0.00,
                "adaptive_lookback_days": 756,
                "drawdown_lookback_days": 20,
                "drawdown_entry_quantile": 0.90,
                "rsi_support_quantile": 0.35,
                "require_bullish_day": False,
                "cooldown_days": 35,
                "max_windows_per_year": 4,
                "monthly_ma_period": 10,
                "weekly_ma_period": 40,
                "weekly_ma_slope_weeks": 4,
                "deep_drawdown_min": 0.04,
                "daily_rsi_cap": 48.0,
                "strong_buy_extra_drawdown": 0.04,
            },
            "silver": {
                "commodity_type": "silver",
                "entry_cost_pct": 0.00,
                "adaptive_lookback_days": 756,
                "drawdown_lookback_days": 20,
                "drawdown_entry_quantile": 0.85,        # raised from 0.82 – silver needs a bigger dip to confirm
                "rsi_support_quantile": 0.38,
                "require_bullish_day": True,
                "cooldown_days": 30,                    # raised from 25 – silver dip clusters are wider
                "max_windows_per_year": 4,
                "monthly_ma_period": 10,
                "weekly_ma_period": 40,
                "weekly_ma_slope_weeks": 4,
                "deep_drawdown_min": 0.12,              # raised from 0.10 – silver moves are larger
                "daily_rsi_cap": 50.0,                  # raised from 48 – more lenient for silver
                "strong_buy_extra_drawdown": 0.06,      # raised from 0.04 – needs bigger overshoot
                # ── silver-specific signal parameters ──────────────────────
                "silver_weekly_rsi_cap_frac": 0.70,     # weekly RSI cap fraction (vs 156-bar range)
                "silver_weekly_macd_lookback": 2,       # weeks of MACD improvement required
                "silver_weekly_deep_dip_min": 0.12,     # min weekly 12w drawdown to open the gate
                "silver_exceptional_dip_multiplier": 1.8,  # lower than gold's 2.0x (silver swings bigger)
                "silver_adaptive_rsi_cap_frac": 0.78,   # daily adaptive RSI cap fraction (vs 252-bar range)
            },
            "industrial": {
                "commodity_type": "industrial",
                "entry_cost_pct": 0.00,
                "adaptive_lookback_days": 756,
                "drawdown_lookback_days": 20,
                "drawdown_entry_quantile": 0.78,
                "rsi_support_quantile": 0.45,
                "require_bullish_day": True,
                "cooldown_days": 20,
                "max_windows_per_year": 5,
                "monthly_ma_period": 10,
                "weekly_ma_period": 40,
                "weekly_ma_slope_weeks": 4,
                "deep_drawdown_min": 0.08,
                "daily_rsi_cap": 50.0,
                "strong_buy_extra_drawdown": 0.03,
            },
            "energy": {
                "commodity_type": "energy",
                "entry_cost_pct": 0.00,
                "adaptive_lookback_days": 756,
                "drawdown_lookback_days": 20,
                "drawdown_entry_quantile": 0.92,
                "rsi_support_quantile": 0.30,
                "require_bullish_day": True,
                "cooldown_days": 35,
                "max_windows_per_year": 3,
                "monthly_ma_period": 10,
                "weekly_ma_period": 40,
                "weekly_ma_slope_weeks": 4,
                "deep_drawdown_min": 0.15,
                "daily_rsi_cap": 40.0,
                "strong_buy_extra_drawdown": 0.06,
            },
        }
    )
