from dataclasses import dataclass
from .types import AssetType


@dataclass(frozen=True)
class Settings:
    # --- Golden weekly ---
    golden_weekly_rsi_min: float = 35.0
    golden_weekly_rsi_max: float = 65.0
    golden_weekly_macd_hist_min: float = 0.0

    golden_exit_weekly_rsi_overbought: float = 70.0
    golden_exit_macd_hist_turns_down: bool = True  # hist < prior hist

    golden_trailing_stop_pct: float = 0.15
    golden_hard_stop_from_entry_pct: float = 0.095

    # --- Golden ADX gating (weekly) ---
    golden_adx_len: int = 14
    golden_on_adx_min: float = 18.0
    golden_exit_block_adx: float = 25.0
    golden_exit_unblock_di_gap: float = 3.5

    # --- Golden DI gating (weekly) ---
    golden_di_near_gap: float = 5.0
    golden_di_confirm_weeks: int = 1

    # --- Golden RSI rolling quantiles (weekly) ---
    golden_rsi_quantile_window_weeks: int = 156
    golden_rsi_quantile_min_weeks: int = 80

    # --- Golden regime controls (used by signals/golden.py) ---
    golden_rsi_entry_buffer: float = 20.0
    golden_relaxed_recovery_weeks: int = 12
    golden_rsi_high_quantile: float = 0.85
    golden_min_weeks_on: int = 8

    #-----Golden Rentry
    golden_trailing_atr_mult: float = 1.4
    golden_enable_reentry_after_hard_exit: bool = True
    golden_reentry_window_weeks: int = 8
    golden_reentry_require_close_above_ma40: bool = True
    golden_reentry_require_rsi_above_q1: bool = True
    golden_reentry_require_positive_macd_slope: bool = True

    # --- Silver daily: Gate A (exhaustion) ---
    silver_rsi_exhaustion: float = 32.0
    silver_atr_percentile_window: int = 252
    silver_atr_percentile_threshold: float = 0.75
    silver_volume_spike_mult: float = 1.8
    silver_vol_ma_window: int = 20

    # return thresholds differ by type
    stock_big_down_day: float = -0.020
    etf_big_down_day: float = -0.012

    # --- Silver daily: Gate B (stabilisation) ---
    hammer_wick_to_body: float = 1.5
    hammer_close_top_frac: float = 0.40
    no_new_low_lookback_days: int = 2

    # --- Silver daily: Gate C (confirmation) ---
    confirm_macd_hist_improves: bool = True
    confirm_macd_slope_days: int = 2

    # MACD params used by silver.py
    silver_macd_fast: int = 12
    silver_macd_slow: int = 26
    silver_macd_signal: int = 9

    # Option 2: Close must be above a short MA (default MA(5))
    silver_short_ma_period: int = 5

    # Option 2: Anti-spike filter (distance from MA)
    silver_max_dist_from_ma: float = 0.10

    silver_adx_entry_min: float = 8.0

    # Break-of-stabilisation-high: lookback used to compute stabilisation high
    stabilisation_lookback: int = 5

    # --- Silver daily: "cheap entry" filters ---
    silver_rsi_cap: float = 60.0
    silver_no_new_low_lookback: int = 10
    silver_no_new_low_days: int = 3
    silver_atr_cooldown_days: int = 3
    silver_adx_max: float = 30.0
    silver_require_adx_rising: bool = False
    silver_require_bullish_di: bool = False   # S8_DI_GUARD: block entry when -DI >= +DI

    # --- Blockers ---
    blocker_lower_low_days: int = 2
    blocker_adx_threshold: float = 30.0
    blocker_adx_rising_days: int = 3
    blocker_atr_rising_days: int = 3

    # --- Execution rule (break of stabilisation high) ---
    use_break_of_stabilisation_high: bool = True

    # --- Golden MA-break robustness buffer (weekly) ---
    golden_ma_break_buffer: float = 0.0

    # --- Golden trend-break confirmation controls ---
    golden_ma_break_confirm_weeks: int = 1
    golden_trend_break_macd_neg_weeks: int = 1

    # --- NEW: Intraday emergency exit overlay ---
    # Keeps the current weekly Golden exit intact, but allows an early
    # emergency exit if an open trade crashes hard intraday.
    use_intraday_emergency_exit: bool = False
    intraday_emergency_interval: str = "60m"
    intraday_emergency_stop_pct: float = 0.07
    intraday_emergency_use_low: bool = True


def settings_for(asset_type: AssetType) -> Settings:
    return Settings()


def settings_for_ticker(ticker: str, asset_type) -> 'Settings':
    """
    Fallback: returns baseline Settings for any ticker.
    The adaptive layer (settings_adaptive) in strategy_select.py
    overrides this at runtime after data is loaded.
    """
    return settings_for(asset_type)
