"""Reversion signal analytics: RSI, z-scores, MA distance, Bollinger, composite scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    MM_BOLLINGER_STD,
    MM_BOLLINGER_WINDOW,
    MM_COMPOSITE_WEIGHTS,
    MM_MA_WINDOWS,
    MM_RSI_WINDOW,
    MM_ZSCORE_WINDOWS,
)


@dataclass
class ReversionSignals:
    """Reversion/oversold signals for the S&P 500 universe."""

    rsi_14: pd.Series  # 14-day RSI (0-100)
    zscore_20d: pd.Series  # z-score vs 20d rolling
    zscore_60d: pd.Series  # z-score vs 60d rolling
    ma_distance_50d: pd.Series  # % distance from 50d MA
    ma_distance_200d: pd.Series  # % distance from 200d MA
    bollinger_pctb: pd.Series  # Bollinger %B (0=lower, 1=upper)
    composite_score: pd.Series  # 0-100 (0=most oversold)
    signals_df: pd.DataFrame  # all signals joined for screening


def compute_rsi(prices: pd.DataFrame, window: int = MM_RSI_WINDOW) -> pd.Series:
    """Compute RSI for each ticker using the latest price data.

    Returns:
        Series indexed by ticker with RSI values (0-100).
    """
    returns = prices.pct_change()
    gains = returns.clip(lower=0)
    losses = (-returns).clip(lower=0)

    # Wilder's smoothing (exponential)
    avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1].rename("RSI(14)")


def compute_zscore(prices: pd.DataFrame, window: int) -> pd.Series:
    """Compute z-score of current price vs rolling mean/std.

    Returns:
        Series indexed by ticker with z-score values.
    """
    rolling_mean = prices.rolling(window, min_periods=window).mean()
    rolling_std = prices.rolling(window, min_periods=window).std()

    zscore = (prices - rolling_mean) / rolling_std.replace(0, np.nan)
    return zscore.iloc[-1].rename(f"Z-Score({window}d)")


def compute_ma_distance(prices: pd.DataFrame, window: int) -> pd.Series:
    """Compute % distance of current price from moving average.

    Returns:
        Series indexed by ticker. Negative = below MA (oversold signal).
    """
    ma = prices.rolling(window, min_periods=window).mean()
    distance = (prices.iloc[-1] / ma.iloc[-1] - 1)
    return distance.rename(f"MA Dist({window}d)")


def compute_bollinger_pctb(
    prices: pd.DataFrame,
    window: int = MM_BOLLINGER_WINDOW,
    num_std: float = MM_BOLLINGER_STD,
) -> pd.Series:
    """Compute Bollinger Band %B for each ticker.

    %B = (Price - Lower Band) / (Upper Band - Lower Band)
    0 = at lower band, 1 = at upper band, <0 = below lower band

    Returns:
        Series indexed by ticker.
    """
    ma = prices.rolling(window, min_periods=window).mean()
    std = prices.rolling(window, min_periods=window).std()

    upper = ma + num_std * std
    lower = ma - num_std * std
    band_width = upper - lower

    pctb = (prices.iloc[-1] - lower.iloc[-1]) / band_width.iloc[-1].replace(0, np.nan)
    return pctb.rename("Bollinger %B")


def _percentile_rank(series: pd.Series, invert: bool = False) -> pd.Series:
    """Convert values to percentile ranks (0-100).

    Args:
        series: Raw signal values.
        invert: If True, lower raw values get higher ranks (e.g., RSI where
                low = oversold = should get low composite score).
    """
    ranked = series.rank(pct=True, na_option="keep") * 100
    if invert:
        ranked = 100 - ranked
    return ranked


def compute_reversion_signals(prices: pd.DataFrame) -> ReversionSignals:
    """Compute all reversion signals for the universe.

    Args:
        prices: DataFrame with DatetimeIndex, ticker columns.

    Returns:
        ReversionSignals with per-ticker signal values and composite score.
    """
    rsi = compute_rsi(prices, MM_RSI_WINDOW)
    zscore_20d = compute_zscore(prices, MM_ZSCORE_WINDOWS[0])
    zscore_60d = compute_zscore(prices, MM_ZSCORE_WINDOWS[1])
    ma_dist_50d = compute_ma_distance(prices, MM_MA_WINDOWS[0])
    ma_dist_200d = compute_ma_distance(prices, MM_MA_WINDOWS[1])
    bollinger = compute_bollinger_pctb(prices, MM_BOLLINGER_WINDOW, MM_BOLLINGER_STD)

    # Composite score: percentile rank each signal, then weighted average
    # All signals: lower raw value = more oversold → lower rank → lower composite
    # No inversion needed since the natural ordering already works
    ranked = pd.DataFrame({
        "rsi_14": _percentile_rank(rsi),
        "zscore_20d": _percentile_rank(zscore_20d),
        "zscore_60d": _percentile_rank(zscore_60d),
        "ma_distance_50d": _percentile_rank(ma_dist_50d),
        "bollinger_pctb": _percentile_rank(bollinger),
    })

    composite = sum(
        ranked[col] * weight
        for col, weight in MM_COMPOSITE_WEIGHTS.items()
    )
    composite = composite.rename("Composite Score")

    # Build combined signals DataFrame
    signals_df = pd.DataFrame({
        "RSI(14)": rsi,
        "Z-Score(20d)": zscore_20d,
        "Z-Score(60d)": zscore_60d,
        "MA Dist(50d)": ma_dist_50d,
        "MA Dist(200d)": ma_dist_200d,
        "Bollinger %B": bollinger,
        "Composite": composite,
    })

    return ReversionSignals(
        rsi_14=rsi,
        zscore_20d=zscore_20d,
        zscore_60d=zscore_60d,
        ma_distance_50d=ma_dist_50d,
        ma_distance_200d=ma_dist_200d,
        bollinger_pctb=bollinger,
        composite_score=composite,
        signals_df=signals_df,
    )
