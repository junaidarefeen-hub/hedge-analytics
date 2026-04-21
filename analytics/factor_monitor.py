"""Factor performance monitoring and stock-level factor exposure estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    ANNUALIZATION_FACTOR,
    MM_FACTOR_BETA_WINDOW,
    MM_FACTOR_TREND_LONG,
    MM_FACTOR_TREND_SHORT,
    MM_PERIODS,
)


@dataclass
class FactorMonitorResult:
    """Factor performance and stock-level exposure data."""

    factor_perf_summary: pd.DataFrame  # factor × {1D, 1W, 1M, 3M, YTD}
    factor_rolling_sharpe: pd.DataFrame  # DatetimeIndex × factor columns
    factor_trends: pd.DataFrame  # factor × {short_ma, long_ma, trend, rolling_sharpe}
    stock_factor_betas: pd.DataFrame  # ticker × factor betas


def compute_factor_summary(factor_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute multi-period cumulative returns for each factor.

    Args:
        factor_returns: DataFrame with DatetimeIndex, factor name columns (daily returns).

    Returns:
        DataFrame with factors as index, period labels as columns.
    """
    if factor_returns.empty:
        return pd.DataFrame()

    results = {}
    for label, days in MM_PERIODS.items():
        if days is None:
            # YTD
            year_idx = factor_returns.index[
                factor_returns.index.year == factor_returns.index[-1].year
            ]
            if len(year_idx) > 0:
                ytd_returns = factor_returns.loc[year_idx]
                results[label] = (1 + ytd_returns).prod() - 1
            else:
                results[label] = pd.Series(np.nan, index=factor_returns.columns)
        else:
            if len(factor_returns) >= days:
                period_rets = factor_returns.iloc[-days:]
                results[label] = (1 + period_rets).prod() - 1
            else:
                results[label] = pd.Series(np.nan, index=factor_returns.columns)

    return pd.DataFrame(results)


def compute_factor_rolling_sharpe(
    factor_returns: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """Compute rolling annualized Sharpe ratio for each factor.

    Returns:
        DataFrame with DatetimeIndex × factor columns.
    """
    rolling_mean = factor_returns.rolling(window, min_periods=window).mean()
    rolling_std = factor_returns.rolling(window, min_periods=window).std()
    return (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(ANNUALIZATION_FACTOR)


def classify_factor_trends(factor_returns: pd.DataFrame) -> pd.DataFrame:
    """Classify each factor as trending up, neutral, or mean-reverting.

    Uses short-term vs long-term moving average of cumulative returns.

    Returns:
        DataFrame with factors as index, columns:
        [Short MA, Long MA, Trend, Rolling Sharpe(60d)]
    """
    if factor_returns.empty:
        return pd.DataFrame()

    # Cumulative returns for MA calculation
    cumulative = (1 + factor_returns).cumprod()

    short_ma = cumulative.rolling(MM_FACTOR_TREND_SHORT, min_periods=MM_FACTOR_TREND_SHORT).mean()
    long_ma = cumulative.rolling(MM_FACTOR_TREND_LONG, min_periods=MM_FACTOR_TREND_LONG).mean()

    # Latest values
    short_latest = short_ma.iloc[-1]
    long_latest = long_ma.iloc[-1]

    # Rolling Sharpe (60d)
    rolling_sharpe = compute_factor_rolling_sharpe(factor_returns, 60)
    sharpe_latest = rolling_sharpe.iloc[-1]

    # Classify
    diff_pct = (short_latest / long_latest - 1) * 100  # % difference

    trends = []
    for factor in factor_returns.columns:
        d = diff_pct.get(factor, 0)
        if d > 0.5:
            trend = "Trending Up"
        elif d < -0.5:
            trend = "Mean Reverting"
        else:
            trend = "Neutral"
        trends.append(trend)

    return pd.DataFrame({
        "Short MA": short_latest,
        "Long MA": long_latest,
        "Diff %": diff_pct,
        "Trend": trends,
        "Rolling Sharpe": sharpe_latest,
    }, index=factor_returns.columns)


def estimate_stock_factor_betas(
    stock_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    market_returns: pd.Series,
    window: int = MM_FACTOR_BETA_WINDOW,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Estimate factor betas for each stock via OLS over a chosen range.

    Model: stock_return = alpha + beta_mkt * market + sum(beta_i * factor_i) + epsilon

    Uses numpy.linalg.lstsq (same pattern as analytics/factor_analytics.py).

    Date selection:
        * If ``start_date`` or ``end_date`` is provided, the OLS is run over
          the aligned daily observations inside that range (inclusive on both ends).
        * Otherwise the trailing ``window`` days are used (backward-compatible
          default for callers like trade_ideas_tab that want a fixed tactical
          lookback).

    Args:
        stock_returns: DataFrame (DatetimeIndex × ticker columns, daily returns).
        factor_returns: DataFrame (DatetimeIndex × factor columns, daily returns).
        market_returns: Series (DatetimeIndex, daily market returns — e.g., SPX).
        window: Trailing lookback used only when both ``start_date`` and
            ``end_date`` are None.
        start_date: Optional lower bound (inclusive) for the OLS window.
        end_date: Optional upper bound (inclusive) for the OLS window.

    Returns:
        DataFrame with tickers as index, factor names as columns (beta values).
    """
    # Align all data
    common_idx = stock_returns.index.intersection(
        factor_returns.index
    ).intersection(market_returns.index)

    if start_date is not None or end_date is not None:
        if start_date is not None:
            common_idx = common_idx[common_idx >= pd.Timestamp(start_date)]
        if end_date is not None:
            common_idx = common_idx[common_idx <= pd.Timestamp(end_date)]
    else:
        common_idx = common_idx[-window:]  # trailing window fallback

    min_obs = 30
    if len(common_idx) < min_obs:
        return pd.DataFrame()

    stock_r = stock_returns.loc[common_idx]
    factor_r = factor_returns.loc[common_idx]
    market_r = market_returns.loc[common_idx]

    # Build design matrix: [1, market, factor1, factor2, ...]
    X = np.column_stack([
        np.ones(len(common_idx)),
        market_r.values,
        factor_r.values,
    ])

    factor_names = ["Market"] + list(factor_r.columns)
    betas = {}

    min_valid = max(20, len(common_idx) // 4)
    for ticker in stock_r.columns:
        y = stock_r[ticker].values
        valid = ~np.isnan(y) & ~np.isnan(X).any(axis=1)

        if valid.sum() < min_valid:
            continue

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X[valid], y[valid], rcond=None)
            # coeffs[0] = alpha, coeffs[1:] = betas
            betas[ticker] = dict(zip(factor_names, coeffs[1:]))
        except np.linalg.LinAlgError:
            continue

    if not betas:
        return pd.DataFrame()

    return pd.DataFrame(betas).T


def compute_factor_monitor(
    factor_returns: pd.DataFrame,
    stock_returns: pd.DataFrame | None = None,
    market_returns: pd.Series | None = None,
) -> FactorMonitorResult:
    """Compute full factor monitor analytics.

    Args:
        factor_returns: GS factor daily returns.
        stock_returns: S&P 500 stock daily returns (optional, for betas).
        market_returns: SPX daily returns (optional, for betas).
    """
    perf_summary = compute_factor_summary(factor_returns)
    rolling_sharpe = compute_factor_rolling_sharpe(factor_returns, 60)
    trends = classify_factor_trends(factor_returns)

    stock_betas = pd.DataFrame()
    if stock_returns is not None and market_returns is not None:
        stock_betas = estimate_stock_factor_betas(
            stock_returns, factor_returns, market_returns,
        )

    return FactorMonitorResult(
        factor_perf_summary=perf_summary,
        factor_rolling_sharpe=rolling_sharpe,
        factor_trends=trends,
        stock_factor_betas=stock_betas,
    )
