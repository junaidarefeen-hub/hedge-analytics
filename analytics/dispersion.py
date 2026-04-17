"""Cross-sectional and sector dispersion metrics for the S&P 500 universe.

Surfaces "dispersion regime" — high cross-sectional stdev days historically
precede factor-rotation opportunities and are a signal that single-name alpha
is available even when index returns are muted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DispersionSnapshot:
    """Latest cross-sectional and sector dispersion vs. historical distribution."""

    date: pd.Timestamp
    cross_sectional_std: float
    sector_std: float
    historical_pctile_cs: float  # 0-100, percentile of latest cs_std in lookback
    historical_pctile_sector: float
    top_decile_flag: bool  # True if either pctile >= 90
    history_cs: pd.Series  # daily cs_std over lookback (for sparkline)
    history_sector: pd.Series  # daily sector_std over lookback (for sparkline)


def _constituent_cols(prices: pd.DataFrame, sector_map: dict[str, str]) -> list[str]:
    """Subset of price columns that have a known sector mapping (excludes SPX, etc.)."""
    return [c for c in prices.columns if c in sector_map]


def compute_dispersion_history(
    prices: pd.DataFrame,
    sector_map: dict[str, str],
    lookback_days: int = 252,
) -> pd.DataFrame:
    """Daily cross-sectional and sector-level dispersion over the lookback window.

    Returns:
        DataFrame indexed by date with columns ``cs_std`` and ``sector_std``.
        ``cs_std`` is the standard deviation of single-name daily returns across
        the universe each day. ``sector_std`` is the standard deviation of the
        equal-weighted sector-mean returns each day (i.e. how spread apart the
        sectors moved).
    """
    cols = _constituent_cols(prices, sector_map)
    if not cols:
        return pd.DataFrame(columns=["cs_std", "sector_std"])

    # Use the last `lookback_days + 1` rows so we get `lookback_days` daily
    # returns after pct_change (the first row becomes NaN).
    window = prices[cols].iloc[-(lookback_days + 1):]
    daily_returns = window.pct_change().iloc[1:]

    cs_std = daily_returns.std(axis=1)

    # Per-day sector-mean returns: tickers x days -> sectors x days via groupby
    # on the column axis. Result columns are dates; rows are sectors.
    by_sector = daily_returns.T.groupby(daily_returns.columns.map(sector_map)).mean()
    sector_std = by_sector.std(axis=0)
    sector_std.index = pd.to_datetime(sector_std.index)

    return pd.DataFrame({"cs_std": cs_std, "sector_std": sector_std}).dropna(how="any")


def _pctile_of_latest(series: pd.Series) -> float:
    """Return the percentile rank (0-100) of the latest value within the series.

    100 = highest in window; 0 = lowest. Using the empirical CDF here rather
    than scipy.stats.percentileofscore to avoid an extra dependency surface
    for what is a one-line operation.
    """
    if series.empty:
        return float("nan")
    latest = series.iloc[-1]
    return float((series <= latest).mean() * 100)


def current_dispersion(
    prices: pd.DataFrame,
    sector_map: dict[str, str],
    lookback_days: int = 252,
    top_decile_threshold: float = 90.0,
) -> DispersionSnapshot:
    """Latest dispersion snapshot with historical context.

    Args:
        prices: Wide-format prices (DatetimeIndex x ticker columns).
        sector_map: ticker -> GICS sector.
        lookback_days: History window for percentile context.
        top_decile_threshold: Percentile cutoff that flips ``top_decile_flag``.
    """
    history = compute_dispersion_history(prices, sector_map, lookback_days)
    if history.empty:
        return DispersionSnapshot(
            date=prices.index[-1] if len(prices) else pd.Timestamp("NaT"),
            cross_sectional_std=float("nan"),
            sector_std=float("nan"),
            historical_pctile_cs=float("nan"),
            historical_pctile_sector=float("nan"),
            top_decile_flag=False,
            history_cs=pd.Series(dtype=float),
            history_sector=pd.Series(dtype=float),
        )

    cs_pctile = _pctile_of_latest(history["cs_std"])
    sector_pctile = _pctile_of_latest(history["sector_std"])

    return DispersionSnapshot(
        date=history.index[-1],
        cross_sectional_std=float(history["cs_std"].iloc[-1]),
        sector_std=float(history["sector_std"].iloc[-1]),
        historical_pctile_cs=cs_pctile,
        historical_pctile_sector=sector_pctile,
        top_decile_flag=bool(
            cs_pctile >= top_decile_threshold or sector_pctile >= top_decile_threshold
        ),
        history_cs=history["cs_std"],
        history_sector=history["sector_std"],
    )
