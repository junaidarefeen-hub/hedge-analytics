"""Market snapshot analytics: breadth, sector aggregation, multi-period returns."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from config import MM_PERIODS


@dataclass
class MarketSnapshotResult:
    """Daily snapshot of S&P 500 market state."""

    snapshot_date: date
    spx_return: float | None  # SPX 1-day return (None if SPX not in data)
    spx_level: float | None
    advance_decline: tuple[int, int]  # (stocks up, stocks down)
    breadth_pct: float  # % of stocks with positive return
    new_highs_lows: tuple[int, int]  # 52-week high/low counts
    sector_returns: pd.DataFrame  # sector × period returns
    top_gainers: pd.DataFrame  # top 10 by 1d return
    top_losers: pd.DataFrame  # bottom 10 by 1d return
    sector_breadth: pd.DataFrame  # per-sector advance/decline/breadth
    multi_period_returns: pd.DataFrame  # all tickers × periods


def compute_multi_period_returns(
    prices: pd.DataFrame,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Compute returns over multiple periods for all tickers.

    Returns:
        DataFrame with tickers as index, period labels as columns.
    """
    if as_of is not None:
        prices = prices.loc[:str(as_of)]

    if prices.empty or len(prices) < 2:
        return pd.DataFrame()

    latest = prices.iloc[-1]
    results = {}

    for label, days in MM_PERIODS.items():
        if days is None:
            # YTD: from first trading day of the year
            year_start = prices.index[prices.index.year == prices.index[-1].year]
            if len(year_start) > 0:
                base = prices.loc[year_start[0]]
                results[label] = (latest / base - 1)
            else:
                results[label] = pd.Series(np.nan, index=prices.columns)
        else:
            if len(prices) > days:
                base = prices.iloc[-(days + 1)]
                results[label] = (latest / base - 1)
            else:
                results[label] = pd.Series(np.nan, index=prices.columns)

    return pd.DataFrame(results)


def compute_sector_returns(
    multi_period: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """Aggregate multi-period returns to sector level (equal-weight average).

    Args:
        multi_period: DataFrame from compute_multi_period_returns().
        sector_map: ticker -> GICS sector.

    Returns:
        DataFrame with sectors as index, period labels as columns.
    """
    mp = multi_period.copy()
    mp["sector"] = mp.index.map(sector_map)
    mp = mp.dropna(subset=["sector"])
    return mp.groupby("sector").mean()


def compute_daily_snapshot(
    prices: pd.DataFrame,
    sector_map: dict[str, str],
    name_map: dict[str, str] | None = None,
    as_of: date | None = None,
) -> MarketSnapshotResult:
    """Compute a full market snapshot for a given date.

    Args:
        prices: DataFrame with DatetimeIndex, ticker columns (including optional "SPX").
        sector_map: ticker -> GICS sector.
        name_map: ticker -> company name (optional).
        as_of: Snapshot date (defaults to latest date in prices).
    """
    if name_map is None:
        name_map = {}

    if as_of is not None:
        prices = prices.loc[:str(as_of)]

    snapshot_date = prices.index[-1].date()

    # Separate SPX from constituents
    spx_col = "SPX" if "SPX" in prices.columns else None
    constituent_cols = [c for c in prices.columns if c != "SPX" and c in sector_map]
    constituent_prices = prices[constituent_cols]

    # 1-day returns
    daily_returns = constituent_prices.pct_change().iloc[-1].dropna()

    # SPX
    spx_return = None
    spx_level = None
    if spx_col:
        spx_series = prices[spx_col].dropna()
        if len(spx_series) >= 2:
            spx_return = float(spx_series.iloc[-1] / spx_series.iloc[-2] - 1)
            spx_level = float(spx_series.iloc[-1])

    # Advance / Decline
    advances = int((daily_returns > 0).sum())
    declines = int((daily_returns < 0).sum())
    total = advances + declines
    breadth_pct = advances / total * 100 if total > 0 else 0.0

    # 52-week highs / lows
    year_window = min(252, len(constituent_prices))
    rolling_prices = constituent_prices.iloc[-year_window:]
    current_prices = constituent_prices.iloc[-1]
    high_52w = rolling_prices.max()
    low_52w = rolling_prices.min()
    new_highs = int((current_prices >= high_52w * 0.99).sum())  # within 1% of high
    new_lows = int((current_prices <= low_52w * 1.01).sum())  # within 1% of low

    # Multi-period returns
    multi_period = compute_multi_period_returns(constituent_prices, as_of)

    # Sector returns
    sector_returns = compute_sector_returns(multi_period, sector_map)

    # Top movers
    movers_df = pd.DataFrame({
        "Ticker": daily_returns.index,
        "Name": [name_map.get(t, t) for t in daily_returns.index],
        "Sector": [sector_map.get(t, "Unknown") for t in daily_returns.index],
        "1D Return": daily_returns.values,
    })
    movers_df = movers_df.sort_values("1D Return", ascending=False)
    top_gainers = movers_df.head(10).reset_index(drop=True)
    top_losers = movers_df.tail(10).sort_values("1D Return").reset_index(drop=True)

    # Sector breadth
    sector_breadth_rows = []
    for sector in sorted(set(sector_map.values())):
        sector_tickers = [t for t in daily_returns.index if sector_map.get(t) == sector]
        sector_rets = daily_returns[sector_tickers]
        if len(sector_rets) == 0:
            continue
        s_adv = int((sector_rets > 0).sum())
        s_dec = int((sector_rets < 0).sum())
        s_total = s_adv + s_dec
        sector_breadth_rows.append({
            "Sector": sector,
            "Advances": s_adv,
            "Declines": s_dec,
            "Breadth %": s_adv / s_total * 100 if s_total > 0 else 0.0,
            "Avg Return": float(sector_rets.mean()),
        })
    sector_breadth_df = pd.DataFrame(sector_breadth_rows)

    return MarketSnapshotResult(
        snapshot_date=snapshot_date,
        spx_return=spx_return,
        spx_level=spx_level,
        advance_decline=(advances, declines),
        breadth_pct=breadth_pct,
        new_highs_lows=(new_highs, new_lows),
        sector_returns=sector_returns,
        top_gainers=top_gainers,
        top_losers=top_losers,
        sector_breadth=sector_breadth_df,
        multi_period_returns=multi_period,
    )


def compute_ma_breadth(
    prices: pd.DataFrame,
    sector_map: dict[str, str],
    ma_window: int = 50,
) -> pd.DataFrame:
    """Compute % of stocks above MA per sector.

    Returns:
        DataFrame with sectors as index, columns: "Above MA %", "Count", "Total".
    """
    constituent_cols = [c for c in prices.columns if c in sector_map]
    constituent_prices = prices[constituent_cols]

    ma = constituent_prices.rolling(ma_window, min_periods=ma_window).mean()
    above_ma = constituent_prices.iloc[-1] > ma.iloc[-1]

    rows = []
    for sector in sorted(set(sector_map.values())):
        sector_tickers = [t for t in constituent_cols if sector_map.get(t) == sector]
        if not sector_tickers:
            continue
        above = int(above_ma[sector_tickers].sum())
        total = len(sector_tickers)
        rows.append({
            "Sector": sector,
            "Above MA %": above / total * 100 if total > 0 else 0.0,
            "Count": above,
            "Total": total,
        })
    return pd.DataFrame(rows)
