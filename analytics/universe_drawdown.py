"""Universe-wide drawdown screening.

Surfaces stocks deep in drawdown that show signs of *basing* — where the
reversion composite score (oversold-ness) is improving even while the price
is still meaningfully below its prior peak. This is a common reversion-
with-catalyst setup: extreme weakness has been bid, but the rebound has
not yet shown up in price.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analytics.drawdown import compute_drawdowns
from analytics.reversion import compute_reversion_signals


@dataclass
class UniverseDrawdownRow:
    ticker: str
    current_dd: float
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    days_underwater: int
    recovery_from_trough_pct: float
    composite_today: float
    composite_60d_improvement: float


def _per_ticker_underwater_summary(price_series: pd.Series) -> dict | None:
    """Return summary of the most recent unrecovered drawdown period or None.

    Uses the existing ``compute_drawdowns`` so we share the same period
    detection logic as the hedge-side Drawdown tab.
    """
    series = price_series.dropna()
    if len(series) < 50:
        return None
    cumulative = series / series.iloc[0]
    analysis = compute_drawdowns(cumulative, top_n=10)

    # Find the unrecovered period whose start is closest to today (the
    # "currently active" drawdown). compute_drawdowns sorts periods by
    # severity, not by date, so we filter explicitly here.
    open_periods = [p for p in analysis.drawdown_periods if p.end is None]
    if not open_periods:
        return None
    active = max(open_periods, key=lambda p: p.trough)

    last_date = series.index[-1]
    days_underwater = int(
        len(series.loc[active.start:last_date]) - 1
    )
    trough_price = float(series.loc[active.trough])
    last_price = float(series.iloc[-1])
    recovery_pct = (last_price / trough_price - 1) if trough_price > 0 else 0.0

    return {
        "current_dd": float(active.max_drawdown),
        "peak_date": active.start,
        "trough_date": active.trough,
        "days_underwater": days_underwater,
        "recovery_from_trough_pct": float(recovery_pct),
    }


def rank_underwater_basing(
    prices: pd.DataFrame,
    signals_df: pd.DataFrame | None = None,
    min_dd: float = -0.15,
    min_days_underwater: int = 40,
    composite_lookback_days: int = 60,
) -> pd.DataFrame:
    """Rank universe by deep-but-improving drawdowns.

    Args:
        prices: wide-format prices (DatetimeIndex x ticker columns).
        signals_df: optional reversion signals DataFrame for "today".
            If ``None``, recomputed via ``compute_reversion_signals``.
        min_dd: only include tickers whose current drawdown is at least
            this deep (e.g. -0.15 = down 15%+ from peak).
        min_days_underwater: minimum length of the active drawdown.
        composite_lookback_days: lookback for the "improvement" delta;
            measures how much the composite score has moved up over this
            window (positive = stock has gotten *less* oversold relative
            to where it was, which often signals basing/accumulation).

    Returns:
        DataFrame with columns matching ``UniverseDrawdownRow``, sorted by
        ``composite_60d_improvement`` descending.
    """
    today_signals = signals_df
    if today_signals is None:
        today_signals = compute_reversion_signals(prices).signals_df

    if len(prices) > composite_lookback_days:
        past_prices = prices.iloc[: -composite_lookback_days]
        past_signals = compute_reversion_signals(past_prices).signals_df
        composite_then = past_signals.get("Composite", pd.Series(dtype=float))
    else:
        composite_then = pd.Series(dtype=float)

    composite_now = today_signals.get("Composite", pd.Series(dtype=float))

    rows: list[dict] = []
    for ticker in prices.columns:
        summary = _per_ticker_underwater_summary(prices[ticker])
        if summary is None:
            continue
        if summary["current_dd"] > min_dd:
            continue
        if summary["days_underwater"] < min_days_underwater:
            continue

        comp_today = float(composite_now.get(ticker, np.nan))
        comp_then = float(composite_then.get(ticker, np.nan))
        improvement = (
            comp_today - comp_then
            if not (np.isnan(comp_today) or np.isnan(comp_then))
            else np.nan
        )
        rows.append({
            "ticker": ticker,
            **summary,
            "composite_today": comp_today,
            "composite_60d_improvement": improvement,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "current_dd", "peak_date", "trough_date",
            "days_underwater", "recovery_from_trough_pct",
            "composite_today", "composite_60d_improvement",
        ])

    df = pd.DataFrame(rows)
    return df.sort_values(
        "composite_60d_improvement",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
