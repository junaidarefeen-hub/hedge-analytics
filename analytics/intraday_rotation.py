"""Sector-rotation candidate ranking: laggers in leading sectors and vice versa.

Two complementary setups:
  * "Lagger in Leader" — single names down today inside the day's top-performing
    sectors. Mean-reversion candidates with sector tailwind.
  * "Leader in Laggard" — single names up today inside the day's weakest
    sectors. Idiosyncratic strength worth investigating (often news-driven).

Both are point-in-time: they read today's daily returns and rank within sectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RotationCandidate:
    """A single rotation idea with sector context."""

    ticker: str
    name: str
    sector: str
    ticker_return_today: float
    sector_return_today: float
    relative_to_sector: float  # ticker_return_today - sector_return_today
    composite_score: float  # from analytics.reversion (0-100, lower = oversold)
    signal: str  # "Lagger in Leader" | "Leader in Laggard"


def _sector_pctile(series: pd.Series) -> pd.Series:
    """Percentile rank (0-100) of a per-sector value series. NaNs propagate."""
    return series.rank(pct=True, na_option="keep") * 100


def rank_rotation_candidates(
    daily_returns: pd.Series,
    sector_returns_today: pd.Series,
    composite_score: pd.Series,
    sector_map: dict[str, str],
    name_map: dict[str, str] | None = None,
    leader_pctile: float = 80.0,
    laggard_pctile: float = 20.0,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rank lagger-in-leader and leader-in-laggard candidates.

    Args:
        daily_returns: per-ticker latest 1-day return.
        sector_returns_today: per-sector latest 1-day return (equal-weight).
        composite_score: per-ticker reversion composite from
            ``analytics.reversion.compute_reversion_signals`` (0-100).
        sector_map: ticker -> GICS sector.
        name_map: ticker -> company name (optional).
        leader_pctile: sectors at/above this pctile are "leading".
        laggard_pctile: sectors at/below this pctile are "lagging".
        top_n: how many of each signal to return.

    Returns:
        DataFrame of ``RotationCandidate`` rows. Columns:
        ``ticker, name, sector, ticker_return_today, sector_return_today,
        relative_to_sector, composite_score, signal``. Sorted with laggers
        first (most-oversold composite ascending), then leaders (largest
        positive ``relative_to_sector`` descending).
    """
    if name_map is None:
        name_map = {}

    if daily_returns.empty or sector_returns_today.empty:
        return pd.DataFrame(columns=[
            "ticker", "name", "sector", "ticker_return_today",
            "sector_return_today", "relative_to_sector",
            "composite_score", "signal",
        ])

    sector_pctile = _sector_pctile(sector_returns_today)
    leading = set(sector_pctile.index[sector_pctile >= leader_pctile])
    lagging = set(sector_pctile.index[sector_pctile <= laggard_pctile])

    rows: list[dict] = []
    for ticker, ret in daily_returns.items():
        sector = sector_map.get(ticker)
        if sector is None or sector not in sector_returns_today.index:
            continue
        sector_ret = float(sector_returns_today.loc[sector])
        rel = float(ret) - sector_ret
        comp = float(composite_score.get(ticker, np.nan))

        if sector in leading and ret < 0:
            signal = "Lagger in Leader"
        elif sector in lagging and ret > 0:
            signal = "Leader in Laggard"
        else:
            continue

        rows.append({
            "ticker": ticker,
            "name": name_map.get(ticker, ticker),
            "sector": sector,
            "ticker_return_today": float(ret),
            "sector_return_today": sector_ret,
            "relative_to_sector": rel,
            "composite_score": comp,
            "signal": signal,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "name", "sector", "ticker_return_today",
            "sector_return_today", "relative_to_sector",
            "composite_score", "signal",
        ])

    df = pd.DataFrame(rows)
    laggers = (
        df[df["signal"] == "Lagger in Leader"]
        .sort_values("composite_score", ascending=True, na_position="last")
        .head(top_n)
    )
    leaders = (
        df[df["signal"] == "Leader in Laggard"]
        .sort_values("relative_to_sector", ascending=False)
        .head(top_n)
    )
    return pd.concat([laggers, leaders], ignore_index=True)


def derive_inputs_from_snapshot(
    prices: pd.DataFrame,
    sector_map: dict[str, str],
) -> tuple[pd.Series, pd.Series]:
    """Helper: pull today's per-ticker and per-sector 1-day returns from a wide
    prices frame.

    Returns:
        ``(daily_returns, sector_returns_today)`` — both Series.
    """
    cols = [c for c in prices.columns if c in sector_map]
    if not cols or len(prices) < 2:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    daily_returns = prices[cols].pct_change(fill_method=None).iloc[-1].dropna()
    sector_returns_today = (
        daily_returns.groupby(daily_returns.index.map(sector_map)).mean()
    )
    return daily_returns, sector_returns_today
