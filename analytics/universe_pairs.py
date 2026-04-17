"""Universe-wide pair-spread scanner — restricted to same-industry candidates.

Bounding the search to within-industry pairs caps the cost from O(n^2) over
504 names to roughly O(sum_i k_i^2) where k_i is the size of industry i. With
~127 GICS industries and an average size of ~4 names, that's ~1500 candidate
pairs total. Only the top-K most-correlated pairs per industry are passed to
``compute_pairs_analysis`` (which runs ADF + half-life), keeping the heavy
work bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from analytics.pairs import compute_pairs_analysis


@dataclass
class PairCandidateRow:
    ticker_a: str
    ticker_b: str
    industry: str
    current_zscore: float
    half_life_days: float
    adf_pvalue: float
    rolling_corr_60d: float


def _select_pair_candidates(
    industry_returns: pd.DataFrame,
    top_k: int,
    min_corr: float = 0.3,
) -> list[tuple[str, str, float]]:
    """Within an industry, return the top-K most-correlated unique pairs."""
    if industry_returns.shape[1] < 2:
        return []
    corr = industry_returns.corr().abs()
    pairs: list[tuple[str, str, float]] = []
    for a, b in combinations(corr.columns, 2):
        c = float(corr.loc[a, b])
        if np.isnan(c) or c < min_corr:
            continue
        pairs.append((a, b, c))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def scan_industry_pairs(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    industry_map: dict[str, str],
    *,
    window: int = 60,
    top_k_per_industry: int = 3,
    min_observations: int = 252,
    min_corr: float = 0.3,
) -> pd.DataFrame:
    """Scan within-industry pairs for spreads at extreme z-scores.

    Args:
        prices: wide-format prices (DatetimeIndex x ticker columns).
        returns: matching daily returns frame.
        industry_map: ticker -> GICS industry.
        window: rolling window for z-score and rolling correlation.
        top_k_per_industry: cap on candidate pairs per industry (selected
            by absolute correlation).
        min_observations: skip pairs with fewer overlapping observations.
        min_corr: skip pairs whose absolute correlation is below this.

    Returns:
        DataFrame with one row per scanned pair. Columns match
        ``PairCandidateRow``. Sorted by ``|current_zscore|`` descending.
    """
    # Group tickers by industry, intersected with the returns universe.
    tickers_by_industry: dict[str, list[str]] = {}
    for ticker, industry in industry_map.items():
        if ticker in returns.columns:
            tickers_by_industry.setdefault(industry, []).append(ticker)

    rows: list[dict] = []
    start = pd.Timestamp(prices.index.min())
    end = pd.Timestamp(prices.index.max())

    for industry, tickers in tickers_by_industry.items():
        if len(tickers) < 2:
            continue
        industry_rets = returns[tickers].dropna(how="any")
        if len(industry_rets) < min_observations:
            continue
        candidates = _select_pair_candidates(
            industry_rets, top_k=top_k_per_industry, min_corr=min_corr,
        )
        for ticker_a, ticker_b, _ in candidates:
            try:
                result = compute_pairs_analysis(
                    prices=prices,
                    returns=returns,
                    ticker_a=ticker_a,
                    ticker_b=ticker_b,
                    window=window,
                    start_date=start,
                    end_date=end,
                    spread_type="log",
                )
            except ValueError:
                # Insufficient overlap after the date slice — skip.
                continue
            roll_corr_60 = float(result.rolling_corr.dropna().iloc[-1]) if not result.rolling_corr.dropna().empty else float("nan")
            rows.append({
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "industry": industry,
                "current_zscore": result.current_zscore,
                "half_life_days": result.half_life,
                "adf_pvalue": result.adf_pvalue,
                "rolling_corr_60d": roll_corr_60,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "ticker_a", "ticker_b", "industry", "current_zscore",
            "half_life_days", "adf_pvalue", "rolling_corr_60d",
        ])

    df = pd.DataFrame(rows)
    df["abs_zscore"] = df["current_zscore"].abs()
    df = df.sort_values("abs_zscore", ascending=False).drop(columns="abs_zscore")
    return df.reset_index(drop=True)
