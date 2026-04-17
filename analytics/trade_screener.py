"""Trade screener: combines reversion signals with factor exposures for actionable candidates."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import MM_FACTOR_BETA_THRESHOLD, MM_REVERSION_THRESHOLD


@dataclass
class TradeCandidate:
    """A screened trade idea with supporting signals."""

    ticker: str
    name: str
    sector: str
    trade_type: str  # "Oversold Reversion", "Factor Play: Momentum", etc.
    signal_strength: float  # 0-100 composite quality score
    reversion_score: float | None
    dominant_factor: str | None
    factor_beta: float | None


def screen_reversion_candidates(
    signals_df: pd.DataFrame,
    sector_map: dict[str, str],
    name_map: dict[str, str],
    threshold: float = MM_REVERSION_THRESHOLD,
    sector_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Screen for oversold reversion candidates.

    Args:
        signals_df: From ReversionSignals.signals_df.
        sector_map: ticker -> sector.
        name_map: ticker -> company name.
        threshold: Max composite score to qualify (lower = more oversold).
        sector_filter: Only include these sectors (None = all).

    Returns:
        DataFrame sorted by Composite score (most oversold first).
    """
    df = signals_df.copy()
    df["Sector"] = df.index.map(sector_map)
    df["Name"] = df.index.map(name_map)
    df = df.dropna(subset=["Composite"])

    if sector_filter:
        df = df[df["Sector"].isin(sector_filter)]

    candidates = df[df["Composite"] <= threshold].sort_values("Composite")
    cols = ["Name", "Sector", "RSI(14)", "Z-Score(20d)", "Z-Score(60d)",
            "MA Dist(50d)", "MA Dist(200d)", "Bollinger %B", "Composite"]
    return candidates[[c for c in cols if c in candidates.columns]]


def screen_factor_candidates(
    stock_betas: pd.DataFrame,
    factor_trends: pd.DataFrame,
    sector_map: dict[str, str],
    name_map: dict[str, str],
    min_beta: float = MM_FACTOR_BETA_THRESHOLD,
    sector_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Screen for stocks with high exposure to trending factors.

    Returns:
        DataFrame with columns: Name, Sector, Factor, Beta, Trend, Rolling Sharpe.
    """
    if stock_betas.empty or factor_trends.empty:
        return pd.DataFrame()

    # Find factors that are trending up
    trending = factor_trends[factor_trends["Trend"] == "Trending Up"].index.tolist()
    if not trending:
        # Fall back to factors with positive rolling Sharpe
        trending = factor_trends[factor_trends["Rolling Sharpe"] > 0].index.tolist()

    if not trending:
        return pd.DataFrame()

    rows = []
    for factor in trending:
        if factor not in stock_betas.columns or factor == "Market":
            continue
        betas = stock_betas[factor].dropna()
        high_exposure = betas[betas.abs() >= min_beta].sort_values(ascending=False)

        for ticker, beta in high_exposure.items():
            sector = sector_map.get(ticker, "Unknown")
            if sector_filter and sector not in sector_filter:
                continue
            rows.append({
                "Ticker": ticker,
                "Name": name_map.get(ticker, ticker),
                "Sector": sector,
                "Factor": factor,
                "Beta": beta,
                "Trend": factor_trends.loc[factor, "Trend"],
                "Factor Sharpe": factor_trends.loc[factor, "Rolling Sharpe"],
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("Beta", ascending=False).reset_index(drop=True)


def screen_factor_reversion(
    signals_df: pd.DataFrame,
    stock_betas: pd.DataFrame,
    factor_trends: pd.DataFrame,
    sector_map: dict[str, str],
    name_map: dict[str, str],
    reversion_threshold: float = MM_REVERSION_THRESHOLD,
    min_beta: float = MM_FACTOR_BETA_THRESHOLD,
    sector_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Screen for the intersection: oversold stocks with high factor exposure.

    These are the strongest candidates — a stock that is technically oversold
    AND has high exposure to a currently trending factor.

    Returns:
        DataFrame sorted by signal strength (combination of reversion + factor).
    """
    if stock_betas.empty or factor_trends.empty or signals_df.empty:
        return pd.DataFrame()

    # Get oversold tickers
    oversold = signals_df[signals_df["Composite"] <= reversion_threshold].index

    # Get trending factors (exclude Market)
    trending = factor_trends[factor_trends["Trend"] == "Trending Up"].index.tolist()
    trending = [f for f in trending if f != "Market"]
    if not trending:
        trending = factor_trends[factor_trends["Rolling Sharpe"] > 0].index.tolist()
        trending = [f for f in trending if f != "Market"]

    rows = []
    for ticker in oversold:
        if ticker not in stock_betas.index:
            continue
        sector = sector_map.get(ticker, "Unknown")
        if sector_filter and sector not in sector_filter:
            continue

        # Find strongest factor exposure among trending factors
        best_factor = None
        best_beta = 0.0
        for factor in trending:
            if factor in stock_betas.columns:
                beta = stock_betas.loc[ticker, factor]
                if abs(beta) >= min_beta and abs(beta) > abs(best_beta):
                    best_factor = factor
                    best_beta = beta

        if best_factor is None:
            continue

        composite = signals_df.loc[ticker, "Composite"]
        # Signal strength: lower reversion score + higher factor beta = stronger signal
        signal_strength = (100 - composite) * 0.6 + min(abs(best_beta) * 20, 100) * 0.4

        rows.append({
            "Ticker": ticker,
            "Name": name_map.get(ticker, ticker),
            "Sector": sector,
            "Trade Type": f"Reversion + {best_factor}",
            "Reversion Score": composite,
            "Factor": best_factor,
            "Factor Beta": best_beta,
            "Signal Strength": signal_strength,
            "RSI(14)": signals_df.loc[ticker, "RSI(14)"],
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("Signal Strength", ascending=False).reset_index(drop=True)
