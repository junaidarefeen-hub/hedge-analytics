"""Backtest engine — hedged vs unhedged portfolio performance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import ANNUALIZATION_FACTOR


@dataclass
class BacktestResult:
    cumulative_unhedged: pd.Series
    cumulative_hedged: pd.Series
    rolling_vol_unhedged: pd.Series
    rolling_vol_hedged: pd.Series
    daily_unhedged: pd.Series
    daily_hedged: pd.Series
    metrics: pd.DataFrame


def _max_drawdown(cumulative: pd.Series) -> float:
    peak = cumulative.cummax()
    dd = (cumulative - peak) / peak
    return float(dd.min())


def _compute_metrics(
    daily: pd.Series, cumulative: pd.Series, risk_free: float, label: str,
) -> dict[str, float]:
    ann_factor = ANNUALIZATION_FACTOR
    total_return = float(cumulative.iloc[-1] / cumulative.iloc[0] - 1)
    n_days = len(daily)
    ann_return = float((1 + total_return) ** (ann_factor / max(n_days, 1)) - 1)
    ann_vol = float(daily.std() * np.sqrt(ann_factor))
    excess = daily - risk_free / ann_factor
    sharpe = float(excess.mean() / daily.std() * np.sqrt(ann_factor)) if daily.std() > 0 else 0.0
    downside = daily[daily < 0].std()
    sortino = float(excess.mean() / downside * np.sqrt(ann_factor)) if downside > 0 else 0.0
    max_dd = _max_drawdown(cumulative)
    calmar = float(ann_return / abs(max_dd)) if max_dd != 0 else 0.0

    return {
        "Total Return": total_return,
        "Ann. Return": ann_return,
        "Ann. Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
    }


def run_backtest(
    returns: pd.DataFrame,
    target: str,
    hedge_instruments: list[str],
    weights: np.ndarray,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    rolling_window: int = 60,
    risk_free: float = 0.05,
) -> BacktestResult:
    """Run backtest of hedged vs unhedged portfolio over date range."""
    cols = [target] + hedge_instruments
    clean = returns[cols].dropna()

    if start_date is not None:
        clean = clean[clean.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        clean = clean[clean.index <= pd.Timestamp(end_date)]

    if len(clean) < 2:
        raise ValueError("Not enough data points for backtest.")

    r_target = clean[target]
    r_hedges = clean[hedge_instruments]

    daily_unhedged = r_target
    daily_hedged = r_target + (r_hedges.values @ weights)
    daily_hedged = pd.Series(daily_hedged, index=clean.index, name="Hedged")

    # Cumulative returns (growth of $1)
    cum_unhedged = (1 + daily_unhedged).cumprod()
    cum_hedged = (1 + daily_hedged).cumprod()

    # Rolling volatility (annualized)
    roll_vol_un = daily_unhedged.rolling(window=rolling_window, min_periods=rolling_window).std() * np.sqrt(ANNUALIZATION_FACTOR)
    roll_vol_h = daily_hedged.rolling(window=rolling_window, min_periods=rolling_window).std() * np.sqrt(ANNUALIZATION_FACTOR)

    # Metrics
    m_un = _compute_metrics(daily_unhedged, cum_unhedged, risk_free, "Unhedged")
    m_h = _compute_metrics(daily_hedged, cum_hedged, risk_free, "Hedged")

    metrics = pd.DataFrame({"Unhedged": m_un, "Hedged": m_h})

    return BacktestResult(
        cumulative_unhedged=cum_unhedged,
        cumulative_hedged=cum_hedged,
        rolling_vol_unhedged=roll_vol_un,
        rolling_vol_hedged=roll_vol_h,
        daily_unhedged=daily_unhedged,
        daily_hedged=daily_hedged,
        metrics=metrics,
    )
