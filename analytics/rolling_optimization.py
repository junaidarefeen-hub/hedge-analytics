"""Rolling (walk-forward) optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analytics.optimization import optimize_hedge
from config import ANNUALIZATION_FACTOR


@dataclass
class RollingOptResult:
    weight_history: pd.DataFrame  # DatetimeIndex, columns = hedge instruments
    vol_history: pd.Series
    dates: pd.DatetimeIndex
    weight_stability: pd.DataFrame  # std of each weight over time
    turnover: pd.Series


def rolling_optimize(
    returns: pd.DataFrame,
    target: str,
    hedge_instruments: list[str],
    strategy: str,
    bounds: tuple[float, float],
    window: int = 120,
    step: int = 20,
    factors: list[str] | None = None,
    confidence: float = 0.95,
    min_names: int = 0,
    notional: float = 1_000_000.0,
    rolling_window: int = 60,
    progress_callback=None,
) -> RollingOptResult:
    """Run walk-forward optimization at regular step intervals.

    Args:
        returns: Full return DataFrame.
        target: Target ticker.
        hedge_instruments: List of hedge instrument tickers.
        strategy: Optimization strategy name.
        bounds: Weight bounds tuple.
        window: Lookback window for each optimization.
        step: Number of trading days between optimizations.
        progress_callback: Optional callable(float) for progress updates.
    """
    cols = [target] + hedge_instruments
    clean = returns[cols].dropna()

    if len(clean) <= window:
        raise ValueError(f"Data length ({len(clean)}) must exceed window ({window}).")

    dates_list = []
    weight_list = []
    vol_list = []

    total_steps = len(range(window, len(clean), step))

    for idx, i in enumerate(range(window, len(clean), step)):
        if progress_callback:
            progress_callback(idx / max(total_steps, 1))

        slice_returns = clean.iloc[i - window:i]
        date = clean.index[i]

        try:
            result = optimize_hedge(
                returns=slice_returns,
                target=target,
                hedge_instruments=hedge_instruments,
                strategy=strategy,
                notional=notional,
                bounds=bounds,
                factors=factors,
                confidence=confidence,
                min_names=min_names,
                rolling_window=min(rolling_window, len(slice_returns)),
            )
            weights = result.weights
            vol = result.hedged_volatility
        except Exception:
            # Fallback: use previous weights if available
            if weight_list:
                weights = weight_list[-1]
                vol = vol_list[-1]
            else:
                n = len(hedge_instruments)
                weights = np.full(n, -1.0 / n) if bounds[0] < 0 else np.full(n, 1.0 / n)
                vol = 0.0

        dates_list.append(date)
        weight_list.append(weights)
        vol_list.append(vol)

    if progress_callback:
        progress_callback(1.0)

    dates_idx = pd.DatetimeIndex(dates_list)
    weight_history = pd.DataFrame(weight_list, index=dates_idx, columns=hedge_instruments)
    vol_history = pd.Series(vol_list, index=dates_idx, name="Hedged Vol (ann.)")

    # Weight stability: std of each weight over time
    weight_stability = weight_history.std().to_frame("Std Dev")
    weight_stability["Mean"] = weight_history.mean()

    # Turnover: sum of absolute weight changes between consecutive optimizations
    turnover_values = []
    for i in range(1, len(weight_history)):
        turnover_values.append(float(np.abs(weight_history.iloc[i].values - weight_history.iloc[i - 1].values).sum()))
    turnover = pd.Series(turnover_values, index=dates_idx[1:], name="Turnover")

    return RollingOptResult(
        weight_history=weight_history,
        vol_history=vol_history,
        dates=dates_idx,
        weight_stability=weight_stability,
        turnover=turnover,
    )
