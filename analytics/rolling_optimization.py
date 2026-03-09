"""Rolling (walk-forward) optimization with out-of-sample backtest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analytics.backtest import _compute_metrics, _tracking_error, _information_ratio
from analytics.optimization import optimize_hedge
from config import ANNUALIZATION_FACTOR


@dataclass
class RollingOptResult:
    weight_history: pd.DataFrame  # DatetimeIndex, columns = hedge instruments
    vol_history: pd.Series        # optimizer-estimated vol at each step
    dates: pd.DatetimeIndex
    weight_stability: pd.DataFrame  # std of each weight over time
    turnover: pd.Series
    # Backtest outputs (walk-forward simulation)
    daily_unhedged: pd.Series
    daily_static: pd.Series
    daily_rolling: pd.Series
    cumulative_unhedged: pd.Series
    cumulative_static: pd.Series
    cumulative_rolling: pd.Series
    rolling_vol_unhedged: pd.Series
    rolling_vol_static: pd.Series
    rolling_vol_rolling: pd.Series
    metrics: pd.DataFrame  # 3-column: Unhedged / Static / Rolling


def rolling_optimize(
    returns: pd.DataFrame,
    target: str,
    hedge_instruments: list[str],
    strategy: str,
    bounds: tuple[float, float],
    static_weights: np.ndarray | None = None,
    window: int = 120,
    step: int = 20,
    factors: list[str] | None = None,
    confidence: float = 0.95,
    min_names: int = 0,
    notional: float = 10_000_000.0,
    rolling_window: int = 60,
    risk_free: float = 0.0,
    progress_callback=None,
    max_gross_notional: float | None = None,
) -> RollingOptResult:
    """Run walk-forward optimization at regular step intervals.

    At each step, optimizes on the trailing window, then applies those weights
    to the out-of-sample period until the next step. Produces full backtest
    metrics comparing unhedged, static hedge, and rolling hedge.
    """
    cols = [target] + hedge_instruments
    clean = returns[cols].dropna()

    if len(clean) <= window:
        raise ValueError(f"Data length ({len(clean)}) must exceed window ({window}).")

    dates_list = []
    weight_list = []
    vol_list = []

    step_indices = list(range(window, len(clean), step))
    total_steps = len(step_indices)

    # --- Phase 1: Optimize at each step ---
    for idx, i in enumerate(step_indices):
        if progress_callback:
            progress_callback(idx / max(total_steps, 1) * 0.6)  # 0-60% for optimization

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
                max_gross_notional=max_gross_notional,
            )
            weights = result.weights
            vol = result.hedged_volatility
        except Exception:
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
        progress_callback(0.6)

    dates_idx = pd.DatetimeIndex(dates_list)
    weight_history = pd.DataFrame(weight_list, index=dates_idx, columns=hedge_instruments)
    vol_history = pd.Series(vol_list, index=dates_idx, name="Hedged Vol (ann.)")

    # Weight stability
    weight_stability = weight_history.std().to_frame("Std Dev")
    weight_stability["Mean"] = weight_history.mean()

    # Turnover
    turnover_values = []
    for i in range(1, len(weight_history)):
        turnover_values.append(float(np.abs(weight_history.iloc[i].values - weight_history.iloc[i - 1].values).sum()))
    turnover = pd.Series(turnover_values, index=dates_idx[1:], name="Turnover")

    # --- Phase 2: Walk-forward backtest simulation ---
    # Apply each step's weights to out-of-sample period until next step
    daily_rolling_parts = []
    for i, step_idx in enumerate(step_indices):
        if i + 1 < len(step_indices):
            end_idx = step_indices[i + 1]
        else:
            end_idx = len(clean)

        period = clean.iloc[step_idx:end_idx]
        if len(period) == 0:
            continue

        w = weight_list[i]
        r_t = period[target].values
        r_h = period[hedge_instruments].values @ w
        daily_rolling_parts.append(pd.Series(r_t + r_h, index=period.index))

    daily_rolling = pd.concat(daily_rolling_parts) if daily_rolling_parts else pd.Series(dtype=float)
    daily_rolling.name = "Rolling"

    # Align to the rolling period
    common_idx = daily_rolling.index
    r_target_full = clean[target]
    daily_unhedged = r_target_full.reindex(common_idx)

    # Static hedge: use first optimization weights (or provided static_weights)
    sw = static_weights if static_weights is not None else weight_list[0]
    daily_static_vals = r_target_full.values + (clean[hedge_instruments].values @ sw)
    daily_static_full = pd.Series(daily_static_vals, index=clean.index, name="Static")
    daily_static = daily_static_full.reindex(common_idx)

    # Cumulative returns
    cum_unhedged = (1 + daily_unhedged).cumprod()
    cum_static = (1 + daily_static).cumprod()
    cum_rolling = (1 + daily_rolling).cumprod()

    # Rolling vol
    roll_vol_un = daily_unhedged.rolling(window=rolling_window, min_periods=rolling_window).std() * np.sqrt(ANNUALIZATION_FACTOR)
    roll_vol_st = daily_static.rolling(window=rolling_window, min_periods=rolling_window).std() * np.sqrt(ANNUALIZATION_FACTOR)
    roll_vol_ro = daily_rolling.rolling(window=rolling_window, min_periods=rolling_window).std() * np.sqrt(ANNUALIZATION_FACTOR)

    # Metrics
    m_un = _compute_metrics(daily_unhedged, cum_unhedged, risk_free, "Unhedged")
    m_st = _compute_metrics(daily_static, cum_static, risk_free, "Static")
    m_ro = _compute_metrics(daily_rolling, cum_rolling, risk_free, "Rolling")

    m_st["Tracking Error"] = _tracking_error(daily_static, daily_unhedged)
    m_st["Information Ratio"] = _information_ratio(daily_static, daily_unhedged)
    m_ro["Tracking Error"] = _tracking_error(daily_rolling, daily_unhedged)
    m_ro["Information Ratio"] = _information_ratio(daily_rolling, daily_unhedged)
    m_un["Tracking Error"] = 0.0
    m_un["Information Ratio"] = 0.0

    metrics = pd.DataFrame({"Unhedged": m_un, "Static": m_st, "Rolling": m_ro})

    if progress_callback:
        progress_callback(1.0)

    return RollingOptResult(
        weight_history=weight_history,
        vol_history=vol_history,
        dates=dates_idx,
        weight_stability=weight_stability,
        turnover=turnover,
        daily_unhedged=daily_unhedged,
        daily_static=daily_static,
        daily_rolling=daily_rolling,
        cumulative_unhedged=cum_unhedged,
        cumulative_static=cum_static,
        cumulative_rolling=cum_rolling,
        rolling_vol_unhedged=roll_vol_un,
        rolling_vol_static=roll_vol_st,
        rolling_vol_rolling=roll_vol_ro,
        metrics=metrics,
    )
