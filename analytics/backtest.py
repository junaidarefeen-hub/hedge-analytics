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


def _tracking_error(daily_hedged: pd.Series, daily_unhedged: pd.Series) -> float:
    """Annualized std of return difference (hedged - unhedged)."""
    diff = daily_hedged - daily_unhedged
    return float(diff.std() * np.sqrt(ANNUALIZATION_FACTOR))


def _information_ratio(daily_hedged: pd.Series, daily_unhedged: pd.Series) -> float:
    """Excess return / tracking error."""
    diff = daily_hedged - daily_unhedged
    te = diff.std()
    if te == 0:
        return 0.0
    return float(diff.mean() / te * np.sqrt(ANNUALIZATION_FACTOR))


def _omega_ratio(daily: pd.Series, threshold: float = 0.0) -> float:
    """Sum of gains above threshold / sum of losses below threshold."""
    excess = daily - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 1.0
    return float(gains / losses)


def _max_drawdown_duration(cumulative: pd.Series) -> int:
    """Longest peak-to-recovery in trading days."""
    peak = cumulative.cummax()
    underwater = cumulative < peak
    if not underwater.any():
        return 0
    # Find contiguous underwater spans
    spans = (~underwater).cumsum()
    underwater_spans = spans[underwater]
    if underwater_spans.empty:
        return 0
    durations = underwater_spans.groupby(underwater_spans).count()
    return int(durations.max())


def _compute_metrics(
    daily: pd.Series, cumulative: pd.Series, risk_free: float, label: str,
) -> dict[str, float]:
    ann_factor = ANNUALIZATION_FACTOR
    total_return = float(cumulative.iloc[-1] - 1)
    n_days = len(daily)
    ann_return = float((1 + total_return) ** (ann_factor / max(n_days, 1)) - 1)
    ann_vol = float(daily.std() * np.sqrt(ann_factor))
    excess = daily - risk_free / ann_factor
    sharpe = float(excess.mean() / daily.std() * np.sqrt(ann_factor)) if daily.std() > 0 else 0.0
    downside = excess[excess < 0].std()
    sortino = float(excess.mean() / downside * np.sqrt(ann_factor)) if downside > 0 else 0.0
    max_dd = _max_drawdown(cumulative)
    calmar = float(ann_return / abs(max_dd)) if max_dd != 0 else 0.0

    omega = _omega_ratio(daily)
    dd_duration = _max_drawdown_duration(cumulative)

    return {
        "Total Return": total_return,
        "Ann. Return": ann_return,
        "Ann. Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
        "Omega Ratio": omega,
        "Max DD Duration (days)": dd_duration,
    }


def run_backtest(
    returns: pd.DataFrame,
    target: str,
    hedge_instruments: list[str],
    weights: np.ndarray,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    rolling_window: int = 60,
    risk_free: float = 0.0,
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

    # Tracking error and information ratio only apply to hedged
    te = _tracking_error(daily_hedged, daily_unhedged)
    ir = _information_ratio(daily_hedged, daily_unhedged)
    m_h["Tracking Error"] = te
    m_h["Information Ratio"] = ir
    m_un["Tracking Error"] = 0.0
    m_un["Information Ratio"] = 0.0

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


# ---------------------------------------------------------------------------
# Dynamic Rebalancing Backtest
# ---------------------------------------------------------------------------


@dataclass
class DynamicBacktestResult:
    cumulative_static: pd.Series
    cumulative_dynamic: pd.Series
    cumulative_unhedged: pd.Series
    daily_static: pd.Series
    daily_dynamic: pd.Series
    daily_unhedged: pd.Series
    weight_history: pd.DataFrame
    rebalance_dates: list
    turnover: pd.Series
    metrics: pd.DataFrame
    rolling_vol_static: pd.Series
    rolling_vol_dynamic: pd.Series
    rolling_vol_unhedged: pd.Series


def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> list[pd.Timestamp]:
    """Return first trading day of each week/month/quarter from the DatetimeIndex."""
    if freq == "weekly":
        iso = index.isocalendar()
        groups = index.to_series().groupby(iso.week.values * 100 + iso.year.values * 10000)
    elif freq == "monthly":
        groups = index.to_series().groupby([index.year, index.month])
    elif freq == "quarterly":
        groups = index.to_series().groupby([index.year, index.quarter])
    else:
        raise ValueError(f"Unknown frequency: {freq}")
    return [g.iloc[0] for _, g in groups]


def run_dynamic_backtest(
    returns: pd.DataFrame,
    target: str,
    hedges: list[str],
    static_weights: np.ndarray,
    strategy: str,
    bounds: tuple[float, float],
    rebalance_freq: str = "monthly",
    lookback_window: int = 120,
    rolling_window: int = 60,
    risk_free: float = 0.0,
    factors: list[str] | None = None,
    confidence: float = 0.95,
    min_names: int = 0,
    notional: float = 10_000_000.0,
    progress_callback=None,
    max_gross_notional: float | None = None,
) -> DynamicBacktestResult:
    """Run dynamic rebalancing backtest with periodic re-optimization."""
    from analytics.optimization import optimize_hedge

    cols = [target] + hedges
    clean = returns[cols].dropna()

    if len(clean) < lookback_window + 2:
        raise ValueError(f"Not enough data ({len(clean)}) for lookback window ({lookback_window}).")

    rebal_dates = _rebalance_dates(clean.index, rebalance_freq)
    # Filter to dates where we have enough lookback
    rebal_dates = [d for d in rebal_dates if clean.index.get_loc(d) >= lookback_window]

    if len(rebal_dates) < 2:
        raise ValueError("Not enough rebalance dates. Try a shorter lookback window or longer date range.")

    # Dynamic backtest
    weight_records = []
    daily_dynamic_parts = []
    prev_weights = None

    for i, reb_date in enumerate(rebal_dates):
        if progress_callback:
            progress_callback(i / len(rebal_dates))

        loc = clean.index.get_loc(reb_date)
        lookback_slice = clean.iloc[max(0, loc - lookback_window):loc]

        try:
            hedge_result = optimize_hedge(
                returns=lookback_slice,
                target=target,
                hedge_instruments=hedges,
                strategy=strategy,
                notional=notional,
                bounds=bounds,
                factors=factors,
                confidence=confidence,
                min_names=min_names,
                rolling_window=min(rolling_window, len(lookback_slice)),
                max_gross_notional=max_gross_notional,
            )
            current_weights = hedge_result.weights
        except Exception:
            # Fallback to equal weights if optimization fails
            current_weights = prev_weights if prev_weights is not None else static_weights

        weight_records.append({"date": reb_date, **{h: w for h, w in zip(hedges, current_weights)}})

        # Apply weights from reb_date to next reb_date (or end)
        if i + 1 < len(rebal_dates):
            period_end_loc = clean.index.get_loc(rebal_dates[i + 1])
        else:
            period_end_loc = len(clean)

        period = clean.iloc[loc:period_end_loc]
        if len(period) > 0:
            r_t = period[target]
            r_h = period[hedges].values @ current_weights
            daily_dynamic_parts.append(pd.Series(r_t.values + r_h, index=period.index))

        prev_weights = current_weights

    if progress_callback:
        progress_callback(1.0)

    # Combine dynamic daily returns
    daily_dynamic = pd.concat(daily_dynamic_parts)
    daily_dynamic.name = "Dynamic"

    # Static backtest for comparison
    r_target_full = clean[target]
    daily_static = r_target_full + (clean[hedges].values @ static_weights)
    daily_static = pd.Series(daily_static.values, index=clean.index, name="Static")

    # Align all series to the dynamic period
    common_idx = daily_dynamic.index
    daily_static = daily_static.reindex(common_idx)
    daily_unhedged = r_target_full.reindex(common_idx)

    # Cumulative returns
    cum_dynamic = (1 + daily_dynamic).cumprod()
    cum_static = (1 + daily_static).cumprod()
    cum_unhedged = (1 + daily_unhedged).cumprod()

    # Rolling vol
    roll_vol_dyn = daily_dynamic.rolling(window=rolling_window, min_periods=rolling_window).std() * np.sqrt(ANNUALIZATION_FACTOR)
    roll_vol_sta = daily_static.rolling(window=rolling_window, min_periods=rolling_window).std() * np.sqrt(ANNUALIZATION_FACTOR)
    roll_vol_un = daily_unhedged.rolling(window=rolling_window, min_periods=rolling_window).std() * np.sqrt(ANNUALIZATION_FACTOR)

    # Weight history
    weight_history = pd.DataFrame(weight_records).set_index("date")

    # Turnover
    turnover_values = []
    for i in range(1, len(weight_history)):
        turnover_values.append(float(np.abs(weight_history.iloc[i].values - weight_history.iloc[i - 1].values).sum()))
    turnover = pd.Series(turnover_values, index=weight_history.index[1:], name="Turnover")

    # Metrics for all three
    m_un = _compute_metrics(daily_unhedged, cum_unhedged, risk_free, "Unhedged")
    m_sta = _compute_metrics(daily_static, cum_static, risk_free, "Static")
    m_dyn = _compute_metrics(daily_dynamic, cum_dynamic, risk_free, "Dynamic")

    m_sta["Tracking Error"] = _tracking_error(daily_static, daily_unhedged)
    m_sta["Information Ratio"] = _information_ratio(daily_static, daily_unhedged)
    m_dyn["Tracking Error"] = _tracking_error(daily_dynamic, daily_unhedged)
    m_dyn["Information Ratio"] = _information_ratio(daily_dynamic, daily_unhedged)
    m_un["Tracking Error"] = 0.0
    m_un["Information Ratio"] = 0.0

    metrics = pd.DataFrame({"Unhedged": m_un, "Static": m_sta, "Dynamic": m_dyn})

    return DynamicBacktestResult(
        cumulative_static=cum_static,
        cumulative_dynamic=cum_dynamic,
        cumulative_unhedged=cum_unhedged,
        daily_static=daily_static,
        daily_dynamic=daily_dynamic,
        daily_unhedged=daily_unhedged,
        weight_history=weight_history,
        rebalance_dates=rebal_dates,
        turnover=turnover,
        metrics=metrics,
        rolling_vol_static=roll_vol_sta,
        rolling_vol_dynamic=roll_vol_dyn,
        rolling_vol_unhedged=roll_vol_un,
    )
