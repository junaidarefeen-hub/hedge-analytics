"""Drawdown analysis — underwater series and period detection."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DrawdownPeriod:
    start: pd.Timestamp
    trough: pd.Timestamp
    end: pd.Timestamp | None  # None if unrecovered
    max_drawdown: float
    duration_days: int
    recovery_days: int | None  # None if unrecovered


@dataclass
class DrawdownAnalysis:
    underwater_series: pd.Series
    drawdown_periods: list[DrawdownPeriod]  # sorted by severity (worst first)
    max_drawdown: float
    avg_drawdown: float
    max_duration: int
    avg_duration: float


def compute_drawdowns(cumulative: pd.Series, top_n: int = 5) -> DrawdownAnalysis:
    """Compute drawdown analysis from a cumulative return series.

    Args:
        cumulative: Cumulative return series (e.g. growth of $1).
        top_n: Number of worst drawdown periods to return.
    """
    peak = cumulative.cummax()
    underwater = (cumulative - peak) / peak

    # Identify contiguous negative spans
    is_underwater = underwater < -1e-10
    periods: list[DrawdownPeriod] = []

    if not is_underwater.any():
        return DrawdownAnalysis(
            underwater_series=underwater,
            drawdown_periods=[],
            max_drawdown=0.0,
            avg_drawdown=0.0,
            max_duration=0,
            avg_duration=0.0,
        )

    # Group contiguous underwater periods
    group_ids = (~is_underwater).cumsum()
    for gid, group in underwater[is_underwater].groupby(group_ids[is_underwater]):
        if len(group) == 0:
            continue

        start_idx = group.index[0]
        trough_idx = group.idxmin()
        max_dd = float(group.min())

        # Find start: the peak date just before this drawdown
        start_loc = cumulative.index.get_loc(start_idx)
        if start_loc > 0:
            peak_date = cumulative.index[start_loc - 1]
        else:
            peak_date = start_idx

        # Find end: first date after trough where underwater returns to 0
        trough_loc = cumulative.index.get_loc(group.index[-1])
        if trough_loc + 1 < len(underwater):
            remaining = underwater.iloc[trough_loc + 1:]
            recovered = remaining[remaining >= -1e-10]
            if len(recovered) > 0:
                end_date = recovered.index[0]
                recovery_days = len(cumulative.loc[trough_idx:end_date]) - 1
            else:
                end_date = None
                recovery_days = None
        else:
            end_date = None
            recovery_days = None

        duration = len(group)

        periods.append(DrawdownPeriod(
            start=peak_date,
            trough=trough_idx,
            end=end_date,
            max_drawdown=max_dd,
            duration_days=duration,
            recovery_days=recovery_days,
        ))

    # Sort by severity (worst first)
    periods.sort(key=lambda p: p.max_drawdown)
    top_periods = periods[:top_n]

    durations = [p.duration_days for p in periods]
    dd_values = [p.max_drawdown for p in periods]

    return DrawdownAnalysis(
        underwater_series=underwater,
        drawdown_periods=top_periods,
        max_drawdown=float(min(dd_values)) if dd_values else 0.0,
        avg_drawdown=float(sum(dd_values) / len(dd_values)) if dd_values else 0.0,
        max_duration=max(durations) if durations else 0,
        avg_duration=float(sum(durations) / len(durations)) if durations else 0.0,
    )
