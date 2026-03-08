"""Stress testing — historical scenario replay and custom shock analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ScenarioResult:
    name: str
    description: str
    is_custom: bool
    start_date: pd.Timestamp | None
    end_date: pd.Timestamp | None
    n_days: int
    unhedged_return: float
    hedged_return: float
    unhedged_pnl: float
    hedged_pnl: float
    hedge_benefit: float
    hedge_benefit_pct: float
    instrument_returns: dict[str, float]
    instrument_pnl: dict[str, float]
    daily_cumulative_hedged: pd.Series | None
    daily_cumulative_unhedged: pd.Series | None


@dataclass
class StressTestResult:
    scenarios: list[ScenarioResult]
    summary_df: pd.DataFrame
    target_ticker: str
    hedge_instruments: list[str]
    weights: np.ndarray
    notional: float
    strategy: str
    skipped: dict[str, str] = field(default_factory=dict)


def _run_historical_scenario(
    returns: pd.DataFrame,
    target: str,
    hedge_instruments: list[str],
    weights: np.ndarray,
    notional: float,
    scenario: dict,
) -> ScenarioResult:
    """Replay a historical crisis period."""
    start = pd.Timestamp(scenario["start"])
    end = pd.Timestamp(scenario["end"])

    cols = [target] + hedge_instruments
    available = [c for c in cols if c in returns.columns]
    if len(available) < len(cols):
        missing = set(cols) - set(available)
        raise ValueError(f"Tickers not in data: {', '.join(missing)}")

    sliced = returns.loc[start:end, cols].dropna()

    if len(sliced) < 2:
        raise ValueError(
            f"Insufficient data for period {scenario['start']} to {scenario['end']}. "
            "Expand sidebar date range to include this period."
        )

    r_target = sliced[target]
    r_hedges = sliced[hedge_instruments]

    daily_unhedged = r_target
    daily_hedged = r_target + (r_hedges.values @ weights)
    daily_hedged = pd.Series(daily_hedged, index=sliced.index)

    cum_unhedged = (1 + daily_unhedged).cumprod()
    cum_hedged = (1 + daily_hedged).cumprod()

    unhedged_ret = float(cum_unhedged.iloc[-1] - 1)
    hedged_ret = float(cum_hedged.iloc[-1] - 1)

    unhedged_pnl = notional * unhedged_ret
    hedged_pnl = notional * hedged_ret

    # Per-instrument cumulative returns and P&L contribution
    inst_returns = {}
    inst_pnl = {}
    for i, inst in enumerate(hedge_instruments):
        inst_ret = float((1 + sliced[inst]).cumprod().iloc[-1] - 1)
        inst_returns[inst] = inst_ret
        inst_pnl[inst] = float(notional * weights[i] * inst_ret)
    inst_returns[target] = float((1 + sliced[target]).cumprod().iloc[-1] - 1)

    return ScenarioResult(
        name=scenario["name"],
        description=scenario["description"],
        is_custom=False,
        start_date=start,
        end_date=end,
        n_days=len(sliced),
        unhedged_return=unhedged_ret,
        hedged_return=hedged_ret,
        unhedged_pnl=unhedged_pnl,
        hedged_pnl=hedged_pnl,
        hedge_benefit=hedged_pnl - unhedged_pnl,
        hedge_benefit_pct=(hedged_ret - unhedged_ret) * 100,
        instrument_returns=inst_returns,
        instrument_pnl=inst_pnl,
        daily_cumulative_hedged=cum_hedged,
        daily_cumulative_unhedged=cum_unhedged,
    )


def _run_custom_scenario(
    target: str,
    hedge_instruments: list[str],
    weights: np.ndarray,
    notional: float,
    shocks: dict[str, float],
    name: str,
    description: str = "",
) -> ScenarioResult:
    """Apply user-defined percentage shocks."""
    target_ret = shocks.get(target, 0.0) / 100
    unhedged_pnl = notional * target_ret

    inst_returns = {target: target_ret}
    inst_pnl = {}
    hedge_pnl_sum = 0.0
    for i, inst in enumerate(hedge_instruments):
        inst_ret = shocks.get(inst, 0.0) / 100
        inst_returns[inst] = inst_ret
        pnl = notional * weights[i] * inst_ret
        inst_pnl[inst] = float(pnl)
        hedge_pnl_sum += pnl

    hedged_pnl = unhedged_pnl + hedge_pnl_sum
    hedged_ret = hedged_pnl / notional if notional > 0 else 0.0

    return ScenarioResult(
        name=name,
        description=description or f"Custom shock: {target} {shocks.get(target, 0):+.0f}%",
        is_custom=True,
        start_date=None,
        end_date=None,
        n_days=1,
        unhedged_return=target_ret,
        hedged_return=hedged_ret,
        unhedged_pnl=float(unhedged_pnl),
        hedged_pnl=float(hedged_pnl),
        hedge_benefit=float(hedged_pnl - unhedged_pnl),
        hedge_benefit_pct=(hedged_ret - target_ret) * 100,
        instrument_returns=inst_returns,
        instrument_pnl=inst_pnl,
        daily_cumulative_hedged=None,
        daily_cumulative_unhedged=None,
    )


def run_stress_test(
    returns: pd.DataFrame,
    target: str,
    hedge_instruments: list[str],
    weights: np.ndarray,
    notional: float,
    strategy: str,
    selected_scenarios: list[dict] | None = None,
    custom_scenarios: list[dict] | None = None,
) -> StressTestResult:
    """Run historical and/or custom stress scenarios."""
    results: list[ScenarioResult] = []
    skipped: dict[str, str] = {}

    for scenario in (selected_scenarios or []):
        try:
            results.append(_run_historical_scenario(
                returns, target, hedge_instruments, weights, notional, scenario,
            ))
        except ValueError as e:
            skipped[scenario["name"]] = str(e)

    for cs in (custom_scenarios or []):
        results.append(_run_custom_scenario(
            target=target,
            hedge_instruments=hedge_instruments,
            weights=weights,
            notional=notional,
            shocks=cs["shocks"],
            name=cs["name"],
            description=cs.get("description", ""),
        ))

    if not results:
        raise ValueError("No scenarios produced results. Check data availability and inputs.")

    # Build summary DataFrame
    rows = []
    for r in results:
        rows.append({
            "Scenario": r.name,
            "Type": "Custom" if r.is_custom else "Historical",
            "Days": r.n_days,
            "Unhedged Return": r.unhedged_return,
            "Hedged Return": r.hedged_return,
            "Unhedged P&L ($)": r.unhedged_pnl,
            "Hedged P&L ($)": r.hedged_pnl,
            "Hedge Benefit ($)": r.hedge_benefit,
            "Hedge Benefit (pp)": r.hedge_benefit_pct,
        })
    summary_df = pd.DataFrame(rows)

    return StressTestResult(
        scenarios=results,
        summary_df=summary_df,
        target_ticker=target,
        hedge_instruments=hedge_instruments,
        weights=weights,
        notional=notional,
        strategy=strategy,
        skipped=skipped,
    )
