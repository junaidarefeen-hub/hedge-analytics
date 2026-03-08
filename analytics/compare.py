"""Strategy comparison — run all 4 strategies, backtest each, rank."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analytics.backtest import BacktestResult, run_backtest
from analytics.optimization import HedgeResult, optimize_hedge
from config import STRATEGY_OPTIONS


@dataclass
class StrategyComparison:
    strategy: str
    hedge_result: HedgeResult
    backtest_result: BacktestResult
    vol_reduction_pct: float


@dataclass
class CompareResult:
    comparisons: list[StrategyComparison]
    ranking_df: pd.DataFrame
    metrics_df: pd.DataFrame
    recommended_strategy: str
    target: str
    failed_strategies: dict[str, str] = field(default_factory=dict)


# Metrics where higher = better
_HIGHER_BETTER = {"Total Return", "Ann. Return", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Vol Reduction"}
# Metrics where higher = better (Max Drawdown is negative, so less negative = higher = better)
_ALL_METRICS = ["Vol Reduction", "Total Return", "Ann. Return", "Ann. Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Calmar Ratio"]


def compare_strategies(
    returns: pd.DataFrame,
    target: str,
    hedge_instruments: list[str],
    notional: float,
    bounds: tuple[float, float],
    factors: list[str],
    confidence: float,
    min_names: int,
    rolling_window: int,
    risk_free: float,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> CompareResult:
    """Run all strategies, backtest each, build comparison."""
    comparisons: list[StrategyComparison] = []
    failed: dict[str, str] = {}

    for strategy in STRATEGY_OPTIONS:
        try:
            hedge_result = optimize_hedge(
                returns=returns,
                target=target,
                hedge_instruments=hedge_instruments,
                strategy=strategy,
                notional=notional,
                bounds=bounds,
                factors=factors,
                confidence=confidence,
                min_names=min_names,
                rolling_window=rolling_window,
            )
            bt_result = run_backtest(
                returns=returns,
                target=target,
                hedge_instruments=hedge_result.hedge_instruments,
                weights=hedge_result.weights,
                start_date=start_date,
                end_date=end_date,
                rolling_window=rolling_window,
                risk_free=risk_free,
            )
            vol_red = (
                (1 - hedge_result.hedged_volatility / hedge_result.unhedged_volatility) * 100
                if hedge_result.unhedged_volatility > 0 else 0.0
            )
            comparisons.append(StrategyComparison(
                strategy=strategy,
                hedge_result=hedge_result,
                backtest_result=bt_result,
                vol_reduction_pct=vol_red,
            ))
        except Exception as e:
            failed[strategy] = str(e)

    if not comparisons:
        raise ValueError("All strategies failed. Check inputs and hedge universe.")

    # Build metrics DataFrame: rows = metrics, columns = strategies
    metrics_data: dict[str, dict[str, float]] = {}
    for comp in comparisons:
        hedged_metrics = comp.backtest_result.metrics["Hedged"].to_dict()
        hedged_metrics["Vol Reduction"] = comp.vol_reduction_pct / 100  # as decimal for consistency
        metrics_data[comp.strategy] = hedged_metrics

    metrics_df = pd.DataFrame(metrics_data)
    # Reorder rows
    row_order = [m for m in _ALL_METRICS if m in metrics_df.index]
    metrics_df = metrics_df.loc[row_order]

    # Add unhedged column (same for all strategies)
    unhedged_metrics = comparisons[0].backtest_result.metrics["Unhedged"].to_dict()
    unhedged_metrics["Vol Reduction"] = 0.0
    metrics_df.insert(0, "Unhedged", pd.Series(unhedged_metrics))

    # Rank strategies (exclude Unhedged column)
    strategy_cols = [c.strategy for c in comparisons]
    rank_data: dict[str, dict[str, float]] = {s: {} for s in strategy_cols}

    for metric in row_order:
        values = {s: metrics_df.loc[metric, s] for s in strategy_cols}
        # For all metrics: higher is better (Max Drawdown is negative, so -0.05 > -0.20)
        # For Ann. Volatility: lower is better
        ascending = metric == "Ann. Volatility"
        sorted_strats = sorted(values, key=lambda s: values[s], reverse=not ascending)
        for rank_pos, s in enumerate(sorted_strats, 1):
            rank_data[s][metric] = rank_pos

    ranking_df = pd.DataFrame(rank_data)
    ranking_df.loc["Composite"] = ranking_df.mean()

    # Recommended = lowest composite rank
    composite_ranks = ranking_df.loc["Composite"]
    recommended = composite_ranks.idxmin()

    return CompareResult(
        comparisons=comparisons,
        ranking_df=ranking_df,
        metrics_df=metrics_df,
        recommended_strategy=recommended,
        target=target,
        failed_strategies=failed,
    )
