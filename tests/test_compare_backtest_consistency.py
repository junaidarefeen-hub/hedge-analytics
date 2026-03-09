"""Verify Compare and Backtest tabs produce identical metrics for the same inputs.

The Strategy Compare tab re-optimizes each strategy and backtests it.
The Backtest tab uses weights from the Optimizer and backtests them.
When inputs match, their metrics must be identical.
"""

import numpy as np
import pandas as pd
import pytest

from analytics.backtest import run_backtest
from analytics.compare import compare_strategies
from analytics.optimization import optimize_hedge


class TestCompareBacktestConsistency:
    """Verify compare_strategies produces the same backtest metrics as standalone run_backtest."""

    @pytest.mark.parametrize("strategy", [
        "Minimum Variance",
        "Beta-Neutral",
        "Tail Risk (CVaR)",
        "Risk Parity",
    ])
    def test_metrics_match_for_same_inputs(self, returns, strategy):
        """Compare tab and Backtest tab should produce identical metrics."""
        target = "AAPL"
        hedges = ["MSFT", "SPY", "QQQ", "XLE"]
        notional = 1_000_000.0
        bounds = (-1.0, 0.0)
        factors = ["SPY", "QQQ"]
        confidence = 0.95
        min_names = 0
        rolling_window = 60
        risk_free = 0.02
        max_gross = notional  # same as default

        # --- Path 1: Compare tab (optimize + backtest in one call) ---
        compare_result = compare_strategies(
            returns=returns,
            target=target,
            hedge_instruments=hedges,
            notional=notional,
            bounds=bounds,
            factors=factors,
            confidence=confidence,
            min_names=min_names,
            rolling_window=rolling_window,
            risk_free=risk_free,
            start_date=None,
            end_date=None,
            max_gross_notional=max_gross,
        )

        # Find the comparison for our strategy
        comp = next(c for c in compare_result.comparisons if c.strategy == strategy)
        compare_metrics = comp.backtest_result.metrics

        # --- Path 2: Backtest tab (separate optimize then backtest) ---
        hedge_result = optimize_hedge(
            returns=returns,
            target=target,
            hedge_instruments=hedges,
            strategy=strategy,
            notional=notional,
            bounds=bounds,
            factors=factors,
            confidence=confidence,
            min_names=min_names,
            rolling_window=rolling_window,
            max_gross_notional=max_gross,
        )

        bt_result = run_backtest(
            returns=returns,
            target=target,
            hedge_instruments=hedge_result.hedge_instruments,
            weights=hedge_result.weights,
            start_date=None,
            end_date=None,
            rolling_window=rolling_window,
            risk_free=risk_free,
        )
        backtest_metrics = bt_result.metrics

        # --- Assert all common metrics are identical ---
        common_rows = compare_metrics.index.intersection(backtest_metrics.index)
        for metric in common_rows:
            for col in ["Unhedged", "Hedged"]:
                if col in compare_metrics.columns and col in backtest_metrics.columns:
                    val_compare = compare_metrics.loc[metric, col]
                    val_backtest = backtest_metrics.loc[metric, col]
                    np.testing.assert_allclose(
                        val_compare, val_backtest, rtol=1e-10,
                        err_msg=f"{strategy} - {metric} ({col}): compare={val_compare}, backtest={val_backtest}",
                    )

    @pytest.mark.parametrize("strategy", [
        "Minimum Variance",
        "Beta-Neutral",
        "Tail Risk (CVaR)",
        "Risk Parity",
    ])
    def test_weights_match(self, returns, strategy):
        """Compare tab's optimization should produce the same weights as standalone optimize_hedge."""
        target = "AAPL"
        hedges = ["MSFT", "SPY", "QQQ", "XLE"]
        notional = 1_000_000.0
        bounds = (-1.0, 0.0)
        factors = ["SPY", "QQQ"]
        confidence = 0.95
        min_names = 0
        rolling_window = 60
        max_gross = notional

        compare_result = compare_strategies(
            returns=returns,
            target=target,
            hedge_instruments=hedges,
            notional=notional,
            bounds=bounds,
            factors=factors,
            confidence=confidence,
            min_names=min_names,
            rolling_window=rolling_window,
            risk_free=0.0,
            max_gross_notional=max_gross,
        )

        comp = next(c for c in compare_result.comparisons if c.strategy == strategy)

        hedge_result = optimize_hedge(
            returns=returns,
            target=target,
            hedge_instruments=hedges,
            strategy=strategy,
            notional=notional,
            bounds=bounds,
            factors=factors,
            confidence=confidence,
            min_names=min_names,
            rolling_window=rolling_window,
            max_gross_notional=max_gross,
        )

        np.testing.assert_allclose(
            comp.hedge_result.weights, hedge_result.weights, atol=1e-8,
            err_msg=f"{strategy}: compare weights differ from standalone optimize_hedge",
        )

    def test_date_filtered_metrics_match(self, returns):
        """With date filtering, compare and backtest should still agree."""
        target = "AAPL"
        hedges = ["MSFT", "SPY", "QQQ"]
        notional = 1_000_000.0
        bounds = (-1.0, 0.0)
        factors = ["SPY"]
        start = pd.Timestamp(returns.index[50])
        end = pd.Timestamp(returns.index[200])

        compare_result = compare_strategies(
            returns=returns,
            target=target,
            hedge_instruments=hedges,
            notional=notional,
            bounds=bounds,
            factors=factors,
            confidence=0.95,
            min_names=0,
            rolling_window=60,
            risk_free=0.02,
            start_date=start,
            end_date=end,
            max_gross_notional=notional,
        )

        comp = next(c for c in compare_result.comparisons if c.strategy == "Minimum Variance")

        # Run standalone with same weights and dates
        bt_result = run_backtest(
            returns=returns,
            target=target,
            hedge_instruments=comp.hedge_result.hedge_instruments,
            weights=comp.hedge_result.weights,
            start_date=start,
            end_date=end,
            rolling_window=60,
            risk_free=0.02,
        )

        for metric in comp.backtest_result.metrics.index:
            for col in ["Unhedged", "Hedged"]:
                np.testing.assert_allclose(
                    comp.backtest_result.metrics.loc[metric, col],
                    bt_result.metrics.loc[metric, col],
                    rtol=1e-10,
                    err_msg=f"Date-filtered: {metric} ({col}) mismatch",
                )

    def test_cumulative_series_match(self, returns):
        """Cumulative return series should also be identical."""
        target = "AAPL"
        hedges = ["MSFT", "SPY", "QQQ"]

        compare_result = compare_strategies(
            returns=returns,
            target=target,
            hedge_instruments=hedges,
            notional=1_000_000.0,
            bounds=(-1.0, 0.0),
            factors=["SPY"],
            confidence=0.95,
            min_names=0,
            rolling_window=60,
            risk_free=0.0,
            max_gross_notional=1_000_000.0,
        )

        comp = next(c for c in compare_result.comparisons if c.strategy == "Minimum Variance")

        bt_result = run_backtest(
            returns=returns,
            target=target,
            hedge_instruments=comp.hedge_result.hedge_instruments,
            weights=comp.hedge_result.weights,
            rolling_window=60,
            risk_free=0.0,
        )

        np.testing.assert_allclose(
            comp.backtest_result.cumulative_hedged.values,
            bt_result.cumulative_hedged.values,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            comp.backtest_result.cumulative_unhedged.values,
            bt_result.cumulative_unhedged.values,
            rtol=1e-10,
        )
