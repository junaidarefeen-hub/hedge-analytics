"""Tests for analytics.compare — compare_strategies()."""

import pytest

from analytics.compare import CompareResult, compare_strategies


class TestCompareStrategies:
    @pytest.fixture()
    def cmp(self, returns):
        return compare_strategies(
            returns=returns,
            target="AAPL",
            hedge_instruments=["MSFT", "SPY", "QQQ", "XLE"],
            notional=1_000_000,
            bounds=(-1.0, 0.0),
            factors=["SPY", "QQQ"],
            confidence=0.95,
            min_names=0,
            rolling_window=60,
            risk_free=0.05,
        )

    def test_returns_compare_result(self, cmp):
        assert isinstance(cmp, CompareResult)

    def test_all_strategies_attempted(self, cmp):
        # At least some strategies should succeed with our synthetic data
        strategies_run = [c.strategy for c in cmp.comparisons]
        failed = list(cmp.failed_strategies.keys())
        assert len(strategies_run) + len(failed) == 4

    def test_ranking_has_composite(self, cmp):
        assert "Composite" in cmp.ranking_df.index

    def test_recommended_is_valid(self, cmp):
        strategies_run = [c.strategy for c in cmp.comparisons]
        assert cmp.recommended_strategy in strategies_run

    def test_metrics_df_has_unhedged(self, cmp):
        assert "Unhedged" in cmp.metrics_df.columns

    def test_vol_reduction_reasonable(self, cmp):
        """Vol reduction should be a finite number (percentage)."""
        for c in cmp.comparisons:
            assert -200 < c.vol_reduction_pct < 200

    def test_ranking_values_are_positive_ints(self, cmp):
        """Per-metric ranks should be positive integers (1, 2, 3, ...)."""
        for col in cmp.ranking_df.columns:
            for metric in cmp.ranking_df.index:
                if metric == "Composite":
                    continue
                val = cmp.ranking_df.loc[metric, col]
                assert val == int(val)
                assert val >= 1


class TestCompareWithBasket:
    def test_compare_with_basket_target(self, returns):
        """compare_strategies works with a synthetic basket column."""
        import numpy as np
        from utils.basket import inject_basket_column
        aug, target = inject_basket_column(returns, ["AAPL", "MSFT"], np.array([0.5, 0.5]))
        result = compare_strategies(
            returns=aug,
            target=target,
            hedge_instruments=["SPY", "QQQ", "XLE"],
            notional=1_000_000,
            bounds=(-1.0, 0.0),
            factors=["SPY", "QQQ"],
            confidence=0.95,
            min_names=0,
            rolling_window=60,
            risk_free=0.05,
        )
        assert isinstance(result, CompareResult)
        assert len(result.comparisons) > 0
