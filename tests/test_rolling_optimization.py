"""Tests for analytics.rolling_optimization — rolling_optimize()."""

import numpy as np
import pandas as pd
import pytest

from analytics.rolling_optimization import RollingOptResult, rolling_optimize


class TestRollingOptimize:
    @pytest.fixture()
    def ro(self, returns):
        return rolling_optimize(
            returns=returns,
            target="AAPL",
            hedge_instruments=["MSFT", "SPY", "QQQ"],
            strategy="Minimum Variance",
            bounds=(-1.0, 0.0),
            window=60,
            step=20,
        )

    def test_returns_correct_type(self, ro):
        assert isinstance(ro, RollingOptResult)

    def test_weight_history_columns(self, ro):
        assert list(ro.weight_history.columns) == ["MSFT", "SPY", "QQQ"]

    def test_weights_sum_to_minus_one(self, ro):
        sums = ro.weight_history.sum(axis=1)
        np.testing.assert_allclose(sums.values, -1.0, atol=0.02)

    def test_turnover_nonnegative(self, ro):
        assert (ro.turnover >= 0).all()

    def test_vol_positive(self, ro):
        assert (ro.vol_history > 0).all()

    def test_window_exceeds_data_raises(self, returns):
        with pytest.raises(ValueError, match="must exceed window"):
            rolling_optimize(
                returns=returns,
                target="AAPL",
                hedge_instruments=["MSFT", "SPY"],
                strategy="Minimum Variance",
                bounds=(-1.0, 0.0),
                window=9999,
                step=20,
            )

    def test_weight_stability_has_all_instruments(self, ro):
        assert "MSFT" in ro.weight_stability.index
        assert "SPY" in ro.weight_stability.index
        assert "QQQ" in ro.weight_stability.index

    def test_turnover_matches_weight_changes(self, ro):
        """Turnover at each step should equal sum of absolute weight diffs."""
        wh = ro.weight_history
        for i in range(1, len(wh)):
            expected = float(np.abs(wh.iloc[i].values - wh.iloc[i - 1].values).sum())
            assert ro.turnover.iloc[i - 1] == pytest.approx(expected, rel=1e-10)

    def test_weight_stability_std_matches(self, ro):
        """Weight stability std should match actual std of weight_history columns."""
        for col in ro.weight_history.columns:
            expected_std = float(ro.weight_history[col].std())
            assert float(ro.weight_stability.loc[col, "Std Dev"]) == pytest.approx(expected_std, rel=1e-10)

    def test_dates_count_matches_expected(self, returns):
        """Number of optimization dates = ceil((len - window) / step)."""
        window, step = 60, 20
        ro = rolling_optimize(
            returns=returns, target="AAPL",
            hedge_instruments=["MSFT", "SPY", "QQQ"],
            strategy="Minimum Variance", bounds=(-1.0, 0.0),
            window=window, step=step,
        )
        clean_len = len(returns[["AAPL", "MSFT", "SPY", "QQQ"]].dropna())
        expected = len(range(window, clean_len, step))
        assert len(ro.dates) == expected

    def test_hedged_vol_lower_than_unhedged(self, ro, returns):
        """Optimized hedged vol should generally be lower than unhedged vol."""
        from config import ANNUALIZATION_FACTOR
        unhedged_vol = float(returns["AAPL"].std() * np.sqrt(ANNUALIZATION_FACTOR))
        # At least some rolling windows should produce lower vol
        assert (ro.vol_history < unhedged_vol).any()
