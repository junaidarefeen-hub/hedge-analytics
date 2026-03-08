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
