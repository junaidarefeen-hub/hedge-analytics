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
            risk_free=0.05,
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


class TestRollingOptBacktest:
    """Tests for the walk-forward backtest simulation outputs."""

    @pytest.fixture()
    def ro(self, returns):
        hedges = ["MSFT", "SPY", "QQQ"]
        static_w = np.array([-0.4, -0.3, -0.3])
        return rolling_optimize(
            returns=returns,
            target="AAPL",
            hedge_instruments=hedges,
            strategy="Minimum Variance",
            bounds=(-1.0, 0.0),
            static_weights=static_w,
            window=60,
            step=20,
            risk_free=0.05,
        )

    def test_metrics_has_three_columns(self, ro):
        assert set(ro.metrics.columns) == {"Unhedged", "Static", "Rolling"}

    def test_metrics_has_all_rows(self, ro):
        expected = {
            "Total Return", "Ann. Return", "Ann. Volatility",
            "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Calmar Ratio",
            "Omega Ratio", "Max DD Duration (days)",
            "Tracking Error", "Information Ratio",
        }
        assert expected == set(ro.metrics.index)

    def test_unhedged_tracking_error_zero(self, ro):
        assert ro.metrics.loc["Tracking Error", "Unhedged"] == 0.0

    def test_cumulative_series_same_length(self, ro):
        assert len(ro.cumulative_unhedged) == len(ro.cumulative_rolling)
        assert len(ro.cumulative_unhedged) == len(ro.cumulative_static)

    def test_cumulative_starts_near_one(self, ro):
        """Cumulative returns should start near 1 + first daily return."""
        np.testing.assert_allclose(
            ro.cumulative_unhedged.iloc[0],
            1.0 + ro.daily_unhedged.iloc[0],
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            ro.cumulative_rolling.iloc[0],
            1.0 + ro.daily_rolling.iloc[0],
            rtol=1e-10,
        )

    def test_daily_rolling_matches_weights_applied(self, ro, returns):
        """Verify first period's daily returns use the first optimization weights."""
        # First optimization weights
        w0 = ro.weight_history.iloc[0].values
        hedges = list(ro.weight_history.columns)
        target = "AAPL"

        # First day of rolling series
        first_date = ro.daily_rolling.index[0]
        cols = [target] + hedges
        clean = returns[cols].dropna()
        r_t = clean.loc[first_date, target]
        r_h = clean.loc[first_date, hedges].values @ w0
        expected = r_t + r_h
        assert ro.daily_rolling.loc[first_date] == pytest.approx(expected, rel=1e-10)

    def test_static_uses_provided_weights(self, ro, returns):
        """Static hedge should use the provided static_weights, not rolling weights."""
        static_w = np.array([-0.4, -0.3, -0.3])
        hedges = list(ro.weight_history.columns)
        target = "AAPL"

        first_date = ro.daily_static.index[0]
        cols = [target] + hedges
        clean = returns[cols].dropna()
        r_t = clean.loc[first_date, target]
        r_h = clean.loc[first_date, hedges].values @ static_w
        expected = r_t + r_h
        assert ro.daily_static.loc[first_date] == pytest.approx(expected, rel=1e-10)

    def test_tracking_error_formula(self, ro):
        """Tracking error should equal annualized std of return differences."""
        diff = ro.daily_rolling - ro.daily_unhedged
        expected_te = float(diff.std() * np.sqrt(252))
        assert ro.metrics.loc["Tracking Error", "Rolling"] == pytest.approx(expected_te, rel=1e-10)

    def test_information_ratio_formula(self, ro):
        """IR = mean(diff)/std(diff) * sqrt(252)."""
        diff = ro.daily_rolling - ro.daily_unhedged
        te = diff.std()
        if te > 0:
            expected_ir = float(diff.mean() / te * np.sqrt(252))
            assert ro.metrics.loc["Information Ratio", "Rolling"] == pytest.approx(expected_ir, rel=1e-10)

    def test_rolling_vol_series_length(self, ro):
        assert len(ro.rolling_vol_unhedged) == len(ro.daily_unhedged)
        assert len(ro.rolling_vol_rolling) == len(ro.daily_rolling)

    def test_max_drawdown_negative_or_zero(self, ro):
        """Max drawdown should be <= 0 for all columns."""
        for col in ro.metrics.columns:
            assert ro.metrics.loc["Max Drawdown", col] <= 0.0

    def test_omega_ratio_positive(self, ro):
        """Omega ratio should be positive (or inf) for all columns."""
        for col in ro.metrics.columns:
            assert ro.metrics.loc["Omega Ratio", col] > 0

    def test_rolling_vol_lower_on_average(self, ro):
        """Rolling hedge vol should on average be <= unhedged (hedge reduces risk)."""
        avg_un = ro.rolling_vol_unhedged.dropna().mean()
        avg_ro = ro.rolling_vol_rolling.dropna().mean()
        # Rolling optimization targets min variance, so should reduce vol
        assert avg_ro <= avg_un * 1.1  # allow 10% slack for edge cases
