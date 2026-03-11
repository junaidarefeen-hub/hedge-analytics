"""Tests for analytics.backtest — run_backtest() & helpers."""

import numpy as np
import pandas as pd
import pytest

from analytics.backtest import (
    BacktestResult,
    DynamicBacktestResult,
    _compute_metrics,
    _information_ratio,
    _max_drawdown,
    _max_drawdown_duration,
    _omega_ratio,
    _tracking_error,
    run_backtest,
    run_dynamic_backtest,
)


class TestMaxDrawdown:
    def test_no_drawdown(self):
        cum = pd.Series([1.0, 1.1, 1.2, 1.3])
        assert _max_drawdown(cum) == pytest.approx(0.0)

    def test_known_drawdown(self):
        cum = pd.Series([1.0, 1.2, 0.9, 1.1])
        # Peak at 1.2, trough at 0.9 → drawdown = (0.9 - 1.2) / 1.2 = -0.25
        assert _max_drawdown(cum) == pytest.approx(-0.25)

    def test_always_negative(self):
        """Max drawdown should be <= 0."""
        rng = np.random.default_rng(7)
        cum = pd.Series(np.cumprod(1 + rng.normal(0, 0.02, 200)))
        assert _max_drawdown(cum) <= 0.0


class TestComputeMetrics:
    def test_keys_present(self):
        daily = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 200))
        cum = (1 + daily).cumprod()
        m = _compute_metrics(daily, cum, 0.05, "Test")
        expected_keys = {
            "Total Return", "Ann. Return", "Ann. Volatility",
            "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Calmar Ratio",
            "Omega Ratio", "Max DD Duration (days)",
        }
        assert expected_keys == set(m.keys())

    def test_zero_vol_sharpe(self):
        """Zero-volatility series → Sharpe should be 0."""
        daily = pd.Series([0.0] * 100)
        cum = pd.Series([1.0] * 100)
        m = _compute_metrics(daily, cum, 0.05, "Flat")
        assert m["Sharpe Ratio"] == 0.0


class TestRunBacktest:
    @pytest.fixture()
    def bt(self, returns):
        hedges = ["MSFT", "SPY", "QQQ"]
        weights = np.array([-0.4, -0.3, -0.3])
        return run_backtest(returns, "AAPL", hedges, weights)

    def test_returns_backtest_result(self, bt):
        assert isinstance(bt, BacktestResult)

    def test_cumulative_starts_near_one(self, bt):
        np.testing.assert_allclose(bt.cumulative_unhedged.iloc[0], 1.0 + bt.daily_unhedged.iloc[0], rtol=1e-10)
        np.testing.assert_allclose(bt.cumulative_hedged.iloc[0], 1.0 + bt.daily_hedged.iloc[0], rtol=1e-10)

    def test_metrics_columns(self, bt):
        assert "Unhedged" in bt.metrics.columns
        assert "Hedged" in bt.metrics.columns

    def test_metrics_rows(self, bt):
        expected_rows = {
            "Total Return", "Ann. Return", "Ann. Volatility",
            "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Calmar Ratio",
            "Omega Ratio", "Max DD Duration (days)", "Tracking Error", "Information Ratio",
        }
        assert expected_rows == set(bt.metrics.index)

    def test_date_filtering(self, returns):
        hedges = ["MSFT", "SPY"]
        weights = np.array([-0.5, -0.5])
        start = returns.index[50]
        end = returns.index[150]
        bt = run_backtest(returns, "AAPL", hedges, weights, start_date=start, end_date=end)
        assert bt.daily_hedged.index[0] >= start
        assert bt.daily_hedged.index[-1] <= end

    def test_insufficient_data_raises(self, returns):
        hedges = ["MSFT"]
        weights = np.array([-1.0])
        with pytest.raises(ValueError, match="Not enough data"):
            run_backtest(
                returns, "AAPL", hedges, weights,
                start_date=returns.index[-1],
                end_date=returns.index[-1],
            )


class TestTrackingError:
    def test_identical_series_zero(self):
        daily = pd.Series(np.random.default_rng(1).normal(0, 0.01, 200))
        assert _tracking_error(daily, daily) == pytest.approx(0.0)

    def test_positive(self):
        rng = np.random.default_rng(2)
        a = pd.Series(rng.normal(0, 0.01, 200))
        b = pd.Series(rng.normal(0, 0.01, 200))
        assert _tracking_error(a, b) > 0

    def test_known_value(self):
        """Hand-calculated: TE = std(diff) * sqrt(252)."""
        a = pd.Series([0.01, 0.02, -0.01, 0.005])
        b = pd.Series([0.005, 0.01, -0.005, 0.002])
        diff = a - b  # [0.005, 0.01, -0.005, 0.003]
        expected = float(diff.std() * np.sqrt(252))
        assert _tracking_error(a, b) == pytest.approx(expected, rel=1e-10)


class TestInformationRatio:
    def test_identical_series_zero(self):
        daily = pd.Series(np.random.default_rng(1).normal(0, 0.01, 200))
        assert _information_ratio(daily, daily) == pytest.approx(0.0)

    def test_known_value(self):
        """IR = mean(diff)/std(diff) * sqrt(252)."""
        a = pd.Series([0.02, 0.03, 0.01, 0.02])
        b = pd.Series([0.01, 0.01, 0.01, 0.01])
        diff = a - b
        expected = float(diff.mean() / diff.std() * np.sqrt(252))
        assert _information_ratio(a, b) == pytest.approx(expected, rel=1e-10)

    def test_negative_when_hedged_worse(self):
        """Hedged underperforms unhedged → negative IR."""
        hedged = pd.Series([-0.01, -0.02, -0.01, -0.01])
        unhedged = pd.Series([0.01, 0.02, 0.01, 0.01])
        assert _information_ratio(hedged, unhedged) < 0


class TestOmegaRatio:
    def test_all_positive_returns(self):
        daily = pd.Series([0.01, 0.02, 0.03, 0.01, 0.005])
        assert _omega_ratio(daily) == float("inf")

    def test_mixed_returns(self):
        daily = pd.Series([0.01, -0.01, 0.02, -0.005])
        omega = _omega_ratio(daily)
        assert omega > 0

    def test_known_value(self):
        """Hand-calculated: omega = sum(gains) / sum(losses) at threshold=0."""
        daily = pd.Series([0.02, -0.01, 0.03, -0.02])
        # gains = 0.02 + 0.03 = 0.05, losses = 0.01 + 0.02 = 0.03
        assert _omega_ratio(daily) == pytest.approx(0.05 / 0.03, rel=1e-10)

    def test_all_negative_returns(self):
        daily = pd.Series([-0.01, -0.02, -0.005])
        # gains = 0, losses > 0 → omega should be 0
        assert _omega_ratio(daily) == pytest.approx(0.0)

    def test_custom_threshold(self):
        """With threshold=0.01, only excess above 0.01 counts as gain."""
        daily = pd.Series([0.02, 0.005, -0.01])
        # excess = [0.01, -0.005, -0.02], gains = 0.01, losses = 0.025
        assert _omega_ratio(daily, threshold=0.01) == pytest.approx(0.01 / 0.025, rel=1e-10)


class TestMaxDrawdownDuration:
    def test_monotonic_series_zero(self):
        cum = pd.Series(np.linspace(1.0, 2.0, 100))
        assert _max_drawdown_duration(cum) == 0

    def test_known_duration(self):
        """Series: up, down for 3 periods, then recover — duration should be 3."""
        cum = pd.Series([1.0, 1.1, 1.0, 0.95, 0.98, 1.1, 1.2])
        # Underwater from index 2-4 (3 periods below peak of 1.1)
        assert _max_drawdown_duration(cum) == 3


class TestDynamicBacktest:
    def test_result_type(self, returns):
        hedges = ["MSFT", "SPY", "QQQ"]
        weights = np.array([-0.4, -0.3, -0.3])
        result = run_dynamic_backtest(
            returns, "AAPL", hedges, weights,
            strategy="Minimum Variance",
            bounds=(-1.0, 0.0),
            rebalance_freq="monthly",
            lookback_window=60,
        )
        assert isinstance(result, DynamicBacktestResult)

    def test_metrics_has_three_columns(self, returns):
        hedges = ["MSFT", "SPY", "QQQ"]
        weights = np.array([-0.4, -0.3, -0.3])
        result = run_dynamic_backtest(
            returns, "AAPL", hedges, weights,
            strategy="Minimum Variance",
            bounds=(-1.0, 0.0),
            rebalance_freq="monthly",
            lookback_window=60,
        )
        assert set(result.metrics.columns) == {"Unhedged", "Static", "Dynamic"}

    def test_turnover_nonnegative(self, returns):
        hedges = ["MSFT", "SPY", "QQQ"]
        weights = np.array([-0.4, -0.3, -0.3])
        result = run_dynamic_backtest(
            returns, "AAPL", hedges, weights,
            strategy="Minimum Variance",
            bounds=(-1.0, 0.0),
            rebalance_freq="monthly",
            lookback_window=60,
        )
        assert (result.turnover >= 0).all()

    def test_weekly_more_rebalances_than_monthly(self, returns):
        hedges = ["MSFT", "SPY"]
        weights = np.array([-0.5, -0.5])
        r_w = run_dynamic_backtest(
            returns, "AAPL", hedges, weights,
            strategy="Minimum Variance", bounds=(-1.0, 0.0),
            rebalance_freq="weekly", lookback_window=60,
        )
        r_m = run_dynamic_backtest(
            returns, "AAPL", hedges, weights,
            strategy="Minimum Variance", bounds=(-1.0, 0.0),
            rebalance_freq="monthly", lookback_window=60,
        )
        assert len(r_w.rebalance_dates) >= len(r_m.rebalance_dates)


class TestBacktestWithBasket:
    def test_backtest_with_basket_target(self, returns):
        """run_backtest works with a synthetic basket column as target."""
        from utils.basket import inject_basket_column
        weights_long = np.array([0.5, 0.5])
        aug_returns, target = inject_basket_column(returns, ["AAPL", "MSFT"], weights_long)
        hedges = ["SPY", "QQQ", "XLE"]
        hedge_weights = np.array([-0.4, -0.3, -0.3])
        bt = run_backtest(aug_returns, target, hedges, hedge_weights)
        assert isinstance(bt, BacktestResult)
        assert bt.metrics.loc["Ann. Volatility", "Hedged"] > 0
        assert len(bt.daily_hedged) > 0

    def test_basket_backtest_different_from_single(self, returns):
        """Basket target should produce different results than single ticker."""
        from utils.basket import inject_basket_column
        # Single ticker
        bt_single = run_backtest(returns, "AAPL", ["SPY", "QQQ"], np.array([-0.5, -0.5]))
        # Basket
        aug, target = inject_basket_column(returns, ["AAPL", "MSFT"], np.array([0.5, 0.5]))
        bt_basket = run_backtest(aug, target, ["SPY", "QQQ"], np.array([-0.5, -0.5]))
        # Metrics should differ (different target)
        assert bt_single.metrics.loc["Ann. Volatility", "Unhedged"] != bt_basket.metrics.loc["Ann. Volatility", "Unhedged"]
