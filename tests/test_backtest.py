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


class TestInformationRatio:
    def test_identical_series_zero(self):
        daily = pd.Series(np.random.default_rng(1).normal(0, 0.01, 200))
        assert _information_ratio(daily, daily) == pytest.approx(0.0)


class TestOmegaRatio:
    def test_all_positive_returns(self):
        daily = pd.Series([0.01, 0.02, 0.03, 0.01, 0.005])
        assert _omega_ratio(daily) == float("inf")

    def test_mixed_returns(self):
        daily = pd.Series([0.01, -0.01, 0.02, -0.005])
        omega = _omega_ratio(daily)
        assert omega > 0


class TestMaxDrawdownDuration:
    def test_monotonic_series_zero(self):
        cum = pd.Series(np.linspace(1.0, 2.0, 100))
        assert _max_drawdown_duration(cum) == 0

    def test_known_drawdown(self):
        cum = pd.Series([1.0, 1.1, 1.0, 0.9, 1.0, 1.1, 1.2])
        duration = _max_drawdown_duration(cum)
        assert duration > 0


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
