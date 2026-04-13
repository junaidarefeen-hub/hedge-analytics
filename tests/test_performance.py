"""Tests for analytics.performance — price performance statistics."""

import numpy as np
import pandas as pd
import pytest

from analytics.performance import PerformanceResult, compute_performance_stats


class TestComputePerformanceStats:
    def test_result_type(self, returns):
        result = compute_performance_stats(
            returns, ["AAPL", "MSFT"], "SPY",
            returns.index.min(), returns.index.max(),
        )
        assert isinstance(result, PerformanceResult)

    def test_absolute_includes_benchmark(self, returns):
        result = compute_performance_stats(
            returns, ["AAPL", "MSFT"], "SPY",
            returns.index.min(), returns.index.max(),
        )
        assert "SPY (benchmark)" in result.absolute.columns

    def test_absolute_includes_all_tickers(self, returns):
        tickers = ["AAPL", "MSFT"]
        result = compute_performance_stats(
            returns, tickers, "SPY",
            returns.index.min(), returns.index.max(),
        )
        for tk in tickers:
            assert tk in result.absolute.columns

    def test_relative_bench_excludes_benchmark(self, returns):
        result = compute_performance_stats(
            returns, ["AAPL", "MSFT"], "SPY",
            returns.index.min(), returns.index.max(),
        )
        assert "SPY" not in result.relative_bench.columns
        assert "SPY (benchmark)" not in result.relative_bench.columns
        assert "SPY" not in result.beta_adjusted.columns

    def test_relative_peers_none_without_peers(self, returns):
        result = compute_performance_stats(
            returns, ["AAPL", "MSFT"], "SPY",
            returns.index.min(), returns.index.max(),
        )
        assert result.relative_peers is None

    def test_relative_peers_computed(self, returns):
        result = compute_performance_stats(
            returns, ["AAPL", "XLE"], "SPY",
            returns.index.min(), returns.index.max(),
            peer_tickers=["MSFT", "QQQ"],
        )
        assert result.relative_peers is not None
        assert "AAPL" in result.relative_peers.columns
        assert result.relative_peers.shape[0] == 4  # 4 relative metrics

    def test_peer_equal_weight(self, returns):
        """Equal-weight peer basket return ≈ mean of peer returns."""
        peer_tks = ["MSFT", "QQQ"]
        result = compute_performance_stats(
            returns, ["AAPL"], "SPY",
            returns.index.min(), returns.index.max(),
            peer_tickers=peer_tks,
        )
        expected_basket = returns[peer_tks].mean(axis=1)
        pd.testing.assert_series_equal(
            result.peer_basket_returns.dropna(),
            expected_basket.dropna(),
            check_names=False,
            atol=1e-12,
        )

    def test_peer_custom_weights(self, returns):
        """Custom-weighted peer basket uses supplied weights."""
        peer_tks = ["MSFT", "QQQ"]
        weights = np.array([0.7, 0.3])
        result = compute_performance_stats(
            returns, ["AAPL"], "SPY",
            returns.index.min(), returns.index.max(),
            peer_tickers=peer_tks,
            peer_weights=weights,
        )
        expected_basket = (returns[peer_tks] * weights).sum(axis=1)
        pd.testing.assert_series_equal(
            result.peer_basket_returns.dropna(),
            expected_basket.dropna(),
            check_names=False,
            atol=1e-12,
        )

    def test_absolute_metrics_present(self, returns):
        result = compute_performance_stats(
            returns, ["AAPL"], "SPY",
            returns.index.min(), returns.index.max(),
        )
        expected_metrics = {
            "Total Return", "Ann. Return", "Ann. Volatility",
            "Sharpe Ratio", "Sortino Ratio", "Max Drawdown",
        }
        assert expected_metrics == set(result.absolute.index)

    def test_beta_adjusted_metrics_present(self, returns):
        result = compute_performance_stats(
            returns, ["AAPL"], "SPY",
            returns.index.min(), returns.index.max(),
        )
        expected = {"Beta", "Beta-Adj Return", "Ann. Alpha", "Residual Volatility"}
        assert expected == set(result.beta_adjusted.index)

    def test_cumulative_shape(self, returns):
        result = compute_performance_stats(
            returns, ["AAPL", "MSFT"], "SPY",
            returns.index.min(), returns.index.max(),
        )
        assert "AAPL" in result.cumulative_abs.columns
        assert "SPY (benchmark)" in result.cumulative_abs.columns
        assert "AAPL" in result.cumulative_beta_adj.columns
        # Beta-adjusted should not have benchmark
        assert "SPY (benchmark)" not in result.cumulative_beta_adj.columns

    def test_date_filtering(self, returns):
        mid = returns.index[len(returns) // 2]
        full = compute_performance_stats(
            returns, ["AAPL"], "SPY",
            returns.index.min(), returns.index.max(),
        )
        subset = compute_performance_stats(
            returns, ["AAPL"], "SPY",
            mid, returns.index.max(),
        )
        assert len(subset.cumulative_abs) < len(full.cumulative_abs)

    def test_insufficient_data_raises(self, returns):
        # Single day range
        single_day = returns.index[0]
        with pytest.raises(ValueError, match="at least 2"):
            compute_performance_stats(
                returns, ["AAPL"], "SPY",
                single_day, single_day,
            )


class TestBetaAccuracy:
    def test_known_beta_2x(self):
        """Synthetic 2x series should produce beta ≈ 2.0."""
        dates = pd.bdate_range("2023-01-01", periods=200)
        rng = np.random.default_rng(99)
        base = rng.normal(0.001, 0.01, 200)
        r = pd.DataFrame({"A": base * 2, "BENCH": base}, index=dates)
        result = compute_performance_stats(
            r, ["A"], "BENCH", dates[0], dates[-1],
        )
        np.testing.assert_allclose(
            result.beta_adjusted.loc["Beta", "A"], 2.0, atol=0.05,
        )

    def test_perfect_tracking_zero_te(self):
        """Identical ticker and benchmark → TE ≈ 0, IR ≈ 0."""
        dates = pd.bdate_range("2023-01-01", periods=200)
        rng = np.random.default_rng(42)
        base = rng.normal(0.001, 0.01, 200)
        r = pd.DataFrame({"CLONE": base, "BENCH": base}, index=dates)
        result = compute_performance_stats(
            r, ["CLONE"], "BENCH", dates[0], dates[-1],
        )
        assert abs(result.relative_bench.loc["Tracking Error", "CLONE"]) < 1e-10
        assert abs(result.relative_bench.loc["Information Ratio", "CLONE"]) < 1e-10
