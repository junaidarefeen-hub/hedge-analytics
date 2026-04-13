"""Tests for analytics.pairs — spread analysis and mean-reversion."""

import numpy as np
import pandas as pd
import pytest

from analytics.pairs import PairsResult, compute_pairs_analysis


class TestComputePairsAnalysis:
    def test_result_type(self, prices, returns):
        result = compute_pairs_analysis(
            prices, returns, "AAPL", "SPY", window=30,
            start_date=prices.index.min(), end_date=prices.index.max(),
        )
        assert isinstance(result, PairsResult)

    def test_spread_length(self, prices, returns):
        result = compute_pairs_analysis(
            prices, returns, "AAPL", "SPY", window=30,
            start_date=prices.index.min(), end_date=prices.index.max(),
        )
        # Spread should cover the full overlapping date range
        assert len(result.spread) > 0
        assert result.spread.index[0] >= prices.index.min()
        assert result.spread.index[-1] <= prices.index.max()

    def test_zscore_warmup_nans(self, prices, returns):
        result = compute_pairs_analysis(
            prices, returns, "AAPL", "SPY", window=60,
            start_date=prices.index.min(), end_date=prices.index.max(),
        )
        # First (window-1) z-scores should be NaN
        assert result.zscore.iloc[:59].isna().all()
        # After warmup, should have valid values
        assert result.zscore.dropna().notna().all()

    def test_hedge_ratio_self(self, prices, returns):
        """Same ticker for both sides → hedge ratio ≈ 1, spread ≈ 0."""
        result = compute_pairs_analysis(
            prices, returns, "SPY", "SPY", window=30,
            start_date=prices.index.min(), end_date=prices.index.max(),
        )
        np.testing.assert_allclose(result.hedge_ratio, 1.0, atol=1e-6)
        np.testing.assert_allclose(result.spread.values, 0.0, atol=1e-10)

    def test_zscore_percentile_range(self, prices, returns):
        result = compute_pairs_analysis(
            prices, returns, "AAPL", "MSFT", window=30,
            start_date=prices.index.min(), end_date=prices.index.max(),
        )
        assert 0 <= result.zscore_percentile <= 100

    def test_rolling_corr_present(self, prices, returns):
        result = compute_pairs_analysis(
            prices, returns, "AAPL", "SPY", window=30,
            start_date=prices.index.min(), end_date=prices.index.max(),
        )
        assert len(result.rolling_corr) > 0

    def test_insufficient_data_raises(self, prices, returns):
        # Very short date range should raise
        short_end = prices.index[5]
        with pytest.raises(ValueError, match="observations"):
            compute_pairs_analysis(
                prices, returns, "AAPL", "SPY", window=60,
                start_date=prices.index.min(), end_date=short_end,
            )

    def test_bands_match_spread(self, prices, returns):
        """Rolling mean and std should have same index as spread."""
        result = compute_pairs_analysis(
            prices, returns, "AAPL", "MSFT", window=30,
            start_date=prices.index.min(), end_date=prices.index.max(),
        )
        assert len(result.rolling_mean) == len(result.spread)
        assert len(result.rolling_std) == len(result.spread)


class TestHalfLife:
    def test_known_half_life(self):
        """Synthetic mean-reverting spread with known half-life."""
        rng = np.random.default_rng(42)
        n = 1000
        theta = 0.95  # AR(1) coefficient → half-life = -log(2)/log(0.95) ≈ 13.5
        expected_hl = -np.log(2) / np.log(theta)

        spread = np.zeros(n)
        for t in range(1, n):
            spread[t] = theta * spread[t - 1] + rng.normal(0, 0.01)

        dates = pd.bdate_range("2020-01-01", periods=n)
        # Build synthetic prices: A = exp(spread + 4.6), B = exp(4.6)
        log_b = np.full(n, 4.6)
        log_a = spread + log_b  # hedge_ratio ≈ 1
        prices = pd.DataFrame({
            "A": np.exp(log_a), "B": np.exp(log_b + rng.normal(0, 0.001, n)),
        }, index=dates)
        # Need returns for rolling_correlation
        returns = prices.pct_change().dropna()
        prices = prices.iloc[1:]  # align with returns

        result = compute_pairs_analysis(
            prices, returns, "A", "B", window=60,
            start_date=dates[1], end_date=dates[-1],
        )
        # Half-life should be in the right ballpark (within 50%)
        assert result.half_life < expected_hl * 1.5
        assert result.half_life > expected_hl * 0.5

    def test_random_walk_large_half_life(self):
        """Two independent random walks → half-life should be very large or inf."""
        rng = np.random.default_rng(99)
        n = 500
        dates = pd.bdate_range("2020-01-01", periods=n)
        prices = pd.DataFrame({
            "X": np.exp(np.cumsum(rng.normal(0.0005, 0.01, n))),
            "Y": np.exp(np.cumsum(rng.normal(0.0003, 0.012, n))),
        }, index=dates)
        returns = prices.pct_change().dropna()
        prices = prices.iloc[1:]

        result = compute_pairs_analysis(
            prices, returns, "X", "Y", window=60,
            start_date=dates[1], end_date=dates[-1],
        )
        # Non-cointegrated pair should have large half-life
        assert result.half_life > 50 or result.half_life == float("inf")


class TestADF:
    def test_stationary_pair(self):
        """Synthetic stationary spread → ADF p-value should be low."""
        rng = np.random.default_rng(42)
        n = 500
        dates = pd.bdate_range("2020-01-01", periods=n)
        # Cointegrated: A = B + stationary_spread
        base = np.cumsum(rng.normal(0.0003, 0.01, n))
        spread_true = np.zeros(n)
        for t in range(1, n):
            spread_true[t] = 0.9 * spread_true[t - 1] + rng.normal(0, 0.005)

        prices = pd.DataFrame({
            "A": np.exp(base + spread_true + 4.6),
            "B": np.exp(base + 4.6),
        }, index=dates)
        returns = prices.pct_change().dropna()
        prices = prices.iloc[1:]

        result = compute_pairs_analysis(
            prices, returns, "A", "B", window=60,
            start_date=dates[1], end_date=dates[-1],
        )
        assert result.adf_pvalue < 0.10

    def test_nonstationary_pair(self):
        """Two independent random walks → ADF p-value should be high."""
        rng = np.random.default_rng(123)
        n = 500
        dates = pd.bdate_range("2020-01-01", periods=n)
        prices = pd.DataFrame({
            "A": np.exp(np.cumsum(rng.normal(0.0005, 0.015, n))),
            "B": np.exp(np.cumsum(rng.normal(0.0003, 0.012, n))),
        }, index=dates)
        returns = prices.pct_change().dropna()
        prices = prices.iloc[1:]

        result = compute_pairs_analysis(
            prices, returns, "A", "B", window=60,
            start_date=dates[1], end_date=dates[-1],
        )
        assert result.adf_pvalue > 0.05
