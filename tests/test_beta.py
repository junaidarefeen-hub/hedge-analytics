"""Tests for analytics.beta — beta_matrix() & rolling_beta()."""

import numpy as np
import pandas as pd

from analytics.beta import beta_matrix, rolling_beta


class TestBetaMatrix:
    def test_shape(self, returns):
        benchmarks = ["SPY", "QQQ"]
        betas = beta_matrix(returns, benchmarks)
        non_bm_tickers = [c for c in returns.columns if c not in benchmarks]
        assert betas.shape == (len(non_bm_tickers), len(benchmarks))

    def test_benchmark_self_beta_excluded(self, returns):
        """Benchmarks should not appear as row tickers (by default)."""
        benchmarks = ["SPY"]
        betas = beta_matrix(returns, benchmarks)
        assert "SPY" not in betas.index

    def test_known_beta_perfectly_correlated(self):
        """Two perfectly correlated series with 2x scale → beta should be 2."""
        dates = pd.bdate_range("2023-01-01", periods=100)
        rng = np.random.default_rng(99)
        base = rng.normal(0, 0.01, 100)
        r = pd.DataFrame({"A": base * 2, "B": base}, index=dates)
        betas = beta_matrix(r, ["B"])
        np.testing.assert_allclose(betas.loc["A", "B"], 2.0, atol=1e-10)

    def test_no_nans_with_sufficient_data(self, returns):
        betas = beta_matrix(returns, ["SPY"])
        assert not betas.isna().any().any()


class TestRollingBeta:
    def test_length(self, returns):
        rb = rolling_beta(returns, "AAPL", "SPY", window=30)
        # dropna inside rolling_beta, so length depends on overlap
        assert len(rb) > 0

    def test_initial_nans(self, returns):
        window = 30
        rb = rolling_beta(returns, "AAPL", "SPY", window=window)
        assert rb.iloc[: window - 1].isna().all()

    def test_name(self, returns):
        rb = rolling_beta(returns, "AAPL", "SPY", window=30)
        assert "AAPL" in rb.name and "SPY" in rb.name
