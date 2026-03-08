"""Tests for analytics.correlation — correlation_matrix() & rolling_correlation()."""

import numpy as np

from analytics.correlation import correlation_matrix, rolling_correlation


class TestCorrelationMatrix:
    def test_shape(self, returns):
        corr = correlation_matrix(returns)
        n = returns.shape[1]
        assert corr.shape == (n, n)

    def test_diagonal_is_one(self, returns):
        corr = correlation_matrix(returns)
        np.testing.assert_allclose(np.diag(corr.values), 1.0)

    def test_symmetric(self, returns):
        corr = correlation_matrix(returns)
        np.testing.assert_allclose(corr.values, corr.values.T)

    def test_values_bounded(self, returns):
        corr = correlation_matrix(returns)
        assert (corr.values >= -1.0 - 1e-10).all()
        assert (corr.values <= 1.0 + 1e-10).all()


class TestRollingCorrelation:
    def test_length(self, returns):
        rc = rolling_correlation(returns, "AAPL", "SPY", window=30)
        assert len(rc) == len(returns)

    def test_initial_nans(self, returns):
        window = 30
        rc = rolling_correlation(returns, "AAPL", "SPY", window=window)
        # First (window - 1) values should be NaN
        assert rc.iloc[: window - 1].isna().all()

    def test_values_bounded(self, returns):
        rc = rolling_correlation(returns, "AAPL", "SPY", window=30).dropna()
        assert (rc >= -1.0 - 1e-10).all()
        assert (rc <= 1.0 + 1e-10).all()
