"""Tests for analytics.correlation — correlation_matrix(), rolling_correlation(), correlation_clustering()."""

import numpy as np
import pytest

from analytics.correlation import correlation_clustering, correlation_matrix, rolling_correlation


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


class TestCorrelationClustering:
    def test_dict_keys(self, returns):
        corr = correlation_matrix(returns)
        result = correlation_clustering(corr)
        assert "linkage_matrix" in result
        assert "labels" in result
        assert "distance_matrix" in result

    def test_linkage_shape(self, returns):
        corr = correlation_matrix(returns)
        result = correlation_clustering(corr)
        n = len(corr)
        assert result["linkage_matrix"].shape == (n - 1, 4)

    def test_labels_match_columns(self, returns):
        corr = correlation_matrix(returns)
        result = correlation_clustering(corr)
        assert result["labels"] == list(corr.columns)

    def test_min_tickers_required(self):
        """Should raise with fewer than 3 tickers."""
        import pandas as pd
        corr = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], columns=["A", "B"], index=["A", "B"])
        with pytest.raises(ValueError, match="3 tickers"):
            correlation_clustering(corr)
