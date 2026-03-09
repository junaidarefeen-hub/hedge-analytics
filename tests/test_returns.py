"""Tests for analytics.returns — compute_returns()."""

import numpy as np
import pandas as pd

from analytics.returns import compute_returns


class TestComputeReturns:
    def test_log_returns_shape(self, prices):
        ret = compute_returns(prices, method="log")
        # One row dropped (shift produces NaN on first row)
        assert ret.shape == (prices.shape[0] - 1, prices.shape[1])

    def test_simple_returns_shape(self, prices):
        ret = compute_returns(prices, method="simple")
        assert ret.shape == (prices.shape[0] - 1, prices.shape[1])

    def test_log_returns_values(self, prices):
        ret = compute_returns(prices, method="log")
        # Manual check for first valid row
        expected = np.log(prices.iloc[1] / prices.iloc[0])
        pd.testing.assert_series_equal(ret.iloc[0], expected, check_names=False)

    def test_simple_returns_values(self, prices):
        ret = compute_returns(prices, method="simple")
        expected = prices.iloc[1] / prices.iloc[0] - 1
        pd.testing.assert_series_equal(ret.iloc[0], expected, check_names=False)

    def test_no_nans_in_output(self, prices):
        ret = compute_returns(prices, method="log")
        assert not ret.isna().all(axis=1).any()

    def test_default_method_is_simple(self, prices):
        ret_default = compute_returns(prices)
        ret_simple = compute_returns(prices, method="simple")
        pd.testing.assert_frame_equal(ret_default, ret_simple)
