"""Tests for utils.basket — basket construction and injection utilities."""

import numpy as np
import pandas as pd
import pytest

from utils.basket import (
    BASKET_COLUMN_NAME,
    basket_display_name,
    exclude_basket_constituents,
    inject_basket_column,
    is_basket,
)


class TestInjectBasketColumn:
    def test_single_ticker_passthrough(self, returns):
        """Single ticker returns original df unmodified."""
        result_df, col = inject_basket_column(returns, ["AAPL"], np.array([1.0]))
        assert result_df is returns  # exact same object, zero copy
        assert col == "AAPL"

    def test_multi_ticker_creates_synthetic(self, returns):
        """Multi-ticker creates a new column with weighted sum."""
        weights = np.array([0.6, 0.4])
        result_df, col = inject_basket_column(returns, ["AAPL", "MSFT"], weights)
        assert col == BASKET_COLUMN_NAME
        assert BASKET_COLUMN_NAME in result_df.columns
        # Original columns still present
        assert "AAPL" in result_df.columns
        assert "MSFT" in result_df.columns

    def test_multi_ticker_weighted_sum_correct(self, returns):
        """Synthetic column equals weighted sum of constituents."""
        weights = np.array([0.6, 0.4])
        result_df, col = inject_basket_column(returns, ["AAPL", "MSFT"], weights)
        expected = returns["AAPL"] * 0.6 + returns["MSFT"] * 0.4
        pd.testing.assert_series_equal(result_df[col], expected, check_names=False)

    def test_equal_weight_basket_is_average(self, returns):
        """Equal-weight basket returns == simple average of constituents."""
        tickers = ["AAPL", "MSFT", "SPY"]
        weights = np.array([1 / 3, 1 / 3, 1 / 3])
        result_df, col = inject_basket_column(returns, tickers, weights)
        expected = returns[tickers].mean(axis=1)
        pd.testing.assert_series_equal(result_df[col], expected, check_names=False, atol=1e-10)

    def test_does_not_modify_original(self, returns):
        """Injection should not modify the original DataFrame."""
        orig_cols = set(returns.columns)
        inject_basket_column(returns, ["AAPL", "MSFT"], np.array([0.5, 0.5]))
        assert set(returns.columns) == orig_cols


class TestExcludeBasketConstituents:
    def test_removes_basket_members(self):
        candidates = ["AAPL", "MSFT", "SPY", "QQQ", "XLE"]
        result = exclude_basket_constituents(candidates, ["AAPL", "MSFT"])
        assert result == ["SPY", "QQQ", "XLE"]

    def test_single_ticker(self):
        candidates = ["AAPL", "MSFT", "SPY"]
        result = exclude_basket_constituents(candidates, ["AAPL"])
        assert result == ["MSFT", "SPY"]

    def test_no_overlap(self):
        candidates = ["SPY", "QQQ"]
        result = exclude_basket_constituents(candidates, ["AAPL", "MSFT"])
        assert result == ["SPY", "QQQ"]

    def test_empty_basket(self):
        candidates = ["AAPL", "MSFT"]
        result = exclude_basket_constituents(candidates, [])
        assert result == ["AAPL", "MSFT"]


class TestBasketDisplayName:
    def test_single_ticker(self):
        assert basket_display_name(["AAPL"], np.array([1.0])) == "AAPL"

    def test_two_tickers_equal(self):
        name = basket_display_name(["AAPL", "MSFT"], np.array([0.5, 0.5]))
        assert name == "AAPL (50%) + MSFT (50%)"

    def test_three_tickers_unequal(self):
        name = basket_display_name(["AAPL", "MSFT", "SPY"], np.array([0.5, 0.3, 0.2]))
        assert name == "AAPL (50%) + MSFT (30%) + SPY (20%)"


class TestIsBasket:
    def test_basket_name_detected(self):
        assert is_basket(BASKET_COLUMN_NAME) is True

    def test_normal_ticker_not_basket(self):
        assert is_basket("AAPL") is False

    def test_empty_string(self):
        assert is_basket("") is False
