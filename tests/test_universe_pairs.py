"""Tests for analytics.universe_pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.universe_pairs import (
    PairCandidateRow,
    _select_pair_candidates,
    scan_industry_pairs,
)


@pytest.fixture()
def industry_universe(rng):
    """Synthetic universe with two industries; intra-industry tickers are
    highly correlated so the scan has something to chew on."""
    n_days = 500
    dates = pd.bdate_range("2023-01-02", periods=n_days, freq="B")

    # Industry A — 4 tickers driven by a common factor + small idio noise.
    factor_a = rng.normal(0.0003, 0.012, n_days)
    industry_a = {
        f"A{i}": np.exp(np.cumsum(factor_a + rng.normal(0, 0.004, n_days))) * 100
        for i in range(4)
    }

    # Industry B — 3 tickers, separate factor.
    factor_b = rng.normal(0.0001, 0.018, n_days)
    industry_b = {
        f"B{i}": np.exp(np.cumsum(factor_b + rng.normal(0, 0.005, n_days))) * 100
        for i in range(3)
    }

    prices = pd.DataFrame({**industry_a, **industry_b}, index=dates)
    returns = prices.pct_change(fill_method=None).dropna()
    industry_map = {**{t: "Industry A" for t in industry_a},
                    **{t: "Industry B" for t in industry_b}}
    return prices, returns, industry_map


class TestSelectPairCandidates:
    def test_returns_top_k(self, industry_universe):
        _, returns, _ = industry_universe
        a_returns = returns[[c for c in returns.columns if c.startswith("A")]]
        pairs = _select_pair_candidates(a_returns, top_k=2)
        assert len(pairs) <= 2

    def test_correlations_descending(self, industry_universe):
        _, returns, _ = industry_universe
        a_returns = returns[[c for c in returns.columns if c.startswith("A")]]
        pairs = _select_pair_candidates(a_returns, top_k=4)
        corrs = [p[2] for p in pairs]
        assert corrs == sorted(corrs, reverse=True)

    def test_below_min_corr_filtered(self):
        # Two perfectly anticorrelated series; abs corr = 1, but adjust
        # min_corr threshold above 1 to verify filtering works.
        n = 100
        idx = pd.bdate_range("2024-01-02", periods=n, freq="B")
        x = np.random.RandomState(0).randn(n)
        df = pd.DataFrame({"X": x, "Y": -x}, index=idx)
        pairs = _select_pair_candidates(df, top_k=5, min_corr=1.5)
        assert pairs == []


class TestScanIndustryPairs:
    def test_columns(self, industry_universe):
        prices, returns, industry_map = industry_universe
        df = scan_industry_pairs(
            prices, returns, industry_map,
            window=60, top_k_per_industry=2,
        )
        for col in ["ticker_a", "ticker_b", "industry", "current_zscore",
                    "half_life_days", "adf_pvalue", "rolling_corr_60d"]:
            assert col in df.columns

    def test_pairs_all_within_industry(self, industry_universe):
        prices, returns, industry_map = industry_universe
        df = scan_industry_pairs(
            prices, returns, industry_map,
            window=60, top_k_per_industry=3,
        )
        for _, row in df.iterrows():
            assert industry_map[row["ticker_a"]] == industry_map[row["ticker_b"]]
            assert industry_map[row["ticker_a"]] == row["industry"]

    def test_top_k_caps_pairs_per_industry(self, industry_universe):
        prices, returns, industry_map = industry_universe
        df = scan_industry_pairs(
            prices, returns, industry_map,
            window=60, top_k_per_industry=2,
        )
        per_industry = df["industry"].value_counts()
        assert (per_industry <= 2).all()

    def test_sorted_by_abs_zscore_desc(self, industry_universe):
        prices, returns, industry_map = industry_universe
        df = scan_industry_pairs(
            prices, returns, industry_map,
            window=60, top_k_per_industry=3,
        )
        if len(df) > 1:
            absz = df["current_zscore"].abs().values
            assert (absz[:-1] >= absz[1:]).all()

    def test_singleton_industry_yields_no_pairs(self):
        n = 300
        idx = pd.bdate_range("2024-01-02", periods=n, freq="B")
        prices = pd.DataFrame(
            {"X": np.cumsum(np.ones(n)) + 100, "Y": np.cumsum(np.ones(n)) + 50},
            index=idx,
        )
        returns = prices.pct_change(fill_method=None).dropna()
        industry_map = {"X": "Solo A", "Y": "Solo B"}  # different industries
        df = scan_industry_pairs(
            prices, returns, industry_map,
            window=60, top_k_per_industry=2, min_observations=50,
        )
        assert df.empty
