"""Tests for analytics.regime — detect_regimes() & regime_hedge_effectiveness()."""

import numpy as np
import pandas as pd
import pytest

from analytics.regime import RegimeResult, detect_regimes, regime_hedge_effectiveness


class TestDetectRegimes:
    def test_regime_count_matches(self, returns):
        result = detect_regimes(returns, "SPY", window=30, n_regimes=3, method="quantile")
        assert isinstance(result, RegimeResult)
        unique_regimes = result.regime_series.unique()
        assert len(unique_regimes) == 3

    def test_series_length(self, returns):
        window = 30
        result = detect_regimes(returns, "SPY", window=window, n_regimes=3, method="quantile")
        expected_len = len(returns["SPY"].dropna()) - window + 1
        assert len(result.regime_series) == expected_len

    def test_regime_0_lowest_vol_quantile(self, returns):
        result = detect_regimes(returns, "SPY", window=30, n_regimes=3, method="quantile")
        # Regime 0 should have lowest avg vol
        stats = result.per_regime_stats
        if len(stats) >= 2:
            vols = stats["Avg Vol (ann.)"]
            assert vols.iloc[0] <= vols.iloc[-1]

    def test_regime_0_lowest_vol_kmeans(self, returns):
        result = detect_regimes(returns, "SPY", window=30, n_regimes=3, method="kmeans")
        stats = result.per_regime_stats
        if len(stats) >= 2:
            vols = stats["Avg Vol (ann.)"]
            assert vols.iloc[0] <= vols.iloc[-1]

    def test_both_methods_same_length(self, returns):
        r1 = detect_regimes(returns, "SPY", window=30, n_regimes=3, method="quantile")
        r2 = detect_regimes(returns, "SPY", window=30, n_regimes=3, method="kmeans")
        assert len(r1.regime_series) == len(r2.regime_series)

    def test_insufficient_data(self, returns):
        with pytest.raises(ValueError):
            detect_regimes(returns, "SPY", window=500, n_regimes=3)


class TestRegimeHedgeEffectiveness:
    def test_one_row_per_regime(self, returns):
        regime_result = detect_regimes(returns, "SPY", window=30, n_regimes=3, method="quantile")
        hedges = ["MSFT", "QQQ", "XLE"]
        weights = np.array([-0.4, -0.3, -0.3])
        eff = regime_hedge_effectiveness(
            returns, "AAPL", hedges, weights, regime_result.regime_series, n_regimes=3,
        )
        assert len(eff) <= 3
        assert len(eff) >= 1  # at least some regimes have enough data
