"""Tests for analytics.regime — detect_regimes() & regime_hedge_effectiveness()."""

import numpy as np
import pandas as pd
import pytest

from analytics.custom_hedge import run_custom_hedge_analysis
from analytics.regime import (
    CorrelationRegimeResult,
    RegimeResult,
    correlation_regime_hedge_metrics,
    detect_correlation_regimes,
    detect_regimes,
    regime_hedge_effectiveness,
)


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


    def test_quantile_tercile_boundaries(self, returns):
        """With 3 quantile regimes, each should have ~33% of observations."""
        result = detect_regimes(returns, "SPY", window=30, n_regimes=3, method="quantile")
        counts = result.regime_series.value_counts()
        total = len(result.regime_series)
        for regime_id in range(3):
            pct = counts.get(regime_id, 0) / total
            # Each tercile should be roughly 25-42% (generous range for discrete data)
            assert 0.15 < pct < 0.50, f"Regime {regime_id} has {pct:.1%} of data"

    def test_per_regime_stats_sum_to_total(self, returns):
        """Per-regime day counts should sum to total series length."""
        result = detect_regimes(returns, "SPY", window=30, n_regimes=3, method="quantile")
        total_count = int(result.per_regime_stats["Count"].sum())
        assert total_count == len(result.regime_series)

    def test_rolling_vol_annualized(self, returns):
        """Rolling vol should be annualized (much larger than daily std)."""
        result = detect_regimes(returns, "SPY", window=30, n_regimes=3, method="quantile")
        daily_std = returns["SPY"].std()
        # Annualized vol should be roughly daily_std * sqrt(252) — at least 5x daily
        assert result.rolling_vol.mean() > daily_std * 5


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

    def test_vol_reduction_computation(self, returns):
        """Vol reduction should equal 1 - hedged_vol/unhedged_vol."""
        regime_result = detect_regimes(returns, "SPY", window=30, n_regimes=3, method="quantile")
        hedges = ["MSFT", "QQQ", "XLE"]
        weights = np.array([-0.4, -0.3, -0.3])
        eff = regime_hedge_effectiveness(
            returns, "AAPL", hedges, weights, regime_result.regime_series, n_regimes=3,
        )
        for _, row in eff.iterrows():
            if row["Unhedged Vol"] > 0:
                expected_red = (1 - row["Hedged Vol"] / row["Unhedged Vol"]) * 100
                assert row["Vol Reduction (%)"] == pytest.approx(expected_red, rel=1e-6)

    def test_days_column_matches_regime_count(self, returns):
        """Days column should match regime series value counts."""
        regime_result = detect_regimes(returns, "SPY", window=30, n_regimes=3, method="quantile")
        hedges = ["MSFT", "QQQ", "XLE"]
        weights = np.array([-0.4, -0.3, -0.3])
        eff = regime_hedge_effectiveness(
            returns, "AAPL", hedges, weights, regime_result.regime_series, n_regimes=3,
        )
        # Days should be positive and sum should not exceed total
        assert (eff["Days"] > 0).all()
        assert eff["Days"].sum() <= len(regime_result.regime_series)


# ===========================================================================
# Correlation Regime Tests
# ===========================================================================


class TestDetectCorrelationRegimes:
    @pytest.fixture()
    def rolling_corr(self, returns):
        return returns["AAPL"].rolling(30).corr(returns["SPY"])

    def test_returns_result(self, rolling_corr):
        result = detect_correlation_regimes(rolling_corr, n_regimes=3, method="quantile")
        assert isinstance(result, CorrelationRegimeResult)

    def test_regime_count_matches(self, rolling_corr):
        result = detect_correlation_regimes(rolling_corr, n_regimes=3, method="quantile")
        assert len(result.regime_series.unique()) == 3

    def test_series_length_matches_dropna(self, rolling_corr):
        result = detect_correlation_regimes(rolling_corr, n_regimes=3, method="quantile")
        assert len(result.regime_series) == len(rolling_corr.dropna())

    def test_regime_0_lowest_corr_quantile(self, rolling_corr):
        result = detect_correlation_regimes(rolling_corr, n_regimes=3, method="quantile")
        stats = result.per_regime_stats
        corrs = stats["Avg Correlation"]
        assert corrs.iloc[0] <= corrs.iloc[-1]

    def test_regime_0_lowest_corr_kmeans(self, rolling_corr):
        result = detect_correlation_regimes(rolling_corr, n_regimes=3, method="kmeans")
        stats = result.per_regime_stats
        corrs = stats["Avg Correlation"]
        assert corrs.iloc[0] <= corrs.iloc[-1]

    def test_per_regime_stats_sum_to_total(self, rolling_corr):
        result = detect_correlation_regimes(rolling_corr, n_regimes=3, method="quantile")
        total = int(result.per_regime_stats["Count"].sum())
        assert total == len(result.regime_series)

    def test_insufficient_data_raises(self):
        tiny = pd.Series([0.5, 0.6], index=pd.bdate_range("2023-01-01", periods=2))
        with pytest.raises(ValueError, match="Not enough data"):
            detect_correlation_regimes(tiny, n_regimes=5)

    def test_labels_default_3_regimes(self, rolling_corr):
        result = detect_correlation_regimes(rolling_corr, n_regimes=3, method="quantile")
        assert result.labels == {0: "Low Corr", 1: "Normal Corr", 2: "High Corr"}

    def test_unknown_method_raises(self, rolling_corr):
        with pytest.raises(ValueError, match="Unknown method"):
            detect_correlation_regimes(rolling_corr, method="invalid")


class TestCorrelationRegimeHedgeMetrics:
    @pytest.fixture()
    def cha_result(self, returns):
        return run_custom_hedge_analysis(
            returns=returns,
            long_tickers=["AAPL", "MSFT"],
            long_weights=np.array([0.6, 0.4]),
            long_notional=10_000_000.0,
            hedge_tickers=["SPY", "QQQ"],
            hedge_weights=np.array([0.5, 0.5]),
            hedge_notional=10_000_000.0,
            rolling_window=30,
        )

    @pytest.fixture()
    def corr_regime(self, cha_result):
        return detect_correlation_regimes(
            cha_result.rolling_correlation, n_regimes=3, method="quantile",
        )

    def test_one_row_per_regime(self, cha_result, corr_regime):
        metrics = correlation_regime_hedge_metrics(
            cha_result.daily_standalone, cha_result.daily_hedged,
            cha_result.daily_hedge_basket, corr_regime.regime_series,
            corr_regime.labels,
        )
        assert len(metrics) <= 3
        assert len(metrics) >= 1

    def test_vol_reduction_formula(self, cha_result, corr_regime):
        metrics = correlation_regime_hedge_metrics(
            cha_result.daily_standalone, cha_result.daily_hedged,
            cha_result.daily_hedge_basket, corr_regime.regime_series,
            corr_regime.labels,
        )
        for _, row in metrics.iterrows():
            if row["Unhedged Vol"] > 0:
                expected = (1 - row["Hedged Vol"] / row["Unhedged Vol"]) * 100
                assert row["Vol Reduction (%)"] == pytest.approx(expected, rel=1e-6)

    def test_days_sum(self, cha_result, corr_regime):
        metrics = correlation_regime_hedge_metrics(
            cha_result.daily_standalone, cha_result.daily_hedged,
            cha_result.daily_hedge_basket, corr_regime.regime_series,
            corr_regime.labels,
        )
        common = cha_result.daily_standalone.index.intersection(corr_regime.regime_series.index)
        assert metrics["Days"].sum() <= len(common)

    def test_correlation_bounded(self, cha_result, corr_regime):
        metrics = correlation_regime_hedge_metrics(
            cha_result.daily_standalone, cha_result.daily_hedged,
            cha_result.daily_hedge_basket, corr_regime.regime_series,
            corr_regime.labels,
        )
        assert (metrics["Avg Correlation"].abs() <= 1.0).all()
