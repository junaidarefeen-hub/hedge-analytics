"""Tests for the Custom Hedge Analyzer analytics module."""

import numpy as np
import pandas as pd
import pytest

from analytics.custom_hedge import CustomHedgeResult, run_custom_hedge_analysis


class TestCustomHedgeBasic:
    """Basic single-ticker and multi-ticker tests."""

    def test_single_ticker_long_single_hedge(self, returns):
        """Single long + single hedge: basic smoke test."""
        result = run_custom_hedge_analysis(
            returns=returns,
            long_tickers=["AAPL"],
            long_weights=np.array([1.0]),
            long_notional=10_000_000.0,
            hedge_tickers=["SPY"],
            hedge_weights=np.array([1.0]),
            hedge_notional=10_000_000.0,
        )
        assert isinstance(result, CustomHedgeResult)
        assert len(result.daily_standalone) > 0
        assert len(result.daily_hedged) > 0
        assert result.hedge_ratio == pytest.approx(1.0)

    def test_multi_ticker_weighted_sum(self, returns):
        """Multi-ticker portfolio return equals hand-computed weighted sum."""
        long_tickers = ["AAPL", "MSFT"]
        long_weights = np.array([0.6, 0.4])

        result = run_custom_hedge_analysis(
            returns=returns,
            long_tickers=long_tickers,
            long_weights=long_weights,
            long_notional=10_000_000.0,
            hedge_tickers=["SPY"],
            hedge_weights=np.array([1.0]),
            hedge_notional=10_000_000.0,
        )

        # Verify standalone = 0.6 * AAPL + 0.4 * MSFT
        clean = returns[["AAPL", "MSFT", "SPY"]].dropna()
        expected = clean["AAPL"] * 0.6 + clean["MSFT"] * 0.4
        pd.testing.assert_series_equal(
            result.daily_standalone.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            atol=1e-10,
        )

    def test_hedge_ratio_scaling(self, returns):
        """Hedge ratio correctly scales when hedge_notional != long_notional."""
        result = run_custom_hedge_analysis(
            returns=returns,
            long_tickers=["AAPL"],
            long_weights=np.array([1.0]),
            long_notional=10_000_000.0,
            hedge_tickers=["SPY"],
            hedge_weights=np.array([1.0]),
            hedge_notional=5_000_000.0,
        )
        assert result.hedge_ratio == pytest.approx(0.5)

        # Verify hedged = standalone - 0.5 * hedge_basket
        expected_hedged = result.daily_standalone - 0.5 * result.daily_hedge_basket
        pd.testing.assert_series_equal(
            result.daily_hedged.reset_index(drop=True),
            expected_hedged.reset_index(drop=True),
            check_names=False,
            atol=1e-10,
        )


class TestCustomHedgeMetrics:
    """Metrics correctness and completeness."""

    def test_all_metric_keys_present(self, returns):
        """All expected metric keys are present in the result."""
        result = run_custom_hedge_analysis(
            returns=returns,
            long_tickers=["AAPL"],
            long_weights=np.array([1.0]),
            long_notional=10_000_000.0,
            hedge_tickers=["SPY"],
            hedge_weights=np.array([1.0]),
            hedge_notional=10_000_000.0,
        )
        expected_keys = {
            "Total Return", "Ann. Return", "Ann. Volatility",
            "Sharpe Ratio", "Sortino Ratio", "Max Drawdown",
            "Calmar Ratio", "Omega Ratio", "Max DD Duration (days)",
            "Tracking Error", "Information Ratio", "Vol Reduction",
        }
        assert expected_keys == set(result.metrics.index)
        assert list(result.metrics.columns) == ["Standalone", "Hedged"]

    def test_vol_reduction_positive_for_correlated_hedge(self, returns):
        """Hedging near the min-variance ratio should reduce volatility."""
        # Synthetic data has low correlation (~0.18) due to high idiosyncratic noise.
        # Optimal min-variance hedge ratio is ~0.25, so use hedge_notional=3M/10M=0.3
        # to stay near the optimal and guarantee positive vol reduction.
        result = run_custom_hedge_analysis(
            returns=returns,
            long_tickers=["AAPL"],
            long_weights=np.array([1.0]),
            long_notional=10_000_000.0,
            hedge_tickers=["SPY"],
            hedge_weights=np.array([1.0]),
            hedge_notional=3_000_000.0,
        )
        vol_reduction = result.metrics.loc["Vol Reduction", "Hedged"]
        assert vol_reduction > 0, "Expected positive vol reduction near min-variance hedge ratio"

    def test_standalone_tracking_error_zero(self, returns):
        """Standalone portfolio should have zero tracking error and IR."""
        result = run_custom_hedge_analysis(
            returns=returns,
            long_tickers=["AAPL"],
            long_weights=np.array([1.0]),
            long_notional=10_000_000.0,
            hedge_tickers=["SPY"],
            hedge_weights=np.array([1.0]),
            hedge_notional=10_000_000.0,
        )
        assert result.metrics.loc["Tracking Error", "Standalone"] == 0.0
        assert result.metrics.loc["Information Ratio", "Standalone"] == 0.0


class TestCustomHedgeValidation:
    """Input validation tests."""

    def test_missing_ticker_raises(self, returns):
        """Missing ticker in returns raises ValueError."""
        with pytest.raises(ValueError, match="Missing tickers"):
            run_custom_hedge_analysis(
                returns=returns,
                long_tickers=["AAPL", "FAKE_TICKER"],
                long_weights=np.array([0.5, 0.5]),
                long_notional=10_000_000.0,
                hedge_tickers=["SPY"],
                hedge_weights=np.array([1.0]),
                hedge_notional=10_000_000.0,
            )

    def test_missing_benchmark_raises(self, returns):
        """Missing benchmark ticker raises ValueError."""
        with pytest.raises(ValueError, match="Missing tickers"):
            run_custom_hedge_analysis(
                returns=returns,
                long_tickers=["AAPL"],
                long_weights=np.array([1.0]),
                long_notional=10_000_000.0,
                hedge_tickers=["SPY"],
                hedge_weights=np.array([1.0]),
                hedge_notional=10_000_000.0,
                benchmarks=["NONEXISTENT"],
            )


class TestCustomHedgeBeta:
    """Net beta computation tests — single benchmark, per-instrument decomposition."""

    def test_net_beta_equals_long_minus_hedge_contributions(self, returns):
        """Net beta = long_beta + sum of per-instrument short contributions."""
        result = run_custom_hedge_analysis(
            returns=returns,
            long_tickers=["AAPL"],
            long_weights=np.array([1.0]),
            long_notional=10_000_000.0,
            hedge_tickers=["SPY"],
            hedge_weights=np.array([1.0]),
            hedge_notional=5_000_000.0,
            benchmarks=["QQQ"],
        )
        # Table should have: Long Portfolio, SPY short, Net Portfolio
        bt = result.beta_table
        assert len(bt) == 3
        long_row = bt[bt["Component"] == "Long Portfolio"].iloc[0]
        spy_row = bt[bt["Component"].str.startswith("SPY")].iloc[0]
        net_row = bt[bt["Component"] == "Net Portfolio"].iloc[0]

        # Verify: net = long_contrib + hedge_contrib
        expected_net = long_row["Beta Contribution"] + spy_row["Beta Contribution"]
        assert net_row["Beta Contribution"] == pytest.approx(expected_net, abs=1e-4)

        # Verify effective hedge ratio = hedge_ratio * weight
        assert spy_row["Eff. Hedge Ratio"] == pytest.approx(
            result.hedge_ratio * 1.0, abs=1e-4,
        )

    def test_multi_instrument_decomposition(self, returns):
        """Each hedge instrument gets its own row with correct effective ratio."""
        result = run_custom_hedge_analysis(
            returns=returns,
            long_tickers=["AAPL"],
            long_weights=np.array([1.0]),
            long_notional=10_000_000.0,
            hedge_tickers=["SPY", "QQQ"],
            hedge_weights=np.array([0.6, 0.4]),
            hedge_notional=5_000_000.0,
            benchmarks=["XLE"],
        )
        bt = result.beta_table
        # Should have: Long Portfolio, SPY short (60%), QQQ short (40%), Net Portfolio
        assert len(bt) == 4
        spy_row = bt[bt["Component"].str.startswith("SPY")].iloc[0]
        qqq_row = bt[bt["Component"].str.startswith("QQQ")].iloc[0]

        # Effective hedge ratios: hedge_ratio=0.5, so SPY=0.5*0.6=0.3, QQQ=0.5*0.4=0.2
        assert spy_row["Eff. Hedge Ratio"] == pytest.approx(0.3, abs=1e-4)
        assert qqq_row["Eff. Hedge Ratio"] == pytest.approx(0.2, abs=1e-4)

        # Net = long + spy_contrib + qqq_contrib
        long_row = bt[bt["Component"] == "Long Portfolio"].iloc[0]
        net_row = bt[bt["Component"] == "Net Portfolio"].iloc[0]
        expected = long_row["Beta Contribution"] + spy_row["Beta Contribution"] + qqq_row["Beta Contribution"]
        assert net_row["Beta Contribution"] == pytest.approx(expected, abs=1e-4)

    def test_net_beta_matches_direct_univariate(self, returns):
        """Net beta from decomposition should match direct cov/var on hedged returns."""
        result = run_custom_hedge_analysis(
            returns=returns,
            long_tickers=["AAPL", "MSFT"],
            long_weights=np.array([0.6, 0.4]),
            long_notional=10_000_000.0,
            hedge_tickers=["SPY", "QQQ"],
            hedge_weights=np.array([0.5, 0.5]),
            hedge_notional=5_000_000.0,
            benchmarks=["SPY"],
        )

        # Direct univariate beta of hedged return to SPY
        clean = returns[["AAPL", "MSFT", "SPY", "QQQ"]].dropna()
        bm_ret = clean["SPY"]
        direct_beta = float(result.daily_hedged.cov(bm_ret) / bm_ret.var())

        # Compare with decomposed net beta
        bt = result.beta_table
        net_row = bt[bt["Component"] == "Net Portfolio"].iloc[0]
        assert net_row["Beta Contribution"] == pytest.approx(direct_beta, abs=1e-4)

    def test_benchmark_is_hedge_constituent(self, returns):
        """When benchmark is a hedge constituent, its beta to itself should be 1.0."""
        result = run_custom_hedge_analysis(
            returns=returns,
            long_tickers=["AAPL"],
            long_weights=np.array([1.0]),
            long_notional=10_000_000.0,
            hedge_tickers=["SPY", "QQQ"],
            hedge_weights=np.array([0.5, 0.5]),
            hedge_notional=10_000_000.0,
            benchmarks=["SPY"],
        )
        bt = result.beta_table
        spy_row = bt[bt["Component"].str.startswith("SPY")].iloc[0]
        # SPY's beta to itself = 1.0
        assert spy_row["Beta"] == pytest.approx(1.0, abs=1e-4)

    def test_no_benchmarks_empty_table(self, returns):
        """No benchmarks → empty beta table."""
        result = run_custom_hedge_analysis(
            returns=returns,
            long_tickers=["AAPL"],
            long_weights=np.array([1.0]),
            long_notional=10_000_000.0,
            hedge_tickers=["SPY"],
            hedge_weights=np.array([1.0]),
            hedge_notional=10_000_000.0,
        )
        assert len(result.beta_table) == 0


class TestCustomHedgeContributions:
    """Constituent P&L contribution tests."""

    def test_contributions_sum_to_hedged_return(self, returns):
        """Sum of all constituent contributions should equal the hedged daily return."""
        result = run_custom_hedge_analysis(
            returns=returns,
            long_tickers=["AAPL", "MSFT"],
            long_weights=np.array([0.6, 0.4]),
            long_notional=10_000_000.0,
            hedge_tickers=["SPY", "QQQ"],
            hedge_weights=np.array([0.5, 0.5]),
            hedge_notional=10_000_000.0,
        )
        contrib_sum = result.constituent_contributions.sum(axis=1)
        pd.testing.assert_series_equal(
            result.daily_hedged.reset_index(drop=True),
            contrib_sum.reset_index(drop=True),
            check_names=False,
            atol=1e-10,
        )

    def test_contribution_columns_correct(self, returns):
        """Contribution columns match long (L) and short (S) labels."""
        result = run_custom_hedge_analysis(
            returns=returns,
            long_tickers=["AAPL"],
            long_weights=np.array([1.0]),
            long_notional=10_000_000.0,
            hedge_tickers=["SPY"],
            hedge_weights=np.array([1.0]),
            hedge_notional=10_000_000.0,
        )
        assert "AAPL (L)" in result.constituent_contributions.columns
        assert "SPY (S)" in result.constituent_contributions.columns


class TestCustomHedgeCorrelation:
    """Rolling and full-period correlation tests."""

    def test_full_period_correlation_in_range(self, returns):
        """Full-period correlation should be in [-1, 1]."""
        result = run_custom_hedge_analysis(
            returns=returns,
            long_tickers=["AAPL"],
            long_weights=np.array([1.0]),
            long_notional=10_000_000.0,
            hedge_tickers=["SPY"],
            hedge_weights=np.array([1.0]),
            hedge_notional=10_000_000.0,
        )
        assert -1.0 <= result.full_period_correlation <= 1.0

    def test_rolling_correlation_length(self, returns):
        """Rolling correlation series length matches daily returns."""
        window = 60
        result = run_custom_hedge_analysis(
            returns=returns,
            long_tickers=["AAPL"],
            long_weights=np.array([1.0]),
            long_notional=10_000_000.0,
            hedge_tickers=["SPY"],
            hedge_weights=np.array([1.0]),
            hedge_notional=10_000_000.0,
            rolling_window=window,
        )
        assert len(result.rolling_correlation) == len(result.daily_standalone)


class TestCustomHedgeDateFiltering:
    """Date range filtering tests."""

    def test_date_filtering_reduces_data(self, returns):
        """Providing start/end dates should reduce the number of data points."""
        full = run_custom_hedge_analysis(
            returns=returns,
            long_tickers=["AAPL"],
            long_weights=np.array([1.0]),
            long_notional=10_000_000.0,
            hedge_tickers=["SPY"],
            hedge_weights=np.array([1.0]),
            hedge_notional=10_000_000.0,
        )
        # Use dates that cover roughly the middle 50% of the data
        mid = returns.index[len(returns) // 4]
        end = returns.index[3 * len(returns) // 4]
        filtered = run_custom_hedge_analysis(
            returns=returns,
            long_tickers=["AAPL"],
            long_weights=np.array([1.0]),
            long_notional=10_000_000.0,
            hedge_tickers=["SPY"],
            hedge_weights=np.array([1.0]),
            hedge_notional=10_000_000.0,
            start_date=pd.Timestamp(mid),
            end_date=pd.Timestamp(end),
        )
        assert len(filtered.daily_standalone) < len(full.daily_standalone)

    def test_date_filtering_respects_bounds(self, returns):
        """All returned dates should be within the specified range."""
        start = returns.index[50]
        end = returns.index[200]
        result = run_custom_hedge_analysis(
            returns=returns,
            long_tickers=["AAPL"],
            long_weights=np.array([1.0]),
            long_notional=10_000_000.0,
            hedge_tickers=["SPY"],
            hedge_weights=np.array([1.0]),
            hedge_notional=10_000_000.0,
            start_date=pd.Timestamp(start),
            end_date=pd.Timestamp(end),
        )
        assert result.daily_standalone.index.min() >= start
        assert result.daily_standalone.index.max() <= end
