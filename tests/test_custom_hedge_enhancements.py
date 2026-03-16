"""Tests for Custom Hedge tab enhancements:
  - MC weight construction and end-to-end simulation
  - Factor analytics wiring for custom hedge positions
  - P&L attribution sign correctness and stackgroup classification
  - Benchmark selector logic (index list vs hedge tickers)
"""

import numpy as np
import pandas as pd
import pytest

from analytics.custom_hedge import CustomHedgeResult, run_custom_hedge_analysis
from analytics.factor_analytics import FactorAnalyticsResult, run_factor_analytics
from analytics.montecarlo import MonteCarloResult, run_monte_carlo
from data.factor_loader import align_factor_returns
from utils.basket import inject_basket_column


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_cha(returns, **overrides):
    """Run a default custom hedge analysis with optional overrides."""
    defaults = dict(
        returns=returns,
        long_tickers=["AAPL", "MSFT"],
        long_weights=np.array([0.6, 0.4]),
        long_notional=10_000_000.0,
        hedge_tickers=["SPY", "QQQ"],
        hedge_weights=np.array([0.5, 0.5]),
        hedge_notional=10_000_000.0,
        benchmarks=["SPY"],
        rolling_window=60,
        risk_free=0.0,
    )
    defaults.update(overrides)
    return run_custom_hedge_analysis(**defaults)


# ===========================================================================
# Monte Carlo weight construction and end-to-end
# ===========================================================================


class TestMCWeightConstruction:
    """Verify MC hedge weights are correctly derived from custom hedge params."""

    def test_equal_notional_equal_weights(self):
        """hedge_ratio=1.0, weights=[0.5, 0.5] → MC weights = [-0.5, -0.5]."""
        hedge_ratio = 10_000_000 / 10_000_000  # 1.0
        hedge_weights = np.array([0.5, 0.5])
        mc_weights = -hedge_ratio * hedge_weights
        np.testing.assert_allclose(mc_weights, [-0.5, -0.5])

    def test_half_notional(self):
        """hedge_ratio=0.5 scales weights by half."""
        hedge_ratio = 5_000_000 / 10_000_000  # 0.5
        hedge_weights = np.array([0.6, 0.4])
        mc_weights = -hedge_ratio * hedge_weights
        np.testing.assert_allclose(mc_weights, [-0.3, -0.2])

    def test_single_hedge_instrument(self):
        """Single instrument hedge → single negative weight."""
        hedge_ratio = 1.0
        hedge_weights = np.array([1.0])
        mc_weights = -hedge_ratio * hedge_weights
        np.testing.assert_allclose(mc_weights, [-1.0])

    def test_mc_weights_are_negative(self):
        """All MC weights must be negative (short positions)."""
        for ratio in [0.3, 0.5, 1.0, 1.5]:
            weights = np.array([0.4, 0.3, 0.3])
            mc_weights = -ratio * weights
            assert np.all(mc_weights < 0), f"MC weights should be negative for ratio={ratio}"


class TestMCEndToEnd:
    """End-to-end Monte Carlo simulation using custom hedge inputs."""

    @pytest.fixture()
    def cha_result(self, returns):
        return _run_cha(returns)

    def test_mc_with_custom_hedge_basket(self, returns, cha_result):
        """MC simulation runs successfully with basket long + hedge instruments."""
        augmented, target_col = inject_basket_column(
            returns, cha_result.long_tickers, cha_result.long_weights,
        )
        mc_weights = -cha_result.hedge_ratio * cha_result.hedge_weights

        mc = run_monte_carlo(
            returns=augmented,
            target=target_col,
            hedge_instruments=cha_result.hedge_tickers,
            weights=mc_weights,
            strategy="Custom Hedge",
            horizon=30,
            num_sims=200,
            initial_value=cha_result.long_notional,
            seed=42,
        )
        assert isinstance(mc, MonteCarloResult)
        assert mc.hedged_paths.shape == (200, 31)
        assert mc.unhedged_paths.shape == (200, 31)

    def test_mc_paths_start_at_initial_value(self, returns, cha_result):
        """All paths start at the initial portfolio value."""
        augmented, target_col = inject_basket_column(
            returns, cha_result.long_tickers, cha_result.long_weights,
        )
        mc_weights = -cha_result.hedge_ratio * cha_result.hedge_weights
        mc = run_monte_carlo(
            augmented, target_col, cha_result.hedge_tickers, mc_weights,
            "Custom Hedge", 20, 100, cha_result.long_notional, seed=42,
        )
        np.testing.assert_allclose(mc.hedged_paths[:, 0], cha_result.long_notional)
        np.testing.assert_allclose(mc.unhedged_paths[:, 0], cha_result.long_notional)

    def test_mc_hedged_differs_from_unhedged(self, returns, cha_result):
        """Hedged and unhedged distributions should differ (hedge has an effect)."""
        augmented, target_col = inject_basket_column(
            returns, cha_result.long_tickers, cha_result.long_weights,
        )
        mc_weights = -cha_result.hedge_ratio * cha_result.hedge_weights
        mc = run_monte_carlo(
            augmented, target_col, cha_result.hedge_tickers, mc_weights,
            "Custom Hedge", 60, 1000, cha_result.long_notional, seed=42,
        )
        # Hedged and unhedged final distributions should not be identical
        assert not np.allclose(mc.hedged_final, mc.unhedged_final)
        # Both should have positive standard deviation (non-degenerate)
        assert np.std(mc.hedged_final) > 0
        assert np.std(mc.unhedged_final) > 0

    def test_mc_metrics_populated(self, returns, cha_result):
        """MC metrics DataFrame has the expected structure."""
        augmented, target_col = inject_basket_column(
            returns, cha_result.long_tickers, cha_result.long_weights,
        )
        mc_weights = -cha_result.hedge_ratio * cha_result.hedge_weights
        mc = run_monte_carlo(
            augmented, target_col, cha_result.hedge_tickers, mc_weights,
            "Custom Hedge", 30, 200, cha_result.long_notional,
            confidence_levels=[0.95, 0.99], seed=42,
        )
        assert "Unhedged" in mc.metrics.columns
        assert "Hedged" in mc.metrics.columns
        assert "Mean Return" in mc.metrics.index
        assert "VaR 95%" in mc.metrics.index
        assert "CVaR 99%" in mc.metrics.index

    def test_mc_reproducibility_with_seed(self, returns, cha_result):
        """Seeded MC produces identical results."""
        augmented, target_col = inject_basket_column(
            returns, cha_result.long_tickers, cha_result.long_weights,
        )
        mc_weights = -cha_result.hedge_ratio * cha_result.hedge_weights
        kwargs = dict(
            returns=augmented, target=target_col,
            hedge_instruments=cha_result.hedge_tickers, weights=mc_weights,
            strategy="Custom Hedge", horizon=20, num_sims=100,
            initial_value=cha_result.long_notional, seed=99,
        )
        r1 = run_monte_carlo(**kwargs)
        r2 = run_monte_carlo(**kwargs)
        np.testing.assert_array_equal(r1.hedged_paths, r2.hedged_paths)


# ===========================================================================
# Factor analytics wiring for custom hedge
# ===========================================================================


@pytest.fixture()
def synthetic_factor_returns(rng):
    """Synthetic GS factor returns for testing factor analytics wiring."""
    n_days = 300
    dates = pd.bdate_range("2022-01-03", periods=n_days, freq="B")
    factors = ["Momentum", "Value", "Quality"]
    data = {}
    for f in factors:
        data[f] = rng.normal(0.0002, 0.008, n_days)
    return pd.DataFrame(data, index=dates)


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


class TestFactorAnalyticsWiring:
    """Test that custom hedge returns feed correctly into the factor analytics engine."""

    def test_factor_analysis_combined_leg(self, returns, synthetic_factor_returns):
        """Factor analysis with custom hedge returns produces all three legs."""
        cha = _run_cha(returns)

        long_ret = cha.daily_standalone
        short_ret = cha.daily_hedge_basket
        combined_ret = cha.daily_hedged
        market_ret = returns["SPY"].loc[long_ret.index]

        fr_aligned, long_aligned, market_aligned = align_factor_returns(
            synthetic_factor_returns, long_ret, market_ret,
        )
        fr_s, short_aligned, _ = align_factor_returns(
            synthetic_factor_returns, short_ret, market_ret,
        )
        fr_c, combined_aligned, _ = align_factor_returns(
            synthetic_factor_returns, combined_ret, market_ret,
        )

        common = fr_aligned.index.intersection(fr_s.index).intersection(fr_c.index)
        assert len(common) >= 30

        result = run_factor_analytics(
            long_returns=long_aligned.loc[common],
            market_returns=market_aligned.loc[common],
            factor_returns=fr_aligned.loc[common],
            market_index="SPY",
            factor_names=list(synthetic_factor_returns.columns),
            dates=common,
            short_returns=short_aligned.loc[common],
            combined_returns=combined_aligned.loc[common],
        )

        assert isinstance(result, FactorAnalyticsResult)
        assert result.has_short is True
        assert result.long is not None
        assert result.short is not None
        assert result.combined is not None
        assert result.market_index == "SPY"

    def test_factor_heatmap_three_rows(self, returns, synthetic_factor_returns):
        """Beta heatmap has 3 rows (Long, Short, Combined) when hedge is present."""
        cha = _run_cha(returns)
        long_ret = cha.daily_standalone
        market_ret = returns["SPY"].loc[long_ret.index]

        fr_aligned, long_aligned, market_aligned = align_factor_returns(
            synthetic_factor_returns, long_ret, market_ret,
        )
        fr_s, short_aligned, _ = align_factor_returns(
            synthetic_factor_returns, cha.daily_hedge_basket, market_ret,
        )
        fr_c, combined_aligned, _ = align_factor_returns(
            synthetic_factor_returns, cha.daily_hedged, market_ret,
        )

        common = fr_aligned.index.intersection(fr_s.index).intersection(fr_c.index)

        result = run_factor_analytics(
            long_returns=long_aligned.loc[common],
            market_returns=market_aligned.loc[common],
            factor_returns=fr_aligned.loc[common],
            market_index="SPY",
            factor_names=list(synthetic_factor_returns.columns),
            dates=common,
            short_returns=short_aligned.loc[common],
            combined_returns=combined_aligned.loc[common],
        )

        # 3 rows (Long, Short, Combined) × (1 market + 3 factors) columns
        assert result.beta_heatmap.shape == (3, 4)
        assert list(result.beta_heatmap.index) == ["Long", "Short", "Combined"]

    def test_factor_decomposition_adds_up(self, returns, synthetic_factor_returns):
        """factor + idio = total daily returns (exact, within machine precision)."""
        cha = _run_cha(returns)
        market_ret = returns["SPY"].loc[cha.daily_standalone.index]

        fr_aligned, long_aligned, market_aligned = align_factor_returns(
            synthetic_factor_returns, cha.daily_hedged, market_ret,
        )
        common = fr_aligned.index

        result = run_factor_analytics(
            long_returns=long_aligned.loc[common],
            market_returns=market_aligned.loc[common],
            factor_returns=fr_aligned.loc[common],
            market_index="SPY",
            factor_names=list(synthetic_factor_returns.columns),
            dates=common,
        )

        leg = result.long
        total = leg.daily_factor + leg.daily_idio
        np.testing.assert_allclose(total.values, leg.daily_total.values, atol=1e-12)

    def test_combined_leg_uses_hedged_returns(self, returns, synthetic_factor_returns):
        """The 'Combined' leg should be regressing on the hedged return series."""
        cha = _run_cha(returns)
        market_ret = returns["SPY"].loc[cha.daily_standalone.index]

        fr_aligned, combined_aligned, market_aligned = align_factor_returns(
            synthetic_factor_returns, cha.daily_hedged, market_ret,
        )
        common = fr_aligned.index

        # Run with combined as "long" only (to get a standalone regression on hedged returns)
        result_standalone = run_factor_analytics(
            long_returns=combined_aligned.loc[common],
            market_returns=market_aligned.loc[common],
            factor_returns=fr_aligned.loc[common],
            market_index="SPY",
            factor_names=list(synthetic_factor_returns.columns),
            dates=common,
        )

        # Run full 3-leg
        fr_l, long_aligned, _ = align_factor_returns(
            synthetic_factor_returns, cha.daily_standalone, market_ret,
        )
        fr_s, short_aligned, _ = align_factor_returns(
            synthetic_factor_returns, cha.daily_hedge_basket, market_ret,
        )
        common3 = common.intersection(fr_l.index).intersection(fr_s.index)

        result_full = run_factor_analytics(
            long_returns=long_aligned.loc[common3],
            market_returns=market_aligned.loc[common3],
            factor_returns=fr_aligned.loc[common3],
            market_index="SPY",
            factor_names=list(synthetic_factor_returns.columns),
            dates=common3,
            short_returns=short_aligned.loc[common3],
            combined_returns=combined_aligned.loc[common3],
        )

        # Combined leg betas should match standalone regression on hedged returns
        np.testing.assert_allclose(
            result_full.combined.ols.betas,
            result_standalone.long.ols.betas,
            atol=1e-10,
        )


# ===========================================================================
# P&L attribution sign correctness
# ===========================================================================


class TestAttributionSigns:
    """Verify that constituent contributions have correct signs."""

    def test_short_contributions_negative_when_hedge_up(self, returns):
        """When hedge appreciates, short contributions should be negative (loss)."""
        result = _run_cha(returns)
        contrib = result.constituent_contributions

        # Short columns end with "(S)"
        short_cols = [c for c in contrib.columns if c.endswith("(S)")]
        long_cols = [c for c in contrib.columns if c.endswith("(L)")]
        assert len(short_cols) == 2  # SPY, QQQ
        assert len(long_cols) == 2  # AAPL, MSFT

        # On days when SPY goes UP, the short SPY contribution should be negative
        spy_up_days = returns["SPY"] > 0
        spy_up_idx = spy_up_days[spy_up_days].index.intersection(contrib.index)
        spy_short_col = [c for c in short_cols if "SPY" in c][0]
        spy_contribs_on_up_days = contrib.loc[spy_up_idx, spy_short_col]
        # All should be <= 0 (or very close, allowing float imprecision)
        assert (spy_contribs_on_up_days <= 1e-15).all(), \
            "Short SPY contribution should be negative when SPY goes up"

    def test_long_contributions_positive_when_stock_up(self, returns):
        """When long stock appreciates, its contribution should be positive."""
        result = _run_cha(returns)
        contrib = result.constituent_contributions

        aapl_up_days = returns["AAPL"] > 0
        aapl_up_idx = aapl_up_days[aapl_up_days].index.intersection(contrib.index)
        aapl_long_col = [c for c in contrib.columns if "AAPL" in c][0]
        aapl_contribs_on_up_days = contrib.loc[aapl_up_idx, aapl_long_col]
        assert (aapl_contribs_on_up_days >= -1e-15).all(), \
            "Long AAPL contribution should be positive when AAPL goes up"

    def test_contributions_sum_to_hedged_return(self, returns):
        """Sum of all constituent contributions = hedged portfolio return."""
        result = _run_cha(returns)
        total_contrib = result.constituent_contributions.sum(axis=1)
        np.testing.assert_allclose(
            total_contrib.values,
            result.daily_hedged.values,
            atol=1e-12,
        )

    def test_column_naming_convention(self, returns):
        """Long cols end with '(L)', short cols end with '(S)'."""
        result = _run_cha(returns)
        cols = result.constituent_contributions.columns.tolist()
        assert "AAPL (L)" in cols
        assert "MSFT (L)" in cols
        assert "SPY (S)" in cols
        assert "QQQ (S)" in cols


# ===========================================================================
# Benchmark selector logic
# ===========================================================================


class TestBenchmarkSelection:
    """Test that net beta works with any index benchmark, not just hedge tickers."""

    def test_net_beta_with_non_hedge_benchmark(self, returns):
        """Net beta can be computed against XLE (not in hedge basket)."""
        from analytics.custom_hedge import compute_net_beta

        live = compute_net_beta(
            returns=returns,
            long_tickers=["AAPL", "MSFT"],
            long_weights=np.array([0.6, 0.4]),
            hedge_tickers=["SPY", "QQQ"],
            hedge_weights=np.array([0.5, 0.5]),
            hedge_ratio=1.0,
            benchmark="XLE",  # not in hedge tickers
        )
        assert live is not None
        assert "long_beta" in live
        assert "net_beta" in live
        assert len(live["instruments"]) == 2

    def test_net_beta_with_spy_default(self, returns):
        """Net beta against SPY (the intended default) works correctly."""
        from analytics.custom_hedge import compute_net_beta

        live = compute_net_beta(
            returns=returns,
            long_tickers=["AAPL"],
            long_weights=np.array([1.0]),
            hedge_tickers=["QQQ"],
            hedge_weights=np.array([1.0]),
            hedge_ratio=1.0,
            benchmark="SPY",
        )
        assert live is not None
        # AAPL beta to SPY should be positive (correlated in synthetic data)
        assert live["long_beta"] > 0
        # QQQ short contribution should be negative (reduces beta)
        assert live["instruments"][0]["contribution"] < 0

    def test_full_analysis_with_index_benchmark(self, returns):
        """Full analysis works when benchmark is an index not in hedge tickers."""
        result = _run_cha(
            returns,
            hedge_tickers=["QQQ"],
            hedge_weights=np.array([1.0]),
            benchmarks=["SPY"],  # SPY is NOT in hedge_tickers
        )
        assert len(result.beta_table) > 0
        assert result.beta_benchmark == "SPY"
        net_rows = result.beta_table[result.beta_table["Component"] == "Net Portfolio"]
        assert len(net_rows) == 1

    def test_benchmark_options_from_params(self):
        """Benchmark options should come from params['benchmarks'], filtered by returns columns."""
        # Simulate params and returns columns
        params = {"benchmarks": ["SPY", "QQQ", "IWM", "MISSING"]}
        returns_columns = {"AAPL", "MSFT", "SPY", "QQQ", "XLE"}

        benchmark_options = [b for b in params.get("benchmarks", []) if b in returns_columns]
        assert benchmark_options == ["SPY", "QQQ"]  # IWM and MISSING filtered out

        # SPY should be default
        default_idx = benchmark_options.index("SPY") if "SPY" in benchmark_options else 0
        assert default_idx == 0
        assert benchmark_options[default_idx] == "SPY"
