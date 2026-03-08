"""Tests for analytics.montecarlo — run_monte_carlo()."""

import numpy as np
import pytest

from analytics.montecarlo import MonteCarloResult, _compute_mc_metrics, run_monte_carlo


class TestComputeMcMetrics:
    def test_keys(self):
        rng = np.random.default_rng(0)
        final_returns = rng.normal(0.05, 0.10, 1000)
        m = _compute_mc_metrics(final_returns, [0.95, 0.99], [0.05, 0.10])
        assert "Mean Return" in m
        assert "VaR 95%" in m
        assert "CVaR 99%" in m
        assert "P(loss > 5%)" in m
        assert "Best Case" in m
        assert "Worst Case" in m

    def test_var_less_than_cvar(self):
        rng = np.random.default_rng(0)
        final_returns = rng.normal(-0.02, 0.15, 5000)
        m = _compute_mc_metrics(final_returns, [0.95], [])
        assert m["VaR 95%"] <= m["CVaR 95%"]


class TestRunMonteCarlo:
    @pytest.fixture()
    def mc(self, returns):
        return run_monte_carlo(
            returns, "AAPL", ["MSFT", "SPY", "QQQ"],
            weights=np.array([-0.4, -0.3, -0.3]),
            strategy="Minimum Variance",
            horizon=30,
            num_sims=500,
            initial_value=1_000_000,
            seed=42,
        )

    def test_returns_result(self, mc):
        assert isinstance(mc, MonteCarloResult)

    def test_path_shapes(self, mc):
        assert mc.hedged_paths.shape == (500, 31)   # num_sims × (horizon + 1)
        assert mc.unhedged_paths.shape == (500, 31)

    def test_paths_start_at_initial(self, mc):
        np.testing.assert_allclose(mc.hedged_paths[:, 0], 1_000_000)
        np.testing.assert_allclose(mc.unhedged_paths[:, 0], 1_000_000)

    def test_final_arrays(self, mc):
        assert mc.hedged_final.shape == (500,)
        np.testing.assert_allclose(mc.hedged_final, mc.hedged_paths[:, -1])

    def test_bands_keys(self, mc):
        for p in [5, 25, 50, 75, 95]:
            assert p in mc.hedged_bands
            assert p in mc.unhedged_bands

    def test_metrics_df_columns(self, mc):
        assert "Unhedged" in mc.metrics.columns
        assert "Hedged" in mc.metrics.columns

    def test_seeded_reproducibility(self, returns):
        kwargs = dict(
            returns=returns, target="AAPL", hedge_instruments=["MSFT", "SPY"],
            weights=np.array([-0.5, -0.5]), strategy="Min Var",
            horizon=20, num_sims=100, initial_value=1_000_000, seed=123,
        )
        r1 = run_monte_carlo(**kwargs)
        r2 = run_monte_carlo(**kwargs)
        np.testing.assert_array_equal(r1.hedged_paths, r2.hedged_paths)

    def test_insufficient_data_raises(self):
        import pandas as pd
        # Only 10 rows — below the 30-row threshold
        idx = pd.bdate_range("2023-01-01", periods=10)
        tiny = pd.DataFrame({"A": range(10), "B": range(10)}, index=idx, dtype=float)
        with pytest.raises(ValueError, match="Not enough data"):
            run_monte_carlo(
                tiny, "A", ["B"], np.array([-1.0]), "X", 30, 100, 1_000_000,
            )
