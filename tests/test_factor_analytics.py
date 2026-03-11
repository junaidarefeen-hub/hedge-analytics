"""Unit tests for the factor analytics OLS engine."""

import numpy as np
import pandas as pd
import pytest

from analytics.factor_analytics import (
    FactorAnalyticsResult,
    LegDecomposition,
    OLSResult,
    _build_regression_table,
    _ols_regression,
    run_factor_analytics,
)


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def dates():
    return pd.bdate_range("2022-01-03", periods=500, freq="B")


class TestOLSPerfectFit:
    """y = 2 + 3*x exactly → R²=1, beta=3, alpha=2."""

    def test_ols_perfect_fit(self):
        T = 200
        x = np.linspace(-1, 1, T)
        y = 2.0 + 3.0 * x
        X = x.reshape(-1, 1)
        result = _ols_regression(y, X, ["x"])

        assert result.alpha == pytest.approx(2.0, abs=1e-10)
        assert result.betas[0] == pytest.approx(3.0, abs=1e-10)
        assert result.r_squared == pytest.approx(1.0, abs=1e-10)
        assert result.n_obs == T
        assert result.df_model == 1
        assert result.df_resid == T - 2


class TestOLSPValues:
    """Strong signal should produce small p-values."""

    def test_ols_p_values(self, rng):
        T = 500
        x = rng.normal(0, 1, T)
        noise = rng.normal(0, 0.01, T)  # very small noise
        y = 1.0 + 5.0 * x + noise
        X = x.reshape(-1, 1)
        result = _ols_regression(y, X, ["x"])

        assert result.p_values[0] < 0.001
        assert result.p_values[1] < 0.001
        assert result.r_squared > 0.99


class TestOLSFStatConsistency:
    """F-stat should be consistent with R² formula."""

    def test_ols_f_stat_consistency(self, rng):
        T = 300
        k = 3
        X = rng.normal(0, 1, (T, k))
        betas_true = np.array([1.5, -0.8, 2.0])
        y = 0.5 + X @ betas_true + rng.normal(0, 0.5, T)
        result = _ols_regression(y, X, ["f1", "f2", "f3"])

        expected_f = (result.r_squared / k) / ((1 - result.r_squared) / (T - k - 1))
        assert result.f_stat == pytest.approx(expected_f, rel=1e-6)


class TestOLSResidualsOrthogonal:
    """Residuals should be uncorrelated with X (by OLS construction)."""

    def test_ols_residuals_orthogonal(self, rng):
        T = 500
        k = 4
        X = rng.normal(0, 1, (T, k))
        y = 1.0 + X @ np.array([2.0, -1.0, 0.5, 3.0]) + rng.normal(0, 1, T)
        result = _ols_regression(y, X, [f"f{i}" for i in range(k)])

        for j in range(k):
            corr = np.corrcoef(result.residuals, X[:, j])[0, 1]
            assert abs(corr) < 0.01, f"Residual correlated with X[{j}]: {corr:.4f}"


class TestResultStructureLongOnly:
    """Long-only analysis: short/combined legs should be None."""

    def test_long_only(self, rng, dates):
        T = len(dates)
        n_factors = 3
        factor_names = ["Momentum", "Value", "Quality"]

        market = pd.Series(rng.normal(0, 0.01, T), index=dates)
        factors = pd.DataFrame(
            rng.normal(0, 0.01, (T, n_factors)),
            index=dates, columns=factor_names,
        )
        long_ret = pd.Series(rng.normal(0.0003, 0.015, T), index=dates)

        result = run_factor_analytics(
            long_returns=long_ret,
            market_returns=market,
            factor_returns=factors,
            market_index="SPY",
            factor_names=factor_names,
            dates=dates,
        )

        assert isinstance(result, FactorAnalyticsResult)
        assert isinstance(result.long, LegDecomposition)
        assert result.short is None
        assert result.combined is None
        assert result.has_short is False
        assert result.leg_names == ["Long"]
        assert result.beta_heatmap.shape == (1, 1 + n_factors)
        assert len(result.long.cum_total) == T


class TestResultStructureWithShort:
    """With short basket: all three legs should be populated."""

    def test_with_short(self, rng, dates):
        T = len(dates)
        n_factors = 3
        factor_names = ["Momentum", "Value", "Quality"]

        market = pd.Series(rng.normal(0, 0.01, T), index=dates)
        factors = pd.DataFrame(
            rng.normal(0, 0.01, (T, n_factors)),
            index=dates, columns=factor_names,
        )
        long_ret = pd.Series(rng.normal(0.0003, 0.015, T), index=dates)
        short_ret = pd.Series(rng.normal(0.0001, 0.012, T), index=dates)
        combined_ret = long_ret - 0.8 * short_ret

        result = run_factor_analytics(
            long_returns=long_ret,
            market_returns=market,
            factor_returns=factors,
            market_index="SPY",
            factor_names=factor_names,
            dates=dates,
            short_returns=short_ret,
            combined_returns=combined_ret,
        )

        assert result.has_short is True
        assert result.short is not None
        assert result.combined is not None
        assert result.leg_names == ["Long", "Short", "Combined"]
        assert result.beta_heatmap.shape == (3, 1 + n_factors)


class TestBetaHeatmapShape:
    """Beta heatmap rows match active legs, cols = Market + factors."""

    def test_long_only_shape(self, rng, dates):
        T = len(dates)
        n_factors = 5
        factor_names = [f"Factor_{i}" for i in range(n_factors)]

        market = pd.Series(rng.normal(0, 0.01, T), index=dates)
        factors = pd.DataFrame(
            rng.normal(0, 0.01, (T, n_factors)),
            index=dates, columns=factor_names,
        )
        long_ret = pd.Series(rng.normal(0, 0.015, T), index=dates)

        result = run_factor_analytics(
            long_returns=long_ret,
            market_returns=market,
            factor_returns=factors,
            market_index="SPY",
            factor_names=factor_names,
            dates=dates,
        )

        assert result.beta_heatmap.shape == (1, 1 + n_factors)
        assert list(result.beta_heatmap.index) == ["Long"]
        assert result.beta_heatmap.columns[0] == "SPY"

    def test_with_short_shape(self, rng, dates):
        T = len(dates)
        n_factors = 5
        factor_names = [f"Factor_{i}" for i in range(n_factors)]

        market = pd.Series(rng.normal(0, 0.01, T), index=dates)
        factors = pd.DataFrame(
            rng.normal(0, 0.01, (T, n_factors)),
            index=dates, columns=factor_names,
        )
        long_ret = pd.Series(rng.normal(0, 0.015, T), index=dates)
        short_ret = pd.Series(rng.normal(0, 0.012, T), index=dates)

        result = run_factor_analytics(
            long_returns=long_ret,
            market_returns=market,
            factor_returns=factors,
            market_index="SPY",
            factor_names=factor_names,
            dates=dates,
            short_returns=short_ret,
            combined_returns=long_ret - short_ret,
        )

        assert result.beta_heatmap.shape == (3, 1 + n_factors)
        assert list(result.beta_heatmap.index) == ["Long", "Short", "Combined"]


class TestDecompositionAddsUp:
    """factor + idio = total daily returns (exactly)."""

    def test_decomposition_adds_up(self, rng, dates):
        T = len(dates)
        n_factors = 2
        factor_names = ["Mom", "Val"]

        market = pd.Series(rng.normal(0, 0.01, T), index=dates)
        factors = pd.DataFrame(
            rng.normal(0, 0.01, (T, n_factors)),
            index=dates, columns=factor_names,
        )
        long_ret = pd.Series(
            0.001 + 0.5 * market.values + rng.normal(0, 0.005, T),
            index=dates,
        )

        result = run_factor_analytics(
            long_returns=long_ret,
            market_returns=market,
            factor_returns=factors,
            market_index="MKT",
            factor_names=factor_names,
            dates=dates,
        )

        # factor + idio = total (exactly)
        np.testing.assert_allclose(
            (result.long.daily_factor + result.long.daily_idio).values,
            long_ret.values,
            atol=1e-12,
        )


class TestVolDecompositionConsistent:
    """factor_vol² + idio_vol² should approximately equal total_vol²."""

    def test_vol_decomposition_consistent(self, rng, dates):
        T = len(dates)
        n_factors = 3
        factor_names = ["A", "B", "C"]

        market = pd.Series(rng.normal(0, 0.01, T), index=dates)
        factors = pd.DataFrame(
            rng.normal(0, 0.01, (T, n_factors)),
            index=dates, columns=factor_names,
        )
        long_ret = pd.Series(rng.normal(0, 0.015, T), index=dates)
        short_ret = pd.Series(rng.normal(0, 0.012, T), index=dates)
        combined_ret = long_ret - short_ret

        result = run_factor_analytics(
            long_returns=long_ret,
            market_returns=market,
            factor_returns=factors,
            market_index="MKT",
            factor_names=factor_names,
            dates=dates,
            short_returns=short_ret,
            combined_returns=combined_ret,
        )

        for name, leg in [("Long", result.long), ("Short", result.short), ("Combined", result.combined)]:
            var_total = leg.vol_total ** 2
            var_sum = leg.vol_factor ** 2 + leg.vol_idio ** 2
            assert var_sum == pytest.approx(var_total, rel=1e-6), (
                f"{name}: factor_var + idio_var = {var_sum:.8f} != total_var = {var_total:.8f}"
            )


class TestBuildRegressionTable:
    """Regression table should have the right structure."""

    def test_table_columns(self, rng):
        T = 100
        X = rng.normal(0, 1, (T, 2))
        y = 1.0 + X @ np.array([2.0, -1.0]) + rng.normal(0, 0.5, T)
        ols = _ols_regression(y, X, ["f1", "f2"])
        table = _build_regression_table(ols)

        assert list(table.columns) == ["Regressor", "Beta", "Std Error", "t-stat", "p-value", "Sig"]
        assert len(table) == 3  # Intercept + 2 regressors
        assert table.iloc[0]["Regressor"] == "Intercept"
        assert table.iloc[1]["Regressor"] == "f1"
        assert table.iloc[2]["Regressor"] == "f2"


class TestDailySeriesStored:
    """LegDecomposition should store daily factor/idio/total series."""

    def test_daily_series_present(self, rng, dates):
        T = len(dates)
        market = pd.Series(rng.normal(0, 0.01, T), index=dates)
        factors = pd.DataFrame(rng.normal(0, 0.01, (T, 2)), index=dates, columns=["A", "B"])
        long_ret = pd.Series(rng.normal(0, 0.015, T), index=dates)

        result = run_factor_analytics(
            long_returns=long_ret,
            market_returns=market,
            factor_returns=factors,
            market_index="MKT",
            factor_names=["A", "B"],
            dates=dates,
        )

        assert len(result.long.daily_factor) == T
        assert len(result.long.daily_idio) == T
        assert len(result.long.daily_total) == T
