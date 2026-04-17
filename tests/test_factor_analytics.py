"""Unit tests for the factor analytics OLS engine."""

import numpy as np
import pandas as pd
import pytest

from analytics.factor_analytics import (
    FactorAnalyticsResult,
    LegDecomposition,
    OLSResult,
    PerTickerExposures,
    _build_regression_table,
    _ols_regression,
    compute_per_ticker_exposures,
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


class TestPerTickerExposures:
    """compute_per_ticker_exposures: one OLS per ticker, shared X."""

    def _build_universe(self, rng, dates, *, betas: dict[str, list[float]]):
        """Construct a synthetic universe where each ticker has known
        betas to [Market, F1, F2] plus idiosyncratic noise."""
        T = len(dates)
        market = rng.normal(0.0003, 0.01, T)
        f1 = rng.normal(0.0001, 0.012, T)
        f2 = rng.normal(0.0002, 0.011, T)
        factors = pd.DataFrame({"F1": f1, "F2": f2}, index=dates)
        market_s = pd.Series(market, index=dates, name="MKT")

        stocks = {}
        for ticker, [bm, b1, b2] in betas.items():
            idio = rng.normal(0, 0.005, T)
            stocks[ticker] = bm * market + b1 * f1 + b2 * f2 + idio
        stock_df = pd.DataFrame(stocks, index=dates)
        return stock_df, market_s, factors

    def test_returns_per_ticker_exposures_dataclass(self, rng, dates):
        stocks, market, factors = self._build_universe(
            rng, dates, betas={"AAA": [1.0, 0.5, 0.0], "BBB": [0.5, 0.0, 0.8]},
        )
        result = compute_per_ticker_exposures(
            stocks, market, factors, market_index="MKT", factor_names=["F1", "F2"],
        )
        assert isinstance(result, PerTickerExposures)
        assert result.market_index == "MKT"
        assert result.factor_names == ["MKT", "F1", "F2"]

    def test_shapes(self, rng, dates):
        stocks, market, factors = self._build_universe(
            rng, dates, betas={"AAA": [1.0, 0.5, 0.0], "BBB": [0.5, 0.0, 0.8], "CCC": [0.7, 0.3, 0.4]},
        )
        result = compute_per_ticker_exposures(
            stocks, market, factors, market_index="MKT", factor_names=["F1", "F2"],
        )
        assert result.betas.shape == (3, 3)  # 3 tickers x [MKT, F1, F2]
        assert result.pvalues.shape == (3, 3)
        assert result.significance.shape == (3, 3)
        assert list(result.betas.columns) == ["MKT", "F1", "F2"]
        assert set(result.betas.index) == {"AAA", "BBB", "CCC"}

    def test_recovers_known_betas(self, rng, dates):
        true_betas = {"AAA": [1.0, 0.5, 0.0], "BBB": [0.5, 0.0, 0.8]}
        stocks, market, factors = self._build_universe(rng, dates, betas=true_betas)
        result = compute_per_ticker_exposures(
            stocks, market, factors, market_index="MKT", factor_names=["F1", "F2"],
        )
        # Betas should be close to the true values (small idio noise).
        for ticker, [bm, b1, b2] in true_betas.items():
            assert result.betas.loc[ticker, "MKT"] == pytest.approx(bm, abs=0.05)
            assert result.betas.loc[ticker, "F1"] == pytest.approx(b1, abs=0.05)
            assert result.betas.loc[ticker, "F2"] == pytest.approx(b2, abs=0.05)

    def test_significance_stars_for_strong_signal(self, rng, dates):
        stocks, market, factors = self._build_universe(
            rng, dates, betas={"AAA": [1.0, 0.5, 0.0]},
        )
        result = compute_per_ticker_exposures(
            stocks, market, factors, market_index="MKT", factor_names=["F1", "F2"],
        )
        # MKT and F1 should be highly significant; F2 should not be.
        assert result.significance.loc["AAA", "MKT"] == "***"
        assert result.significance.loc["AAA", "F1"] == "***"
        assert result.significance.loc["AAA", "F2"] == ""

    def test_r_squared_in_zero_one(self, rng, dates):
        stocks, market, factors = self._build_universe(
            rng, dates, betas={"AAA": [1.0, 0.5, 0.0], "BBB": [0.5, 0.0, 0.8]},
        )
        result = compute_per_ticker_exposures(
            stocks, market, factors, market_index="MKT", factor_names=["F1", "F2"],
        )
        assert (result.r_squared >= 0).all()
        assert (result.r_squared <= 1).all()

    def test_per_ticker_legs_have_decomposition(self, rng, dates):
        stocks, market, factors = self._build_universe(
            rng, dates, betas={"AAA": [1.0, 0.5, 0.0]},
        )
        result = compute_per_ticker_exposures(
            stocks, market, factors, market_index="MKT", factor_names=["F1", "F2"],
        )
        leg = result.per_ticker_legs["AAA"]
        # factor + idio = total (additive decomposition invariant from _build_leg)
        sum_decomp = leg.daily_factor + leg.daily_idio
        np.testing.assert_allclose(sum_decomp.values, leg.daily_total.values, atol=1e-12)

    def test_short_history_ticker_skipped(self, rng, dates):
        stocks, market, factors = self._build_universe(
            rng, dates, betas={"GOOD": [1.0, 0.5, 0.0]},
        )
        # Inject a ticker with only 5 valid observations (below k+5=8).
        short_series = pd.Series(np.nan, index=dates)
        short_series.iloc[-5:] = rng.normal(0, 0.01, 5)
        stocks["IPO"] = short_series

        result = compute_per_ticker_exposures(
            stocks, market, factors, market_index="MKT", factor_names=["F1", "F2"],
        )
        assert "GOOD" in result.tickers
        assert "IPO" not in result.tickers

    def test_empty_overlap_returns_empty(self, rng):
        # Date ranges that don't intersect.
        d1 = pd.bdate_range("2020-01-02", periods=100, freq="B")
        d2 = pd.bdate_range("2024-01-02", periods=100, freq="B")
        stocks = pd.DataFrame({"AAA": rng.normal(0, 0.01, 100)}, index=d1)
        market = pd.Series(rng.normal(0, 0.01, 100), index=d2, name="MKT")
        factors = pd.DataFrame({"F1": rng.normal(0, 0.01, 100)}, index=d2)
        result = compute_per_ticker_exposures(
            stocks, market, factors, market_index="MKT", factor_names=["F1"],
        )
        assert result.tickers == []
        assert result.betas.empty
