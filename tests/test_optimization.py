"""Tests for analytics.optimization — all 4 strategies + helpers."""

import numpy as np
import pandas as pd
import pytest

from analytics.optimization import (
    HedgeResult,
    _apply_min_names,
    _covariance_pieces,
    _portfolio_vol,
    _weight_sum_constraint,
    compute_risk_parity,
    optimize_cvar,
    optimize_hedge,
    optimize_min_variance,
    optimize_beta_neutral,
)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestCovariancePieces:
    def test_shapes(self, returns):
        hedges = ["MSFT", "SPY", "QQQ"]
        sigma_hh, sigma_ht, var_t = _covariance_pieces(returns, "AAPL", hedges)
        assert sigma_hh.shape == (3, 3)
        assert sigma_ht.shape == (3,)
        assert isinstance(var_t, float)

    def test_variance_positive(self, returns):
        _, _, var_t = _covariance_pieces(returns, "AAPL", ["SPY"])
        assert var_t > 0


class TestPortfolioVol:
    def test_zero_weights_equals_target_vol(self, returns):
        hedges = ["SPY", "QQQ"]
        sigma_hh, sigma_ht, var_t = _covariance_pieces(returns, "AAPL", hedges)
        w = np.zeros(len(hedges))
        vol = _portfolio_vol(w, sigma_hh, sigma_ht, var_t)
        expected = np.sqrt(var_t * 252)
        np.testing.assert_allclose(vol, expected, rtol=1e-10)

    def test_non_negative(self, returns):
        hedges = ["SPY", "QQQ"]
        sigma_hh, sigma_ht, var_t = _covariance_pieces(returns, "AAPL", hedges)
        w = np.array([-0.5, -0.5])
        vol = _portfolio_vol(w, sigma_hh, sigma_ht, var_t)
        assert vol >= 0


class TestWeightSumConstraint:
    def test_short_only(self):
        c = _weight_sum_constraint((-1.0, 0.0))
        assert c["fun"](np.array([-0.5, -0.5])) == pytest.approx(0.0)

    def test_long_only(self):
        c = _weight_sum_constraint((0.0, 1.0))
        assert c["fun"](np.array([0.5, 0.5])) == pytest.approx(0.0)


class TestApplyMinNames:
    def test_no_effect_when_min_names_leq_1(self):
        assert _apply_min_names((-1.0, 0.0), 5, 0) == (-1.0, 0.0)
        assert _apply_min_names((-1.0, 0.0), 5, 1) == (-1.0, 0.0)

    def test_no_effect_when_min_names_exceeds_n(self):
        assert _apply_min_names((-1.0, 0.0), 3, 5) == (-1.0, 0.0)

    def test_short_only_caps_lower_bound(self):
        lb, ub = _apply_min_names((-1.0, 0.0), 5, 3)
        assert lb == pytest.approx(-1.0 / 3)
        assert ub == 0.0

    def test_long_only_caps_upper_bound(self):
        lb, ub = _apply_min_names((0.0, 1.0), 4, 2)
        assert lb == 0.0
        assert ub == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

class TestMinVariance:
    def test_weights_sum_short(self, returns):
        w = optimize_min_variance(returns, "AAPL", ["MSFT", "SPY", "QQQ"], (-1.0, 0.0))
        np.testing.assert_allclose(w.sum(), -1.0, atol=1e-6)

    def test_weights_within_bounds(self, returns):
        w = optimize_min_variance(returns, "AAPL", ["MSFT", "SPY", "QQQ"], (-1.0, 0.0))
        assert (w >= -1.0 - 1e-6).all()
        assert (w <= 0.0 + 1e-6).all()

    def test_weights_sum_long(self, returns):
        w = optimize_min_variance(returns, "AAPL", ["MSFT", "SPY", "QQQ"], (0.0, 1.0))
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)


class TestBetaNeutral:
    def test_returns_tuple(self, returns):
        result = optimize_beta_neutral(
            returns, "AAPL", ["MSFT", "SPY", "QQQ", "XLE"], ["SPY"], (-1.0, 0.0),
        )
        assert isinstance(result, tuple) and len(result) == 2

    def test_weights_sum(self, returns):
        w, feasible = optimize_beta_neutral(
            returns, "AAPL", ["MSFT", "SPY", "QQQ", "XLE"], ["SPY"], (-1.0, 0.0),
        )
        np.testing.assert_allclose(w.sum(), -1.0, atol=1e-4)

    def test_fallback_no_factors(self, returns):
        """With no valid factors, should fall back to min-variance."""
        w = optimize_beta_neutral(
            returns, "AAPL", ["MSFT", "SPY"], [], (-1.0, 0.0),
        )
        # Falls back to min-variance, returns ndarray directly
        assert isinstance(w, np.ndarray)


class TestCVaR:
    def test_returns_tuple(self, returns):
        w, cvar = optimize_cvar(returns, "AAPL", ["MSFT", "SPY", "QQQ"], (-1.0, 0.0))
        assert isinstance(w, np.ndarray)
        assert isinstance(cvar, float)

    def test_weights_sum(self, returns):
        w, _ = optimize_cvar(returns, "AAPL", ["MSFT", "SPY", "QQQ"], (-1.0, 0.0))
        np.testing.assert_allclose(w.sum(), -1.0, atol=1e-6)

    def test_cvar_positive(self, returns):
        _, cvar = optimize_cvar(returns, "AAPL", ["MSFT", "SPY", "QQQ"], (-1.0, 0.0))
        assert cvar > 0  # CVaR is a positive loss value


class TestRiskParity:
    def test_weights_sum_short(self, returns):
        w = compute_risk_parity(returns, ["MSFT", "SPY", "QQQ"], (-1.0, 0.0))
        np.testing.assert_allclose(w.sum(), -1.0, atol=1e-10)

    def test_weights_sum_long(self, returns):
        w = compute_risk_parity(returns, ["MSFT", "SPY", "QQQ"], (0.0, 1.0))
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-10)

    def test_inverse_vol_ordering(self, returns):
        """Lower-vol instrument should get higher absolute weight."""
        hedges = ["MSFT", "SPY", "QQQ"]
        w = compute_risk_parity(returns, hedges, (-1.0, 0.0))
        vols = returns[hedges].std().values
        abs_w = np.abs(w)
        # Instrument with lowest vol should have highest weight
        assert abs_w[np.argmin(vols)] == pytest.approx(abs_w.max())


# ---------------------------------------------------------------------------
# Main entry point — optimize_hedge()
# ---------------------------------------------------------------------------

class TestOptimizeHedge:
    @pytest.mark.parametrize("strategy", [
        "Minimum Variance",
        "Beta-Neutral",
        "Tail Risk (CVaR)",
        "Risk Parity",
    ])
    def test_returns_hedge_result(self, returns, strategy):
        result = optimize_hedge(
            returns=returns,
            target="AAPL",
            hedge_instruments=["MSFT", "SPY", "QQQ", "XLE"],
            strategy=strategy,
            notional=1_000_000,
            bounds=(-1.0, 0.0),
            factors=["SPY", "QQQ"],
        )
        assert isinstance(result, HedgeResult)
        assert result.strategy == strategy
        assert result.target_ticker == "AAPL"

    def test_weights_sum_constraint(self, returns):
        result = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ"], "Minimum Variance",
            1_000_000, (-1.0, 0.0),
        )
        np.testing.assert_allclose(result.weights.sum(), -1.0, atol=1e-6)

    def test_notionals_match_weights(self, returns):
        notional = 1_000_000
        result = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ"], "Minimum Variance",
            notional, (-1.0, 0.0),
        )
        np.testing.assert_allclose(result.notionals, result.weights * notional)

    def test_volatility_fields_positive(self, returns):
        result = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ"], "Minimum Variance",
            1_000_000, (-1.0, 0.0),
        )
        assert result.hedged_volatility > 0
        assert result.unhedged_volatility > 0

    def test_hedged_vol_is_finite(self, returns):
        """Hedged volatility should be a finite positive number."""
        result = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ"], "Minimum Variance",
            1_000_000, (-1.0, 0.0),
        )
        assert 0 < result.hedged_volatility < float("inf")

    def test_correlation_bounded(self, returns):
        result = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ"], "Minimum Variance",
            1_000_000, (-1.0, 0.0),
        )
        assert -1.0 <= result.portfolio_correlation <= 1.0

    def test_target_filtered_from_hedges(self, returns):
        """If target is in hedge_instruments, it should be excluded."""
        result = optimize_hedge(
            returns, "AAPL", ["AAPL", "MSFT", "SPY"], "Minimum Variance",
            1_000_000, (-1.0, 0.0),
        )
        assert "AAPL" not in result.hedge_instruments

    def test_unknown_strategy_raises(self, returns):
        with pytest.raises(ValueError, match="Unknown strategy"):
            optimize_hedge(
                returns, "AAPL", ["MSFT", "SPY"], "FooBar",
                1_000_000, (-1.0, 0.0),
            )

    def test_no_valid_hedges_raises(self, returns):
        with pytest.raises(ValueError, match="No valid hedge"):
            optimize_hedge(
                returns, "AAPL", ["AAPL"], "Minimum Variance",
                1_000_000, (-1.0, 0.0),
            )

    def test_min_names_enforcement(self, returns):
        result = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ", "XLE"],
            "Minimum Variance", 1_000_000, (-1.0, 0.0), min_names=3,
        )
        active = np.sum(np.abs(result.weights) > 1e-6)
        assert active >= 3

    def test_betas_populated_with_factors(self, returns):
        result = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ", "XLE"],
            "Minimum Variance", 1_000_000, (-1.0, 0.0), factors=["SPY", "QQQ"],
        )
        assert len(result.portfolio_betas) > 0
        assert len(result.unhedged_betas) > 0

    def test_cvar_fields_populated(self, returns):
        result = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ"], "Tail Risk (CVaR)",
            1_000_000, (-1.0, 0.0), confidence=0.95,
        )
        assert result.cvar is not None
        assert result.confidence_level == 0.95


# ---------------------------------------------------------------------------
# Basket metadata on HedgeResult
# ---------------------------------------------------------------------------

class TestHedgeResultBasketMetadata:
    def test_defaults_none(self, returns):
        result = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ"], "Minimum Variance",
            1_000_000, (-1.0, 0.0),
        )
        assert result.target_tickers is None
        assert result.target_weights is None

    def test_stamp_single_ticker(self, returns):
        result = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ"], "Minimum Variance",
            1_000_000, (-1.0, 0.0),
        )
        result.target_tickers = ["AAPL"]
        result.target_weights = np.array([1.0])
        assert result.target_tickers == ["AAPL"]

    def test_stamp_multi_ticker(self, returns):
        from utils.basket import inject_basket_column
        weights = np.array([0.6, 0.4])
        aug_returns, target = inject_basket_column(returns, ["AAPL", "MSFT"], weights)
        result = optimize_hedge(
            aug_returns, target, ["SPY", "QQQ", "XLE"], "Minimum Variance",
            1_000_000, (-1.0, 0.0),
        )
        result.target_tickers = ["AAPL", "MSFT"]
        result.target_weights = weights
        assert result.target_tickers == ["AAPL", "MSFT"]
        assert isinstance(result, HedgeResult)
        assert result.hedged_volatility > 0

    def test_optimize_with_basket_column(self, returns):
        """Full optimization works with synthetic basket column as target."""
        from utils.basket import inject_basket_column
        weights = np.array([0.5, 0.5])
        aug_returns, target = inject_basket_column(returns, ["AAPL", "MSFT"], weights)
        result = optimize_hedge(
            aug_returns, target, ["SPY", "QQQ", "XLE"], "Minimum Variance",
            1_000_000, (-1.0, 0.0),
        )
        assert result.hedged_volatility > 0
        assert result.unhedged_volatility > 0
        np.testing.assert_allclose(result.weights.sum(), -1.0, atol=1e-6)
