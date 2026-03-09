"""Cross-check optimization results against manual / analytical calculations.

These tests go beyond constraint-satisfaction to verify that the optimizer
actually finds the *correct* solution — by comparing against closed-form
formulas, perturbation checks, and independent recomputation.
"""

import numpy as np
import pandas as pd
import pytest

from analytics.optimization import (
    _covariance_pieces,
    _multivariate_beta_matrix,
    _portfolio_vol,
    compute_risk_parity,
    optimize_beta_neutral,
    optimize_cvar,
    optimize_hedge,
    optimize_min_variance,
)
from config import ANNUALIZATION_FACTOR


# ---------------------------------------------------------------------------
# 1. Covariance sanity — verify _covariance_pieces against numpy directly
# ---------------------------------------------------------------------------

class TestCovarianceCrossCheck:
    """Verify _covariance_pieces matches manual numpy computation."""

    def test_matches_numpy_cov(self, returns):
        hedges = ["MSFT", "SPY", "QQQ"]
        sigma_hh, sigma_ht, var_t = _covariance_pieces(returns, "AAPL", hedges)

        # Manual: aligned dropna, then numpy covariance
        cols = ["AAPL"] + hedges
        clean = returns[cols].dropna()
        full_cov = np.cov(clean.values, rowvar=False, ddof=1)  # pandas default ddof=1

        np.testing.assert_allclose(var_t, full_cov[0, 0], rtol=1e-10)
        np.testing.assert_allclose(sigma_ht, full_cov[1:, 0], rtol=1e-10)
        np.testing.assert_allclose(sigma_hh, full_cov[1:, 1:], rtol=1e-10)

    def test_symmetry(self, returns):
        sigma_hh, _, _ = _covariance_pieces(returns, "AAPL", ["MSFT", "SPY", "QQQ"])
        np.testing.assert_allclose(sigma_hh, sigma_hh.T, atol=1e-15)

    def test_positive_semidefinite(self, returns):
        sigma_hh, _, _ = _covariance_pieces(returns, "AAPL", ["MSFT", "SPY", "QQQ"])
        eigenvalues = np.linalg.eigvalsh(sigma_hh)
        assert (eigenvalues >= -1e-12).all(), f"Negative eigenvalue: {eigenvalues.min()}"


# ---------------------------------------------------------------------------
# 2. Portfolio vol — verify formula against manual computation
# ---------------------------------------------------------------------------

class TestPortfolioVolCrossCheck:
    """Verify _portfolio_vol matches direct return-series computation."""

    def test_matches_return_series_vol(self, returns):
        hedges = ["MSFT", "SPY", "QQQ"]
        sigma_hh, sigma_ht, var_t = _covariance_pieces(returns, "AAPL", hedges)

        w = np.array([-0.4, -0.3, -0.3])
        formula_vol = _portfolio_vol(w, sigma_hh, sigma_ht, var_t)

        # Manual: compute from actual return series
        clean = returns[["AAPL"] + hedges].dropna()
        port_returns = clean["AAPL"] + (clean[hedges].values @ w)
        manual_vol = float(port_returns.std() * np.sqrt(ANNUALIZATION_FACTOR))

        # ddof difference (pandas std uses ddof=1, formula uses population-like)
        # Allow ~1% tolerance for the small ddof bias
        np.testing.assert_allclose(formula_vol, manual_vol, rtol=0.01)

    def test_min_variance_is_global_minimum(self, returns):
        """Optimizer weights should produce the lowest vol of any feasible portfolio.

        With equality constraint sum=-1, the hedge may or may not reduce vol
        (depends on correlation structure). But it should be the *minimum*
        variance portfolio given the constraint — no other feasible weights
        should produce a lower vol.
        """
        hedges = ["MSFT", "SPY", "QQQ"]
        sigma_hh, sigma_ht, var_t = _covariance_pieces(returns, "AAPL", hedges)

        w_opt = optimize_min_variance(returns, "AAPL", hedges, (-1.0, 0.0))
        vol_opt = _portfolio_vol(w_opt, sigma_hh, sigma_ht, var_t)

        # Check against a grid of feasible portfolios
        rng = np.random.default_rng(99)
        for _ in range(500):
            raw = -rng.dirichlet(np.ones(len(hedges)))  # random weights summing to -1
            vol_rand = _portfolio_vol(raw, sigma_hh, sigma_ht, var_t)
            assert vol_rand >= vol_opt - 1e-8, (
                f"Random portfolio vol {vol_rand:.8f} < optimal {vol_opt:.8f}"
            )


# ---------------------------------------------------------------------------
# 3. Min-Variance analytical cross-check (2-asset closed form)
# ---------------------------------------------------------------------------

class TestMinVarianceAnalytical:
    """For a 2-asset case with sum(w)=-1, the closed-form solution exists.

    Portfolio variance = var_t + 2*w*sigma_ht + w^2*sigma_hh
    Minimize subject to w = -1:
        d/dw (var_t + 2*w*sigma_ht + w^2*sigma_hh) = 2*sigma_ht + 2*w*sigma_hh = 0
        => w* = -sigma_ht / sigma_hh
    But with constraint w = -1 and bounds [-1, 0], solution is just w = -1.

    For unconstrained (inequality) mode, the optimizer can choose w freely.
    """

    def test_two_asset_equality_mode(self, returns):
        """With 1 hedge and sum=-1 constraint, weight must be -1."""
        w = optimize_min_variance(returns, "AAPL", ["SPY"], (-1.0, 0.0))
        np.testing.assert_allclose(w, [-1.0], atol=1e-6)

    def test_two_asset_inequality_optimal(self, returns):
        """With inequality mode, verify optimizer finds the calculus minimum."""
        hedges = ["SPY"]
        sigma_hh, sigma_ht, var_t = _covariance_pieces(returns, "AAPL", hedges)

        # Analytical optimal (unconstrained): w* = -sigma_ht / sigma_hh
        w_analytical = -sigma_ht[0] / sigma_hh[0, 0]
        # Clip to bounds
        w_analytical = np.clip(w_analytical, -1.0, 0.0)

        w_opt = optimize_min_variance(
            returns, "AAPL", hedges, (-1.0, 0.0),
            max_hedge_ratio=1.0, use_inequality=True,
        )

        np.testing.assert_allclose(w_opt[0], w_analytical, atol=1e-4)

    def test_multi_asset_optimality_by_perturbation(self, returns):
        """Perturb optimal weights — all perturbations should increase vol."""
        hedges = ["MSFT", "SPY", "QQQ"]
        w_opt = optimize_min_variance(returns, "AAPL", hedges, (-1.0, 0.0))
        sigma_hh, sigma_ht, var_t = _covariance_pieces(returns, "AAPL", hedges)
        vol_opt = _portfolio_vol(w_opt, sigma_hh, sigma_ht, var_t)

        rng = np.random.default_rng(123)
        for _ in range(200):
            # Random perturbation that preserves sum=-1 and bounds
            delta = rng.normal(0, 0.05, len(hedges))
            delta -= delta.mean()  # zero-sum perturbation preserves sum constraint
            w_perturbed = w_opt + delta
            # Clip to bounds
            w_perturbed = np.clip(w_perturbed, -1.0, 0.0)
            # Re-enforce sum = -1 (redistribute among non-clipped)
            slack = -1.0 - w_perturbed.sum()
            free = (w_perturbed > -1.0 + 1e-8) & (w_perturbed < -1e-8)
            if free.any():
                w_perturbed[free] += slack / free.sum()
            w_perturbed = np.clip(w_perturbed, -1.0, 0.0)
            if not np.isclose(w_perturbed.sum(), -1.0, atol=0.01):
                continue  # skip invalid perturbations

            vol_pert = _portfolio_vol(w_perturbed, sigma_hh, sigma_ht, var_t)
            assert vol_pert >= vol_opt - 1e-8, (
                f"Perturbation found lower vol: {vol_pert:.8f} < {vol_opt:.8f}"
            )


# ---------------------------------------------------------------------------
# 4. Beta-Neutral verification — portfolio beta actually ≈ 0
# ---------------------------------------------------------------------------

class TestBetaNeutralCorrectness:
    """After beta-neutral optimization, recompute portfolio beta independently."""

    def test_portfolio_beta_near_zero(self, returns):
        hedges = ["MSFT", "SPY", "QQQ", "XLE"]
        factors = ["SPY"]
        w, feasible = optimize_beta_neutral(
            returns, "AAPL", hedges, factors, (-1.0, 0.0),
        )

        # Independently compute portfolio beta using no-intercept OLS
        # (matching _multivariate_beta_matrix which uses lstsq without intercept)
        all_cols = list(dict.fromkeys(["AAPL"] + hedges + factors))  # dedupe, preserve order
        clean = returns[all_cols].dropna()
        port_returns = clean["AAPL"].values + clean[hedges].values @ w
        X = clean[factors].values

        betas, _, _, _ = np.linalg.lstsq(X, port_returns, rcond=None)
        portfolio_beta = betas[0]

        if feasible:
            np.testing.assert_allclose(portfolio_beta, 0.0, atol=0.05)
        else:
            # Soft constraint — beta should at least be reduced
            betas_uh, _, _, _ = np.linalg.lstsq(X, clean["AAPL"].values, rcond=None)
            assert abs(portfolio_beta) < abs(betas_uh[0])

    def test_multi_factor_betas_near_zero(self, returns):
        hedges = ["MSFT", "SPY", "QQQ", "XLE"]
        factors = ["SPY", "QQQ"]
        w, feasible = optimize_beta_neutral(
            returns, "AAPL", hedges, factors, (-1.0, 0.0),
        )

        # Independent multivariate OLS (no intercept, matching _multivariate_beta_matrix)
        all_cols = list(dict.fromkeys(["AAPL"] + hedges + factors))
        clean = returns[all_cols].dropna()
        port_returns = clean["AAPL"].values + clean[hedges].values @ w
        X = clean[factors].values

        betas, _, _, _ = np.linalg.lstsq(X, port_returns, rcond=None)

        if feasible:
            for j, f in enumerate(factors):
                np.testing.assert_allclose(
                    betas[j], 0.0, atol=0.1,
                    err_msg=f"Portfolio beta to {f} not near zero: {betas[j]:.4f}",
                )

    def test_beta_neutral_vol_vs_unconstrained(self, returns):
        """Beta-neutral vol should be >= min-variance vol (extra constraint can only hurt).

        Beta-neutral may or may not reduce vol vs unhedged — the beta=0 constraint
        can be more costly than the variance reduction. But it should never beat
        unconstrained min-variance, which has strictly more freedom.
        """
        hedges = ["MSFT", "SPY", "QQQ", "XLE"]
        result_bn = optimize_hedge(
            returns, "AAPL", hedges,
            "Beta-Neutral", 1_000_000, (-1.0, 0.0), factors=["SPY"],
        )
        result_mv = optimize_hedge(
            returns, "AAPL", hedges,
            "Minimum Variance", 1_000_000, (-1.0, 0.0),
        )
        assert result_bn.hedged_volatility >= result_mv.hedged_volatility - 1e-6


# ---------------------------------------------------------------------------
# 5. CVaR manual cross-check
# ---------------------------------------------------------------------------

class TestCVaRCorrectness:
    """Verify CVaR value by recomputing from the loss distribution."""

    def test_cvar_matches_manual_tail_mean(self, returns):
        hedges = ["MSFT", "SPY", "QQQ"]
        confidence = 0.95
        w, cvar_reported = optimize_cvar(
            returns, "AAPL", hedges, (-1.0, 0.0), confidence=confidence,
        )

        # Manual CVaR: mean of losses beyond the VaR threshold
        clean = returns[["AAPL"] + hedges].dropna()
        port_returns = clean["AAPL"].values + clean[hedges].values @ w
        losses = -port_returns
        var_threshold = np.percentile(losses, confidence * 100)
        manual_cvar = float(np.mean(losses[losses >= var_threshold]))

        np.testing.assert_allclose(cvar_reported, manual_cvar, rtol=1e-6)

    def test_cvar_exceeds_var(self, returns):
        """CVaR (expected shortfall) should be >= VaR by definition."""
        hedges = ["MSFT", "SPY", "QQQ"]
        confidence = 0.95
        w, cvar = optimize_cvar(returns, "AAPL", hedges, (-1.0, 0.0), confidence=confidence)

        clean = returns[["AAPL"] + hedges].dropna()
        port_returns = clean["AAPL"].values + clean[hedges].values @ w
        losses = -port_returns
        var = np.percentile(losses, confidence * 100)

        assert cvar >= var - 1e-10, f"CVaR ({cvar}) should be >= VaR ({var})"

    def test_cvar_optimality_by_perturbation(self, returns):
        """Perturbed weights should not improve CVaR."""
        hedges = ["MSFT", "SPY", "QQQ"]
        confidence = 0.95
        w_opt, cvar_opt = optimize_cvar(
            returns, "AAPL", hedges, (-1.0, 0.0), confidence=confidence,
        )

        clean = returns[["AAPL"] + hedges].dropna().values
        r_target = clean[:, 0]
        r_hedges = clean[:, 1:]

        def compute_cvar(w):
            port = r_target + r_hedges @ w
            losses = -port
            threshold = np.percentile(losses, confidence * 100)
            return float(np.mean(losses[losses >= threshold]))

        cvar_baseline = compute_cvar(w_opt)

        rng = np.random.default_rng(456)
        for _ in range(100):
            delta = rng.normal(0, 0.05, len(hedges))
            delta -= delta.mean()
            w_pert = np.clip(w_opt + delta, -1.0, 0.0)
            slack = -1.0 - w_pert.sum()
            free = (w_pert > -1.0 + 1e-8) & (w_pert < -1e-8)
            if free.any():
                w_pert[free] += slack / free.sum()
            w_pert = np.clip(w_pert, -1.0, 0.0)
            if not np.isclose(w_pert.sum(), -1.0, atol=0.01):
                continue

            cvar_pert = compute_cvar(w_pert)
            # Allow small tolerance since CVaR landscape can be flat
            assert cvar_pert >= cvar_baseline - 1e-5, (
                f"Perturbation found lower CVaR: {cvar_pert:.8f} < {cvar_baseline:.8f}"
            )


# ---------------------------------------------------------------------------
# 6. Risk Parity — equal risk contribution check
# ---------------------------------------------------------------------------

class TestRiskParityCorrectness:
    """Verify risk parity weights are inverse-volatility proportional."""

    def test_weights_proportional_to_inverse_vol(self, returns):
        hedges = ["MSFT", "SPY", "QQQ"]
        w = compute_risk_parity(returns, hedges, (-1.0, 0.0))

        vols = returns[hedges].std().values
        expected_raw = 1.0 / vols
        expected_raw = expected_raw / expected_raw.sum()
        expected = -expected_raw  # short-only

        np.testing.assert_allclose(w, expected, atol=1e-10)

    def test_vol_weighted_contribution_equal(self, returns):
        """Each instrument's |w_i| * vol_i should be equal (equal risk contribution)."""
        hedges = ["MSFT", "SPY", "QQQ"]
        w = compute_risk_parity(returns, hedges, (-1.0, 0.0))

        vols = returns[hedges].std().values
        risk_contributions = np.abs(w) * vols

        # All risk contributions should be the same
        np.testing.assert_allclose(
            risk_contributions,
            np.full_like(risk_contributions, risk_contributions.mean()),
            rtol=1e-8,
        )

    def test_risk_parity_with_bounds_clipping(self, returns):
        """When min_names forces tight bounds, weights still respect bounds."""
        hedges = ["MSFT", "SPY", "QQQ", "XLE"]
        tight_bounds = (-0.3, 0.0)
        w = compute_risk_parity(returns, hedges, tight_bounds)

        assert (w >= tight_bounds[0] - 1e-10).all()
        assert (w <= tight_bounds[1] + 1e-10).all()
        # Sum should be capped at -1.0 (or less if bounds limit it)
        assert w.sum() >= -1.0 - 1e-10

    def test_long_only_risk_parity(self, returns):
        hedges = ["MSFT", "SPY", "QQQ"]
        w = compute_risk_parity(returns, hedges, (0.0, 1.0))

        vols = returns[hedges].std().values
        expected_raw = 1.0 / vols
        expected = expected_raw / expected_raw.sum()

        np.testing.assert_allclose(w, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# 7. Max gross notional — inequality mode behavior
# ---------------------------------------------------------------------------

class TestMaxGrossNotional:
    """Verify max_gross_notional constrains total hedge size."""

    def test_half_budget_uses_less_hedge(self, returns):
        """With max_gross = 50% of notional, hedge ratio should be <= 0.5."""
        notional = 1_000_000
        result = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ"],
            "Minimum Variance", notional, (-1.0, 0.0),
            max_gross_notional=500_000,
        )
        assert result.hedge_ratio <= 0.5 + 1e-4
        assert result.total_hedge_notional <= 500_000 * 1.001

    def test_full_budget_matches_equality_mode(self, returns):
        """With max_gross = notional, result should be similar to equality mode."""
        notional = 1_000_000
        hedges = ["MSFT", "SPY", "QQQ"]

        result_eq = optimize_hedge(
            returns, "AAPL", hedges, "Minimum Variance",
            notional, (-1.0, 0.0), max_gross_notional=None,
        )
        result_ineq = optimize_hedge(
            returns, "AAPL", hedges, "Minimum Variance",
            notional, (-1.0, 0.0), max_gross_notional=notional,
        )

        # Inequality mode should find vol <= equality mode (it has more freedom)
        assert result_ineq.hedged_volatility <= result_eq.hedged_volatility + 1e-6

    def test_larger_budget_no_worse_than_smaller(self, returns):
        """More hedge budget should never increase optimal hedged vol."""
        notional = 1_000_000

        result_50 = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ"],
            "Minimum Variance", notional, (-1.0, 0.0),
            max_gross_notional=500_000,
        )
        result_100 = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ"],
            "Minimum Variance", notional, (-1.0, 0.0),
            max_gross_notional=1_000_000,
        )

        assert result_100.hedged_volatility <= result_50.hedged_volatility + 1e-6

    def test_bounds_scale_with_large_budget(self, returns):
        """With max_gross > notional, per-instrument bounds should scale up."""
        notional = 1_000_000
        result = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ"],
            "Minimum Variance", notional, (-1.0, 0.0),
            max_gross_notional=2_000_000,
        )
        # Weights can go below -1.0 because bounds are scaled
        # Total hedge ratio can be up to 2.0
        assert result.hedge_ratio <= 2.0 + 1e-4

    def test_risk_parity_respects_budget_cap(self, returns):
        """Risk parity with capped budget uses effective_ratio = min(1, max_hedge_ratio)."""
        notional = 1_000_000
        result = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ"],
            "Risk Parity", notional, (-1.0, 0.0),
            max_gross_notional=300_000,
        )
        # hedge_ratio should be ~0.3
        np.testing.assert_allclose(
            abs(result.weights.sum()), 0.3, atol=0.01,
        )

    def test_post_enforcement_clipping(self, returns):
        """When beta-neutral constraints cause overshoot, weights get scaled down."""
        notional = 1_000_000
        result = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ", "XLE"],
            "Beta-Neutral", notional, (-1.0, 0.0),
            factors=["SPY"],
            max_gross_notional=500_000,
        )
        # Total hedge should respect the cap
        assert result.total_hedge_notional <= 500_000 * 1.002


# ---------------------------------------------------------------------------
# 8. Multivariate beta matrix — cross-check OLS
# ---------------------------------------------------------------------------

class TestBetaMatrixCrossCheck:
    """Verify _multivariate_beta_matrix against manual OLS."""

    def test_matches_manual_ols(self, returns):
        tickers = ["AAPL", "MSFT"]
        factors = ["SPY", "QQQ"]
        beta_mat, X = _multivariate_beta_matrix(returns, tickers, factors)

        # Manual OLS for AAPL
        clean = returns[tickers + factors].dropna()
        X_manual = clean[factors].values
        y_aapl = clean["AAPL"].values
        betas_manual, _, _, _ = np.linalg.lstsq(X_manual, y_aapl, rcond=None)

        np.testing.assert_allclose(beta_mat[0], betas_manual, rtol=1e-8)

    def test_beta_additivity(self, returns):
        """Portfolio beta should equal target_beta + w @ hedge_betas."""
        hedges = ["MSFT", "QQQ", "XLE"]
        factors = ["SPY"]
        w = np.array([-0.3, -0.4, -0.3])

        all_tickers = ["AAPL"] + hedges
        beta_mat, _ = _multivariate_beta_matrix(returns, all_tickers, factors)

        # Portfolio beta via beta additivity
        target_beta = beta_mat[0, 0]
        hedge_betas = beta_mat[1:, 0]
        port_beta_additive = target_beta + w @ hedge_betas

        # Portfolio beta via direct OLS on portfolio returns (deduplicated columns)
        all_cols = list(dict.fromkeys(["AAPL"] + hedges + factors))
        clean = returns[all_cols].dropna()
        port_returns = clean["AAPL"].values + clean[hedges].values @ w
        X = clean[factors].values
        port_beta_direct, _, _, _ = np.linalg.lstsq(X, port_returns, rcond=None)

        np.testing.assert_allclose(
            port_beta_additive, port_beta_direct[0], atol=1e-8,
            err_msg="Beta additivity violated — multivariate OLS betas are not additive",
        )


# ---------------------------------------------------------------------------
# 9. HedgeResult fields consistency
# ---------------------------------------------------------------------------

class TestHedgeResultConsistency:
    """Verify derived fields in HedgeResult are self-consistent."""

    @pytest.mark.parametrize("strategy", [
        "Minimum Variance", "Beta-Neutral", "Tail Risk (CVaR)", "Risk Parity",
    ])
    def test_hedge_ratio_matches_notionals(self, returns, strategy):
        notional = 1_000_000
        result = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ", "XLE"],
            strategy, notional, (-1.0, 0.0), factors=["SPY", "QQQ"],
        )
        expected_ratio = result.total_hedge_notional / notional
        np.testing.assert_allclose(result.hedge_ratio, expected_ratio, rtol=1e-10)

    @pytest.mark.parametrize("strategy", [
        "Minimum Variance", "Beta-Neutral", "Tail Risk (CVaR)", "Risk Parity",
    ])
    def test_total_hedge_notional_is_abs_sum(self, returns, strategy):
        notional = 1_000_000
        result = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ", "XLE"],
            strategy, notional, (-1.0, 0.0), factors=["SPY", "QQQ"],
        )
        expected = float(np.sum(np.abs(result.notionals)))
        np.testing.assert_allclose(result.total_hedge_notional, expected, rtol=1e-10)

    @pytest.mark.parametrize("strategy", [
        "Minimum Variance", "Beta-Neutral", "Tail Risk (CVaR)", "Risk Parity",
    ])
    def test_unhedged_vol_matches_target_vol(self, returns, strategy):
        """Unhedged vol should equal standalone target vol."""
        notional = 1_000_000
        result = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ", "XLE"],
            strategy, notional, (-1.0, 0.0), factors=["SPY", "QQQ"],
        )
        _, _, var_t = _covariance_pieces(returns, "AAPL", result.hedge_instruments)
        expected_vol = float(np.sqrt(var_t * ANNUALIZATION_FACTOR))
        np.testing.assert_allclose(result.unhedged_volatility, expected_vol, rtol=1e-10)

    @pytest.mark.parametrize("strategy", [
        "Minimum Variance", "Beta-Neutral", "Tail Risk (CVaR)", "Risk Parity",
    ])
    def test_hedged_vol_recomputation(self, returns, strategy):
        """Recompute hedged vol from weights and covariance; must match result."""
        notional = 1_000_000
        result = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ", "XLE"],
            strategy, notional, (-1.0, 0.0), factors=["SPY", "QQQ"],
        )
        sigma_hh, sigma_ht, var_t = _covariance_pieces(
            returns, "AAPL", result.hedge_instruments,
        )
        expected_vol = _portfolio_vol(result.weights, sigma_hh, sigma_ht, var_t)
        np.testing.assert_allclose(result.hedged_volatility, expected_vol, rtol=1e-10)

    def test_correlation_recomputation(self, returns):
        """Recompute portfolio correlation independently."""
        result = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ"],
            "Minimum Variance", 1_000_000, (-1.0, 0.0),
        )
        clean = returns[["AAPL"] + result.hedge_instruments].dropna()
        basket = (clean[result.hedge_instruments] * np.abs(result.weights)).sum(axis=1)
        expected_corr = float(clean["AAPL"].corr(basket))
        np.testing.assert_allclose(result.portfolio_correlation, expected_corr, rtol=1e-10)

    def test_portfolio_betas_match_formula(self, returns):
        """Verify portfolio_betas = unhedged_betas + w @ hedge_betas."""
        factors = ["SPY", "QQQ"]
        result = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ", "XLE"],
            "Minimum Variance", 1_000_000, (-1.0, 0.0), factors=factors,
        )
        all_tickers = ["AAPL"] + result.hedge_instruments
        beta_mat, _ = _multivariate_beta_matrix(returns, all_tickers, factors)

        for j, f in enumerate(factors):
            if f not in result.portfolio_betas:
                continue
            target_beta = beta_mat[0, j]
            hedge_betas = beta_mat[1:, j]
            expected = float(target_beta + result.weights @ hedge_betas)
            np.testing.assert_allclose(
                result.portfolio_betas[f], round(expected, 4), atol=1e-4,
            )


# ---------------------------------------------------------------------------
# 10. Min-names diversification constraint
# ---------------------------------------------------------------------------

class TestMinNamesDiversification:
    """Verify min_names forces the optimizer to spread weight across instruments."""

    @pytest.mark.parametrize("min_names", [2, 3, 4])
    def test_active_instruments_meet_minimum(self, returns, min_names):
        result = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ", "XLE"],
            "Minimum Variance", 1_000_000, (-1.0, 0.0), min_names=min_names,
        )
        active = int(np.sum(np.abs(result.weights) > 1e-6))
        assert active >= min_names, (
            f"Only {active} active instruments, expected >= {min_names}"
        )

    def test_weight_cap_enforced(self, returns):
        """With min_names=4, no single weight should exceed 1/4 = 25%."""
        result = optimize_hedge(
            returns, "AAPL", ["MSFT", "SPY", "QQQ", "XLE"],
            "Minimum Variance", 1_000_000, (-1.0, 0.0), min_names=4,
        )
        max_abs_weight = np.max(np.abs(result.weights))
        assert max_abs_weight <= 1.0 / 4 + 1e-6, (
            f"Max |weight| = {max_abs_weight:.4f}, expected <= 0.25"
        )
