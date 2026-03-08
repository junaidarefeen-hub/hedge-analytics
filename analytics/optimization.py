"""Hedge optimization — 4 strategy solvers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import ANNUALIZATION_FACTOR


@dataclass
class HedgeResult:
    strategy: str
    target_ticker: str
    target_notional: float
    hedge_instruments: list[str]
    weights: np.ndarray
    notionals: np.ndarray
    total_hedge_notional: float
    hedged_volatility: float
    unhedged_volatility: float
    portfolio_betas: dict[str, float]
    unhedged_betas: dict[str, float]
    portfolio_correlation: float
    rolling_correlation: pd.Series
    beta_neutral_feasible: bool | None = None
    cvar: float | None = None
    confidence_level: float | None = None


def _covariance_pieces(
    returns: pd.DataFrame, target: str, hedges: list[str],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (Sigma_hh, sigma_ht, var_t) from aligned returns."""
    cols = [target] + hedges
    clean = returns[cols].dropna()
    cov = clean.cov().values
    var_t = cov[0, 0]
    sigma_ht = cov[1:, 0]
    sigma_hh = cov[1:, 1:]
    return sigma_hh, sigma_ht, var_t


def _portfolio_vol(w: np.ndarray, sigma_hh: np.ndarray, sigma_ht: np.ndarray, var_t: float) -> float:
    """Annualized vol of target + w @ hedges."""
    port_var = var_t + 2 * w @ sigma_ht + w @ sigma_hh @ w
    return float(np.sqrt(max(port_var, 0.0) * ANNUALIZATION_FACTOR))


def _min_variance_objective(w, sigma_hh, sigma_ht, var_t):
    return var_t + 2 * w @ sigma_ht + w @ sigma_hh @ w


def _multivariate_beta_matrix(
    returns: pd.DataFrame, tickers: list[str], factors: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Multivariate OLS betas on a single aligned sample.

    Regresses each ticker on all factors simultaneously, using the same
    dropna'd sample across all tickers for consistency (betas are additive).

    Returns:
        beta_matrix: shape (n_tickers, n_factors) — row i = betas of tickers[i]
        factor_returns: the aligned factor return matrix (for verification)
    """
    all_cols = list(set(tickers + factors))
    clean = returns[all_cols].dropna()
    X = clean[factors].values  # (T, n_factors)

    beta_matrix = np.zeros((len(tickers), len(factors)))
    for i, tk in enumerate(tickers):
        y = clean[tk].values
        betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        beta_matrix[i] = betas
    return beta_matrix, X


def _weight_sum_constraint(bounds: tuple[float, float]) -> dict:
    """Equality constraint: weights sum to -1 (short-only) or +1 (long-only)."""
    target_sum = -1.0 if bounds[1] <= 0 else 1.0
    return {"type": "eq", "fun": lambda w: np.sum(w) - target_sum}


def optimize_min_variance(
    returns: pd.DataFrame,
    target: str,
    hedges: list[str],
    bounds: tuple[float, float],
) -> np.ndarray:
    n = len(hedges)
    sigma_hh, sigma_ht, var_t = _covariance_pieces(returns, target, hedges)
    w0 = np.full(n, -1.0 / n) if bounds[0] < 0 else np.full(n, 1.0 / n)
    res = minimize(
        _min_variance_objective,
        w0,
        args=(sigma_hh, sigma_ht, var_t),
        method="SLSQP",
        bounds=[bounds] * n,
        constraints=[_weight_sum_constraint(bounds)],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return res.x


def optimize_beta_neutral(
    returns: pd.DataFrame,
    target: str,
    hedges: list[str],
    factors: list[str],
    bounds: tuple[float, float],
) -> np.ndarray:
    """Min-variance with equality constraints: portfolio beta = 0 for each factor.

    Uses multivariate OLS betas (all factors regressed simultaneously) on a single
    aligned sample to properly account for multicollinearity between factors.
    """
    n = len(hedges)
    valid_factors = [f for f in factors if f in returns.columns]
    sigma_hh, sigma_ht, var_t = _covariance_pieces(returns, target, hedges)
    w0 = np.full(n, -1.0 / n) if bounds[0] < 0 else np.zeros(n)

    if not valid_factors:
        return optimize_min_variance(returns, target, hedges, bounds)

    # Compute all betas on a single aligned sample
    all_tickers = [target] + hedges
    beta_matrix, _ = _multivariate_beta_matrix(returns, all_tickers, valid_factors)
    # beta_matrix[0] = target betas, beta_matrix[1:] = hedge betas
    target_betas = beta_matrix[0]           # shape (n_factors,)
    hedge_betas = beta_matrix[1:]           # shape (n_hedges, n_factors)

    # Limit number of beta constraints to n_hedges - 1 to avoid over-determination
    # (we already have the sum constraint using 1 degree of freedom)
    max_beta_constraints = n - 1
    active_factors = valid_factors[:max_beta_constraints]

    # One constraint per active factor: w @ hedge_betas_f + target_beta_f = 0
    constraints = [_weight_sum_constraint(bounds)]
    for j, factor in enumerate(active_factors):
        bt = float(target_betas[j])
        hb = hedge_betas[:, j].copy()
        constraints.append({
            "type": "eq",
            "fun": lambda w, hb=hb, bt=bt: float(w @ hb + bt),
        })

    sum_constraint = constraints[0]  # always first
    beta_constraints = constraints[1:]

    if not beta_constraints:
        # No beta constraints — fall back to min variance
        return optimize_min_variance(returns, target, hedges, bounds), True

    # Try hard constraints first (sum + all betas as equalities)
    res = minimize(
        _min_variance_objective,
        w0,
        args=(sigma_hh, sigma_ht, var_t),
        method="SLSQP",
        bounds=[bounds] * n,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    if res.success:
        return res.x, True

    # Fallback: keep sum as hard constraint, penalize beta constraints
    penalty = 1e6

    def penalized(w):
        obj = float(_min_variance_objective(w, sigma_hh, sigma_ht, var_t))
        for bc in beta_constraints:
            obj += penalty * float(bc["fun"](w)) ** 2
        return obj

    res2 = minimize(
        penalized,
        w0,
        method="SLSQP",
        bounds=[bounds] * n,
        constraints=[sum_constraint],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return res2.x, False


def optimize_cvar(
    returns: pd.DataFrame,
    target: str,
    hedges: list[str],
    bounds: tuple[float, float],
    confidence: float = 0.95,
) -> tuple[np.ndarray, float]:
    """Minimize CVaR via Rockafellar-Uryasev linearization. Returns (weights, cvar)."""
    cols = [target] + hedges
    clean = returns[cols].dropna().values
    r_target = clean[:, 0]
    r_hedges = clean[:, 1:]
    n_hedge = r_hedges.shape[1]
    T = len(r_target)

    w0_hedge = np.full(n_hedge, -1.0 / n_hedge) if bounds[0] < 0 else np.zeros(n_hedge)
    # x = [w_1..w_n, alpha]  where alpha is the VaR auxiliary variable
    x0 = np.concatenate([w0_hedge, [0.0]])

    def cvar_obj(x):
        w = x[:n_hedge]
        alpha = x[n_hedge]
        port_returns = r_target + r_hedges @ w
        losses = -port_returns - alpha
        return alpha + np.mean(np.maximum(losses, 0)) / (1 - confidence)

    target_sum = -1.0 if bounds[1] <= 0 else 1.0
    sum_constr = {"type": "eq", "fun": lambda x: np.sum(x[:n_hedge]) - target_sum}
    bnds = [bounds] * n_hedge + [(-np.inf, np.inf)]
    res = minimize(
        cvar_obj,
        x0,
        method="SLSQP",
        bounds=bnds,
        constraints=[sum_constr],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w_opt = res.x[:n_hedge]
    # Compute final CVaR
    port_returns = r_target + r_hedges @ w_opt
    var_level = np.percentile(-port_returns, confidence * 100)
    cvar = float(np.mean(-port_returns[-port_returns >= -var_level]) if np.any(-port_returns >= var_level) else var_level)
    # Simpler: recompute from losses exceeding VaR
    losses = -port_returns
    cvar = float(np.mean(losses[losses >= np.percentile(losses, confidence * 100)]))
    return w_opt, cvar


def compute_risk_parity(
    returns: pd.DataFrame,
    hedges: list[str],
    bounds: tuple[float, float],
) -> np.ndarray:
    """Inverse-volatility weighting, scaled to bounds."""
    vols = returns[hedges].std().values
    # Avoid division by zero
    vols = np.where(vols == 0, 1e-10, vols)
    raw = 1.0 / vols
    raw = raw / raw.sum()  # normalize to sum=1
    # Scale so weights sum to -1 (short-only) or +1 (long-only)
    if bounds[1] <= 0:
        weights = -raw  # sum to -1
    else:
        weights = raw   # sum to +1
    return weights


def _apply_min_names(bounds: tuple[float, float], n: int, min_names: int) -> tuple[float, float]:
    """Tighten bounds to enforce minimum number of active instruments.

    Caps max absolute weight per instrument to 1/min_names, so at least
    min_names instruments must be used to reach the sum constraint.
    """
    if min_names <= 1 or min_names > n:
        return bounds
    lb, ub = bounds
    max_abs = 1.0 / min_names
    if ub <= 0:
        # Short-only: cap lower bound at -max_abs
        return (max(lb, -max_abs), ub)
    else:
        # Long-only: cap upper bound at max_abs
        return (lb, min(ub, max_abs))


def optimize_hedge(
    returns: pd.DataFrame,
    target: str,
    hedge_instruments: list[str],
    strategy: str,
    notional: float,
    bounds: tuple[float, float],
    factors: list[str] | None = None,
    confidence: float = 0.95,
    min_names: int = 0,
    rolling_window: int = 60,
) -> HedgeResult:
    """Main entry point: run chosen strategy and return HedgeResult."""
    hedges = [h for h in hedge_instruments if h != target and h in returns.columns]
    if not hedges:
        raise ValueError("No valid hedge instruments after filtering.")

    if min_names > len(hedges):
        raise ValueError(
            f"Min names ({min_names}) exceeds available hedge instruments ({len(hedges)})."
        )

    effective_bounds = _apply_min_names(bounds, len(hedges), min_names)

    cvar_val = None
    conf_val = None
    beta_feasible = None

    if strategy == "Minimum Variance":
        weights = optimize_min_variance(returns, target, hedges, effective_bounds)
    elif strategy == "Beta-Neutral":
        sel_factors = factors or []
        weights, beta_feasible = optimize_beta_neutral(returns, target, hedges, sel_factors, effective_bounds)
    elif strategy == "Tail Risk (CVaR)":
        weights, cvar_val = optimize_cvar(returns, target, hedges, effective_bounds, confidence)
        conf_val = confidence
    elif strategy == "Risk Parity":
        weights = compute_risk_parity(returns, hedges, effective_bounds)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    notionals = weights * notional
    total_hedge = float(np.sum(np.abs(notionals)))

    sigma_hh, sigma_ht, var_t = _covariance_pieces(returns, target, hedges)
    hedged_vol = _portfolio_vol(weights, sigma_hh, sigma_ht, var_t)
    unhedged_vol = float(np.sqrt(var_t * ANNUALIZATION_FACTOR))

    # Multivariate betas vs factor/index benchmarks (single aligned sample)
    bench_tickers = factors or []
    bench_tickers = [b for b in bench_tickers if b in returns.columns and b != target]
    port_betas = {}
    unhedged_betas = {}
    if bench_tickers:
        all_tickers = [target] + hedges
        beta_mat, _ = _multivariate_beta_matrix(returns, all_tickers, bench_tickers)
        target_mv_betas = beta_mat[0]
        hedge_mv_betas = beta_mat[1:]
        for j, bm in enumerate(bench_tickers):
            unhedged_betas[bm] = round(float(target_mv_betas[j]), 4)
            weighted_hedge_beta = float(weights @ hedge_mv_betas[:, j])
            port_betas[bm] = round(float(target_mv_betas[j] + weighted_hedge_beta), 4)

    # Correlation: target vs hedge basket (using absolute weights so direction is ignored)
    clean = returns[[target] + hedges].dropna()
    r_target_s = clean[target]
    hedge_basket_s = (clean[hedges] * np.abs(weights)).sum(axis=1)
    port_corr = float(r_target_s.corr(hedge_basket_s))

    # Rolling correlation: target vs hedge basket
    rolling_corr = r_target_s.rolling(window=rolling_window, min_periods=rolling_window).corr(hedge_basket_s)
    rolling_corr.name = f"{target} vs Hedge Basket"

    return HedgeResult(
        strategy=strategy,
        target_ticker=target,
        target_notional=notional,
        hedge_instruments=hedges,
        weights=weights,
        notionals=notionals,
        total_hedge_notional=total_hedge,
        hedged_volatility=hedged_vol,
        unhedged_volatility=unhedged_vol,
        portfolio_betas=port_betas,
        unhedged_betas=unhedged_betas,
        portfolio_correlation=port_corr,
        rolling_correlation=rolling_corr,
        beta_neutral_feasible=beta_feasible,
        cvar=cvar_val,
        confidence_level=conf_val,
    )
