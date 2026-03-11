"""Factor regression engine — OLS with numpy/scipy, no statsmodels."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from config import ANNUALIZATION_FACTOR, FA_SIGNIFICANCE_LEVELS


@dataclass
class OLSResult:
    regressor_names: list[str]  # ["Market", "Momentum", "Value", ...]
    alpha: float
    betas: np.ndarray  # (k,) slopes
    std_errors: np.ndarray  # (k+1,) including intercept
    t_stats: np.ndarray  # (k+1,)
    p_values: np.ndarray  # (k+1,)
    r_squared: float
    r_squared_adj: float
    f_stat: float
    f_pvalue: float
    residuals: np.ndarray  # (T,) — idiosyncratic daily returns
    fitted: np.ndarray  # (T,) — factor-explained daily returns
    n_obs: int
    df_model: int
    df_resid: int


@dataclass
class LegDecomposition:
    """Return and vol decomposition for a single portfolio leg."""
    ols: OLSResult
    table: pd.DataFrame  # regression table
    cum_factor: pd.Series
    cum_idio: pd.Series
    cum_total: pd.Series
    vol_total: float
    vol_factor: float
    vol_idio: float
    # Cumulative sum (additive, exact decomposition)
    cumsum_factor: pd.Series
    cumsum_idio: pd.Series
    cumsum_total: pd.Series
    # Daily series for vol chart
    daily_factor: pd.Series
    daily_idio: pd.Series
    daily_total: pd.Series


@dataclass
class FactorAnalyticsResult:
    long: LegDecomposition
    short: LegDecomposition | None  # None when no short basket
    combined: LegDecomposition | None  # None when no short basket
    has_short: bool
    market_index: str
    factor_names: list[str]
    # Beta heatmap: rows = active legs, cols = Market + factors
    beta_heatmap: pd.DataFrame
    dates: pd.DatetimeIndex
    p_threshold: float
    # Ordered list of active leg names for iteration
    leg_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# OLS implementation
# ---------------------------------------------------------------------------

def _ols_regression(y: np.ndarray, X: np.ndarray, regressor_names: list[str]) -> OLSResult:
    """OLS: y = alpha + X @ betas + epsilon, using numpy.linalg.lstsq."""
    T, k = X.shape

    # Design matrix with intercept
    X_full = np.column_stack([np.ones(T), X])  # (T, k+1)

    # Least squares solve
    coeffs, _, _, _ = np.linalg.lstsq(X_full, y, rcond=None)
    alpha = coeffs[0]
    betas = coeffs[1:]

    # Fitted values and residuals
    fitted = X_full @ coeffs
    residuals = y - fitted

    # Degrees of freedom
    df_model = k
    df_resid = T - k - 1

    # Sum of squares
    ss_res = float(residuals @ residuals)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    s_squared = ss_res / df_resid if df_resid > 0 else np.inf

    # Coefficient covariance (pinv for numerical stability with correlated factors)
    XtX_inv = np.linalg.pinv(X_full.T @ X_full)
    cov_beta = s_squared * XtX_inv

    # Standard errors, t-stats, p-values
    std_errors = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))
    t_stats = np.where(std_errors > 0, coeffs / std_errors, 0.0)
    if df_resid > 0:
        p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df_resid))
    else:
        p_values = np.ones_like(t_stats)

    # R-squared
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r_squared_adj = 1.0 - (1.0 - r_squared) * (T - 1) / df_resid if df_resid > 0 else 0.0

    # F-statistic
    ss_reg = ss_tot - ss_res
    if k > 0 and df_resid > 0 and ss_res > 0:
        f_stat = (ss_reg / k) / (ss_res / df_resid)
        f_pvalue = float(stats.f.sf(f_stat, k, df_resid))
    else:
        f_stat = 0.0
        f_pvalue = 1.0

    # Multicollinearity warning
    cond = np.linalg.cond(X_full)
    if cond > 1e10:
        warnings.warn(
            f"High multicollinearity detected (condition number: {cond:.0f}). "
            "Factor regression results may be unstable.",
            stacklevel=2,
        )

    return OLSResult(
        regressor_names=regressor_names,
        alpha=alpha,
        betas=betas,
        std_errors=std_errors,
        t_stats=t_stats,
        p_values=p_values,
        r_squared=r_squared,
        r_squared_adj=r_squared_adj,
        f_stat=f_stat,
        f_pvalue=f_pvalue,
        residuals=residuals,
        fitted=fitted,
        n_obs=T,
        df_model=df_model,
        df_resid=df_resid,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _significance_star(p: float) -> str:
    """Return significance stars based on p-value thresholds."""
    for threshold in sorted(FA_SIGNIFICANCE_LEVELS.keys()):
        if p < threshold:
            return FA_SIGNIFICANCE_LEVELS[threshold]
    return ""


def _build_regression_table(ols: OLSResult) -> pd.DataFrame:
    """Build a formatted regression results table from OLS output."""
    names = ["Intercept"] + ols.regressor_names
    rows = []
    for i, name in enumerate(names):
        rows.append({
            "Regressor": name,
            "Beta": ols.alpha if i == 0 else ols.betas[i - 1],
            "Std Error": ols.std_errors[i],
            "t-stat": ols.t_stats[i],
            "p-value": ols.p_values[i],
            "Sig": _significance_star(ols.p_values[i]),
        })
    return pd.DataFrame(rows)


def _build_leg(
    ols: OLSResult, y: pd.Series, dates: pd.DatetimeIndex,
) -> LegDecomposition:
    """Build decomposition for a single portfolio leg."""
    factor_daily = pd.Series(ols.fitted, index=dates)
    idio_daily = pd.Series(ols.residuals, index=dates)
    total_daily = y

    cum_factor = (1 + factor_daily).cumprod()
    cum_idio = (1 + idio_daily).cumprod()
    cum_total = (1 + total_daily).cumprod()

    cumsum_factor = factor_daily.cumsum()
    cumsum_idio = idio_daily.cumsum()
    cumsum_total = total_daily.cumsum()

    sqrt_ann = np.sqrt(ANNUALIZATION_FACTOR)
    vol_total = float(total_daily.std() * sqrt_ann)
    vol_factor = float(factor_daily.std() * sqrt_ann)
    vol_idio = float(idio_daily.std() * sqrt_ann)

    return LegDecomposition(
        ols=ols,
        table=_build_regression_table(ols),
        cum_factor=cum_factor,
        cum_idio=cum_idio,
        cum_total=cum_total,
        vol_total=vol_total,
        vol_factor=vol_factor,
        vol_idio=vol_idio,
        cumsum_factor=cumsum_factor,
        cumsum_idio=cumsum_idio,
        cumsum_total=cumsum_total,
        daily_factor=factor_daily,
        daily_idio=idio_daily,
        daily_total=total_daily,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_factor_analytics(
    long_returns: pd.Series,
    market_returns: pd.Series,
    factor_returns: pd.DataFrame,
    market_index: str,
    factor_names: list[str],
    dates: pd.DatetimeIndex,
    short_returns: pd.Series | None = None,
    combined_returns: pd.Series | None = None,
    p_threshold: float = 0.05,
) -> FactorAnalyticsResult:
    """Run factor regressions for long (and optionally short/combined) legs.

    Parameters
    ----------
    long_returns : pd.Series
        Daily returns for the long basket (aligned with dates).
    market_returns : pd.Series
        Market index daily returns (aligned with dates).
    factor_returns : pd.DataFrame
        GS factor daily returns, columns = selected factor names (aligned).
    market_index : str
        Name of the market index ticker.
    factor_names : list[str]
        Display names of the selected GS factors.
    dates : pd.DatetimeIndex
        Aligned date index.
    short_returns : pd.Series | None
        Daily returns for the short basket (None if long-only).
    combined_returns : pd.Series | None
        Daily returns for the combined basket (None if long-only).
    p_threshold : float
        Significance threshold (for UI display only).
    """
    all_regressor_names = [market_index] + factor_names
    has_short = short_returns is not None and combined_returns is not None

    # Build X matrix: [Market | F1 | F2 | ...]
    X = np.column_stack([
        market_returns.values,
        factor_returns.values,
    ])

    # Long leg (always present)
    long_ols = _ols_regression(long_returns.values, X, all_regressor_names)
    long_leg = _build_leg(long_ols, long_returns, dates)

    # Short and combined legs (optional)
    short_leg: LegDecomposition | None = None
    combined_leg: LegDecomposition | None = None
    if has_short:
        short_ols = _ols_regression(short_returns.values, X, all_regressor_names)
        short_leg = _build_leg(short_ols, short_returns, dates)
        combined_ols = _ols_regression(combined_returns.values, X, all_regressor_names)
        combined_leg = _build_leg(combined_ols, combined_returns, dates)

    # Beta heatmap: rows = active legs, cols = [Market, F1, ...]
    heatmap_data = {"Long": long_ols.betas}
    leg_names = ["Long"]
    if has_short:
        heatmap_data["Short"] = short_leg.ols.betas
        heatmap_data["Combined"] = combined_leg.ols.betas
        leg_names += ["Short", "Combined"]

    beta_heatmap = pd.DataFrame(heatmap_data, index=all_regressor_names).T

    return FactorAnalyticsResult(
        long=long_leg,
        short=short_leg,
        combined=combined_leg,
        has_short=has_short,
        market_index=market_index,
        factor_names=factor_names,
        beta_heatmap=beta_heatmap,
        dates=dates,
        p_threshold=p_threshold,
        leg_names=leg_names,
    )
