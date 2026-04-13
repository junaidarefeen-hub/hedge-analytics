"""Pairs / spread analysis — cointegration, z-scores, and mean-reversion."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from analytics.correlation import rolling_correlation

logger = logging.getLogger(__name__)


@dataclass
class PairsResult:
    spread: pd.Series  # log price spread: log(A) - beta * log(B)
    zscore: pd.Series  # rolling z-score of the spread
    rolling_corr: pd.Series  # rolling correlation between A and B returns
    hedge_ratio: float  # OLS beta from log(A) ~ log(B)
    half_life: float  # mean-reversion half-life in trading days
    adf_stat: float  # ADF test statistic
    adf_pvalue: float  # approximate p-value
    current_zscore: float  # latest z-score
    zscore_percentile: float  # percentile rank of current z-score (0-100)
    mean_spread: float  # long-run mean of the spread
    spread_std: float  # long-run std of the spread
    rolling_mean: pd.Series  # for charting bands
    rolling_std: pd.Series  # for charting bands


# ---------------------------------------------------------------------------
# ADF test (lightweight, no statsmodels)
# ---------------------------------------------------------------------------

# MacKinnon critical values for ADF "constant, no trend" at common sample sizes.
# Interpolation is used for in-between sizes.
_ADF_CRITICAL = {
    # (1%, 5%, 10%) for various n
    25:  (-3.724, -2.986, -2.633),
    50:  (-3.568, -2.921, -2.599),
    100: (-3.498, -2.891, -2.583),
    250: (-3.457, -2.873, -2.573),
    500: (-3.443, -2.867, -2.570),
    1000: (-3.437, -2.864, -2.568),
}


def _adf_pvalue(stat: float, n: int) -> float:
    """Approximate ADF p-value via interpolation of critical values."""
    # Find the critical values for the closest sample size
    sizes = sorted(_ADF_CRITICAL.keys())
    if n <= sizes[0]:
        cv = _ADF_CRITICAL[sizes[0]]
    elif n >= sizes[-1]:
        cv = _ADF_CRITICAL[sizes[-1]]
    else:
        # Linear interpolation between bracketing sizes
        for i in range(len(sizes) - 1):
            if sizes[i] <= n <= sizes[i + 1]:
                lo, hi = sizes[i], sizes[i + 1]
                w = (n - lo) / (hi - lo)
                cv_lo = _ADF_CRITICAL[lo]
                cv_hi = _ADF_CRITICAL[hi]
                cv = tuple(cv_lo[j] + w * (cv_hi[j] - cv_lo[j]) for j in range(3))
                break
        else:
            cv = _ADF_CRITICAL[sizes[-1]]

    # cv = (1% critical, 5% critical, 10% critical)
    # More negative stat → more evidence of stationarity → lower p-value
    if stat <= cv[0]:
        return 0.005  # below 1% critical → p < 0.01
    elif stat <= cv[1]:
        # Interpolate between 1% and 5%
        return 0.01 + (stat - cv[0]) / (cv[1] - cv[0]) * (0.05 - 0.01)
    elif stat <= cv[2]:
        # Interpolate between 5% and 10%
        return 0.05 + (stat - cv[1]) / (cv[2] - cv[1]) * (0.10 - 0.05)
    else:
        # Above 10% critical → not significant
        # Rough extrapolation
        return min(0.10 + (stat - cv[2]) * 0.15, 1.0)


def _adf_test(series: np.ndarray) -> tuple[float, float]:
    """Run ADF test (constant, no trend, 1 lag) on a 1-D array.

    Returns (adf_statistic, approx_p_value).
    """
    y = series.copy()
    n = len(y)
    if n < 10:
        return 0.0, 1.0

    dy = np.diff(y)  # Δy_t = y_t - y_{t-1}
    # Align: dy[t] corresponds to y[t+1] - y[t]
    # Regressors: y_{t-1} (lagged level), Δy_{t-1} (lagged diff), intercept
    # We use indices [1:] for dy (to have lagged dy available)
    y_lag = y[1:-1]  # y_{t-1} for t = 2..n-1
    dy_dep = dy[1:]  # Δy_t for t = 2..n-1
    dy_lag = dy[:-1]  # Δy_{t-1} for t = 2..n-1

    T = len(dy_dep)
    X = np.column_stack([y_lag, dy_lag, np.ones(T)])

    # OLS: dy_dep = gamma * y_lag + c1 * dy_lag + const + eps
    coeffs, _, _, _ = np.linalg.lstsq(X, dy_dep, rcond=None)
    gamma = coeffs[0]

    # Residuals and standard error of gamma
    fitted = X @ coeffs
    residuals = dy_dep - fitted
    s_squared = float(residuals @ residuals) / (T - 3) if T > 3 else 1e-10

    # Covariance of coefficients
    XtX_inv = np.linalg.pinv(X.T @ X)
    se_gamma = np.sqrt(max(s_squared * XtX_inv[0, 0], 0.0))

    if se_gamma > 0:
        adf_stat = gamma / se_gamma
    else:
        adf_stat = 0.0

    pvalue = _adf_pvalue(adf_stat, T)
    return float(adf_stat), float(pvalue)


# ---------------------------------------------------------------------------
# Half-life estimation
# ---------------------------------------------------------------------------

def _estimate_half_life(spread: np.ndarray) -> float:
    """Estimate mean-reversion half-life from an AR(1) fit on the spread.

    Model: spread_t - spread_{t-1} = phi * spread_{t-1} + eps
    Half-life = -log(2) / log(1 + phi)

    Returns half-life in trading days, or inf if non-mean-reverting.
    """
    y = np.diff(spread)  # Δs_t
    x = spread[:-1]  # s_{t-1}

    # OLS: Δs = phi * s_{t-1} + intercept
    T = len(y)
    if T < 5:
        return float("inf")

    X = np.column_stack([x, np.ones(T)])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    phi = coeffs[0]

    # phi should be negative for mean reversion
    theta = 1 + phi  # AR(1) coefficient on s_t = theta * s_{t-1}
    if theta <= 0 or theta >= 1:
        return float("inf")

    return -np.log(2) / np.log(theta)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_pairs_analysis(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window: int,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
) -> PairsResult:
    """Compute pairs/spread analysis between two tickers.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily prices with DatetimeIndex and ticker columns.
    returns : pd.DataFrame
        Daily returns (for rolling correlation).
    ticker_a, ticker_b : str
        The two tickers to analyse.
    window : int
        Rolling window for z-score and correlation.
    start_date, end_date : date-like
        Date range for analysis.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Slice and align — use unique column list to avoid duplicate-column issues
    cols = list(dict.fromkeys([ticker_a, ticker_b]))
    mask = (prices.index >= start) & (prices.index <= end)
    p = prices.loc[mask, cols].dropna()

    if len(p) < window + 2:
        raise ValueError(
            f"Only {len(p)} overlapping observations — need at least {window + 2} "
            f"for a {window}-day rolling window."
        )

    # Log prices
    log_a = np.log(p[ticker_a])
    log_b = np.log(p[ticker_b])

    # Hedge ratio via OLS: log(A) = alpha + beta * log(B)
    T = len(log_a)
    X = np.column_stack([log_b.values, np.ones(T)])
    coeffs, _, _, _ = np.linalg.lstsq(X, log_a.values, rcond=None)
    hedge_ratio = float(coeffs[0])

    # Spread: log(A) - beta * log(B)
    spread = log_a - hedge_ratio * log_b
    spread.name = f"{ticker_a} / {ticker_b} spread"

    # Rolling statistics
    roll_mean = spread.rolling(window=window, min_periods=window).mean()
    roll_std = spread.rolling(window=window, min_periods=window).std()
    zscore = (spread - roll_mean) / roll_std
    zscore.name = "Z-Score"

    # Rolling correlation (on returns)
    ret_mask = (returns.index >= start) & (returns.index <= end)
    ret_sliced = returns.loc[ret_mask]
    roll_corr = rolling_correlation(ret_sliced, ticker_a, ticker_b, window)

    # Half-life
    spread_arr = spread.dropna().values
    half_life = _estimate_half_life(spread_arr)

    # ADF test
    adf_stat, adf_pvalue = _adf_test(spread_arr)

    # Current stats
    zs_clean = zscore.dropna()
    current_zs = float(zs_clean.iloc[-1]) if len(zs_clean) > 0 else 0.0
    zs_pctile = float((zs_clean < current_zs).mean() * 100) if len(zs_clean) > 1 else 50.0

    return PairsResult(
        spread=spread,
        zscore=zscore,
        rolling_corr=roll_corr,
        hedge_ratio=hedge_ratio,
        half_life=half_life,
        adf_stat=adf_stat,
        adf_pvalue=adf_pvalue,
        current_zscore=current_zs,
        zscore_percentile=zs_pctile,
        mean_spread=float(spread.mean()),
        spread_std=float(spread.std()),
        rolling_mean=roll_mean,
        rolling_std=roll_std,
    )
