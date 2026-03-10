"""Custom Hedge Analyzer — user-defined long/hedge portfolios with full analytics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analytics.backtest import (
    _compute_metrics,
    _information_ratio,
    _tracking_error,
)
from config import ANNUALIZATION_FACTOR


@dataclass
class CustomHedgeResult:
    # Portfolio composition
    long_tickers: list[str]
    long_weights: np.ndarray
    hedge_tickers: list[str]
    hedge_weights: np.ndarray
    long_notional: float
    hedge_notional: float
    hedge_ratio: float

    # Daily return series
    daily_standalone: pd.Series
    daily_hedge_basket: pd.Series
    daily_hedged: pd.Series

    # Cumulative return series
    cumulative_standalone: pd.Series
    cumulative_hedged: pd.Series

    # Rolling volatility
    rolling_vol_standalone: pd.Series
    rolling_vol_hedged: pd.Series

    # Metrics table (rows = metric names, cols = ["Standalone", "Hedged"])
    metrics: pd.DataFrame

    # Rolling correlation between long and hedge portfolios
    rolling_correlation: pd.Series
    full_period_correlation: float

    # Net portfolio beta — decomposed by constituent for a single benchmark
    # columns: Component, Beta, Eff. Hedge Ratio, Beta Contribution
    beta_table: pd.DataFrame

    # Hedge efficiency: vol reduction per unit of return sacrifice
    hedge_efficiency: float

    # Constituent P&L contributions
    constituent_contributions: pd.DataFrame

    # Rolling net beta to benchmark (None if no benchmark selected)
    rolling_net_beta: pd.Series | None


def compute_net_beta(
    returns: pd.DataFrame,
    long_tickers: list[str],
    long_weights: np.ndarray,
    hedge_tickers: list[str],
    hedge_weights: np.ndarray,
    hedge_ratio: float,
    benchmark: str,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> dict:
    """Lightweight net beta computation for live display (no full analysis).

    Returns dict with keys: long_beta, net_beta, and per-instrument entries.
    Returns None if computation fails.
    """
    long_weights = np.asarray(long_weights, dtype=float)
    hedge_weights = np.asarray(hedge_weights, dtype=float)

    all_cols = list(set(long_tickers + hedge_tickers + [benchmark]))
    missing = set(all_cols) - set(returns.columns)
    if missing:
        return None

    clean = returns[all_cols].dropna()
    if start_date is not None:
        clean = clean[clean.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        clean = clean[clean.index <= pd.Timestamp(end_date)]

    if len(clean) < 2:
        return None

    daily_standalone = clean[long_tickers].values @ long_weights
    bm_ret = clean[benchmark].values
    bm_var = float(np.var(bm_ret, ddof=1))
    if bm_var <= 0:
        return None

    long_beta = float(np.cov(daily_standalone, bm_ret, ddof=1)[0, 1] / bm_var)

    instruments = []
    running_hedge_contrib = 0.0
    for i, tk in enumerate(hedge_tickers):
        tk_ret = clean[tk].values
        tk_beta = float(np.cov(tk_ret, bm_ret, ddof=1)[0, 1] / bm_var)
        eff_ratio = hedge_ratio * hedge_weights[i]
        contribution = -eff_ratio * tk_beta
        running_hedge_contrib += contribution
        instruments.append({
            "ticker": tk,
            "beta": tk_beta,
            "eff_ratio": eff_ratio,
            "contribution": contribution,
        })

    return {
        "long_beta": long_beta,
        "net_beta": long_beta + running_hedge_contrib,
        "instruments": instruments,
    }


def run_custom_hedge_analysis(
    returns: pd.DataFrame,
    long_tickers: list[str],
    long_weights: np.ndarray,
    long_notional: float,
    hedge_tickers: list[str],
    hedge_weights: np.ndarray,
    hedge_notional: float,
    benchmarks: list[str] | None = None,
    rolling_window: int = 60,
    risk_free: float = 0.0,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> CustomHedgeResult:
    """Analyze a user-defined long portfolio hedged by a user-defined short portfolio.

    Args:
        returns: DataFrame of daily returns for all tickers.
        long_tickers: Tickers in the long portfolio.
        long_weights: Weights for the long portfolio (must sum to ~1).
        long_notional: Dollar notional of the long portfolio.
        hedge_tickers: Tickers in the hedge (short) portfolio.
        hedge_weights: Weights for the hedge portfolio (must sum to ~1).
        hedge_notional: Dollar notional of the hedge portfolio.
        benchmarks: Tickers to compute net beta against.
        rolling_window: Window for rolling vol and correlation.
        risk_free: Annual risk-free rate.
        start_date: Start of analysis window (inclusive). None = use all data.
        end_date: End of analysis window (inclusive). None = use all data.

    Returns:
        CustomHedgeResult with full analytics.
    """
    long_weights = np.asarray(long_weights, dtype=float)
    hedge_weights = np.asarray(hedge_weights, dtype=float)

    # Validate tickers exist in returns
    all_needed = set(long_tickers) | set(hedge_tickers)
    if benchmarks:
        all_needed |= set(benchmarks)
    missing = all_needed - set(returns.columns)
    if missing:
        raise ValueError(f"Missing tickers in returns data: {sorted(missing)}")

    # Align and drop NaN rows
    all_cols = list(set(long_tickers + hedge_tickers + (benchmarks or [])))
    clean = returns[all_cols].dropna()

    if start_date is not None:
        clean = clean[clean.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        clean = clean[clean.index <= pd.Timestamp(end_date)]

    if len(clean) < 2:
        raise ValueError("Not enough data points for analysis.")

    # Core portfolio return computation
    daily_standalone = pd.Series(
        clean[long_tickers].values @ long_weights,
        index=clean.index,
        name="Standalone",
    )
    daily_hedge_basket = pd.Series(
        clean[hedge_tickers].values @ hedge_weights,
        index=clean.index,
        name="Hedge Basket",
    )

    hedge_ratio = hedge_notional / long_notional if long_notional > 0 else 0.0
    daily_hedged = pd.Series(
        daily_standalone - hedge_ratio * daily_hedge_basket,
        index=clean.index,
        name="Hedged",
    )

    # Cumulative returns (growth of $1)
    cum_standalone = (1 + daily_standalone).cumprod()
    cum_hedged = (1 + daily_hedged).cumprod()

    # Rolling volatility (annualized)
    roll_vol_standalone = daily_standalone.rolling(
        window=rolling_window, min_periods=rolling_window,
    ).std() * np.sqrt(ANNUALIZATION_FACTOR)
    roll_vol_hedged = daily_hedged.rolling(
        window=rolling_window, min_periods=rolling_window,
    ).std() * np.sqrt(ANNUALIZATION_FACTOR)

    # Performance metrics
    m_standalone = _compute_metrics(daily_standalone, cum_standalone, risk_free, "Standalone")
    m_hedged = _compute_metrics(daily_hedged, cum_hedged, risk_free, "Hedged")

    te = _tracking_error(daily_hedged, daily_standalone)
    ir = _information_ratio(daily_hedged, daily_standalone)
    m_hedged["Tracking Error"] = te
    m_hedged["Information Ratio"] = ir
    m_standalone["Tracking Error"] = 0.0
    m_standalone["Information Ratio"] = 0.0

    # Vol reduction metric
    standalone_vol = m_standalone["Ann. Volatility"]
    hedged_vol = m_hedged["Ann. Volatility"]
    vol_reduction = (1 - hedged_vol / standalone_vol) if standalone_vol > 0 else 0.0
    m_standalone["Vol Reduction"] = 0.0
    m_hedged["Vol Reduction"] = vol_reduction

    metrics = pd.DataFrame({"Standalone": m_standalone, "Hedged": m_hedged})

    # Rolling correlation
    rolling_corr = daily_standalone.rolling(
        window=rolling_window, min_periods=rolling_window,
    ).corr(daily_hedge_basket)
    full_corr = float(daily_standalone.corr(daily_hedge_basket))

    # Net beta decomposition — one benchmark at a time, per-instrument breakdown.
    # With a single benchmark, beta = cov(r, bm) / var(bm). The decomposition
    # net_beta = long_beta - Σ(eff_ratio_i × beta_i) is exact because
    # portfolio beta is linear in weights regardless of inter-instrument correlation.
    beta_rows = []
    if benchmarks:
        for bm in benchmarks:
            if bm not in clean.columns:
                continue
            bm_ret = clean[bm]
            bm_var = float(bm_ret.var())
            if bm_var <= 0:
                continue

            # Long portfolio beta
            long_beta = float(daily_standalone.cov(bm_ret) / bm_var)
            beta_rows.append({
                "Component": "Long Portfolio",
                "Beta": round(long_beta, 4),
                "Eff. Hedge Ratio": "",
                "Beta Contribution": round(long_beta, 4),
            })

            # Per-instrument hedge betas
            running_hedge_contrib = 0.0
            for i, tk in enumerate(hedge_tickers):
                tk_beta = float(clean[tk].cov(bm_ret) / bm_var)
                eff_ratio = hedge_ratio * hedge_weights[i]
                contribution = -eff_ratio * tk_beta
                running_hedge_contrib += contribution
                pct_label = f"{hedge_weights[i] * 100:.0f}%"
                beta_rows.append({
                    "Component": f"{tk} short ({pct_label})",
                    "Beta": round(tk_beta, 4),
                    "Eff. Hedge Ratio": round(eff_ratio, 4),
                    "Beta Contribution": round(contribution, 4),
                })

            # Net row
            net_beta = round(long_beta + running_hedge_contrib, 4)
            beta_rows.append({
                "Component": "Net Portfolio",
                "Beta": "",
                "Eff. Hedge Ratio": "",
                "Beta Contribution": net_beta,
            })

    beta_table = pd.DataFrame(beta_rows) if beta_rows else pd.DataFrame(
        columns=["Component", "Beta", "Eff. Hedge Ratio", "Beta Contribution"],
    )

    # Rolling net beta to benchmark
    rolling_net_beta = None
    if benchmarks:
        bm = benchmarks[0]
        if bm in clean.columns:
            bm_ret = clean[bm]
            roll_bm_var = bm_ret.rolling(
                window=rolling_window, min_periods=rolling_window,
            ).var()
            roll_long_beta = daily_standalone.rolling(
                window=rolling_window, min_periods=rolling_window,
            ).cov(bm_ret) / roll_bm_var
            roll_net = roll_long_beta.copy()
            for i, tk in enumerate(hedge_tickers):
                roll_tk_beta = clean[tk].rolling(
                    window=rolling_window, min_periods=rolling_window,
                ).cov(bm_ret) / roll_bm_var
                roll_net = roll_net - (hedge_ratio * hedge_weights[i]) * roll_tk_beta
            rolling_net_beta = roll_net

    # Hedge efficiency: vol reduction % / |return sacrifice %|
    standalone_return = m_standalone["Ann. Return"]
    hedged_return = m_hedged["Ann. Return"]
    return_sacrifice = standalone_return - hedged_return
    vol_reduction_pct = vol_reduction * 100
    hedge_efficiency = (
        vol_reduction_pct / abs(return_sacrifice * 100)
        if abs(return_sacrifice) > 1e-10
        else 0.0
    )

    # Constituent P&L contributions
    contrib_data = {}
    for i, tk in enumerate(long_tickers):
        contrib_data[f"{tk} (L)"] = clean[tk] * long_weights[i]
    for i, tk in enumerate(hedge_tickers):
        contrib_data[f"{tk} (S)"] = -hedge_ratio * clean[tk] * hedge_weights[i]
    constituent_contributions = pd.DataFrame(contrib_data, index=clean.index)

    return CustomHedgeResult(
        long_tickers=long_tickers,
        long_weights=long_weights,
        hedge_tickers=hedge_tickers,
        hedge_weights=hedge_weights,
        long_notional=long_notional,
        hedge_notional=hedge_notional,
        hedge_ratio=hedge_ratio,
        daily_standalone=daily_standalone,
        daily_hedge_basket=daily_hedge_basket,
        daily_hedged=daily_hedged,
        cumulative_standalone=cum_standalone,
        cumulative_hedged=cum_hedged,
        rolling_vol_standalone=roll_vol_standalone,
        rolling_vol_hedged=roll_vol_hedged,
        metrics=metrics,
        rolling_correlation=rolling_corr,
        full_period_correlation=full_corr,
        beta_table=beta_table,
        hedge_efficiency=hedge_efficiency,
        constituent_contributions=constituent_contributions,
        rolling_net_beta=rolling_net_beta,
    )
