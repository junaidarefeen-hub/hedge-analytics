"""Monte Carlo simulation — forward-looking hedge efficacy analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import MC_LOSS_THRESHOLDS, MC_PERCENTILE_BANDS


@dataclass
class MonteCarloResult:
    horizon: int
    num_sims: int
    initial_value: float
    hedged_paths: np.ndarray        # (num_sims, horizon+1)
    unhedged_paths: np.ndarray      # (num_sims, horizon+1)
    hedged_final: np.ndarray        # (num_sims,)
    unhedged_final: np.ndarray      # (num_sims,)
    hedged_bands: dict[int, np.ndarray]
    unhedged_bands: dict[int, np.ndarray]
    metrics: pd.DataFrame
    target_ticker: str
    hedge_instruments: list[str]
    weights: np.ndarray
    strategy: str


def _compute_mc_metrics(
    final_returns: np.ndarray,
    confidence_levels: list[float],
    loss_thresholds: list[float],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics["Mean Return"] = float(np.mean(final_returns))
    metrics["Median Return"] = float(np.median(final_returns))
    metrics["Std Dev"] = float(np.std(final_returns))
    for cl in confidence_levels:
        losses = -final_returns
        var = float(np.percentile(losses, cl * 100))
        metrics[f"VaR {cl:.0%}"] = var
        tail = losses[losses >= var]
        metrics[f"CVaR {cl:.0%}"] = float(np.mean(tail)) if len(tail) > 0 else var
    for th in loss_thresholds:
        metrics[f"P(loss > {th:.0%})"] = float(np.mean(final_returns < -th))
    metrics["Best Case"] = float(np.max(final_returns))
    metrics["Worst Case"] = float(np.min(final_returns))
    return metrics


def run_monte_carlo(
    returns: pd.DataFrame,
    target: str,
    hedge_instruments: list[str],
    weights: np.ndarray,
    strategy: str,
    horizon: int,
    num_sims: int,
    initial_value: float,
    confidence_levels: list[float] | None = None,
    loss_thresholds: list[float] | None = None,
    seed: int | None = None,
) -> MonteCarloResult:
    """Simulate forward paths for hedged and unhedged portfolios."""
    conf_levels = confidence_levels or [0.95, 0.99]
    thresholds = loss_thresholds or MC_LOSS_THRESHOLDS

    cols = [target] + hedge_instruments
    clean = returns[cols].dropna()

    if len(clean) < 30:
        raise ValueError("Not enough data points for Monte Carlo simulation (need >= 30).")

    mu = clean.mean().values
    cov = clean.cov().values

    # Ensure covariance matrix is positive semidefinite
    min_eig = np.min(np.linalg.eigvalsh(cov))
    if min_eig < 0:
        cov -= 1.1 * min_eig * np.eye(len(cov))

    rng = np.random.default_rng(seed)
    # (num_sims, horizon, n_assets)
    simulated_returns = rng.multivariate_normal(mu, cov, size=(num_sims, horizon))

    # Unhedged: target only (index 0)
    unhedged_daily = simulated_returns[:, :, 0]

    # Hedged: target + weights @ hedges
    hedge_daily = simulated_returns[:, :, 1:]
    hedged_daily = unhedged_daily + np.einsum("ijk,k->ij", hedge_daily, weights)

    # Cumulative paths (growth of initial_value)
    unhedged_growth = np.cumprod(1 + unhedged_daily, axis=1)
    unhedged_paths = np.column_stack([np.ones(num_sims), unhedged_growth]) * initial_value

    hedged_growth = np.cumprod(1 + hedged_daily, axis=1)
    hedged_paths = np.column_stack([np.ones(num_sims), hedged_growth]) * initial_value

    # Percentile bands
    bands = MC_PERCENTILE_BANDS
    hedged_bands = {p: np.percentile(hedged_paths, p, axis=0) for p in bands}
    unhedged_bands = {p: np.percentile(unhedged_paths, p, axis=0) for p in bands}

    # Final values
    hedged_final = hedged_paths[:, -1]
    unhedged_final = unhedged_paths[:, -1]

    # Return distributions
    hedged_returns_final = hedged_final / initial_value - 1
    unhedged_returns_final = unhedged_final / initial_value - 1

    h_metrics = _compute_mc_metrics(hedged_returns_final, conf_levels, thresholds)
    u_metrics = _compute_mc_metrics(unhedged_returns_final, conf_levels, thresholds)
    metrics_df = pd.DataFrame({"Unhedged": u_metrics, "Hedged": h_metrics})

    return MonteCarloResult(
        horizon=horizon,
        num_sims=num_sims,
        initial_value=initial_value,
        hedged_paths=hedged_paths,
        unhedged_paths=unhedged_paths,
        hedged_final=hedged_final,
        unhedged_final=unhedged_final,
        hedged_bands=hedged_bands,
        unhedged_bands=unhedged_bands,
        metrics=metrics_df,
        target_ticker=target,
        hedge_instruments=hedge_instruments,
        weights=weights,
        strategy=strategy,
    )
