"""Regime detection — volatility-based regime classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

from config import ANNUALIZATION_FACTOR


@dataclass
class RegimeResult:
    regime_series: pd.Series
    labels: dict[int, str]
    rolling_vol: pd.Series
    per_regime_stats: pd.DataFrame  # rows = regimes, cols = avg_return, vol, count, pct_of_time


def detect_regimes(
    returns: pd.DataFrame,
    reference_ticker: str,
    window: int = 60,
    n_regimes: int = 3,
    method: str = "quantile",
    labels: dict[int, str] | None = None,
) -> RegimeResult:
    """Detect volatility regimes from a reference ticker's returns.

    Args:
        returns: DataFrame of daily returns.
        reference_ticker: Ticker to compute rolling vol from.
        window: Rolling window for volatility computation.
        n_regimes: Number of regimes to detect.
        method: 'quantile' or 'kmeans'.
        labels: Optional label mapping {0: 'Low', 1: 'Normal', ...}.
    """
    ref = returns[reference_ticker].dropna()
    rolling_vol = ref.rolling(window=window, min_periods=window).std() * np.sqrt(ANNUALIZATION_FACTOR)
    rolling_vol = rolling_vol.dropna()

    if len(rolling_vol) < n_regimes:
        raise ValueError(f"Not enough data points ({len(rolling_vol)}) for {n_regimes} regimes.")

    if method == "quantile":
        quantiles = np.linspace(0, 1, n_regimes + 1)[1:-1]
        thresholds = np.quantile(rolling_vol.values, quantiles)
        regime_vals = np.zeros(len(rolling_vol), dtype=int)
        for thresh in thresholds:
            regime_vals += (rolling_vol.values > thresh).astype(int)
        regime_series = pd.Series(regime_vals, index=rolling_vol.index, name="Regime")

    elif method == "kmeans":
        vol_data = rolling_vol.values.reshape(-1, 1)
        centroids, regime_vals = kmeans2(vol_data, n_regimes, minit="points")
        # Sort centroids so regime 0 = lowest vol
        order = np.argsort(centroids.flatten())
        remap = {old: new for new, old in enumerate(order)}
        regime_vals = np.array([remap[v] for v in regime_vals])
        regime_series = pd.Series(regime_vals, index=rolling_vol.index, name="Regime")
    else:
        raise ValueError(f"Unknown method: {method}")

    # Default labels
    if labels is None:
        if n_regimes == 3:
            labels = {0: "Low Vol", 1: "Normal", 2: "High Vol"}
        else:
            labels = {i: f"Regime {i}" for i in range(n_regimes)}

    # Per-regime stats
    aligned_returns = ref.reindex(rolling_vol.index)
    stats_rows = []
    for regime_id in range(n_regimes):
        mask = regime_series == regime_id
        regime_rets = aligned_returns[mask]
        regime_vols = rolling_vol[mask]
        count = int(mask.sum())
        stats_rows.append({
            "Regime": labels.get(regime_id, f"Regime {regime_id}"),
            "Avg Return (ann.)": float(regime_rets.mean() * ANNUALIZATION_FACTOR) if count > 0 else 0.0,
            "Avg Vol (ann.)": float(regime_vols.mean()) if count > 0 else 0.0,
            "Count": count,
            "% of Time": float(count / len(regime_series) * 100) if len(regime_series) > 0 else 0.0,
        })

    per_regime_stats = pd.DataFrame(stats_rows).set_index("Regime")

    return RegimeResult(
        regime_series=regime_series,
        labels=labels,
        rolling_vol=rolling_vol,
        per_regime_stats=per_regime_stats,
    )


def regime_hedge_effectiveness(
    returns: pd.DataFrame,
    target: str,
    hedges: list[str],
    weights: np.ndarray,
    regime_series: pd.Series,
    n_regimes: int = 3,
    labels: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Compute hedge effectiveness metrics per regime.

    Returns DataFrame with regime rows x metric columns.
    """
    cols = [target] + hedges
    clean = returns[cols].dropna()
    common_idx = clean.index.intersection(regime_series.index)
    clean = clean.loc[common_idx]
    regimes = regime_series.loc[common_idx]

    r_target = clean[target]
    r_hedged = r_target + (clean[hedges].values @ weights)

    if labels is None:
        labels = {i: f"Regime {i}" for i in range(n_regimes)}

    rows = []
    for regime_id in range(n_regimes):
        mask = regimes == regime_id
        if mask.sum() < 2:
            continue
        un_vol = float(r_target[mask].std() * np.sqrt(ANNUALIZATION_FACTOR))
        h_vol = float(r_hedged[mask].std() * np.sqrt(ANNUALIZATION_FACTOR))
        vol_red = (1 - h_vol / un_vol) * 100 if un_vol > 0 else 0.0
        avg_ret = float(r_hedged[mask].mean() * ANNUALIZATION_FACTOR)
        corr = float(r_target[mask].corr(pd.Series(r_hedged[mask], index=r_target[mask].index)))
        rows.append({
            "Regime": labels.get(regime_id, f"Regime {regime_id}"),
            "Unhedged Vol": un_vol,
            "Hedged Vol": h_vol,
            "Vol Reduction (%)": vol_red,
            "Avg Hedged Return (ann.)": avg_ret,
            "Correlation": corr,
            "Days": int(mask.sum()),
        })

    return pd.DataFrame(rows).set_index("Regime")
