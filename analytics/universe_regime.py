"""Universe-wide regime-conditional return statistics.

Reuses ``analytics.regime.detect_regimes`` to label each historical day with a
volatility regime based on SPX, then groups every constituent's returns by that
regime label and computes per-regime stats. Surfaces names whose alpha
concentrates in (or hides during) the *current* market regime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analytics.regime import detect_regimes
from config import ANNUALIZATION_FACTOR, REGIME_LABELS


@dataclass
class RegimeConditionalRow:
    ticker: str
    regime_label: str
    avg_return_ann: float
    vol_ann: float
    sharpe: float
    vs_spx_alpha: float
    days: int


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(a.corr(b))


def regime_conditional_stats(
    prices: pd.DataFrame,
    spx_returns: pd.Series,
    regime_series: pd.Series | None = None,
    min_days_per_regime: int = 5,
    labels: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Per-ticker stats split by SPX volatility regime.

    Args:
        prices: wide-format prices for the universe.
        spx_returns: daily SPX returns. Used as the regime reference series
            (and as the alpha benchmark).
        regime_series: optional pre-computed regime classification. If
            ``None``, computed via ``detect_regimes`` on SPX with default
            settings (60d vol window, 3 quantile-based regimes).
        min_days_per_regime: drop ticker/regime cells with fewer days than
            this from the output.
        labels: optional regime label override.

    Returns:
        Long-format DataFrame with columns matching ``RegimeConditionalRow``.
    """
    if labels is None:
        labels = REGIME_LABELS

    if regime_series is None:
        spx_df = pd.DataFrame({"SPX": spx_returns})
        regime_series = detect_regimes(
            spx_df, "SPX", method="quantile", n_regimes=3, labels=labels,
        ).regime_series

    returns = prices.pct_change(fill_method=None).dropna(how="all")
    common_idx = returns.index.intersection(regime_series.index)
    if common_idx.empty:
        return pd.DataFrame(columns=[
            "ticker", "regime_label", "avg_return_ann", "vol_ann",
            "sharpe", "vs_spx_alpha", "days",
        ])
    returns = returns.loc[common_idx]
    regimes = regime_series.loc[common_idx]
    spx_aligned = spx_returns.reindex(common_idx)

    rows: list[dict] = []
    sqrt_ann = np.sqrt(ANNUALIZATION_FACTOR)
    for regime_id, label in labels.items():
        mask = regimes == regime_id
        if mask.sum() < min_days_per_regime:
            continue
        regime_rets = returns.loc[mask]
        spx_in_regime = spx_aligned.loc[mask]
        spx_mean = float(spx_in_regime.mean()) * ANNUALIZATION_FACTOR

        for ticker in regime_rets.columns:
            r = regime_rets[ticker].dropna()
            if len(r) < min_days_per_regime:
                continue
            mu = float(r.mean()) * ANNUALIZATION_FACTOR
            sigma = float(r.std()) * sqrt_ann
            sharpe = mu / sigma if sigma > 0 else float("nan")
            rows.append({
                "ticker": ticker,
                "regime_label": label,
                "avg_return_ann": mu,
                "vol_ann": sigma,
                "sharpe": sharpe,
                "vs_spx_alpha": mu - spx_mean,
                "days": int(len(r)),
            })

    return pd.DataFrame(rows)


def latest_regime_label(
    spx_returns: pd.Series,
    labels: dict[int, str] | None = None,
) -> str:
    """Convenience: detect regimes and return today's label."""
    if labels is None:
        labels = REGIME_LABELS
    spx_df = pd.DataFrame({"SPX": spx_returns})
    result = detect_regimes(
        spx_df, "SPX", method="quantile", n_regimes=3, labels=labels,
    )
    return labels.get(int(result.regime_series.iloc[-1]), "Unknown")
