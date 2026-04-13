"""Price performance statistics — absolute, relative, and beta-adjusted."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import ANNUALIZATION_FACTOR

logger = logging.getLogger(__name__)

ANN = ANNUALIZATION_FACTOR  # 252


@dataclass
class PerformanceResult:
    absolute: pd.DataFrame  # rows=metrics, cols=tickers (including benchmark/peer basket)
    relative_bench: pd.DataFrame  # rows=metrics, cols=tickers (excluding benchmark)
    relative_peers: pd.DataFrame | None  # rows=metrics, cols=tickers (None if no peers)
    beta_adjusted: pd.DataFrame  # rows=metrics, cols=tickers (excluding benchmark)
    cumulative_abs: pd.DataFrame  # DatetimeIndex, cols=tickers+benchmark
    cumulative_beta_adj: pd.DataFrame  # DatetimeIndex, cols=tickers only
    peer_basket_returns: pd.Series | None  # daily returns of weighted peer basket
    benchmark: str
    peer_label: str


# ---------------------------------------------------------------------------
# Per-ticker metric helpers
# ---------------------------------------------------------------------------

def _absolute_metrics(daily: pd.Series) -> dict[str, float]:
    """Compute absolute performance metrics for a single return series."""
    n = len(daily)
    cum = (1 + daily).cumprod()
    total = float(cum.iloc[-1] - 1) if n > 0 else 0.0
    ann_ret = (1 + total) ** (ANN / n) - 1 if n > 0 else 0.0
    ann_vol = float(daily.std() * np.sqrt(ANN))

    # Use excess returns (daily - risk_free) for Sharpe/Sortino, matching backtest.py
    excess = daily  # risk_free = 0 for performance tab (consistent with DEFAULT_RISK_FREE_RATE)
    mean_d = float(excess.mean())
    std_d = float(daily.std())
    sharpe = (mean_d / std_d * np.sqrt(ANN)) if std_d > 0 else 0.0

    downside = excess[excess < 0]
    down_std = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = (mean_d / down_std * np.sqrt(ANN)) if down_std > 0 else 0.0

    # Max drawdown from high-water mark
    cum_full = (1 + daily).cumprod()
    hwm = cum_full.cummax()
    dd = (cum_full - hwm) / hwm
    max_dd = float(dd.min())

    return {
        "Total Return": total,
        "Ann. Return": ann_ret,
        "Ann. Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
    }


def _relative_metrics(ticker_daily: pd.Series, ref_daily: pd.Series) -> dict[str, float]:
    """Compute relative performance metrics (ticker vs a reference series)."""
    excess = ticker_daily - ref_daily
    n = len(excess)

    cum_excess = (1 + excess).cumprod()
    excess_total = float(cum_excess.iloc[-1] / cum_excess.iloc[0] - 1) if n > 0 else 0.0
    ann_excess = (1 + excess_total) ** (ANN / n) - 1 if n > 0 else 0.0

    te = float(excess.std() * np.sqrt(ANN))
    ir = (float(excess.mean()) / float(excess.std()) * np.sqrt(ANN)) if float(excess.std()) > 0 else 0.0

    return {
        "Excess Return": excess_total,
        "Ann. Excess Return": ann_excess,
        "Tracking Error": te,
        "Information Ratio": ir,
    }


def _beta_adjusted_metrics(
    ticker_daily: pd.Series, bench_daily: pd.Series,
) -> dict[str, float]:
    """Compute beta and beta-adjusted performance metrics."""
    bench_var = float(bench_daily.var())
    if bench_var > 0:
        beta = float(ticker_daily.cov(bench_daily) / bench_var)
    else:
        beta = 0.0

    adj = ticker_daily - beta * bench_daily
    n = len(adj)

    cum_adj = (1 + adj).cumprod()
    ba_total = float(cum_adj.iloc[-1] / cum_adj.iloc[0] - 1) if n > 0 else 0.0
    ann_alpha = (1 + ba_total) ** (ANN / n) - 1 if n > 0 else 0.0
    resid_vol = float(adj.std() * np.sqrt(ANN))

    return {
        "Beta": beta,
        "Beta-Adj Return": ba_total,
        "Ann. Alpha": ann_alpha,
        "Residual Volatility": resid_vol,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_performance_stats(
    returns: pd.DataFrame,
    tickers: list[str],
    benchmark: str,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    peer_tickers: list[str] | None = None,
    peer_weights: np.ndarray | None = None,
) -> PerformanceResult:
    """Compute absolute, relative (bench + peers), and beta-adjusted stats.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns with DatetimeIndex and ticker columns.
    tickers : list[str]
        Tickers to analyse (columns in ``returns``).
    benchmark : str
        Benchmark ticker (column in ``returns``).
    start_date, end_date : date-like
        Date range for analysis.
    peer_tickers : list[str] | None
        Optional peer group tickers for peer-relative stats.
    peer_weights : np.ndarray | None
        Decimal weights (sums to ~1.0) for peer basket. If *None*,
        equal-weight across ``peer_tickers``.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Slice to date range
    mask = (returns.index >= start) & (returns.index <= end)
    ret = returns.loc[mask].copy()

    if len(ret) < 2:
        raise ValueError(
            f"Only {len(ret)} observation(s) in the selected date range — need at least 2."
        )

    # Benchmark daily returns
    bench = ret[benchmark].dropna()

    # Build peer basket if provided
    peer_basket: pd.Series | None = None
    peer_label = ""
    has_peers = peer_tickers is not None and len(peer_tickers) > 0
    if has_peers:
        if peer_weights is None:
            pw = np.ones(len(peer_tickers)) / len(peer_tickers)
        else:
            pw = peer_weights
        peer_basket = (ret[peer_tickers] * pw).sum(axis=1)
        peer_basket.name = "Peer Group"
        # Build display label
        parts = []
        for tk, w in zip(peer_tickers, pw):
            parts.append(f"{tk} ({w:.0%})")
        peer_label = " + ".join(parts)

    # Collect per-ticker stats
    abs_cols: dict[str, dict] = {}
    rel_bench_cols: dict[str, dict] = {}
    rel_peers_cols: dict[str, dict] = {}
    beta_adj_cols: dict[str, dict] = {}
    cum_abs_series: dict[str, pd.Series] = {}
    cum_ba_series: dict[str, pd.Series] = {}

    # Include benchmark in absolute table
    bench_clean = bench.dropna()
    if len(bench_clean) >= 2:
        abs_cols[f"{benchmark} (benchmark)"] = _absolute_metrics(bench_clean)
        cum_abs_series[f"{benchmark} (benchmark)"] = (1 + bench_clean).cumprod()

    # Include peer basket in absolute table
    if has_peers and peer_basket is not None:
        pb_clean = peer_basket.dropna()
        if len(pb_clean) >= 2:
            abs_cols["Peer Group"] = _absolute_metrics(pb_clean)
            cum_abs_series["Peer Group"] = (1 + pb_clean).cumprod()

    for tk in tickers:
        if tk not in ret.columns:
            logger.warning(f"Ticker {tk} not in returns — skipping.")
            continue

        tk_daily = ret[tk].dropna()

        # Overlap with benchmark
        common_b = tk_daily.index.intersection(bench.index)
        if len(common_b) < 2:
            logger.warning(f"Ticker {tk}: <2 overlapping dates with benchmark — skipping.")
            continue

        tk_b = tk_daily.loc[common_b]
        bn_b = bench.loc[common_b]

        # Absolute
        abs_cols[tk] = _absolute_metrics(tk_b)
        cum_abs_series[tk] = (1 + tk_b).cumprod()

        # Relative vs benchmark
        rel_bench_cols[tk] = _relative_metrics(tk_b, bn_b)

        # Beta-adjusted
        ba_metrics = _beta_adjusted_metrics(tk_b, bn_b)
        beta_adj_cols[tk] = ba_metrics

        # Cumulative beta-adjusted
        beta_val = ba_metrics["Beta"]
        ba_daily = tk_b - beta_val * bn_b
        cum_ba_series[tk] = (1 + ba_daily).cumprod()

        # Relative vs peers
        if has_peers and peer_basket is not None:
            common_p = tk_daily.index.intersection(peer_basket.dropna().index)
            if len(common_p) >= 2:
                rel_peers_cols[tk] = _relative_metrics(
                    tk_daily.loc[common_p], peer_basket.loc[common_p],
                )

    # Assemble DataFrames (rows=metrics, cols=tickers)
    absolute = pd.DataFrame(abs_cols)
    relative_bench = pd.DataFrame(rel_bench_cols)
    relative_peers = pd.DataFrame(rel_peers_cols) if rel_peers_cols else None
    beta_adjusted = pd.DataFrame(beta_adj_cols)

    cumulative_abs = pd.DataFrame(cum_abs_series).sort_index()
    cumulative_beta_adj = pd.DataFrame(cum_ba_series).sort_index()

    return PerformanceResult(
        absolute=absolute,
        relative_bench=relative_bench,
        relative_peers=relative_peers,
        beta_adjusted=beta_adjusted,
        cumulative_abs=cumulative_abs,
        cumulative_beta_adj=cumulative_beta_adj,
        peer_basket_returns=peer_basket,
        benchmark=benchmark,
        peer_label=peer_label,
    )
