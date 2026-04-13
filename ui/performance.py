"""Price Performance tab — absolute, relative, and beta-adjusted stats."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.performance import compute_performance_stats
from ui.style import PLOTLY_LAYOUT, render_metrics_table
from ui.weight_helpers import (
    handle_normalize,
    render_weight_inputs,
    sync_weights,
    weights_array,
)


# ---------------------------------------------------------------------------
# Quick-period helpers
# ---------------------------------------------------------------------------

_PERIODS = {
    "1M": timedelta(days=30),
    "3M": timedelta(days=91),
    "6M": timedelta(days=182),
    "1Y": timedelta(days=365),
}


def _apply_pending_dates() -> None:
    """Apply deferred date changes set by quick-period buttons."""
    pending_start = st.session_state.pop("perf_pending_start", None)
    pending_end = st.session_state.pop("perf_pending_end", None)
    if pending_start is not None:
        st.session_state["perf_start"] = pending_start
    if pending_end is not None:
        st.session_state["perf_end"] = pending_end


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

_PCT_METRICS = {
    "Total Return", "Ann. Return", "Ann. Volatility", "Max Drawdown",
    "Excess Return", "Ann. Excess Return", "Tracking Error",
    "Beta-Adj Return", "Ann. Alpha", "Residual Volatility",
}


def _format_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format a metrics DataFrame for display (rows=metrics, cols=tickers)."""
    fmt = df.copy()
    for col in fmt.columns:
        fmt[col] = fmt.index.map(
            lambda idx, c=col: f"{df.loc[idx, c]:.2%}"
            if idx in _PCT_METRICS
            else f"{df.loc[idx, c]:.2f}"
        )
    return fmt


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _cumulative_chart(
    data: pd.DataFrame,
    title: str,
    chart_tickers: list[str],
    benchmark_col: str | None = None,
    peer_col: str | None = None,
) -> go.Figure:
    """Build a cumulative return line chart."""
    fig = go.Figure()

    for tk in chart_tickers:
        if tk in data.columns:
            series = data[tk].dropna()
            fig.add_trace(go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=tk,
                line=dict(width=2),
                hovertemplate="%{x|%b %d, %Y}<br>%{y:.2%}<extra></extra>",
            ))

    if benchmark_col and benchmark_col in data.columns:
        series = data[benchmark_col].dropna()
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=benchmark_col,
            line=dict(width=2, dash="dash", color="#64748b"),
            hovertemplate="%{x|%b %d, %Y}<br>%{y:.2%}<extra></extra>",
        ))

    if peer_col and peer_col in data.columns:
        series = data[peer_col].dropna()
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=peer_col,
            line=dict(width=2, dash="dot", color="#ca8a04"),
            hovertemplate="%{x|%b %d, %Y}<br>%{y:.2%}<extra></extra>",
        ))

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="Cumulative Return",
        yaxis_tickformat=".0%",
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_performance_tab(returns: pd.DataFrame, params: dict) -> None:
    """Render the Price Performance tab."""
    st.caption(
        "Absolute, relative (vs benchmark and peer group), and beta-adjusted "
        "performance statistics for all loaded tickers."
    )

    # Apply any deferred date updates from quick-period buttons
    _apply_pending_dates()

    # Also apply deferred peer weight normalize
    peer_tks_current = st.session_state.get("perf_peer_prev_tickers", [])
    handle_normalize("perf_peer", peer_tks_current)

    all_tickers = params["all_tickers"]
    benchmarks = params.get("benchmarks", [])

    # --- Controls row ---
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        default_bench_idx = 0
        if benchmarks:
            for i, tk in enumerate(all_tickers):
                if tk == benchmarks[0]:
                    default_bench_idx = i
                    break
        benchmark = st.selectbox(
            "Benchmark Index",
            options=all_tickers,
            index=default_bench_idx,
            key="perf_benchmark",
        )

    min_date = returns.index.min().date()
    max_date = returns.index.max().date()

    # Clamp sidebar defaults to the actual data range (returns drops the
    # first price row via pct_change, so start_date can precede min_date).
    default_start = max(params["start_date"], min_date)
    default_end = min(params["end_date"], max_date)

    with c2:
        start = st.date_input(
            "Start date",
            value=default_start,
            min_value=min_date,
            max_value=max_date,
            key="perf_start",
        )
    with c3:
        end = st.date_input(
            "End date",
            value=default_end,
            min_value=min_date,
            max_value=max_date,
            key="perf_end",
        )

    # Quick-period buttons
    btn_cols = st.columns(6)
    ref_end = max_date
    for i, (label, delta) in enumerate(_PERIODS.items()):
        with btn_cols[i]:
            if st.button(label, key=f"perf_qp_{label}", use_container_width=True):
                st.session_state["perf_pending_start"] = ref_end - delta
                st.session_state["perf_pending_end"] = ref_end
                st.rerun()
    with btn_cols[4]:
        if st.button("YTD", key="perf_qp_ytd", use_container_width=True):
            st.session_state["perf_pending_start"] = date(ref_end.year, 1, 1)
            st.session_state["perf_pending_end"] = ref_end
            st.rerun()
    with btn_cols[5]:
        if st.button("Max", key="perf_qp_max", use_container_width=True):
            st.session_state["perf_pending_start"] = min_date
            st.session_state["perf_pending_end"] = ref_end
            st.rerun()

    # --- Peer Group controls ---
    stock_tickers = [t for t in params.get("stock_tickers", []) if t != benchmark]
    available_peers = [t for t in all_tickers if t != benchmark]

    with st.expander("Peer Group Settings", expanded=False):
        peer_tickers = st.multiselect(
            "Peer tickers",
            options=available_peers,
            default=stock_tickers[:10] if stock_tickers else available_peers[:5],
            key="perf_peers",
        )
        if peer_tickers:
            sync_weights("perf_peer", peer_tickers)
            peer_weights_raw = render_weight_inputs(
                "perf_peer", peer_tickers, "perf_peer_normalize",
            )
            pw = weights_array(peer_weights_raw, peer_tickers)
        else:
            pw = None

    # --- Determine tickers to analyse (all loaded tickers except benchmark) ---
    analyse_tickers = [t for t in all_tickers if t != benchmark]

    # --- Compute ---
    try:
        result = compute_performance_stats(
            returns,
            tickers=analyse_tickers,
            benchmark=benchmark,
            start_date=start,
            end_date=end,
            peer_tickers=peer_tickers if peer_tickers else None,
            peer_weights=pw,
        )
    except ValueError as e:
        st.error(str(e))
        return

    # --- Absolute Performance ---
    st.subheader("Absolute Performance")
    render_metrics_table(_format_table(result.absolute))

    # --- Relative vs Benchmark ---
    st.subheader(f"Relative Performance vs {benchmark}")
    if result.relative_bench.empty:
        st.info("No relative benchmark data available.")
    else:
        render_metrics_table(_format_table(result.relative_bench))

    # --- Relative vs Peers ---
    if result.relative_peers is not None and not result.relative_peers.empty:
        st.subheader("Relative Performance vs Peer Group")
        render_metrics_table(_format_table(result.relative_peers))

    # --- Beta-Adjusted ---
    st.subheader(f"Beta-Adjusted Performance vs {benchmark}")
    if result.beta_adjusted.empty:
        st.info("No beta-adjusted data available.")
    else:
        render_metrics_table(_format_table(result.beta_adjusted))

    # --- Charts ---
    st.subheader("Cumulative Performance")

    # Ticker selector for charts
    chartable = [c for c in result.cumulative_abs.columns
                 if c not in {f"{benchmark} (benchmark)", "Peer Group"}]
    default_chart = chartable[:5]
    chart_tickers = st.multiselect(
        "Tickers to chart",
        options=chartable,
        default=default_chart,
        key="perf_chart_tickers",
    )

    bench_col = f"{benchmark} (benchmark)"
    peer_col = "Peer Group" if result.peer_basket_returns is not None else None

    # Chart 1: Absolute cumulative
    st.plotly_chart(
        _cumulative_chart(
            result.cumulative_abs,
            "Cumulative Returns (Absolute)",
            chart_tickers,
            benchmark_col=bench_col,
            peer_col=peer_col,
        ),
        use_container_width=True,
    )

    # Chart 2: Beta-adjusted cumulative
    st.plotly_chart(
        _cumulative_chart(
            result.cumulative_beta_adj,
            "Cumulative Returns (Beta-Adjusted)",
            chart_tickers,
        ),
        use_container_width=True,
    )
