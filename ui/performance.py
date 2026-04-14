"""Price Performance tab — absolute, relative, and beta-adjusted stats."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.performance import compute_performance_stats
from ui.style import METRIC_DESCRIPTIONS, PLOTLY_LAYOUT
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
    "Excess Return (vs Index)", "Ann. Excess Return (vs Index)", "Tracking Error (vs Index)",
    "Excess Return (vs Peers)", "Ann. Excess Return (vs Peers)", "Tracking Error (vs Peers)",
    "Beta-Adj Return", "Ann. Alpha", "Residual Volatility",
}

_PLACEHOLDER = "\u2014"  # em-dash for non-applicable cells


def _build_consolidated_table(result, show_peers: bool = True) -> pd.DataFrame:
    """Merge absolute, relative, and beta-adjusted into one table.

    Rows = metrics (grouped by section), cols = tickers + benchmark + peer group.
    Non-applicable cells (e.g. benchmark's excess return vs itself) show an em-dash.
    """
    all_cols = list(result.absolute.columns)

    # Optionally strip peer group column
    if not show_peers:
        all_cols = [c for c in all_cols if c != "Peer Group"]

    # Rename relative metrics to distinguish index vs peers
    rel_bench = result.relative_bench.rename(index={
        "Excess Return": "Excess Return (vs Index)",
        "Ann. Excess Return": "Ann. Excess Return (vs Index)",
        "Tracking Error": "Tracking Error (vs Index)",
        "Information Ratio": "Information Ratio (vs Index)",
    })

    sections = [result.absolute, rel_bench, result.beta_adjusted]

    if show_peers and result.relative_peers is not None and not result.relative_peers.empty:
        rel_peers = result.relative_peers.rename(index={
            "Excess Return": "Excess Return (vs Peers)",
            "Ann. Excess Return": "Ann. Excess Return (vs Peers)",
            "Tracking Error": "Tracking Error (vs Peers)",
            "Information Ratio": "Information Ratio (vs Peers)",
        })
        # Insert peer-relative right after bench-relative
        sections = [result.absolute, rel_bench, rel_peers, result.beta_adjusted]

    # Reindex each section to all_cols (fills missing benchmark/peer cols with NaN)
    aligned = [s.reindex(columns=all_cols) for s in sections]
    merged = pd.concat(aligned)

    return merged


def _fmt_pct(val: float) -> str:
    """Format a percentage with parentheses for negatives: 12.34% or (12.34)%."""
    if val < 0:
        return f"({abs(val):.2%})"
    return f"{val:.2%}"


def _fmt_ratio(val: float) -> str:
    """Format a ratio with parentheses for negatives: 1.23 or (1.23)."""
    if val < 0:
        return f"({abs(val):.2f})"
    return f"{val:.2f}"


def _color_class(val: float) -> str:
    """Return CSS class for positive/negative coloring."""
    if val > 0:
        return "mt-pos"
    elif val < 0:
        return "mt-neg"
    return ""


def _render_performance_table(df: pd.DataFrame) -> None:
    """Render the consolidated performance table with color-coded values.

    Green for positive, red for negative. Percentages use (x.xx)% for negatives.
    """
    import html as html_mod

    rows_html = []
    for idx in df.index:
        desc = METRIC_DESCRIPTIONS.get(idx, "")
        escaped_idx = html_mod.escape(str(idx))
        escaped_desc = html_mod.escape(desc)

        cells = []
        for col in df.columns:
            val = df.loc[idx, col]
            if pd.isna(val):
                cells.append(f'<td class="mt-val">{_PLACEHOLDER}</td>')
            elif idx in _PCT_METRICS:
                cls = _color_class(val)
                cells.append(f'<td class="mt-val {cls}">{_fmt_pct(val)}</td>')
            else:
                cls = _color_class(val)
                cells.append(f'<td class="mt-val {cls}">{_fmt_ratio(val)}</td>')

        if desc:
            metric_cell = (
                f'<td class="mt-metric" title="{escaped_desc}">'
                f'<span class="mt-tip" data-tip="{escaped_desc}">{escaped_idx}</span></td>'
            )
        else:
            metric_cell = f'<td class="mt-metric">{escaped_idx}</td>'
        rows_html.append(f'<tr>{metric_cell}{"".join(cells)}</tr>')

    header_cells = "".join(
        f'<th class="mt-hdr mt-hdr-val">{html_mod.escape(str(col))}</th>'
        for col in df.columns
    )

    table_html = f"""
    <style>
    .mt-wrap {{ overflow-x:auto; border:1px solid #e2e8f0; border-radius:8px; }}
    .mt-wrap table {{ width:100%; border-collapse:collapse; font-family:Inter,system-ui,sans-serif; }}
    .mt-hdr {{ padding:8px 12px; border-bottom:2px solid #cbd5e1; font-size:13px;
               font-weight:600; color:#475569; text-align:left; }}
    .mt-hdr-val {{ text-align:right; }}
    .mt-metric {{ padding:6px 12px; border-bottom:1px solid #e2e8f0; font-weight:500;
                  font-size:13px; position:relative; }}
    .mt-val {{ padding:6px 12px; border-bottom:1px solid #e2e8f0; text-align:right; font-size:13px; }}
    .mt-pos {{ color: #16a34a; }}
    .mt-neg {{ color: #dc2626; }}
    .mt-tip {{ cursor:help; text-decoration:underline dotted #94a3b8;
               text-underline-offset:3px; position:relative; display:inline-block; }}
    .mt-tip::after {{
        content: attr(data-tip);
        position: absolute; left: 0; bottom: 100%; margin-bottom: 6px;
        background: #1e293b; color: #f8fafc; padding: 6px 10px;
        border-radius: 6px; font-size: 12px; font-weight: 400;
        line-height: 1.4; white-space: normal; width: max-content; max-width: 320px;
        opacity: 0; pointer-events: none; transition: opacity 0.15s; z-index: 1000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    .mt-tip:hover::after {{ opacity: 1; }}
    .mt-hint {{ font-size:11px; color:#94a3b8; margin-top:4px; padding-left:4px; }}
    </style>
    <div class="mt-wrap">
    <table>
        <thead><tr><th class="mt-hdr">Metric</th>{header_cells}</tr></thead>
        <tbody>{''.join(rows_html)}</tbody>
    </table>
    </div>
    <p class="mt-hint">Hover over metric names for descriptions.</p>
    """
    st.html(table_html)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _add_endpoint_label(fig: go.Figure, series: pd.Series, name: str) -> None:
    """Add a text annotation at the final point of a series."""
    if series.empty:
        return
    last_x = series.index[-1]
    last_y = float(series.iloc[-1])
    fig.add_annotation(
        x=last_x,
        y=last_y,
        text=f"  {name} {last_y:.1%}",
        showarrow=False,
        xanchor="left",
        font=dict(size=11, color="#334155"),
    )


def _cumulative_chart(
    data: pd.DataFrame,
    title: str,
    chart_tickers: list[str],
    benchmark_col: str | None = None,
    peer_col: str | None = None,
) -> go.Figure:
    """Build a cumulative return line chart with endpoint labels."""
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
            _add_endpoint_label(fig, series, tk)

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
        _add_endpoint_label(fig, series, benchmark_col)

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
        _add_endpoint_label(fig, series, peer_col)

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="Cumulative Return",
        yaxis_tickformat=".0%",
        height=380,
        # Add right margin so endpoint labels aren't clipped
        margin=dict(r=120),
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

    # --- Beta estimation controls ---
    with st.expander("Beta Settings", expanded=False):
        st.caption(
            "Beta is estimated on a separate date window (default: trailing 1 year from "
            "the performance end date). Adjust to control the lookback used for beta."
        )
        bc1, bc2 = st.columns(2)
        # Default beta window: 1 year trailing from the performance end date
        default_beta_end = end
        default_beta_start = max(
            min_date,
            (pd.Timestamp(end) - pd.Timedelta(days=365)).date()
            if hasattr(end, "year") else min_date,
        )
        with bc1:
            beta_start = st.date_input(
                "Beta start date", value=default_beta_start,
                min_value=min_date, max_value=max_date, key="perf_beta_start",
            )
        with bc2:
            beta_end = st.date_input(
                "Beta end date", value=default_beta_end,
                min_value=min_date, max_value=max_date, key="perf_beta_end",
            )

    # --- Peer Group controls ---
    stock_tickers = [t for t in params.get("stock_tickers", []) if t != benchmark]
    available_peers = [t for t in all_tickers if t != benchmark]

    with st.expander("Peer Group Settings", expanded=False):
        show_peers = st.checkbox("Show peer group results", value=False, key="perf_show_peers")
        peer_tickers: list[str] = []
        pw = None
        if show_peers:
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
            beta_start_date=beta_start,
            beta_end_date=beta_end,
        )
    except ValueError as e:
        st.error(str(e))
        return

    # --- Consolidated Performance Table ---
    st.subheader("Performance Statistics")
    consolidated = _build_consolidated_table(result, show_peers=show_peers)
    _render_performance_table(consolidated)

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
    peer_col = "Peer Group" if (show_peers and result.peer_basket_returns is not None) else None

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
