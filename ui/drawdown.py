"""Drawdown Analysis tab — underwater charts + worst drawdown periods."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.drawdown import compute_drawdowns
from analytics.returns import compute_returns
from ui.style import PLOTLY_LAYOUT


def _underwater_chart(underwater: pd.Series, title: str = "Underwater Chart") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=underwater.index,
        y=underwater.values,
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(220, 38, 38, 0.3)",
        line=dict(color="#dc2626", width=1.5),
        name="Drawdown",
        hovertemplate="%{x|%b %d, %Y}<br>Drawdown: %{y:.2%}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", line_width=1)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title=title,
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        height=350,
        showlegend=False,
    )
    return fig


def _drawdown_table(analysis) -> pd.DataFrame:
    """Format worst drawdown periods into a display DataFrame."""
    if not analysis.drawdown_periods:
        return pd.DataFrame()

    rows = []
    for i, p in enumerate(analysis.drawdown_periods, 1):
        rows.append({
            "#": i,
            "Start": p.start.strftime("%Y-%m-%d"),
            "Trough": p.trough.strftime("%Y-%m-%d"),
            "End": p.end.strftime("%Y-%m-%d") if p.end else "Unrecovered",
            "Max Drawdown": f"{p.max_drawdown:.2%}",
            "Duration (days)": p.duration_days,
            "Recovery (days)": p.recovery_days if p.recovery_days is not None else "N/A",
        })
    return pd.DataFrame(rows)


def render_drawdown_tab(returns: pd.DataFrame, params: dict):
    """Render the Drawdown Analysis tab."""
    all_tickers = list(returns.columns)

    mode = st.radio(
        "Drawdown mode",
        options=["Standalone (any ticker)", "Hedged vs Unhedged"],
        horizontal=True,
        key="dd_mode",
    )

    if mode == "Standalone (any ticker)":
        ticker = st.selectbox("Select ticker", options=all_tickers, index=0, key="dd_ticker")
        top_n = st.slider("Worst N drawdowns", min_value=1, max_value=10, value=5, key="dd_topn")

        # Compute cumulative returns for the selected ticker
        cum = (1 + returns[ticker]).cumprod()
        analysis = compute_drawdowns(cum, top_n=top_n)

        st.plotly_chart(
            _underwater_chart(analysis.underwater_series, f"Underwater Chart: {ticker}"),
            use_container_width=True,
        )

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Max Drawdown", f"{analysis.max_drawdown:.2%}")
        m2.metric("Avg Drawdown", f"{analysis.avg_drawdown:.2%}")
        m3.metric("Max Duration", f"{analysis.max_duration} days")
        m4.metric("Avg Duration", f"{analysis.avg_duration:.0f} days")

        # Worst drawdowns table
        if analysis.drawdown_periods:
            st.subheader(f"Worst {len(analysis.drawdown_periods)} Drawdown Periods")
            st.dataframe(_drawdown_table(analysis), use_container_width=True, hide_index=True)

    else:
        # Hedged vs Unhedged mode
        bt_result = st.session_state.get("backtest_result")
        if bt_result is None:
            st.info("Run a **Backtest** first to compare hedged vs unhedged drawdowns.")
            return

        top_n = st.slider("Worst N drawdowns", min_value=1, max_value=10, value=5, key="dd_topn_h")

        analysis_un = compute_drawdowns(bt_result.cumulative_unhedged, top_n=top_n)
        analysis_h = compute_drawdowns(bt_result.cumulative_hedged, top_n=top_n)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                _underwater_chart(analysis_un.underwater_series, "Unhedged Underwater"),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                _underwater_chart(analysis_h.underwater_series, "Hedged Underwater"),
                use_container_width=True,
            )

        # Side-by-side summary
        col1, col2 = st.columns(2)
        with col1:
            st.caption("**Unhedged**")
            st.metric("Max Drawdown", f"{analysis_un.max_drawdown:.2%}")
            st.metric("Max Duration", f"{analysis_un.max_duration} days")
            if analysis_un.drawdown_periods:
                st.dataframe(_drawdown_table(analysis_un), use_container_width=True, hide_index=True)

        with col2:
            st.caption("**Hedged**")
            st.metric("Max Drawdown", f"{analysis_h.max_drawdown:.2%}")
            st.metric("Max Duration", f"{analysis_h.max_duration} days")
            if analysis_h.drawdown_periods:
                st.dataframe(_drawdown_table(analysis_h), use_container_width=True, hide_index=True)
