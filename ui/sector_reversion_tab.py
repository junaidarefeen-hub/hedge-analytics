"""Sector & Reversion tab: sector rotation, drill-down, oversold/overbought screening."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from analytics.market_snapshot import compute_multi_period_returns, compute_sector_returns
from analytics.reversion import compute_reversion_signals
from data.market_monitor.constituents import GICS_SECTORS, get_name_map, get_sector_map
from ui.style import PLOTLY_LAYOUT


def _sector_rotation_chart(sector_returns) -> go.Figure:
    """Sector rotation quadrant chart: X=1M, Y=3M relative strength."""
    if sector_returns.empty or "1M" not in sector_returns.columns or "3M" not in sector_returns.columns:
        return go.Figure()

    x = sector_returns["1M"] * 100
    y = sector_returns["3M"] * 100

    fig = go.Figure()

    # Quadrant backgrounds (via shapes)
    fig.add_shape(type="rect", x0=0, y0=0, x1=max(x.max(), 5), y1=max(y.max(), 5),
                  fillcolor="rgba(22,163,74,0.05)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=min(x.min(), -5), y0=0, x1=0, y1=max(y.max(), 5),
                  fillcolor="rgba(234,88,12,0.05)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=min(x.min(), -5), y0=min(y.min(), -5), x1=0, y1=0,
                  fillcolor="rgba(220,38,38,0.05)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, y0=min(y.min(), -5), x1=max(x.max(), 5), y1=0,
                  fillcolor="rgba(37,99,235,0.05)", line_width=0, layer="below")

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers+text",
        text=sector_returns.index,
        textposition="top center",
        textfont=dict(size=10),
        marker=dict(
            size=12,
            color=x + y,
            colorscale="RdYlGn",
            showscale=False,
        ),
        hovertemplate="<b>%{text}</b><br>1M: %{x:.1f}%<br>3M: %{y:.1f}%<extra></extra>",
    ))

    # Quadrant labels
    fig.add_annotation(x=0.98, y=0.98, text="Leading", showarrow=False,
                       xref="paper", yref="paper", font=dict(size=11, color="#16a34a"))
    fig.add_annotation(x=0.02, y=0.98, text="Weakening", showarrow=False,
                       xref="paper", yref="paper", font=dict(size=11, color="#ea580c"))
    fig.add_annotation(x=0.02, y=0.02, text="Lagging", showarrow=False,
                       xref="paper", yref="paper", font=dict(size=11, color="#dc2626"))
    fig.add_annotation(x=0.98, y=0.02, text="Improving", showarrow=False,
                       xref="paper", yref="paper", font=dict(size=11, color="#2563eb"))

    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", line_width=1)
    fig.add_vline(x=0, line_dash="dash", line_color="#94a3b8", line_width=1)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Sector Rotation (1M vs 3M Return)",
        xaxis_title="1-Month Return (%)",
        yaxis_title="3-Month Return (%)",
        height=450,
        showlegend=False,
    )
    return fig


def _sector_bar_chart(sector_returns, period: str) -> go.Figure:
    """Horizontal bar chart of sector returns for a given period."""
    if sector_returns.empty or period not in sector_returns.columns:
        return go.Figure()

    data = sector_returns[period].sort_values()

    fig = go.Figure(go.Bar(
        y=data.index,
        x=data.values * 100,
        orientation="h",
        marker_color=["#16a34a" if v > 0 else "#dc2626" for v in data.values],
        text=[f"{v:.2f}%" for v in data.values * 100],
        textposition="auto",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Sector Returns ({period})",
        xaxis_title="Return (%)",
        height=400,
    )
    return fig


def render_sector_reversion_tab(prices, mm_data_available: bool) -> None:
    """Render the Sector & Reversion tab."""
    st.caption(
        "Sector rotation analysis and oversold/overbought screening across the S&P 500. "
        "Lower composite score = more oversold."
    )

    if not mm_data_available or prices is None or prices.empty:
        st.info("No market data loaded. Click **Refresh Market Data** in the sidebar.")
        return

    sector_map = get_sector_map()
    name_map = get_name_map()

    # Exclude SPX from constituent analysis
    constituent_cols = [c for c in prices.columns if c != "SPX" and c in sector_map]
    constituent_prices = prices[constituent_cols]

    # ---- SECTOR ANALYSIS ----
    st.subheader("Sector Analysis")

    multi_period = compute_multi_period_returns(constituent_prices)
    sector_returns = compute_sector_returns(multi_period, sector_map)

    col_rotation, col_bars = st.columns([1, 1])
    with col_rotation:
        st.plotly_chart(_sector_rotation_chart(sector_returns), use_container_width=True)

    with col_bars:
        period = st.radio(
            "Period", ["1D", "1W", "1M", "3M", "YTD"],
            horizontal=True, key="mm_sector_period",
        )
        st.plotly_chart(_sector_bar_chart(sector_returns, period), use_container_width=True)

    # Sector drill-down
    with st.expander("Sector Drill-Down", expanded=False):
        selected_sector = st.selectbox(
            "Select Sector", GICS_SECTORS, key="mm_drilldown_sector",
        )
        sector_tickers = [t for t in constituent_cols if sector_map.get(t) == selected_sector]
        if sector_tickers:
            sector_mp = multi_period.loc[multi_period.index.isin(sector_tickers)].copy()
            sector_mp.insert(0, "Name", sector_mp.index.map(name_map))
            styled = sector_mp.style.format(
                {c: "{:.2%}" for c in sector_mp.columns if c != "Name"},
                na_rep="—",
            )
            st.dataframe(styled, use_container_width=True, height=400)

    # ---- REVERSION SIGNALS ----
    st.divider()
    st.subheader("Reversion Signals")

    signals = compute_reversion_signals(constituent_prices)
    signals_df = signals.signals_df.copy()
    signals_df["Sector"] = signals_df.index.map(sector_map)
    signals_df["Name"] = signals_df.index.map(name_map)

    # Filters
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    with col_filter1:
        sector_filter = st.multiselect(
            "Filter by Sector", GICS_SECTORS, default=[], key="mm_rev_sector_filter",
        )
    with col_filter2:
        sort_col = st.selectbox(
            "Sort by", ["Composite", "RSI(14)", "Z-Score(20d)", "MA Dist(50d)", "Bollinger %B"],
            key="mm_rev_sort",
        )
    with col_filter3:
        sort_dir = st.radio("Direction", ["Ascending (most oversold)", "Descending"],
                            key="mm_rev_dir", horizontal=True)

    filtered = signals_df.copy()
    if sector_filter:
        filtered = filtered[filtered["Sector"].isin(sector_filter)]

    ascending = sort_dir.startswith("Ascending")
    filtered = filtered.sort_values(sort_col, ascending=ascending)

    # Signal distribution histogram
    col_hist, col_stats = st.columns([2, 1])
    with col_hist:
        fig = go.Figure(go.Histogram(
            x=signals_df["Composite"].dropna(),
            nbinsx=50,
            marker_color="#2563eb",
            opacity=0.7,
        ))
        fig.add_vline(x=25, line_dash="dash", line_color="#dc2626",
                      annotation_text="Oversold threshold (25)")
        fig.add_vline(x=75, line_dash="dash", line_color="#16a34a",
                      annotation_text="Overbought threshold (75)")
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Composite Score Distribution",
            xaxis_title="Composite Score (0=most oversold)",
            yaxis_title="Count",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        n_oversold = (signals_df["Composite"] <= 25).sum()
        n_overbought = (signals_df["Composite"] >= 75).sum()
        n_neutral = len(signals_df) - n_oversold - n_overbought
        st.metric("Oversold (score <= 25)", n_oversold)
        st.metric("Neutral", n_neutral)
        st.metric("Overbought (score >= 75)", n_overbought)
        avg_rsi = signals_df["RSI(14)"].mean()
        st.metric("Avg RSI", f"{avg_rsi:.1f}")

    # Screener table
    display_cols = ["Name", "Sector", "RSI(14)", "Z-Score(20d)", "Z-Score(60d)",
                    "MA Dist(50d)", "MA Dist(200d)", "Bollinger %B", "Composite"]
    display = filtered[[c for c in display_cols if c in filtered.columns]].copy()

    format_dict = {
        "RSI(14)": "{:.1f}",
        "Z-Score(20d)": "{:.2f}",
        "Z-Score(60d)": "{:.2f}",
        "MA Dist(50d)": "{:.2%}",
        "MA Dist(200d)": "{:.2%}",
        "Bollinger %B": "{:.2f}",
        "Composite": "{:.1f}",
    }

    def _color_composite(val):
        if isinstance(val, (int, float)) and not np.isnan(val):
            if val <= 25:
                return "background-color: rgba(220,38,38,0.15)"
            elif val >= 75:
                return "background-color: rgba(22,163,74,0.15)"
        return ""

    styled = display.style.format(
        {k: v for k, v in format_dict.items() if k in display.columns},
        na_rep="—",
    ).map(_color_composite, subset=["Composite"] if "Composite" in display.columns else [])

    st.dataframe(styled, use_container_width=True, height=500)
