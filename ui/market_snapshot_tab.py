"""Market Snapshot tab: daily overview of S&P 500 breadth, sector heatmap, top movers."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from analytics.market_snapshot import (
    MarketSnapshotResult,
    compute_daily_snapshot,
    compute_ma_breadth,
)
from data.market_monitor.constituents import get_name_map, get_sector_map
from ui.style import PLOTLY_LAYOUT


def _sector_treemap(snapshot: MarketSnapshotResult) -> go.Figure:
    """Build a sector performance treemap."""
    df = snapshot.sector_breadth.copy()
    if df.empty:
        return go.Figure()

    # Size by number of stocks, color by average return
    fig = go.Figure(go.Treemap(
        labels=df["Sector"],
        parents=["S&P 500"] * len(df),
        values=df["Advances"] + df["Declines"],
        text=[f"{r:.2%}" for r in df["Avg Return"]],
        textinfo="label+text",
        marker=dict(
            colors=df["Avg Return"],
            colorscale=[
                [0, "#dc2626"],    # red for negative
                [0.5, "#f8fafc"],  # neutral
                [1, "#16a34a"],    # green for positive
            ],
            cmid=0,
            colorbar=dict(title="Avg Return", tickformat=".1%"),
        ),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Avg Return: %{text}<br>"
            "Stocks: %{value}<br>"
            "Breadth: %{customdata:.1f}%"
            "<extra></extra>"
        ),
        customdata=df["Breadth %"],
    ))
    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"},
        title="Sector Performance Heatmap",
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _format_return(val: float | None) -> str:
    """Format return as colored percentage string."""
    if val is None:
        return "N/A"
    return f"{val:+.2%}"


def render_market_snapshot_tab(prices, mm_data_available: bool) -> None:
    """Render the Market Snapshot tab."""
    st.caption(
        "Daily snapshot of S&P 500: market breadth, sector performance, and top movers. "
        "Use the sidebar refresh button to update market data from RVX."
    )

    if not mm_data_available or prices is None or prices.empty:
        st.info("No market data loaded. Click **Refresh Market Data** in the sidebar to fetch S&P 500 prices from RVX.")
        return

    sector_map = get_sector_map()
    name_map = get_name_map()

    snapshot = compute_daily_snapshot(prices, sector_map, name_map)

    # --- Header metrics ---
    cols = st.columns(5)
    with cols[0]:
        spx_label = f"{snapshot.spx_level:,.0f}" if snapshot.spx_level else "N/A"
        spx_delta = _format_return(snapshot.spx_return) if snapshot.spx_return is not None else None
        st.metric("S&P 500", spx_label, delta=spx_delta)
    with cols[1]:
        adv, dec = snapshot.advance_decline
        st.metric("Advance / Decline", f"{adv} / {dec}")
    with cols[2]:
        st.metric("Breadth", f"{snapshot.breadth_pct:.1f}%")
    with cols[3]:
        highs, lows = snapshot.new_highs_lows
        st.metric("Near 52w High / Low", f"{highs} / {lows}")
    with cols[4]:
        st.metric("Snapshot Date", str(snapshot.snapshot_date))

    # --- Sector treemap ---
    st.plotly_chart(_sector_treemap(snapshot), use_container_width=True)

    # --- Top movers ---
    col_gain, col_lose = st.columns(2)

    with col_gain:
        st.subheader("Top 10 Gainers")
        gainers = snapshot.top_gainers.copy()
        gainers["1D Return"] = gainers["1D Return"].map(lambda x: f"{x:+.2%}")
        st.dataframe(gainers, use_container_width=True, hide_index=True, height=390)

    with col_lose:
        st.subheader("Top 10 Losers")
        losers = snapshot.top_losers.copy()
        losers["1D Return"] = losers["1D Return"].map(lambda x: f"{x:+.2%}")
        st.dataframe(losers, use_container_width=True, hide_index=True, height=390)

    # --- Sector returns table ---
    st.subheader("Sector Returns (Equal-Weight)")
    sector_rets = snapshot.sector_returns
    if not sector_rets.empty:
        styled = sector_rets.style.format("{:.2%}").background_gradient(
            cmap="RdYlGn", axis=None, vmin=-0.1, vmax=0.1,
        )
        st.dataframe(styled, use_container_width=True, height=440)

    # --- MA breadth ---
    st.subheader("Sector Breadth: % Above Moving Average")
    col_50, col_200 = st.columns(2)

    with col_50:
        ma_50 = compute_ma_breadth(prices, sector_map, 50)
        if not ma_50.empty:
            fig = go.Figure(go.Bar(
                y=ma_50["Sector"],
                x=ma_50["Above MA %"],
                orientation="h",
                marker_color=["#16a34a" if v > 50 else "#dc2626" for v in ma_50["Above MA %"]],
                text=[f"{v:.0f}%" for v in ma_50["Above MA %"]],
                textposition="auto",
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title="% Above 50-Day MA",
                xaxis_title="% of Stocks",
                xaxis_range=[0, 100],
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_200:
        ma_200 = compute_ma_breadth(prices, sector_map, 200)
        if not ma_200.empty:
            fig = go.Figure(go.Bar(
                y=ma_200["Sector"],
                x=ma_200["Above MA %"],
                orientation="h",
                marker_color=["#16a34a" if v > 50 else "#dc2626" for v in ma_200["Above MA %"]],
                text=[f"{v:.0f}%" for v in ma_200["Above MA %"]],
                textposition="auto",
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title="% Above 200-Day MA",
                xaxis_title="% of Stocks",
                xaxis_range=[0, 100],
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
