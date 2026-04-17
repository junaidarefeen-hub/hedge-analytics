"""Market Snapshot tab: daily overview of S&P 500 breadth, sector heatmap, top movers."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from analytics.dispersion import DispersionSnapshot, current_dispersion
from analytics.intraday_rotation import (
    derive_inputs_from_snapshot,
    rank_rotation_candidates,
)
from analytics.market_snapshot import (
    MarketSnapshotResult,
    compute_daily_snapshot,
    compute_ma_breadth,
)
from analytics.reversion import compute_reversion_signals
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


def _dispersion_sparkline(history: "pd.Series", title: str) -> go.Figure:
    """Tiny line chart of dispersion history with the latest point highlighted."""
    fig = go.Figure(go.Scatter(
        x=history.index,
        y=history.values,
        mode="lines",
        line=dict(color="#2563eb", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(37, 99, 235, 0.08)",
        hovertemplate="%{x|%b %d}<br>%{y:.4f}<extra></extra>",
        name=title,
    ))
    if len(history):
        fig.add_trace(go.Scatter(
            x=[history.index[-1]],
            y=[history.values[-1]],
            mode="markers",
            marker=dict(color="#0f172a", size=8),
            showlegend=False,
            hoverinfo="skip",
        ))
    fig.update_layout(
        font=PLOTLY_LAYOUT["font"],
        paper_bgcolor=PLOTLY_LAYOUT["paper_bgcolor"],
        plot_bgcolor=PLOTLY_LAYOUT["plot_bgcolor"],
        title=dict(text=title, font=PLOTLY_LAYOUT["title_font"]),
        margin=dict(l=40, r=10, t=36, b=24),
        height=140,
        showlegend=False,
        xaxis=dict(showticklabels=False, gridcolor="#f1f5f9", zeroline=False),
        yaxis=dict(gridcolor="#f1f5f9", zeroline=False, tickformat=".3f"),
    )
    return fig


def _render_dispersion_section(disp: DispersionSnapshot) -> None:
    """Cross-sectional + sector dispersion gauges with 1y context."""
    import pandas as pd  # local — avoids module-load overhead during tests
    import numpy as np

    if disp.history_cs.empty:
        return

    st.subheader("Dispersion Regime")
    st.caption(
        "Cross-sectional std measures how spread apart single names moved today; "
        "sector std measures how spread apart the 11 GICS sectors moved. "
        "High percentiles often precede single-name and rotation opportunities."
    )

    cols = st.columns([1, 1, 2, 2])
    with cols[0]:
        st.metric(
            "Cross-Sectional Std",
            f"{disp.cross_sectional_std:.4f}",
            delta=f"{disp.historical_pctile_cs:.0f}th pctile",
            delta_color="off",
        )
        if disp.historical_pctile_cs >= 90:
            st.markdown(
                "<span style='color:#dc2626;font-weight:600;font-size:12px;'>"
                "Top decile dispersion</span>",
                unsafe_allow_html=True,
            )
    with cols[1]:
        st.metric(
            "Sector Std",
            f"{disp.sector_std:.4f}",
            delta=f"{disp.historical_pctile_sector:.0f}th pctile",
            delta_color="off",
        )
        if disp.historical_pctile_sector >= 90:
            st.markdown(
                "<span style='color:#dc2626;font-weight:600;font-size:12px;'>"
                "Top decile rotation</span>",
                unsafe_allow_html=True,
            )
    with cols[2]:
        st.plotly_chart(
            _dispersion_sparkline(disp.history_cs, "Single-name dispersion (1y)"),
            use_container_width=True,
        )
    with cols[3]:
        st.plotly_chart(
            _dispersion_sparkline(disp.history_sector, "Sector dispersion (1y)"),
            use_container_width=True,
        )


def _format_rotation_table(df: "pd.DataFrame") -> "pd.DataFrame":
    """Pretty-print the per-row return columns as percentages."""
    out = df.copy()
    out["Stock"] = out["ticker_return_today"].map(lambda v: f"{v:+.2%}")
    out["Sector"] = out["sector_return_today"].map(lambda v: f"{v:+.2%}")
    out["vs Sector"] = out["relative_to_sector"].map(lambda v: f"{v:+.2%}")
    out["Composite"] = out["composite_score"].map(
        lambda v: "—" if v != v else f"{v:.0f}"
    )
    return out[[
        "ticker", "name", "sector", "Stock", "Sector", "vs Sector", "Composite",
    ]].rename(columns={
        "ticker": "Ticker",
        "name": "Name",
        "sector": "GICS Sector",
    })


def _render_rotation_section(prices, sector_map, name_map) -> None:
    """Lagger-in-Leader / Leader-in-Laggard ranked tables."""
    daily_returns, sector_returns_today = derive_inputs_from_snapshot(
        prices, sector_map,
    )
    if daily_returns.empty or sector_returns_today.empty:
        return

    # Composite score is needed for the lagger ranking; derive it once here so
    # the snapshot tab does not recompute it elsewhere.
    rev = compute_reversion_signals(prices)
    candidates = rank_rotation_candidates(
        daily_returns,
        sector_returns_today,
        rev.composite_score,
        sector_map,
        name_map,
    )

    st.subheader("Sector Rotation Candidates")
    st.caption(
        "**Laggers in Leading Sectors** — single names down today inside the "
        "best-performing sectors. Mean-reversion candidates with sector tailwind. "
        "**Leaders in Lagging Sectors** — names bucking a weak sector, often "
        "signalling idiosyncratic news."
    )

    laggers = candidates[candidates["signal"] == "Lagger in Leader"]
    leaders = candidates[candidates["signal"] == "Leader in Laggard"]

    col_lag, col_lead = st.columns(2)
    with col_lag:
        st.markdown("**Laggers in Leading Sectors**")
        if laggers.empty:
            st.info("No qualifying laggers today.")
        else:
            st.dataframe(
                _format_rotation_table(laggers),
                use_container_width=True,
                hide_index=True,
                height=380,
            )
    with col_lead:
        st.markdown("**Leaders in Lagging Sectors**")
        if leaders.empty:
            st.info("No qualifying leaders today.")
        else:
            st.dataframe(
                _format_rotation_table(leaders),
                use_container_width=True,
                hide_index=True,
                height=380,
            )


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

    # --- Dispersion regime + sector rotation candidates ---
    st.divider()
    disp = current_dispersion(prices, sector_map)
    _render_dispersion_section(disp)
    _render_rotation_section(prices, sector_map, name_map)
