"""Trade Ideas tab: ranked candidates combining reversion signals and factor exposures."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from analytics.factor_monitor import compute_factor_monitor
from analytics.reversion import compute_reversion_signals
from analytics.trade_screener import (
    screen_factor_candidates,
    screen_factor_reversion,
    screen_reversion_candidates,
)
from data.market_monitor.constituents import GICS_SECTORS, get_name_map, get_sector_map
from ui.style import PLOTLY_LAYOUT


def _signal_strength_bar(candidates, title: str) -> go.Figure:
    """Horizontal bar chart of top candidates by signal strength."""
    if candidates.empty:
        return go.Figure()

    top = candidates.head(25)
    labels = top["Ticker"] if "Ticker" in top.columns else top.index

    fig = go.Figure(go.Bar(
        y=labels,
        x=top.get("Signal Strength", top.get("Composite", [0] * len(top))),
        orientation="h",
        marker_color="#2563eb",
        text=[f"{v:.1f}" for v in top.get("Signal Strength", top.get("Composite", []))],
        textposition="auto",
    ))
    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "yaxis"},
        title=title,
        xaxis_title="Signal Strength",
        height=max(350, len(top) * 22),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def render_trade_ideas_tab(prices, factor_data, mm_data_available: bool) -> None:
    """Render the Trade Ideas tab."""
    st.caption(
        "Ranked trade candidates combining reversion signals with factor exposures. "
        "Three screening modes: pure reversion, factor plays, and the combined signal."
    )

    if not mm_data_available or prices is None or prices.empty:
        st.info("No market data loaded. Click **Refresh Market Data** in the sidebar.")
        return

    sector_map = get_sector_map()
    name_map = get_name_map()

    # Compute reversion signals
    constituent_cols = [c for c in prices.columns if c != "SPX" and c in sector_map]
    constituent_prices = prices[constituent_cols]
    signals = compute_reversion_signals(constituent_prices)

    # Compute factor betas (if factor data available)
    stock_betas = None
    factor_trends = None
    if factor_data is not None:
        factor_returns = factor_data.returns
        stock_returns = constituent_prices.pct_change().dropna(how="all")
        market_returns = prices["SPX"].pct_change().dropna() if "SPX" in prices.columns else None

        if market_returns is not None:
            fm = compute_factor_monitor(factor_returns, stock_returns, market_returns)
            stock_betas = fm.stock_factor_betas
            factor_trends = fm.factor_trends

    # --- Mode selector ---
    mode = st.radio(
        "Screening Mode",
        ["Oversold Reversion", "Factor Plays", "Factor + Reversion Combo"],
        horizontal=True,
        key="mm_trade_mode",
    )

    # --- Filters ---
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        sector_filter = st.multiselect(
            "Sector Filter", GICS_SECTORS, default=[], key="mm_trade_sector",
        )
    with col_f2:
        if mode == "Oversold Reversion":
            threshold = st.slider("Max Composite Score", 5, 50, 25, key="mm_trade_threshold")
        elif mode == "Factor Plays":
            min_beta = st.slider("Min Factor Beta", 0.5, 3.0, 1.0, 0.1, key="mm_trade_min_beta")
        else:
            threshold = st.slider("Max Composite Score", 5, 50, 30, key="mm_trade_combo_thresh")

    sector_list = sector_filter if sector_filter else None

    # --- Screen candidates ---
    if mode == "Oversold Reversion":
        candidates = screen_reversion_candidates(
            signals.signals_df, sector_map, name_map,
            threshold=threshold, sector_filter=sector_list,
        )
        if candidates.empty:
            st.info("No oversold candidates found with current filters. Try raising the threshold.")
        else:
            st.subheader(f"Oversold Candidates ({len(candidates)} found)")

            col_chart, col_table = st.columns([1, 2])
            with col_chart:
                # Use composite as signal strength proxy for bar chart
                chart_df = candidates.copy()
                chart_df["Ticker"] = chart_df.index
                chart_df["Signal Strength"] = 100 - chart_df["Composite"]
                st.plotly_chart(
                    _signal_strength_bar(chart_df, "Reversion Signal Strength"),
                    use_container_width=True,
                )

            with col_table:
                format_dict = {
                    "RSI(14)": "{:.1f}",
                    "Z-Score(20d)": "{:.2f}",
                    "Z-Score(60d)": "{:.2f}",
                    "MA Dist(50d)": "{:.2%}",
                    "MA Dist(200d)": "{:.2%}",
                    "Bollinger %B": "{:.2f}",
                    "Composite": "{:.1f}",
                }
                styled = candidates.style.format(
                    {k: v for k, v in format_dict.items() if k in candidates.columns},
                    na_rep="—",
                )
                st.dataframe(styled, use_container_width=True, height=550)

    elif mode == "Factor Plays":
        if stock_betas is None or factor_trends is None:
            st.warning("Factor data required for factor plays screening. Ensure factor data is loaded.")
            return

        candidates = screen_factor_candidates(
            stock_betas, factor_trends, sector_map, name_map,
            min_beta=min_beta, sector_filter=sector_list,
        )
        if candidates.empty:
            st.info("No factor play candidates found. Try lowering the minimum beta.")
        else:
            st.subheader(f"Factor Play Candidates ({len(candidates)} found)")

            # Group by factor
            for factor in candidates["Factor"].unique():
                with st.expander(f"Factor: {factor}", expanded=True):
                    factor_df = candidates[candidates["Factor"] == factor].head(20)
                    styled = factor_df.style.format({
                        "Beta": "{:.3f}",
                        "Factor Sharpe": "{:.2f}",
                    }, na_rep="—")
                    st.dataframe(styled, use_container_width=True, hide_index=True, height=400)

    else:  # Factor + Reversion Combo
        if stock_betas is None or factor_trends is None:
            st.warning("Factor data required for combo screening.")
            return

        candidates = screen_factor_reversion(
            signals.signals_df, stock_betas, factor_trends,
            sector_map, name_map,
            reversion_threshold=threshold,
            sector_filter=sector_list,
        )
        if candidates.empty:
            st.info("No combo candidates found. These are the highest-conviction signals — try widening filters.")
        else:
            st.subheader(f"Factor + Reversion Candidates ({len(candidates)} found)")

            col_chart, col_table = st.columns([1, 2])
            with col_chart:
                st.plotly_chart(
                    _signal_strength_bar(candidates, "Combined Signal Strength"),
                    use_container_width=True,
                )

            with col_table:
                format_dict = {
                    "Reversion Score": "{:.1f}",
                    "Factor Beta": "{:.3f}",
                    "Signal Strength": "{:.1f}",
                    "RSI(14)": "{:.1f}",
                }
                styled = candidates.style.format(
                    {k: v for k, v in format_dict.items() if k in candidates.columns},
                    na_rep="—",
                )
                st.dataframe(styled, use_container_width=True, hide_index=True, height=550)
